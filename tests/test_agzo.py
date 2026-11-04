import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import DataCollatorForSeq2Seq
from dominant_layer_zo.config import DEFAULTS, validate
from dominant_layer_zo.methods.agzo import activation_basis, AGZOStep
from dominant_layer_zo.train import make_trainer, train as run_train
from dominant_layer_zo.candidates import _build_mezo_classification_dataset
from dominant_layer_zo.collators import DataCollatorWithPaddingAndNesting
from test_core import tiny, tokenizer, settings, rows
from test_resume import InterruptAfterSave


class AGZOTests(unittest.TestCase):
    def test_basis_recovers_leading_subspace(self):
        x = torch.diag(torch.tensor([10., 1., .1]))
        basis = activation_basis(x, 1, 3)
        torch.testing.assert_close(basis.T @ basis, torch.eye(1))
        self.assertGreater(abs(basis[0, 0].item()), .99999)
        basis = activation_basis(x, 9, 1)
        self.assertEqual(basis.shape, (3, 3))
        torch.testing.assert_close(basis.T @ basis, torch.eye(3))
        with self.assertRaises(FloatingPointError):
            activation_basis(x * float('nan'), 1, 1)

    def test_update_matches_quadratic_directional_derivative(self):
        torch.manual_seed(19)
        model = torch.nn.Linear(3, 2)
        x = torch.tensor([[1., 2., 3.], [4., -2., 1.]])
        target = torch.tensor([[.2, .7], [-.1, -.4]])
        stepper = AGZOStep(model, model.named_parameters(), rank=2, power_iterations=2)
        loss_fn = lambda: (model(x) - target).square().mean()
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        rng = torch.get_rng_state()
        with torch.no_grad(), stepper.collect():
            loss_fn()
        torch.manual_seed(37)
        directions = {n: stepper.noise(n, p) for n, p in model.named_parameters()}
        gradients = torch.autograd.grad(loss_fn(), tuple(model.parameters()))
        coefficient = sum((g * directions[n]).sum().item()
                          for (n, _), g in zip(model.named_parameters(), gradients))
        torch.set_rng_state(rng)
        _, info = stepper.step(loss_fn, seed=37, epsilon=.01, lr=.003)
        self.assertAlmostEqual(info['agzo_coefficient'], coefficient, places=3)
        self.assertEqual(info['agzo_forwards'], 3)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, base[n] - .003 * coefficient * directions[n],
                                       rtol=1e-5, atol=1e-6)
        self.assertFalse(stepper.bases)
        self.assertFalse(model._forward_hooks)

    def test_nonfinite_probe_removes_hooks_and_restores(self):
        model = torch.nn.Linear(3, 2)
        stepper = AGZOStep(model, model.named_parameters())
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        calls = 0

        def loss_fn():
            nonlocal calls
            calls += 1
            loss = model(torch.ones(2, 3)).square().mean()
            return loss if calls == 1 else loss * float('nan')

        with self.assertRaises(FloatingPointError):
            stepper.step(loss_fn, seed=0, epsilon=.001, lr=.01)
        self.assertFalse(model._forward_hooks)
        self.assertFalse(stepper.bases)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=1e-7)

    def test_three_model_scopes_and_precisions(self):
        for family in ('llama2', 'qwen3', 'mistral'):
            for scope in ('full',):
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(family=family, scope=scope, dtype=dtype), tempfile.TemporaryDirectory() as td:
                        c = settings(family, td, scope)
                        c['training'].update(method='agzo', scope='full', layer=None, epsilon=.01,
                                             noise='gaussian', logging_steps=1)
                        c['model']['dtype'] = str(dtype).split('.')[-1]
                        model = tiny(family).to(dtype)
                        # Use fp32 attention for fp16 probes.
                        if family == 'llama2':
                            model.config.fp32_attention_scores = True
                        tok = tokenizer()
                        trainer = make_trainer(validate(c), model, tok, rows(), rows(),
                                               DataCollatorForSeq2Seq(tok))
                        before = {n: p.detach().clone() for n, p in model.named_parameters()}
                        trainer.train()
                        rates = [row['learning_rate'] for row in trainer.state.log_history
                                 if 'learning_rate' in row]
                        self.assertTrue(rates)
                        self.assertEqual(set(rates), {c['training']['lr']})
                        changes = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
                        self.assertTrue(changes)
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss'])))
                        self.assertFalse(trainer.agzo_stepper.bases)
                        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_classification_objective(self):
        for family in ('llama2', 'qwen3', 'mistral'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as td:
                tok = tokenizer()
                data = _build_mezo_classification_dataset(
                    [{'prompt': 'This was', 'full_prompt': 'This was great'}] * 4,
                    'sst2_mezo', tok, 32)
                c = settings(family, td)
                c['data']['task'] = 'sst2'
                c['training'].update(method='agzo', scope='full', layer=None, steps=1, noise='gaussian')
                trainer = make_trainer(c, tiny(family), tok, data, data,
                                       DataCollatorWithPaddingAndNesting(tok))
                trainer.train()
                self.assertEqual(trainer.state.global_step, 1)
                self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss'])))

    def test_resume_matches_uninterrupted(self):
        for family in ('llama2', 'qwen3', 'mistral'):
            for scope in ('full',):
                with self.subTest(family=family, scope=scope), tempfile.TemporaryDirectory() as td:
                    c = settings(family, Path(td) / 'reference', scope)
                    c['training'].update(method='agzo', scope='full', layer=None, noise='gaussian')
                    tok = tokenizer()
                    collator = DataCollatorForSeq2Seq(tok)
                    expected = tiny(family)
                    make_trainer(c, expected, tok, rows(), rows(), collator).train()
                    c['output_dir'] = str(Path(td) / 'interrupted')
                    c['checkpoint'].update(strategy='steps', save_steps=1)
                    first = make_trainer(c, tiny(family), tok, rows(), rows(), collator)
                    first.add_callback(InterruptAfterSave())
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        first.train()
                    model = tiny(family)
                    resumed = make_trainer(c, model, tok, rows(), rows(), collator)
                    resumed.train(resume_from_checkpoint=str(Path(c['output_dir']) / 'checkpoint-1'))
                    for (n, p), (_, q) in zip(expected.named_parameters(), model.named_parameters()):
                        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=n)

    def test_config_rejects_unimplemented_variants(self):
        for section, key, value in [
            ('training', 'noise', 'rademacher'), ('training', 'directions', 2),
            ('training', 'non_diff', True), ('training', 'max_zo_coeff', 10),
            ('agzo', 'rank', 0), ('agzo', 'power_iterations', -1),
            ('agzo', 'estimator', 'unknown'), ('agzo', 'restoration', 'unknown'),
            ('model', 'dtype', 'bfloat16'),
        ]:
            with self.subTest(section=section, key=key):
                c = copy.deepcopy(DEFAULTS)
                c['training'].update(method='agzo', scope='full', layer=None, noise='gaussian')
                c['model']['fp32_attention_scores'] = True
                c[section][key] = value
                with self.assertRaises(ValueError):
                    validate(c)

    def test_exact_fp16_restore_zero_update_and_failure(self):
        torch.manual_seed(28)
        model = torch.nn.Linear(16, 16).half()
        inputs = torch.randn(4, 16).half()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        stepper = AGZOStep(model, model.named_parameters())
        _, info = stepper.step(lambda: model(inputs).sum() * 0, seed=7, epsilon=.001, lr=1e-6)
        self.assertEqual(info['agzo_coefficient'], 0.)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        calls = 0

        def loss():
            nonlocal calls
            calls += 1
            result = model(inputs).float().square().mean()
            return result if calls == 1 else result * float('nan')

        with self.assertRaises(FloatingPointError):
            stepper.step(loss, seed=8, epsilon=.001, lr=1e-6)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        self.assertFalse(stepper.bases)
        self.assertFalse(model._forward_hooks)

    def test_exact_restore_trainer_resume(self):
        with tempfile.TemporaryDirectory() as td:
            c = settings('qwen3', Path(td) / 'reference')
            c['training'].update(method='agzo', scope='full', layer=None, noise='gaussian')
            c['model']['dtype'] = 'float16'
            tok = tokenizer()
            collator = DataCollatorForSeq2Seq(tok)
            expected = tiny('qwen3').half()
            make_trainer(c, expected, tok, rows(), rows(), collator).train()
            c['output_dir'] = str(Path(td) / 'interrupted')
            c['checkpoint'].update(strategy='steps', save_steps=1)
            first = make_trainer(c, tiny('qwen3').half(), tok, rows(), rows(), collator)
            first.add_callback(InterruptAfterSave())
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                first.train()
            model = tiny('qwen3').half()
            resumed = make_trainer(c, model, tok, rows(), rows(), collator)
            resumed.train(resume_from_checkpoint=str(Path(c['output_dir']) / 'checkpoint-1'))
            for (n, p), (_, q) in zip(expected.named_parameters(), model.named_parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0, msg=n)

    def test_artifacts_record_method_and_cost(self):
        with tempfile.TemporaryDirectory() as td:
            c = settings('llama2', Path(td) / 'run')
            c['training'].update(method='agzo', scope='full', layer=None, noise='gaussian')
            c['evaluation']['after_training'] = True
            tok = tokenizer()
            with patch('dominant_layer_zo.train.load_model', return_value=(tiny('llama2'), tok)), \
                 patch('dominant_layer_zo.train.make_data',
                       return_value=(rows(), rows(), DataCollatorForSeq2Seq(tok), {'fixture': True})), \
                 patch('dominant_layer_zo.train.evaluate_model', return_value={'fixture_metric': 0.5}):
                run_train(c, 'cpu')
            out = Path(c['output_dir'])
            record = json.loads((out / 'run.json').read_text())
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['agzo']['training_objective_evaluations'], 6)
            self.assertEqual(record['agzo']['basis_refreshes'], 2)
            self.assertGreater(record['agzo']['invocation_training_seconds'], 0)
            self.assertTrue((out / 'downstream_eval.json').is_file())
            self.assertFalse((out / 'final').exists())


if __name__ == '__main__':
    unittest.main()
