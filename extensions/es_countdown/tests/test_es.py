import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from dominant_layer_zo.methods.scope import select_parameters
from extensions.es_countdown.optimizer import EvolutionStrategies, normalized_rewards
from extensions.es_countdown.countdown import score_response, load_examples, check_disjoint, validate_row
from extensions.es_countdown.config import DEFAULTS
from extensions.es_countdown.train import run, Evaluator, ValidateScores

torch.set_num_threads(1)


def tiny():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=12,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=64, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config)
    raw = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'prompt': 3}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                        eos_token='[EOS]', unk_token='[UNK]')
    return model, tokenizer


class ESTests(unittest.TestCase):
    def test_population_update_matches_independent_oracle(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                param = torch.nn.Parameter(torch.tensor([.5, -.7, 1.2], dtype=dtype))
                original = param.detach().clone()
                es = EvolutionStrategies([('weight', param)], population_size=6, sigma=.05, alpha=.03, seed=17)
                target = torch.tensor([1., 2., -1.])
                observed = []

                def reward():
                    observed.append(param.detach().clone())
                    return -(param.float() - target).square().sum().item()

                result = es.step(reward)
                noises = [es.noise('weight', param, seed) for seed in result['population_seeds']]
                rewards = np.array([-(p.float() - target).square().sum().item() for p in observed], dtype=np.float32)
                scores = (rewards - rewards.mean()) / (rewards.std(ddof=0) + 1e-8)
                expected = original.float() + .03 / 6 * sum(float(s) * z.float() for s, z in zip(scores, noises))
                torch.testing.assert_close(param, expected.to(dtype), rtol=1e-6, atol=1e-7)
                for candidate, noise in zip(observed, noises):
                    predicted = original.clone().add_(noise, alpha=.05)
                    torch.testing.assert_close(candidate, predicted, rtol=0, atol=0)
                self.assertEqual(result['objective_evaluations'], 6)
                self.assertAlmostEqual(result['actual_update_l2'],
                                       (param.float() - original.float()).norm().item(), places=6)
                self.assertEqual(result['changed_parameter_entries'], int(torch.count_nonzero(param != original)))
                np.testing.assert_array_equal(scores, result['normalized_rewards'])

    def test_constant_reward_exact_restore_fp16(self):
        torch.manual_seed(33)
        p = torch.nn.Parameter(torch.randn(257, dtype=torch.float16))
        original = p.detach().clone()
        es = EvolutionStrategies([('p', p)], population_size=16)
        for _ in range(3):
            result = es.step(lambda: 1.1)
            self.assertEqual(result['actual_update_l2'], 0.)
            self.assertEqual(result['changed_parameter_entries'], 0)
            torch.testing.assert_close(p, original, rtol=0, atol=0)
        np.testing.assert_array_equal(normalized_rewards([0., 0., 0.]), [0., 0., 0.])
        for reward in (.1, .7, 1.1):
            for size in (3, 7, 16, 30):
                np.testing.assert_array_equal(normalized_rewards([reward] * size), np.zeros(size))

    def test_failure_rolls_back_all_parameters(self):
        for reason in ('exception', 'nan', 'overflow'):
            with self.subTest(reason=reason):
                p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float16))
                q = torch.nn.Parameter(torch.tensor([3., 4.], dtype=torch.float16))
                originals = [p.detach().clone(), q.detach().clone()]
                es = EvolutionStrategies([('p', p), ('q', q)], population_size=4,
                                         alpha=1e10 if reason == 'overflow' else .0005)
                calls = 0

                def reward():
                    nonlocal calls
                    calls += 1
                    if calls == 2 and reason == 'exception':
                        raise RuntimeError('failed candidate')
                    return float('nan') if reason == 'nan' else float(calls)

                with self.assertRaises((RuntimeError, FloatingPointError)):
                    es.step(reward)
                for current, old in zip((p, q), originals):
                    torch.testing.assert_close(current, old, rtol=0, atol=0)

    def test_scope_freezes_other_layers_and_noise_is_independent(self):
        model, _ = tiny()
        params = select_parameters(model, 'layer', 0)
        self.assertTrue(all(n.startswith('model.layers.0.') for n, _ in params))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        es = EvolutionStrategies(params, population_size=4, sigma=.05, alpha=.1)
        es.step(lambda: sum(float(p.float().sum()) for _, p in params))
        changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
        self.assertTrue(changed)
        self.assertTrue(all(n.startswith('model.layers.0.') for n in changed))
        p = torch.zeros(50)
        self.assertFalse(torch.equal(es.noise('a', p, 42), es.noise('b', p, 42)))
        torch.testing.assert_close(es.noise('a', p, 42), es.noise('a', p, 42), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            select_parameters(model, 'layer', 34)
        with self.assertRaises(ValueError):
            select_parameters(model.bfloat16(), 'full')

    def test_countdown_correctness_and_format(self):
        good = 'Work</think>\n<answer>(8 - 2) * 3</answer>'
        self.assertEqual(score_response(good, [2, 3, 8], 18),
                         {'correct': True, 'format_reward': 1., 'reward': 1.1})
        self.assertAlmostEqual(score_response('<answer>1 + 2</answer>', [1, 2], 3)['reward'], 1.05)
        self.assertTrue(score_response('<answer>2+2</answer>', [2, 2], 4)['correct'])
        self.assertTrue(score_response('<answer>2+3</answer><answer>2*3</answer>', [2, 3], 6)['correct'])
        self.assertFalse(score_response('<answer>2*3</answer><answer>2+3</answer>', [2, 3], 6)['correct'])
        for expression, numbers, target in [
            ('2+2', [2, 3], 4), ('8**2', [8, 2], 64), ('8//3', [8, 3], 2),
            ('1/0', [1, 0], 1), ('__import__("os")', [1], 1),
            ('1+2+3', [1, 2], 6), ('1e2', [1, 2], 100), ('1.2', [1, 2], 1),
        ]:
            self.assertFalse(score_response(f'<answer>{expression}</answer>', numbers, target)['correct'])

    def test_data_guard_detects_reordered_puzzle_leakage(self):
        a = {'prompt': 'prompt', 'numbers': [1, 2], 'target': 3}
        b = {'prompt': 'different wording', 'numbers': [2, 1], 'target': 3}
        with self.assertRaises(ValueError):
            check_disjoint([a], [b])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'data.jsonl'
            path.write_text(json.dumps(a) + '\n')
            data, meta = load_examples(path)
            self.assertEqual(data, [a])
            self.assertEqual(meta['examples'], 1)

    def test_signed_and_fractional_targets(self):
        row = {'context': 'prompt', 'numbers': [3, 4], 'target': '-0.75'}
        normalized = validate_row(row)
        self.assertEqual(normalized['target'], -.75)
        self.assertTrue(score_response('<answer>-3/4</answer>', [3, 4], normalized['target'])['correct'])
        for target in ('nan', 'inf', True, None):
            with self.assertRaises(ValueError):
                validate_row({**row, 'target': target})

    def test_tiny_runner_all_artifacts_and_collision_guard(self):
        for condition in ('base', 'full', 'layer0'):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                train = [{'prompt': 'prompt', 'numbers': [1, 2], 'target': 3}]
                test = [{'prompt': 'prompt', 'numbers': [4, 5], 'target': 9},
                        {'prompt': 'prompt', 'numbers': [6, 7], 'target': 13}]
                for split, rows in [('train', train), ('test', test)]:
                    (root / f'{split}.json').write_text(json.dumps(rows))
                c = copy.deepcopy(DEFAULTS)
                c['data'].update(train=str(root / 'train.json'), test=str(root / 'test.json'),
                                 expected_test_examples=2)
                c['es'].update(iterations=1, population_size=2, prompts_per_member=1)
                c['generation'].update(batch_size=2, max_new_tokens=2)
                c['output_root'] = str(root / 'runs')
                c['model']['dtype'] = 'float32'
                model, tokenizer = tiny()
                with patch('extensions.es_countdown.train.AutoModelForCausalLM.from_pretrained', return_value=model), \
                     patch('extensions.es_countdown.train.AutoTokenizer.from_pretrained', return_value=tokenizer):
                    metrics = run(c, condition, 'cpu')
                out = Path(c['output_root']) / condition
                record = json.loads((out / 'run.json').read_text())
                self.assertEqual(record['status'], 'complete')
                self.assertEqual(metrics['examples'], 2)
                self.assertEqual(record['objective_evaluations'], 0 if condition == 'base' else 2)
                self.assertEqual(len(json.loads((out / 'predictions.json').read_text())), 2)
                self.assertTrue((out / 'downstream_eval.json').is_file())
                if condition != 'base':
                    self.assertEqual(len((out / 'history.jsonl').read_text().splitlines()), 1)
                    self.assertEqual(len((out / 'population.jsonl').read_text().splitlines()), 2)
                self.assertFalse((out / 'final').exists())
                with self.assertRaises(FileExistsError):
                    run(c, condition, 'cpu')

    def test_generation_scores_only_new_tokens(self):
        model, tok = tiny()
        config = copy.deepcopy(DEFAULTS['generation'])
        config['max_new_tokens'] = 2
        evaluator = Evaluator(model, tok, config)
        # The real decoder is replaced only to make a valid answer available in a tiny vocabulary.
        with patch.object(tok, 'batch_decode', return_value=['ok</think>\n<answer>1+2</answer>']) as decoder:
            metrics, _ = evaluator.evaluate([{'prompt': 'prompt prompt', 'numbers': [1, 2], 'target': 3}])
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertLessEqual(decoder.call_args.args[0].shape[1], 2)

    def test_fixed_dev_does_not_change_training_updates(self):
        class DeterministicEvaluator(Evaluator):
            def evaluate(self, rows):
                self.examples_processed += len(rows)
                self.generation_calls += 1
                reward = float(self.model.model.layers[0].self_attn.q_proj.weight.flatten()[0])
                return {'accuracy': 0., 'mean_reward': reward, 'examples': len(rows)}, [
                    {'response': 'fixture', 'reward': reward} for _ in rows]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for split, numbers in [('train', [1, 2]), ('dev', [2, 3]), ('test', [4, 5])]:
                (root / (split + '.json')).write_text(json.dumps([
                    {'prompt': 'prompt', 'numbers': numbers, 'target': sum(numbers)}]))
            c = copy.deepcopy(DEFAULTS)
            c['data'].update(train=str(root / 'train.json'), test=str(root / 'test.json'),
                             expected_test_examples=1)
            c['es'].update(iterations=2, population_size=2, prompts_per_member=1, sigma=.05, alpha=.1)
            c['model']['dtype'] = 'float32'
            final_parameters = []
            for enabled in (False, True):
                c['output_root'] = str(root / str(enabled))
                c['data']['dev'] = str(root / 'dev.json') if enabled else None
                c['evaluation']['dev_every'] = 1 if enabled else 0
                model, tok = tiny()
                with patch('extensions.es_countdown.train.AutoModelForCausalLM.from_pretrained', return_value=model), \
                     patch('extensions.es_countdown.train.AutoTokenizer.from_pretrained', return_value=tok), \
                     patch('extensions.es_countdown.train.Evaluator', DeterministicEvaluator):
                    run(c, 'layer0', 'cpu')
                final_parameters.append({n: p.detach().clone() for n, p in model.named_parameters()})
                out = Path(c['output_root']) / 'layer0'
                history = [json.loads(x) for x in (out / 'history.jsonl').read_text().splitlines()]
                self.assertTrue(all(h['changed_parameter_entries'] > 0 for h in history))
                if enabled:
                    dev = [json.loads(x) for x in (out / 'dev_history.jsonl').read_text().splitlines()]
                    self.assertEqual([x['iteration'] for x in dev], [0, 1, 2])
                    self.assertTrue(all(x['examples'] == 1 for x in dev))
                else:
                    self.assertFalse((out / 'dev_history.jsonl').exists())
            for name in final_parameters[0]:
                torch.testing.assert_close(final_parameters[0][name], final_parameters[1][name], rtol=0, atol=0)

            c['output_root'] = str(root / 'leak')
            c['data']['dev'] = c['data']['train']
            with patch('extensions.es_countdown.train.AutoModelForCausalLM.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'overlapping'):
                    run(c, 'layer0', 'cpu')
                loader.assert_not_called()
            record = json.loads((Path(c['output_root']) / 'layer0/run.json').read_text())
            self.assertEqual(record['status'], 'failed')

    def test_nonfinite_generation_fails_instead_of_becoming_a_reward(self):
        processor = ValidateScores()
        ids, logits = torch.ones(1, 2, dtype=torch.long), torch.tensor([[.1, .2]])
        self.assertIs(processor(ids, logits), logits)
        masked = torch.tensor([[-float('inf'), .2]])
        self.assertIs(processor(ids, masked), masked)
        for value in (float('nan'), float('inf')):
            with self.assertRaises(FloatingPointError):
                processor(ids, torch.tensor([[value, .2]]))
        with self.assertRaises(FloatingPointError):
            processor(ids, torch.full((1, 2), -float('inf')))


if __name__ == '__main__':
    unittest.main()
