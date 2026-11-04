"""SparseMeZO numerical contracts and integration across all supported families."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import DataCollatorForSeq2Seq
from dominant_layer_zo.config import load_config, validate
from dominant_layer_zo.models import backend
from dominant_layer_zo.sparse import (SparseMeZOConfig, configure_sparse_mezo_groups,
    build_step_masks, SparseMeZOStep, sparse_metadata)
from dominant_layer_zo.train import make_trainer
from test_core import tiny, tokenizer, rows, settings
from test_resume import InterruptAfterSave

FAMILIES = ('llama2', 'qwen3', 'mistral')


def config(family, path, scope='full'):
    c = settings(family, path, scope)
    c['training'].update(method='sparse_mezo', epsilon=2**-10, noise='gaussian')
    return validate(c)


def trainer(c, model=None):
    tok = tokenizer()
    return make_trainer(c, model if model is not None else tiny(c['model']['family']),
                        tok, rows(), rows(), DataCollatorForSeq2Seq(tok))


class SparseTests(unittest.TestCase):
    def test_fp16_all_families_scope_norm_and_unselected_coordinates(self):
        for family in FAMILIES:
            for scope in ('full',):
                for noise in ('gaussian',):
                    with self.subTest(family=family, scope=scope, noise=noise), tempfile.TemporaryDirectory() as td:
                        c=config(family,td,scope); c['model']['dtype']='float16'; c['training']['noise']=noise
                        c['training']['logging_steps']=1
                        tr=trainer(c,tiny(family).half())
                        before={n:p.detach().clone() for n,p in tr.model.named_parameters()}
                        meta=sparse_metadata(tr)
                        self.assertEqual(meta['initial_selected_entries'],tr.trainable_dim)
                        self.assertLess(meta['initial_selected_entries'],meta['eligible_entries'])
                        names={g.name for g in tr.sparse_param_groups}
                        self.assertTrue(all('norm' not in n for n in names))
                        tr.train()
                        rates = [row['learning_rate'] for row in tr.state.log_history if 'learning_rate' in row]
                        self.assertTrue(rates)
                        self.assertEqual(set(rates), {c['training']['lr']})
                        self.assertEqual(tr._skipped_zo_steps,0)
                        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in tr.model.named_parameters() if n in names))
                        for n,p in tr.model.named_parameters():
                            self.assertTrue(torch.isfinite(p).all(),n)
                            if n not in names: torch.testing.assert_close(p,before[n],rtol=0,atol=0,msg=n)
                        for g in tr.sparse_param_groups:
                            outside=before[g.name].abs()>=g.threshold
                            torch.testing.assert_close(g.param[outside],before[g.name][outside],rtol=0,atol=0,msg=g.name)

    def test_mistral_resume_preserves_thresholds_and_rng(self):
        for scope in ('full',):
            with self.subTest(scope=scope),tempfile.TemporaryDirectory() as td:
                c=config('mistral',Path(td)/'reference',scope);ref=trainer(c);ref.train()
                c['output_dir']=str(Path(td)/'interrupted');c['checkpoint'].update(strategy='steps',save_steps=1,save_total_limit=2)
                interrupted=trainer(c);interrupted.add_callback(InterruptAfterSave())
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):interrupted.train()
                resumed=trainer(c);resumed.train(resume_from_checkpoint=str(Path(c['output_dir'])/'checkpoint-1'))
                for (n,p),(_,q) in zip(ref.model.named_parameters(),resumed.model.named_parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0,msg=n)
                for a,b in zip(ref.sparse_param_groups,resumed.sparse_param_groups):torch.testing.assert_close(a.threshold,b.threshold,rtol=0,atol=0)
                self.assertEqual(ref._py_rng.getstate(),resumed._py_rng.getstate())


    def test_quadratic_derivative_and_fixed_step_mask(self):
        p = torch.nn.Parameter(torch.tensor([1., 2., 4., 8.]))
        stepper = SparseMeZOStep([('weight', p)], .75)
        initial = p.detach().clone()
        torch.manual_seed(7)
        direction = torch.randn(p.shape) * (initial.abs() < 4)
        coefficient = (initial * direction).sum().item()
        expected = initial - .01 * coefficient * direction
        _, info = stepper.step(lambda: .5 * p.square().sum(), seed=7, epsilon=.01, lr=.01)
        torch.testing.assert_close(p, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(info['zo/sparse_selected_entries'], 2)
        torch.testing.assert_close(p[2:], initial[2:], rtol=0, atol=0)

    def test_ratio_one_empty_selection_and_exclusions(self):
        for family in FAMILIES:
            with self.subTest(family=family),tempfile.TemporaryDirectory() as td:
                c=config(family,td);c['sparse'].update(selected_ratio=1.);tr=trainer(c)
                masks,count=build_step_masks(tr.sparse_param_groups)
                self.assertLess(count,sum(p.numel() for p in tr.model.parameters()))
                self.assertTrue(all(mask is not None for mask in masks.values()))
                c['sparse'].update(selected_ratio=1e-6)
                with self.assertRaisesRegex(ValueError,'no trainable coordinates'):trainer(c)

    def test_explicit_mistral_route_and_fp16_presets(self):
        root=Path(__file__).resolve().parents[1]
        for family in FAMILIES:
            for scope in ('full',):
                c=load_config(root/f'configs/{family}/sparse_mezo_{scope}.yaml')
                self.assertEqual(c['training']['method'],'sparse_mezo');self.assertEqual(c['sparse']['selected_ratio'],.25)
                self.assertEqual(c['model']['dtype'],'float16');self.assertEqual(len(c['model']['revision']),40)
                self.assertFalse(c['checkpoint']['save_final']);self.assertEqual(c['checkpoint']['strategy'],'no')
                bad=copy.deepcopy(c);bad['model']['dtype']='bfloat16'
                with self.assertRaisesRegex(ValueError,'bf16'):validate(bad)
        for method in ('fo','zo_lora','fo_lora'):
            with self.assertRaises(ValueError):backend('mistral',method+'_full')


if __name__=='__main__':unittest.main()
