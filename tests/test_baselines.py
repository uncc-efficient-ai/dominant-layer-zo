import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import DataCollatorForSeq2Seq
from dominant_layer_zo.adapters import attach_adapter, restore_adapter, save_adapter, load_checkpoint
from dominant_layer_zo.config import validate
from dominant_layer_zo.train import make_trainer
from test_core import tokenizer, tiny, settings, rows
from test_resume import InterruptAfterSave

METHODS = ('fo', 'fo_lora', 'zo_lora', 'sparse_mezo')


def setup(family, path, method, scope='layer'):
    c = settings(family,path,scope)
    c['training']['method'] = method
    if method == 'sparse_mezo': c['training'].update(noise='gaussian', scope='full', layer=None)
    c['lora'].update(rank=2, alpha=4.)
    return validate(c)


def trainer(c, model=None):
    tok=tokenizer()
    return make_trainer(c, model if model is not None else tiny(c['model']['family']), tok,
                        rows(), rows(), DataCollatorForSeq2Seq(tok))


class BaselineTests(unittest.TestCase):
    def test_methods_train_only_requested_scope(self):
        for family in ('llama2','qwen3'):
            for method in METHODS:
                for scope in (('full',) if method == 'sparse_mezo' else ('layer','full')):
                    with self.subTest(family=family,method=method,scope=scope), tempfile.TemporaryDirectory() as td:
                        c=setup(family,td,method,scope); tr=trainer(c)
                        before={n:p.detach().clone() for n,p in tr.model.named_parameters()}
                        names={n for n,p in tr.model.named_parameters() if p.requires_grad}
                        self.assertTrue(names)
                        if scope=='layer': self.assertTrue(all(n.startswith('model.layers.1.') for n in names))
                        if method.endswith('_lora'): self.assertTrue(all('lora_' in n for n in names))
                        tr.train()
                        self.assertTrue(torch.isfinite(torch.tensor(tr.evaluate()['eval_loss'])))
                        changed=set()
                        for n,p in tr.model.named_parameters():
                            if not torch.equal(p,before[n]): changed.add(n)
                            if n not in names: torch.testing.assert_close(p,before[n],rtol=0,atol=0,msg=n)
                        self.assertTrue(changed)
                        if method=='sparse_mezo':
                            self.assertTrue(all('norm' not in n for n in names))
                            for group in tr.sparse_param_groups:
                                # Coordinates above the fixed threshold cannot change at any step.
                                outside=before[group.name].abs()>=group.threshold
                                torch.testing.assert_close(group.param[outside],before[group.name][outside],rtol=0,atol=0)

    def test_resume_complete_state_all_baselines(self):
        for family in ('llama2','qwen3'):
            for method in METHODS:
                for scope in (('full',) if method == 'sparse_mezo' else ('layer','full')):
                    with self.subTest(family=family,method=method,scope=scope), tempfile.TemporaryDirectory() as td:
                        c=setup(family,Path(td)/'reference',method,scope)
                        expected=trainer(c); expected.train()
                        c['output_dir']=str(Path(td)/'interrupted')
                        c['checkpoint'].update(strategy='steps',save_steps=1,save_total_limit=2)
                        first=trainer(c); first.add_callback(InterruptAfterSave())
                        with self.assertRaisesRegex(RuntimeError,'simulated interruption'): first.train()
                        cp=Path(c['output_dir'])/'checkpoint-1'
                        resumed=trainer(c); resumed.train(resume_from_checkpoint=str(cp))
                        for (n,p),(_,q) in zip(expected.model.named_parameters(),resumed.model.named_parameters()):
                            torch.testing.assert_close(p,q,rtol=0,atol=0,msg=n)
                        if method=='sparse_mezo':
                            for a,b in zip(expected.sparse_param_groups,resumed.sparse_param_groups):
                                torch.testing.assert_close(a.threshold,b.threshold,rtol=0,atol=0)
                        if method.endswith('_lora'):
                            self.assertFalse((cp/'model.safetensors').exists())
                            self.assertTrue((cp/'adapter.safetensors').exists())

    def test_fo_matches_manual_adamw_and_accumulation(self):
        for family in ('llama2','qwen3'):
            for method in ('fo','fo_lora'):
                for accumulation in (1,2):
                    with self.subTest(family=family,method=method,accumulation=accumulation), tempfile.TemporaryDirectory() as td:
                        c=setup(family,td,method)
                        c['training']['steps']=1
                        c['fo'].update(max_grad_norm=0., scheduler='constant',gradient_accumulation_steps=accumulation)
                        tr=trainer(c); reference=copy.deepcopy(tr.model)
                        opt=torch.optim.AdamW((p for p in reference.parameters() if p.requires_grad),lr=c['training']['lr'],
                                              betas=(c['fo']['adam_beta1'],c['fo']['adam_beta2']),eps=c['fo']['adam_epsilon'],weight_decay=0.)
                        batch=DataCollatorForSeq2Seq(tokenizer())(rows())
                        reference.train(); reference(**batch).loss.backward(); opt.step()
                        tr.train()
                        for (n,p),(_,q) in zip(tr.model.named_parameters(),reference.named_parameters()):
                            torch.testing.assert_close(p,q,rtol=2e-5,atol=1e-7,msg=n)

    def test_fo_classification_accumulation_has_correct_gradients(self):
        from unittest.mock import patch
        for family in ('llama2','qwen3'):
            for method in ('fo','fo_lora'):
                with self.subTest(family=family,method=method),tempfile.TemporaryDirectory() as td:
                    c=setup(family,td,method); c['data']['task']='sst2'
                    c['training'].update(steps=1,batch_size=1)
                    c['fo'].update(gradient_accumulation_steps=2,max_grad_norm=0.,scheduler='constant')
                    # One logical item consists of its two candidate completions.
                    item={'input_ids':[[1,3,4,5],[1,3,4,6]], 'attention_mask':[[1]*4]*2,
                          'labels':[0,0], 'option_len':[1,1], 'num_options':[2,2]}
                    def collate(batch):
                        return {key:torch.tensor([x for row in batch for x in row[key]]) for key in item}
                    tr=make_trainer(c,tiny(family),tokenizer(),[item]*2,[item]*2,collate)
                    reference=copy.deepcopy(tr.model); reference.train()
                    reference(**collate([item]*2)).loss.backward()
                    expected={n:p.grad.clone() for n,p in reference.named_parameters() if p.requires_grad}
                    captured={}
                    original_step=torch.optim.AdamW.step
                    def step(optimizer,*args,**kwargs):
                        captured.update({n:p.grad.clone() for n,p in tr.model.named_parameters() if p.requires_grad})
                        return original_step(optimizer,*args,**kwargs)
                    with patch.object(torch.optim.AdamW,'step',step): tr.train()
                    self.assertEqual(captured.keys(),expected.keys())
                    for n,g in expected.items(): torch.testing.assert_close(captured[n],g,rtol=2e-5,atol=1e-7,msg=n)

    def test_baseline_config_and_actual_dtype_guards(self):
        cases=[('fo','model','dtype','float16'),('fo','training','non_diff',True),
               ('zo_lora','model','dtype','bfloat16'),('sparse_mezo','model','dtype','bfloat16'),
               ('zo_lora','training','steps',None),('zo_lora','lora','targets',['o_proj']),
               ('sparse_mezo','sparse','selected_ratio',0.)]
        with tempfile.TemporaryDirectory() as td:
            for method,section,key,value in cases:
                c=setup('llama2',td,method);c[section][key]=value
                with self.assertRaises(ValueError): validate(c)
            for family in ('llama2','qwen3'):
                c=setup(family,td,'zo_lora')
                with self.assertRaisesRegex(ValueError,'bf16'): trainer(c,tiny(family).bfloat16())


    def test_zero_qa_f1_is_valid_in_every_fp16_zo_method(self):
        for family in ('llama2','qwen3'):
            for method in ('zo','zo_lora','sparse_mezo'):
                for scope in (('full',) if method == 'sparse_mezo' else ('layer','full')):
                    with self.subTest(family=family,method=method,scope=scope),tempfile.TemporaryDirectory() as td:
                        c=setup(family,td,method,scope)
                        c['model']['dtype']='float16';c['data']['task']='squad'
                        c['training'].update(non_diff=True,lr=1e-4)
                        tr=trainer(c,tiny(family).half())
                        values=iter([0.,-0.5])
                        tr.compute_loss_from_batch=lambda model,inputs:torch.tensor(next(values))
                        loss=tr.training_step(tr.model,DataCollatorForSeq2Seq(tokenizer())(rows()))
                        self.assertEqual(loss.item(),0. if method=='sparse_mezo' else -0.25)
                        self.assertEqual(tr._skipped_zo_steps,0)

    def test_lora_zero_initialization_and_compact_reload(self):
        for family in ('llama2','qwen3'):
            for scope in ('layer','full'):
                with self.subTest(family=family,scope=scope), tempfile.TemporaryDirectory() as td:
                    root=Path(td); base=tiny(family); base.save_pretrained(root/'base'); tokenizer().save_pretrained(root/'base')
                    c=setup(family,root/'run','zo_lora',scope)
                    c['model']['name']=str(root/'base')
                    batch=DataCollatorForSeq2Seq(tokenizer())(rows())
                    base.eval(); original=base(**batch).logits.detach()
                    tr=trainer(c,base); base.eval()
                    torch.testing.assert_close(base(**batch).logits,original,rtol=0,atol=0)
                    tr.train(); base.eval()
                    expected=base(**batch).logits.detach()
                    save_adapter(base,root/'final',tokenizer())
                    restored,tok=load_checkpoint(c,'cpu',root/'final')
                    torch.testing.assert_close(restored(**batch).logits,expected,rtol=0,atol=0)
                    self.assertLess((root/'final'/'adapter.safetensors').stat().st_size,(root/'base'/'model.safetensors').stat().st_size)
                    bad=tiny(family); wrong=copy.deepcopy(c); wrong['lora']['rank']=3; attach_adapter(wrong,bad)
                    with self.assertRaisesRegex(ValueError,'does not match'): restore_adapter(bad,root/'final')

    def test_lora_best_checkpoint_restores_adapter(self):
        for family in ('llama2','qwen3'):
            for method in ('fo_lora','zo_lora'):
                with self.subTest(family=family,method=method),tempfile.TemporaryDirectory() as td:
                    c=setup(family,td,method)
                    c['checkpoint'].update(strategy='steps',save_steps=1,load_best_model_at_end=True,save_total_limit=2)
                    tr=trainer(c)
                    # Force step 1 to be the best, so loading the last checkpoint would fail.
                    original_evaluate=tr.evaluate
                    def evaluate(*args,**kwargs):
                        result=original_evaluate(*args,**kwargs)
                        result['eval_loss']=float(tr.state.global_step)
                        return result
                    tr.evaluate=evaluate; tr.train()
                    self.assertTrue(tr.state.best_model_checkpoint.endswith('checkpoint-1'))
                    expected=trainer(c).model; restore_adapter(expected,tr.state.best_model_checkpoint)
                    for (n,p),(_,q) in zip(tr.model.named_parameters(),expected.named_parameters()):
                        torch.testing.assert_close(p,q,rtol=0,atol=0,msg=n)

    def test_qwen_multiple_directions_matches_quadratic_gradient_estimator(self):
        for scope in ('layer','full'):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as td:
                c=setup('qwen3',td,'zo_lora',scope)
                c['training'].update(directions=3,epsilon=0.05,lr=0.01)
                tr=trainer(c); params=tr.trainable; base=[p.detach().clone() for p in params]
                def loss(model,inputs): return 1+sum(p.square().sum() for p in params)
                tr.compute_loss_from_batch=loss
                expected=[p.clone() for p in base]
                rng=random.Random(c['training']['seed'])
                for _ in range(3):
                    torch.manual_seed(rng.randint(0,2**31-1))
                    noise=[torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in params]
                    derivative=2*sum((p*z).sum() for p,z in zip(base,noise))
                    for value,z in zip(expected,noise): value.add_(z,alpha=-0.01/3*float(derivative))
                tr.training_step(tr.model,DataCollatorForSeq2Seq(tokenizer())(rows()))
                for p,q in zip(params,expected): torch.testing.assert_close(p,q,rtol=3e-5,atol=2e-6)

    def test_sparse_strict_fixed_threshold_and_same_step_masks(self):
        from dominant_layer_zo.sparse import SparseMeZOConfig,configure_sparse_mezo_groups,build_step_masks
        p=torch.nn.Parameter(torch.tensor([1.,2.,2.,9.]))
        groups,_=configure_sparse_mezo_groups([('weight',p)],SparseMeZOConfig(0.75),'test')
        masks,count=build_step_masks(groups)
        self.assertEqual(count,1); self.assertEqual(groups[0].threshold.item(),2.)
        with torch.no_grad(): p[0]=3.; p[3]=0.5
        changed,count=build_step_masks(groups)
        self.assertEqual(groups[0].threshold.item(),2.)
        self.assertEqual(masks[id(p)].tolist(),[True,False,False,False])
        self.assertEqual(changed[id(p)].tolist(),[False,False,False,True])

    def test_fo_gradient_checkpointing_with_frozen_embeddings(self):
        for family in ('llama2','qwen3'):
            for method in ('fo','fo_lora'):
                with self.subTest(family=family,method=method),tempfile.TemporaryDirectory() as td:
                    c=setup(family,td,method); c['fo']['gradient_checkpointing']=True
                    tr=trainer(c); before={n:p.detach().clone() for n,p in tr.model.named_parameters()}
                    tr.train()
                    self.assertTrue(any(not torch.equal(before[n],p) for n,p in tr.model.named_parameters() if p.requires_grad))
                    torch.testing.assert_close(tr.model.model.embed_tokens.weight,before['model.embed_tokens.weight'],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
