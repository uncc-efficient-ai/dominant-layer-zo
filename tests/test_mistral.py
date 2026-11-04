import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import MistralConfig, MistralForCausalLM as HFModel, DataCollatorForSeq2Seq
from dominant_layer_zo.config import DEFAULTS, validate
from dominant_layer_zo.models import backend, load_model
from dominant_layer_zo.train import make_trainer
from test_core import tokenizer, rows, settings


def tiny_mistral(flag=False, dtype=torch.float32):
    torch.manual_seed(10)
    cfg=MistralConfig(vocab_size=16,hidden_size=16,intermediate_size=24,num_hidden_layers=3,
                      num_attention_heads=4,num_key_value_heads=2,sliding_window=4,
                      max_position_embeddings=64,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    cfg._attn_implementation='eager';cfg.fp32_attention_scores=flag
    return backend('mistral','modeling').MistralForCausalLM(cfg).to(dtype).eval()


class MistralTests(unittest.TestCase):
    def test_native_parity_gqa_sliding_window_and_generation(self):
        ids=torch.tensor([[1,3,4,5,6,7]])
        for flag in [False,True]:
            with self.subTest(fp32_scores=flag),torch.no_grad():
                model=tiny_mistral(flag)
                cfg=copy.deepcopy(model.config);cfg._attn_implementation='eager'
                reference=HFModel(cfg).eval();reference.load_state_dict(model.state_dict())
                a=model(ids,labels=ids,output_attentions=True)
                b=reference(ids,labels=ids,output_attentions=True)
                torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
                torch.testing.assert_close(a.loss,b.loss,rtol=0,atol=0)
                # Last query cannot see keys at positions 0/1 with window=4.
                self.assertEqual(a.attentions[0][...,5,:2].count_nonzero(),0)
                self.assertEqual(model.model.layers[0].self_attn.num_key_value_groups,2)
                kwargs=dict(attention_mask=torch.ones_like(ids),max_new_tokens=2,do_sample=False,use_cache=True)
                torch.testing.assert_close(model.generate(ids,**kwargs),reference.generate(ids,**kwargs))

    def test_opt_in_attention_prevents_score_overflow(self):
        from types import SimpleNamespace
        from transformers.models.mistral.modeling_mistral import eager_attention_forward
        from dominant_layer_zo.backends.mistral.modeling import fp32_attention_forward
        module=SimpleNamespace(num_key_value_groups=2,training=False)
        q=torch.full((1,4,2,4),200.,dtype=torch.float16)
        k=torch.full((1,2,2,4),200.,dtype=torch.float16)
        v=torch.ones_like(k)
        self.assertFalse(torch.isfinite(eager_attention_forward(module,q,k,v,None,.5)[0]).all())
        result,_=fp32_attention_forward(module,q,k,v,None,.5)
        torch.testing.assert_close(result,torch.ones_like(result))
        self.assertEqual(result.dtype,torch.float16)
        # A large future key must remain masked after fp32 QK promotion.
        k[:,:,0,:]=1.;v[:,:,0,:]=0.
        mask=torch.zeros((1,1,2,2),dtype=torch.float16)
        mask[:,:,0,1]=torch.finfo(torch.float16).min
        result,weights=fp32_attention_forward(module,q,k,v,mask,.5)
        self.assertEqual(weights[:,:,0,1].count_nonzero(),0)
        self.assertEqual(result[:,0].count_nonzero(),0)

    def test_dense_zo_scopes_freezing_and_rng_replay(self):
        for scope in ['layer','full']:
            with self.subTest(scope=scope),tempfile.TemporaryDirectory() as td:
                c=settings('mistral',td,scope);model=tiny_mistral();tok=tokenizer()
                trainer=make_trainer(c,model,tok,rows(),rows(),DataCollatorForSeq2Seq(tok))
                before={n:p.detach().clone() for n,p in model.named_parameters()}
                trainer._save_rng_state(td)
                expected=[trainer._py_rng.randint(0,2**31-1) for _ in range(3)]
                trainer._load_rng_state(td)
                self.assertEqual(expected,[trainer._py_rng.randint(0,2**31-1) for _ in range(3)])
                trainer.train()
                changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
                self.assertTrue(changed)
                if scope=='layer':self.assertTrue(all(n.startswith('model.layers.1.') for n in changed))
                self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss'])))

    def test_checkpoint_loader_and_config_guards(self):
        with tempfile.TemporaryDirectory() as td:
            model=tiny_mistral(True,torch.float16);tok=tokenizer()
            model.save_pretrained(td);tok.save_pretrained(td)
            cfg=copy.deepcopy(DEFAULTS['model']);cfg.update(family='mistral',fp32_attention_scores=True)
            loaded,tok=load_model(cfg,'cpu',checkpoint=td)
            self.assertEqual(loaded.config.model_type,'mistral')
            self.assertEqual(loaded.config._attn_implementation,'dlzo_mistral_fp32')
            self.assertEqual(tok.padding_side,'left');self.assertEqual(tok.pad_token_id,0)
            with torch.no_grad():
                ids=torch.tensor([[1,3,4]])
                torch.testing.assert_close(model(ids).logits,loaded(ids).logits,rtol=0,atol=0)
        for changes in [{'training':{'method':'zo_lora'}},{'model':{'fp32_residual_stream':True}}]:
            c=copy.deepcopy(DEFAULTS);c['model']['family']='mistral'
            for k,v in changes.items():c[k].update(v)
            with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
