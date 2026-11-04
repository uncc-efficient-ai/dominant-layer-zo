import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LlamaConfig, Qwen3Config, MistralConfig
from datasets import Dataset, DatasetDict
from dominant_layer_zo.activation import AbsActivationStat, apply_selection_rule
from dominant_layer_zo.config import DEFAULTS, validate
from dominant_layer_zo.models import backend, load_model
from dominant_layer_zo.train import make_trainer, train as run_train
from dominant_layer_zo.data import make_data
from dominant_layer_zo.candidates import _build_mezo_classification_dataset
from dominant_layer_zo.collators import DataCollatorWithPaddingAndNesting
from dominant_layer_zo.runtime import run_lock

torch.set_num_threads(1)


def tokenizer():
    vocab = {x:i for i,x in enumerate(['[PAD]','[UNK]','[EOS]','This','was','great','terrible',
                                     'Yes','No','Maybe','answer','prompt','because','so','.'])}
    raw = Tokenizer(WordLevel(vocab,unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
    result.padding_side = 'left'
    result.model_input_names = ['input_ids','attention_mask']
    return result


def tiny(family):
    torch.manual_seed(10)
    cfg_cls = {'llama2':LlamaConfig,'qwen3':Qwen3Config,'mistral':MistralConfig}[family]
    cfg = cfg_cls(vocab_size=16,hidden_size=16,intermediate_size=24,num_hidden_layers=3,
                  num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
                  pad_token_id=0,bos_token_id=1,eos_token_id=2)
    cfg._attn_implementation='eager'
    module = backend(family,'modeling')
    cls = getattr(module, {'llama2':'LlamaForCausalLM','qwen3':'Qwen3ForCausalLM','mistral':'MistralForCausalLM'}[family])
    return cls(cfg)


def settings(family,path,scope='layer'):
    c = copy.deepcopy(DEFAULTS)
    c['model'].update(family=family,dtype='float32',fp32_attention_scores=True)
    c['output_dir'] = str(path)
    c['training'].update(scope=scope,layer=1 if scope=='layer' else None,
                         lr=0.01,steps=2,batch_size=2,eval_steps=1,logging_steps=100)
    c['evaluation'].update(dtype='float32',after_training=False)
    c['checkpoint']['save_final']=False
    return validate(c)


def rows():
    return [{'input_ids':[1,3,4,5,2], 'attention_mask':[1]*5,'labels':[-100,-100,-100,5,2]}]*4


class CoreTests(unittest.TestCase):
    def test_masked_activation_and_abstention(self):
        stat=AbsActivationStat()
        stat.update(torch.tensor([[[1.,-3.],[10000.,10000.]]]),torch.tensor([[1,0]]))
        self.assertEqual(stat.finalize(),{'count':2,'abs_mean':2.,'abs_max':3.})
        with self.assertRaises(FloatingPointError):
            stat.update(torch.tensor([[[float('nan')]]]),None)
        r=[{'layer_idx_zero_based':i,'abs_max':a,'abs_mean':m}
           for i,a,m in [(0,1,1),(1,2000,1),(2,3000,1)]]
        selected=apply_selection_rule(r,10,100,1000)
        self.assertEqual(selected['selected_layer_zero_based'],1)
        self.assertFalse(selected['layers'][2]['qualifies'])
        self.assertTrue(apply_selection_rule(r,10000,100,1000)['abstained'])

    def test_config_guards(self):
        for section,key,value in [('model','dtype','bfloat16'),('training','directions',0),
                                  ('training','layer',-1),('training','lr',float('nan'))]:
            c=copy.deepcopy(DEFAULTS); c[section][key]=value
            with self.assertRaises(ValueError): validate(c)

    def test_nested_candidates(self):
        tok=tokenizer()
        ds=_build_mezo_classification_dataset([{'prompt':'This was','full_prompt':'This was great'}],
                                              'sst2_mezo',tok,32)
        batch=DataCollatorWithPaddingAndNesting(tok)([ds[0]])
        self.assertEqual(batch['labels'].tolist(),[1,1])
        self.assertEqual(batch['option_len'].tolist(),[1,1])
        self.assertEqual(batch['num_options'].tolist(),[2,2])

    def test_both_models_scopes_training_and_generation(self):
        for family in ['llama2','qwen3']:
            for scope in ['layer','full']:
                with self.subTest(family=family,scope=scope), tempfile.TemporaryDirectory() as td:
                    tok=tokenizer(); model=tiny(family); c=settings(family,td,scope)
                    from transformers import DataCollatorForSeq2Seq
                    trainer=make_trainer(c,model,tok,rows(),rows(),DataCollatorForSeq2Seq(tok))
                    before={n:p.detach().clone() for n,p in model.named_parameters()}
                    trainer.train()
                    self.assertEqual(trainer.state.global_step,2)
                    changes=[n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
                    self.assertTrue(changes)
                    if scope=='layer': self.assertTrue(all(n.startswith('model.layers.1.') for n in changes))
                    self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss'])))
                    generated=model.generate(torch.tensor([[1,3]]),attention_mask=torch.ones(1,2,dtype=torch.long),max_new_tokens=2)
                    self.assertGreater(generated.shape[1],2)

    def test_rng_replay_and_precision_guards(self):
        for family in ['llama2','qwen3']:
            for scope in ['layer','full']:
                with self.subTest(family=family,scope=scope),tempfile.TemporaryDirectory() as td:
                    from transformers import DataCollatorForSeq2Seq
                    tok=tokenizer(); c=settings(family,td,scope); model=tiny(family)
                    trainer=make_trainer(c,model,tok,rows(),rows(),DataCollatorForSeq2Seq(tok))
                    trainer._save_rng_state(td)
                    expected=[trainer._py_rng.randint(0,2**31-1) for _ in range(4)]
                    trainer._load_rng_state(td)
                    self.assertEqual(expected,[trainer._py_rng.randint(0,2**31-1) for _ in range(4)])
                    with self.assertRaises(ValueError):
                        make_trainer(c,tiny(family).bfloat16(),tok,rows(),rows(),DataCollatorForSeq2Seq(tok))

    def test_checkpoint_roundtrip(self):
        for family in ['llama2','qwen3']:
            with self.subTest(family=family),tempfile.TemporaryDirectory() as td:
                model=tiny(family).eval(); tok=tokenizer()
                model.save_pretrained(td); tok.save_pretrained(td)
                loaded,_=load_model(settings(family,td)['model'],'cpu',checkpoint=td)
                x=torch.tensor([[1,3,4]])
                with torch.no_grad():
                    torch.testing.assert_close(model(x).logits,loaded(x).logits,rtol=0,atol=0)

    def test_data_splits_and_no_pickle(self):
        for family in ['llama2','qwen3']:
            with self.subTest(family=family),tempfile.TemporaryDirectory() as td:
                c=settings(family,td)
                c['data']['loader_kwargs']={'dev_n':2}
                raw=Dataset.from_dict({'premise':[f'This {i}.' for i in range(8)],
                    'question':['cause']*8,'choice1':['great']*8,'choice2':['terrible']*8,'label':[0]*8})
                tasks=backend(family,'tasks')
                with patch.object(tasks,'load_dataset',return_value=DatasetDict(train=raw,validation=raw.select([0,1]))):
                    tr,dv,collator,info=make_data(c,tokenizer())
                self.assertEqual((len(tr),len(dv)),(6,2))
                self.assertEqual(info['splits']['test']['n'],2)
                self.assertFalse(list(Path(td).glob('*.pkl')))

    def test_nondiff_dev_uses_generation_objective(self):
        for family in ['llama2','qwen3']:
            with self.subTest(family=family),tempfile.TemporaryDirectory() as td:
                c=settings(family,td); c['data']['task']='squad'; c['training']['non_diff']=True
                from dominant_layer_zo.collators import NondiffQACollator
                tok=tokenizer()
                data=[{'input_ids':[1,3],'attention_mask':[1,1],'gold':['answer']}]*2
                trainer=make_trainer(c,tiny(family),tok,data,data,NondiffQACollator(tok))
                with patch.object(trainer,'compute_nondiff_loss',return_value=torch.tensor(-0.5)):
                    self.assertEqual(trainer.evaluate()['eval_loss'],-0.5)

    def test_run_outputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as td:
            c=settings('llama2',Path(td)/'run')
            from transformers import DataCollatorForSeq2Seq
            tok=tokenizer()
            with patch('dominant_layer_zo.train.load_model',return_value=(tiny('llama2'),tok)), \
                 patch('dominant_layer_zo.train.make_data',return_value=(rows(),rows(),DataCollatorForSeq2Seq(tok),{'test':1})):
                run_train(c,'cpu')
            result=json.loads((Path(c['output_dir'])/'run.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['skipped_zo_steps'],0)
            with self.assertRaises(FileExistsError):
                with run_lock(c['output_dir']): pass


if __name__=='__main__': unittest.main()
