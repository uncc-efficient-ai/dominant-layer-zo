import tempfile
import unittest
from pathlib import Path
import torch
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from dominant_layer_zo.train import make_trainer
from test_core import tokenizer, tiny, settings, rows


class InterruptAfterSave(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if state.global_step == 1:
            raise RuntimeError('simulated interruption')


class ResumeTests(unittest.TestCase):
    def test_interruption_resume_matches_uninterrupted(self):
        for family in ['llama2','qwen3','mistral']:
            for scope in ['layer','full']:
                with self.subTest(family=family,scope=scope),tempfile.TemporaryDirectory() as td:
                    tok=tokenizer(); collator=DataCollatorForSeq2Seq(tok)
                    c=settings(family,Path(td)/'reference',scope)
                    expected=tiny(family)
                    reference=make_trainer(c,expected,tok,rows(),rows(),collator)
                    reference.train()
                    c['output_dir']=str(Path(td)/'interrupted')
                    c['checkpoint'].update(strategy='steps',save_steps=1)
                    first=make_trainer(c,tiny(family),tok,rows(),rows(),collator)
                    first.add_callback(InterruptAfterSave())
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        first.train()
                    model=tiny(family)
                    second=make_trainer(c,model,tok,rows(),rows(),collator)
                    second.train(resume_from_checkpoint=str(Path(c['output_dir'])/'checkpoint-1'))
                    self.assertEqual(second.state.global_step,2)
                    for (name,p),(_,q) in zip(expected.named_parameters(),model.named_parameters()):
                        torch.testing.assert_close(p,q,rtol=0,atol=0,msg=name)


if __name__=='__main__': unittest.main()
