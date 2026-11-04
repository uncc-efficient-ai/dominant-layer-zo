import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datasets import Dataset
from dominant_layer_zo.config import DEFAULTS, signature
from dominant_layer_zo.models import backend
from test_core import tiny,tokenizer


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(path,lr,seed=0,loss=1):
    path.mkdir()
    c=copy.deepcopy(DEFAULTS);c['training'].update(lr=lr,seed=seed);c['output_dir']=str(path)
    (path/'run.json').write_text(json.dumps({'status':'complete','config':c,'signature':signature(c)}))
    (path/'validation.json').write_text(json.dumps({'split':'held_out_train','value':loss}))
    (path/'downstream_eval.json').write_text(json.dumps({'copa_mezo':{'accuracy':.5,'n':4}}))


class ReportingTests(unittest.TestCase):
    def test_select_uses_dev_and_rejects_seed_change(self):
        mod=script('select_lr')
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b'
            run(a,1e-6,loss=2);run(b,5e-6,loss=1)
            self.assertEqual(mod.select([a,b])[1],5e-6)
            d=json.loads((b/'run.json').read_text());d['config']['training']['seed']=42
            d['signature']=signature(d['config']);(b/'run.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):mod.select([a,b])

    def test_summary_rejects_per_seed_lr(self):
        mod=script('summarize')
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b'
            run(a,1e-6,seed=0);run(b,1e-6,seed=42)
            self.assertEqual(mod.summarize([a,b])['mean'],.5)
            d=json.loads((b/'run.json').read_text());d['config']['training']['lr']=5e-6
            d['signature']=signature(d['config']);(b/'run.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):mod.summarize([a,b])

    def test_actual_evaluator_supplies_count(self):
        raw=Dataset.from_dict({'premise':['This was.']*3,'question':['cause']*3,
                              'choice1':['great']*3,'choice2':['terrible']*3,'label':[0]*3})
        for family in ['llama2','qwen3']:
            with self.subTest(family=family):
                mod=backend(family,'evaluation')
                with patch.object(mod,'load_dataset',return_value={'validation':raw}):
                    value=mod.run_downstream_eval(tiny(family),tokenizer(),['copa_mezo'],32,max_samples=2)
                self.assertEqual(value['copa_mezo']['n'],2)
                self.assertGreaterEqual(value['copa_mezo']['accuracy'],0)


if __name__=='__main__': unittest.main()
