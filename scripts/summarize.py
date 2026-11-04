"""Summarize only complete, comparable reporting runs with fixed hyperparameters."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from dominant_layer_zo.config import signature


def summarize(folders):
    rows = []
    common = None
    for folder in folders:
        path = Path(folder)
        run = json.loads((path/'run.json').read_text())
        metrics = json.loads((path/'downstream_eval.json').read_text())
        c = run['config']
        if run['status'] != 'complete' or run['signature'] != signature(c):
            raise ValueError(f'{path}: incomplete or mismatched run')
        comparable = copy.deepcopy(c)
        comparable.pop('output_dir')
        comparable['training'].pop('seed')
        if common is None: common = comparable
        if comparable != common:
            raise ValueError('Reporting runs must share all settings except training seed/output_dir')
        metric = metrics[c['data']['task']+'_mezo']
        key = 'accuracy' if 'accuracy' in metric else 'f1'
        value = float(metric[key])
        if not math.isfinite(value): raise ValueError('Non-finite final metric')
        rows.append({'seed':c['training']['seed'],'metric':key,'value':value,'n':metric['n']})
    if not rows or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Supply distinct reporting seeds')
    if len({(r['n'],r['metric']) for r in rows}) != 1:
        raise ValueError('Metric or evaluation sample count differs')
    values = [r['value'] for r in rows]
    return {'metric':rows[0]['metric'],'num_seeds':len(rows),'mean':statistics.mean(values),
            'sample_std':statistics.stdev(values) if len(rows)>1 else None,'runs':rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    a = p.parse_args()
    print(json.dumps(summarize(a.runs),indent=2))
