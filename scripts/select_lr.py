"""Select one LR from pilot internal-dev losses, then lock it across reporting seeds."""
import argparse
import copy
import json
import math
from pathlib import Path
import yaml
from dominant_layer_zo.config import signature


def select(runs):
    choices = []
    comparison = None
    for folder in runs:
        path = Path(folder)
        run = json.loads((path/'run.json').read_text())
        validation = json.loads((path/'validation.json').read_text())
        if run['status'] != 'complete' or validation['split'] != 'held_out_train':
            raise ValueError(f'{path}: need a complete pilot with internal-dev validation')
        c = run['config']
        if run['signature'] != signature(c):
            raise ValueError(f'{path}: signature mismatch')
        comparable = copy.deepcopy(c)
        comparable.pop('output_dir')
        comparable['training'].pop('lr')
        if comparison is None:
            comparison = comparable
        if comparable != comparison:
            raise ValueError('Pilots must differ only in LR and output_dir; keep seeds/protocol fixed')
        score = float(validation['value'])
        if not math.isfinite(score):
            raise ValueError('Invalid dev loss')
        choices.append((score,c['training']['lr'],str(path),c))
    if not choices:
        raise ValueError('Supply at least one pilot')
    return min(choices,key=lambda x:(x[0],x[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    p.add_argument('--output',required=True,help='New locked YAML config')
    p.add_argument('--run-output',required=True,help='Output_dir for the reporting run')
    a = p.parse_args()
    score,lr,pilot,c = select(a.runs)
    c = copy.deepcopy(c)
    c['output_dir'] = a.run_output
    c['evaluation']['after_training'] = True
    with open(a.output,'x') as f:
        f.write(f'# Selected on held-out train loss {score}; pilot {Path(pilot).name}\n')
        yaml.safe_dump(c,f,sort_keys=False)
    print(f'Locked LR={lr}; reuse this LR for every reporting seed')


if __name__ == '__main__': main()
