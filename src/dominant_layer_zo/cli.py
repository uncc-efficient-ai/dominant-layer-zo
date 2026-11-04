import argparse
import json
from pathlib import Path
from .config import load_config, signature


def main():
    parser = argparse.ArgumentParser(description='Dominant-Layer ZO: measure, train, evaluate')
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('measure',help='Forward-only activation measurement and layer selection')
    p.add_argument('--family',choices=['llama2','qwen3','mistral'],required=True)
    p.add_argument('--model',help='Optional local checkpoint or HF model ID')
    p.add_argument('--revision',default=None)
    p.add_argument('--dtype',choices=['float16','float32'],default='float16')
    p.add_argument('--tasks',default='copa,wsc')
    p.add_argument('--seeds',default='0,42,2026')
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-length',type=int,default=256)
    p.add_argument('--jump-threshold',type=float,default=10.)
    p.add_argument('--min-abs-max',type=float,default=100.)
    p.add_argument('--min-max-to-mean',type=float,default=1000.)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True,help='New output directory')
    p = sub.add_parser('train',help='Single-layer or full-model ZO, FO, LoRA, Sparse-MeZO, or AGZO')
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[],metavar='SECTION.KEY=VALUE')
    p.add_argument('--selection',help='Completed measurement.json; overrides training.layer')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',help='Explicit checkpoint directory inside output_dir')
    p.add_argument('--dry-run',action='store_true',help='Validate and print settings without model/data/GPU loading')
    p = sub.add_parser('evaluate',help='Evaluate a base model or saved final/checkpoint model')
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[],metavar='SECTION.KEY=VALUE')
    p.add_argument('--checkpoint',help='Omit for zero-shot base evaluation')
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True,help='New output directory')
    args = parser.parse_args()
    if args.command == 'measure':
        from .measure import measure
        measure(args)
        return
    c = load_config(args.config,args.set)
    if args.command == 'train':
        if args.selection:
            from .measure import selected_layer
            layer, meta = selected_layer(args.selection)
            if c['training']['scope'] != 'layer':
                raise ValueError('--selection requires layer scope')
            if meta['family'] != c['model']['family'] or meta['model'] != c['model']['name']:
                raise ValueError('Measurement model does not match training configuration')
            if c['model']['revision'] and meta['revision'] != c['model']['revision']:
                raise ValueError('Use the same immutable model revision for selection and training')
            c['training']['layer'] = layer
        if args.dry_run:
            print(json.dumps({'signature':signature(c),'config':c},indent=2))
            return
        from .train import train
        train(c,args.device,args.resume)
    else:
        from .adapters import load_checkpoint
        from .train import evaluate_model
        from .runtime import atomic_json, environment, run_lock
        with run_lock(args.output) as out:
            model, tokenizer = load_checkpoint(c,args.device,args.checkpoint)
            metrics = evaluate_model(c,model,tokenizer)
            atomic_json(out/'downstream_eval.json',metrics)
            atomic_json(out/'run.json',{'status':'complete','config':c,'checkpoint':args.checkpoint,
                                      'model_revision':getattr(model.config,'_commit_hash',None),
                                      'environment':environment(),'metrics':metrics})
            print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
