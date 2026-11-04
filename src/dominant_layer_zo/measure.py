"""Measure block outputs on unlabeled calibration prompts, then select or abstain."""
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .activation import build_prompt_only_inputs, prompt_digest, LayerCollector, apply_selection_rule
from .models import MODEL_IDS, backend
from .runtime import atomic_json, environment, run_lock


@torch.inference_mode()
def measure_batch(model, tokenizer, texts, device, batch_size, max_length):
    collector = LayerCollector(model)
    try:
        for start in range(0,len(texts),batch_size):
            batch = tokenizer(texts[start:start+batch_size],padding=True,truncation=True,
                              max_length=max_length,return_tensors='pt')
            batch.pop('token_type_ids',None)
            batch = {k:v.to(device) for k,v in batch.items()}
            collector.set_mask(batch['attention_mask'])
            model.model(**batch,use_cache=False)
        return collector.rows()
    finally:
        collector.close()


def measure(args):
    if args.samples <= 0 or args.batch_size <= 0 or args.max_length < 2:
        raise ValueError('samples/batch_size must be positive and max_length >= 2')
    if args.jump_threshold <= 1 or args.min_abs_max <= 0 or args.min_max_to_mean <= 1:
        raise ValueError('Invalid selection thresholds')
    datasets = args.tasks.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError('Repeated calibration conditions')
    if any(x not in {'copa','wsc'} for x in datasets) or any(s < 0 for s in seeds):
        raise ValueError('Calibration uses COPA/WSC and nonnegative seeds')
    with run_lock(args.output) as out:
        name = args.model or MODEL_IDS[args.family]
        device = torch.device(args.device)
        cls = (AutoModelForCausalLM if args.family in ('llama2','mistral') else
               backend('qwen3','modeling').Qwen3ForCausalLM)
        # Use HF LLaMA/Mistral and local Qwen3 models with automatic attention selection.
        model = cls.from_pretrained(name,revision=args.revision,torch_dtype=getattr(torch,args.dtype),
                                   low_cpu_mem_usage=True).to(device).eval()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(name,revision=args.revision,use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        report = {'complete':False, 'meta':{'family':args.family,'model':name,'dtype':args.dtype,
                  'revision':getattr(model.config,'_commit_hash',None),
                  'attention':model.config._attn_implementation,'environment':environment()},'conditions':[]}
        atomic_json(out/'measurement.json',report)
        for task in datasets:
            for seed in seeds:
                texts, provenance = build_prompt_only_inputs(task,args.samples,seed)
                started = time.monotonic()
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                rows = measure_batch(model,tokenizer,texts,device,args.batch_size,args.max_length)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                selection = apply_selection_rule(rows,args.jump_threshold,args.min_abs_max,args.min_max_to_mean)
                report['conditions'].append({'condition':{'dataset':task,'data_seed':seed,
                    'n_samples_requested':args.samples,'n_samples_processed':len(texts),
                    'max_length':args.max_length,'batch_size':args.batch_size,'split':'train',
                    'input_mode':'prompt_only_no_gold_answer_or_choice','padding_side':'right',
                    'prompt_sha256':prompt_digest(texts),**provenance},'selection':selection,
                    'runtime':{'seconds':time.monotonic()-started,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}})
                atomic_json(out/'measurement.json',report)
                print(f'{task} seed={seed}: layer={selection["selected_layer_zero_based"]}',flush=True)
        report['complete'] = True
        atomic_json(out/'measurement.json',report)


def selected_layer(path):
    report = __import__('json').loads(Path(path).read_text())
    if not report.get('complete') or not report.get('conditions'):
        raise ValueError('Measurement is incomplete')
    indices = {x['selection']['selected_layer_zero_based'] for x in report['conditions']}
    if len(indices) != 1 or None in indices:
        raise ValueError('Selector abstained or disagreed across calibration conditions')
    return indices.pop(), report['meta']
