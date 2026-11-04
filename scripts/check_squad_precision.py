"""Read-only, matched-batch SQuAD LM loss diagnostic; never trains or saves weights."""
import argparse
import hashlib
import json
import math
import time

import torch
from transformers import set_seed
from dominant_layer_zo.config import load_config, signature
from dominant_layer_zo.data import make_data
from dominant_layer_zo.models import load_model
from dominant_layer_zo.runtime import atomic_json, environment, implementation_signature, run_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--round-weights-fp16', action='store_true',
                   help='Round fp32 parameters through fp16 to isolate forward arithmetic')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True, help='New diagnostic directory')
    args = p.parse_args()
    c = load_config(args.config, args.set)
    if c['data']['task'] != 'squad' or c['training']['non_diff']:
        p.error('This diagnostic requires SQuAD with the LM objective')
    if args.round_weights_fp16 and c['model']['dtype'] != 'float32':
        p.error('--round-weights-fp16 requires model.dtype=float32')
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(c['data']['seed'])
    with run_lock(args.output) as out:
        report = {'status': 'running', 'config': c, 'config_signature': signature(c),
                  'environment': environment(), 'implementation_signature': implementation_signature(),
                  'reference_weights_rounded_to_fp16': args.round_weights_fp16,
                  'matmul_allow_tf32': False, 'batches': []}
        atomic_json(out/'precision.json', report)
        try:
            model, tok = load_model(c['model'], args.device)
            if args.round_weights_fp16:
                with torch.no_grad():
                    for param in model.parameters():
                        param.copy_(param.to(torch.float16).to(torch.float32))
            _, dev, collator, info = make_data(c, tok)
            report.update(data=info, attention=model.config._attn_implementation,
                          resolved_revision=getattr(model.config, '_commit_hash', None),
                          parameter_dtypes=sorted({str(param.dtype) for param in model.parameters()}),
                          dev_tokenized_sha256=hashlib.sha256(
                              json.dumps([dict(row) for row in dev], sort_keys=True).encode()).hexdigest())
            bs = c['training']['batch_size']
            device = torch.device(args.device)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            with torch.inference_mode():
                for start in range(0, len(dev), bs):
                    batch = {k: v.to(device) for k, v in collator(dev[start:start+bs]).items()}
                    result = model(**batch, use_cache=False)
                    loss = result.loss.float().item()
                    row = {'index': start//bs, 'offset': start, 'n': min(bs, len(dev)-start),
                           'shape': list(batch['input_ids'].shape), 'finite': math.isfinite(loss),
                           'loss': loss if math.isfinite(loss) else None,
                           'loss_dtype': str(result.loss.dtype), 'logits_dtype': str(result.logits.dtype)}
                    report['batches'].append(row)
                    print(json.dumps(row), flush=True)
                    del result, batch
                    atomic_json(out/'precision.json', report)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            finite = all(row['finite'] for row in report['batches'])
            report.update(status='complete', all_finite=finite, n_examples=len(dev),
                          seconds=time.monotonic()-started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                          example_weighted_mean_batch_loss=sum(row['loss']*row['n'] for row in report['batches'])/len(dev) if finite else None)
            atomic_json(out/'precision.json', report)
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            atomic_json(out/'precision.json', report)
            raise


if __name__ == '__main__':
    main()
