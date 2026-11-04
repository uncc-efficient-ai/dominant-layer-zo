"""Explicit, portable experiment configuration; no machine-specific paths."""
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml

TASKS = ('sst2', 'rte', 'cb', 'boolq', 'wsc', 'multirc', 'copa', 'squad', 'drop', 'wic', 'record')
DEFAULTS = {
    'model': {'family': 'llama2', 'name': 'meta-llama/Llama-2-7b-hf', 'dtype': 'float16',
              'revision': None, 'tokenizer_revision': None, 'fp32_residual_stream': False,
              'fp32_attention_scores': False},
    'data': {'task': 'copa', 'seed': 0, 'cutoff_len': 256, 'train_on_inputs': False,
             'loader_kwargs': {}},
    'training': {'method': 'zo', 'scope': 'layer', 'layer': 1, 'part': 'layer', 'lr': 1e-6,
                 'epsilon': 1e-3, 'directions': 1, 'noise': 'rademacher', 'seed': 0,
                 'steps': 10000, 'batch_size': 16, 'eval_steps': 2000, 'logging_steps': 10,
                 'max_zo_coeff': None, 'non_diff': False, 'forward_autocast': 'none',
                 'upcast_loss_logits_fp32': True, 'upcast_zo_scalar_fp32': True},
    'fo': {'epochs': 5., 'gradient_accumulation_steps': 1, 'gradient_checkpointing': False,
           'amp': 'none', 'scheduler': 'linear', 'warmup_ratio': 0., 'weight_decay': 0.,
           'max_grad_norm': 1., 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-8},
    'lora': {'rank': 8, 'alpha': 16., 'dropout': 0., 'targets': ['q_proj', 'v_proj'],
             'init_seed': 0, 'perturb_math_fp32': True},
    'agzo': {'rank': 1, 'power_iterations': 5},
    'sparse': {'selected_ratio': 0.25},
    'evaluation': {'dtype': 'float16', 'seed': 0, 'max_length': 2048,
                   'batch_size': 16, 'max_samples': None, 'max_new_tokens': 32,
                   'after_training': True},
    'checkpoint': {'strategy': 'no', 'save_steps': 2000, 'save_total_limit': 1,
                   'load_best_model_at_end': False, 'save_final': True},
    'output_dir': 'runs/example',
}


def merge(default, incoming, prefix=''):
    result = copy.deepcopy(default)
    if not isinstance(incoming, dict):
        raise ValueError(f'{prefix or "config"} must be a mapping')
    for key, value in incoming.items():
        if key not in result:
            raise ValueError(f'Unknown config field: {prefix}{key}')
        if isinstance(result[key], dict) and key != 'loader_kwargs':
            result[key] = merge(result[key], value, prefix + key + '.')
        else:
            result[key] = value
    return result


def validate(c):
    merge(DEFAULTS, c)
    m, d, t, e, k = (c[x] for x in ('model', 'data', 'training', 'evaluation', 'checkpoint'))
    for key, value, choices in [
        ('family', m['family'], ('llama2', 'qwen3', 'mistral')),
        ('model dtype', m['dtype'], ('float16', 'float32', 'bfloat16')),
        ('eval dtype', e['dtype'], ('float16', 'float32', 'bfloat16')),
        ('method', t['method'], ('zo', 'fo', 'zo_lora', 'fo_lora', 'sparse_mezo', 'agzo')),
        ('task', d['task'], TASKS), ('scope', t['scope'], ('layer', 'full')),
        ('part', t['part'], ('layer', 'mlp', 'attn')),
        ('noise', t['noise'], ('gaussian', 'rademacher')),
        ('forward_autocast', t['forward_autocast'], ('none', 'fp16', 'bf16')),
        ('checkpoint strategy', k['strategy'], ('no', 'steps')),
    ]:
        if value not in choices:
            raise ValueError(f'{key} must be one of {choices}; got {value!r}')
    for obj, keys in [(d, ['cutoff_len']), (t, ['directions','batch_size','eval_steps','logging_steps']),
                      (e, ['max_length','batch_size','max_new_tokens']), (k, ['save_steps','save_total_limit'])]:
        for key in keys:
            if type(obj[key]) is not int or obj[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
    fo, lora, sparse = c['fo'], c['lora'], c['sparse']
    is_fo = t['method'] in ('fo', 'fo_lora')
    if t['steps'] is None:
        if not is_fo:
            raise ValueError('ZO requires a positive training.steps budget')
    elif type(t['steps']) is not int or t['steps'] <= 0:
        raise ValueError('steps must be positive, or null for FO epoch training')
    if m['dtype'] == 'bfloat16' and not is_fo:
        raise ValueError('Raw bf16 ZO is unsupported; use float16 or float32')
    if is_fo and m['dtype'] == 'float16':
        raise ValueError('FO AdamW requires float32 or bfloat16 weights; fp16 states are not supported')
    if is_fo and t['non_diff']:
        raise ValueError('FO requires a differentiable objective')
    if fo['amp'] not in ('none', 'bf16', 'fp16') or fo['scheduler'] not in ('linear', 'constant', 'cosine'):
        raise ValueError('Invalid FO AMP or scheduler')
    if fo['amp'] == 'fp16' and m['dtype'] != 'float32' and is_fo:
        raise ValueError('FP16 AMP requires float32 master weights')
    for key in ['epochs', 'adam_epsilon']:
        fo[key] = float(fo[key])
        if not math.isfinite(fo[key]) or fo[key] <= 0:
            raise ValueError(f'fo.{key} must be finite and positive')
    for key in ['warmup_ratio', 'adam_beta1', 'adam_beta2']:
        fo[key] = float(fo[key])
        if not math.isfinite(fo[key]) or not 0 <= fo[key] < 1:
            raise ValueError(f'fo.{key} must be in [0, 1)')
    for key in ['weight_decay', 'max_grad_norm']:
        fo[key] = float(fo[key])
        if not math.isfinite(fo[key]) or fo[key] < 0:
            raise ValueError(f'fo.{key} must be nonnegative and finite')
    if type(fo['gradient_accumulation_steps']) is not int or fo['gradient_accumulation_steps'] < 1:
        raise ValueError('FO gradient_accumulation_steps must be positive')
    if type(fo['gradient_checkpointing']) is not bool:
        raise ValueError('gradient_checkpointing must be a boolean')
    if type(lora['rank']) is not int or lora['rank'] < 1 or type(lora['init_seed']) is not int or lora['init_seed'] < 0:
        raise ValueError('LoRA requires positive integer rank and nonnegative init_seed')
    lora['alpha'], lora['dropout'] = float(lora['alpha']), float(lora['dropout'])
    if not math.isfinite(lora['alpha']) or lora['alpha'] <= 0 or not 0 <= lora['dropout'] < 1:
        raise ValueError('Invalid LoRA alpha or dropout')
    if type(lora['perturb_math_fp32']) is not bool:
        raise ValueError('perturb_math_fp32 must be a boolean')
    if not isinstance(lora['targets'], list) or not lora['targets'] or any(not isinstance(x,str) or not x for x in lora['targets']):
        raise ValueError('LoRA targets must be a nonempty list of projection names')
    if t['method'].endswith('_lora'):
        allowed = {'q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'}
        if m['family'] == 'qwen3': allowed.add('k_proj')
        if set(lora['targets']) - allowed or len(set(lora['targets'])) != len(lora['targets']):
            raise ValueError(f'LoRA targets must be unique and drawn from {sorted(allowed)}')
        part_targets = {'attn': {'q_proj','k_proj','v_proj'}, 'mlp': {'gate_proj','up_proj','down_proj'}}
        if t['part'] in part_targets and set(lora['targets']) - part_targets[t['part']]:
            raise ValueError('LoRA targets conflict with training.part')
    agzo = c['agzo']
    if type(agzo['rank']) is not int or agzo['rank'] < 1:
        raise ValueError('agzo.rank must be a positive integer')
    if type(agzo['power_iterations']) is not int or agzo['power_iterations'] < 1:
        raise ValueError('agzo.power_iterations must be a positive integer')
    if t['method'] == 'agzo':
        if t['noise'] != 'gaussian' or t['directions'] != 1:
            raise ValueError('AGZO requires Gaussian noise and one direction')
        if t['non_diff'] or t['max_zo_coeff'] is not None:
            raise ValueError('AGZO supports the task loss without coefficient clipping; non_diff is unsupported')
    sparse['selected_ratio'] = float(sparse['selected_ratio'])
    if not math.isfinite(sparse['selected_ratio']) or not 0 < sparse['selected_ratio'] <= 1:
        raise ValueError('sparse.selected_ratio must be in (0, 1]')
    if t['method'] in ('sparse_mezo', 'agzo'):
        if t['scope'] != 'full':
            raise ValueError('SparseMeZO and AGZO are full-model baselines; use training.scope=full')
        if t['noise'] != 'gaussian' or t['directions'] != 1 or t['max_zo_coeff'] is not None:
            raise ValueError('SparseMeZO and AGZO require Gaussian noise, one direction and no coefficient clipping')
        if not m['fp32_attention_scores']:
            raise ValueError('SparseMeZO and AGZO require model.fp32_attention_scores=true')
        if type(t['seed']) is not int or not 0 <= t['seed'] < 2**32:
            raise ValueError('Perturbation seed must be an integer in [0, 2**32)')
    if t['method'] != 'zo' and (t['forward_autocast'] != 'none' or not t['upcast_loss_logits_fp32'] or not t['upcast_zo_scalar_fp32']):
        raise ValueError('These precision switches belong to dense ZO; baselines retain their documented precision')
    for obj in [d, t, e]:
        if type(obj['seed']) is not int or obj['seed'] < 0:
            raise ValueError('Seeds must be nonnegative integers')
    for key in ['lr','epsilon']:
        t[key] = float(t[key])
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if t['max_zo_coeff'] is not None:
        t['max_zo_coeff'] = float(t['max_zo_coeff'])
        if not math.isfinite(t['max_zo_coeff']) or t['max_zo_coeff'] <= 0:
            raise ValueError('max_zo_coeff must be null or finite and positive')
    if t['scope'] == 'layer' and (type(t['layer']) is not int or t['layer'] < 0):
        raise ValueError('layer must be a nonnegative, zero-based integer')
    if t['scope'] == 'full' and (t['layer'] is not None or t['part'] != 'layer'):
        raise ValueError('full scope requires layer: null and part: layer')
    if t['non_diff'] and d['task'] != 'squad':
        raise ValueError('non_diff is supported only for SQuAD')
    if m['family'] == 'qwen3' and t['forward_autocast'] != 'none':
        raise ValueError('Qwen3 backend uses its original forward precision; forward_autocast must be none')
    if m['family'] != 'llama2' and m['fp32_residual_stream']:
        raise ValueError('fp32_residual_stream is a LLaMA-only option')
    if m['family'] == 'mistral' and t['method'] not in ('zo', 'sparse_mezo', 'agzo'):
        raise ValueError('Mistral supports dense ZO, SparseMeZO and AGZO only')
    if e['max_samples'] is not None and (type(e['max_samples']) is not int or e['max_samples'] <= 0):
        raise ValueError('max_samples must be null or a positive integer')
    if d['cutoff_len'] < 2 or e['max_length'] <= e['max_new_tokens']:
        raise ValueError('Invalid sequence length or generation budget')
    if k['load_best_model_at_end'] and (k['strategy'] != 'steps' or k['save_steps'] % t['eval_steps']):
        raise ValueError('Best checkpoint selection requires steps strategy and save_steps divisible by eval_steps')
    if not isinstance(d['loader_kwargs'], dict) or set(d['loader_kwargs']) - {'train_n','dev_n','test_n'}:
        raise ValueError('loader_kwargs supports train_n, dev_n, test_n only')
    for key, value in d['loader_kwargs'].items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'loader_kwargs.{key} must be positive')
    for obj, keys in [(m,['fp32_residual_stream','fp32_attention_scores']), (d,['train_on_inputs']),
                      (t,['non_diff','upcast_loss_logits_fp32','upcast_zo_scalar_fp32']),
                      (e,['after_training']), (k,['load_best_model_at_end','save_final'])]:
        for key in keys:
            if type(obj[key]) is not bool:
                raise ValueError(f'{key} must be a YAML boolean')
    if not isinstance(c['output_dir'], str) or not c['output_dir']:
        raise ValueError('output_dir must be a nonempty path')
    return c


def load_config(path, overrides=()):
    incoming = yaml.safe_load(Path(path).read_text())
    c = merge(DEFAULTS, incoming)
    for assignment in overrides:
        key, sep, value = assignment.partition('=')
        if not sep:
            raise ValueError('--set requires section.key=value')
        parts = key.split('.')
        obj = c
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                raise ValueError(f'Unknown field: {key}')
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError(f'Unknown field: {key}')
        obj[parts[-1]] = yaml.safe_load(value)
    return validate(c)


def signature(c):
    # Relocation is allowed; numerical settings are not.
    c = copy.deepcopy(c)
    c.pop('output_dir', None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
