"""Compact checkpoints for the model-specific, unmerged MeZO LoRA layers."""
import copy
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .models import backend, load_model
from .runtime import atomic_json


def attach_adapter(c, model):
    """Attach only the requested projections, with an independent initialization seed."""
    if not c['training']['method'].endswith('_lora'):
        return model
    if hasattr(model, '_dlz_adapter_spec'):
        spec = model._dlz_adapter_spec
        if spec['lora'] != c['lora'] or any(spec[key] != c['training'][key] for key in ('scope','layer','part')) or spec['model']['family'] != c['model']['family']:
            raise ValueError('Model already has a different adapter')
        return model
    t, l = c['training'], c['lora']
    layers = model.model.layers
    if t['scope'] == 'layer' and not 0 <= t['layer'] < len(layers):
        raise ValueError('LoRA layer is outside the model')
    indices = range(len(layers)) if t['scope'] == 'full' else [t['layer']]
    paths = sorted(f'model.layers.{i}.{"self_attn" if target in {"q_proj","k_proj","v_proj"} else "mlp"}.{target}'
                   for i in indices for target in l['targets'])
    module = backend(c['model']['family'], 'lora')
    attach = module.attach_mezo_llama_lora if c['model']['family'] == 'llama2' else module.attach_mezo_qwen_lora
    # The custom constructor initializes on CPU before casting to the base device.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(l['init_seed'])
        attach(model, target_module_paths=paths, r=l['rank'], lora_alpha=l['alpha'],
               lora_dropout=l['dropout'], fan_in_fan_out=False)
    expected = {p+'.'+suffix for p in paths for suffix in ['lora_A', 'lora_B']}
    actual = {n for n,p in model.named_parameters() if p.requires_grad}
    if actual != expected:
        raise ValueError('Adapter trainable parameters differ from the requested scope')
    base = copy.deepcopy(c['model'])
    base['revision'] = getattr(model.config, '_commit_hash', None) or base['revision']
    base['tokenizer_revision'] = base['tokenizer_revision'] or base['revision']
    model._dlz_adapter_spec = {
        'format_version': 1, 'implementation': 'custom_mezo_unmerged', 'model': base,
        'lora': copy.deepcopy(l), 'scope': t['scope'], 'layer': t['layer'], 'part': t['part'],
        'target_paths': paths,
    }
    return model


def adapter_parameters(model):
    spec = model._dlz_adapter_spec
    names = {p+'.'+suffix for p in spec['target_paths'] for suffix in ['lora_A', 'lora_B']}
    params = dict(model.named_parameters())
    if names - params.keys():
        raise ValueError('Missing adapter parameters')
    return {n:params[n] for n in sorted(names)}


def save_adapter(model, output_dir, tokenizer=None):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    state = {n:p.detach().cpu().contiguous() for n,p in adapter_parameters(model).items()}
    save_file(state, str(out/'adapter.safetensors'))
    atomic_json(out/'adapter.json', model._dlz_adapter_spec)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)


def restore_adapter(model, checkpoint):
    cp = Path(checkpoint)
    spec = json.loads((cp/'adapter.json').read_text())
    if spec != model._dlz_adapter_spec:
        raise ValueError('Adapter checkpoint does not match the attached model/configuration')
    state = load_file(str(cp/'adapter.safetensors'))
    params = adapter_parameters(model)
    if state.keys() != params.keys():
        raise ValueError('Adapter tensor names do not match')
    # Validate everything before changing the model.
    for name, p in params.items():
        value = state[name]
        if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
            raise ValueError(f'Invalid adapter tensor shape/dtype/value: {name}')
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state[name].to(p.device))


def load_checkpoint(c, device='cuda', checkpoint=None):
    """A checkpoint reconstructs trained adapters; omitting it means zero-shot base."""
    if checkpoint and (Path(checkpoint)/'adapter.json').is_file():
        spec = json.loads((Path(checkpoint)/'adapter.json').read_text())
        if spec.get('format_version') != 1 or spec.get('implementation') != 'custom_mezo_unmerged':
            raise ValueError('Unsupported adapter checkpoint format')
        if spec['model']['family'] != c['model']['family'] or spec['model']['name'] != c['model']['name']:
            raise ValueError('Adapter base model differs from the evaluation configuration')
        if c['model']['revision'] and spec['model']['revision'] != c['model']['revision']:
            raise ValueError('Adapter base revision differs from the evaluation configuration')
        saved = copy.deepcopy(c)
        saved['model'] = spec['model']; saved['lora'] = spec['lora']
        saved['training'].update(method='zo_lora', scope=spec['scope'], layer=spec['layer'], part=spec['part'])
        model, tokenizer = load_model(saved['model'], device)
        attach_adapter(saved, model)
        restore_adapter(model, checkpoint)
        # Tokenizer is part of the saved run; do not silently use a changed base tokenizer.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        tokenizer.padding_side = 'left'
        model.eval()
        return model, tokenizer
    if checkpoint and not (Path(checkpoint)/'config.json').is_file():
        raise ValueError('Checkpoint must contain a saved HF model or adapter.json')
    return load_model(c['model'], device, checkpoint)
