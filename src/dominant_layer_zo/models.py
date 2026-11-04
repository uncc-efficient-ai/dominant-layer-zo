"""Keep model-specific loading and tokenizer conventions explicit."""
import importlib
import torch
from transformers import AutoTokenizer, LlamaConfig, Qwen3Config, MistralConfig

MODEL_IDS = {'llama2': 'meta-llama/Llama-2-7b-hf', 'qwen3': 'Qwen/Qwen3-8B-Base',
             'mistral': 'mistralai/Mistral-7B-v0.1'}


def backend(family, module):
    if family not in MODEL_IDS:
        raise ValueError(f'Unsupported model family: {family}')
    if family == 'mistral' and module != 'modeling':
        # These operate on tensors, model.model.layers, or text task protocols.
        # Explicitly reuse the established LM/candidate objectives and ZO updates.
        if module not in {'tasks', 'tokenization', 'evaluation', 'zo_layer', 'zo_full'}:
            raise ValueError(f'Unsupported Mistral backend: {module}')
        family = 'llama2'
    return importlib.import_module(f'.backends.{family}.{module}', 'dominant_layer_zo')


def load_model(config, device='cuda', checkpoint=None, measurement=False):
    family = config['family']
    module = backend(family, 'modeling')
    class_name, cfg_cls = {'llama2': ('LlamaForCausalLM', LlamaConfig),
                           'qwen3': ('Qwen3ForCausalLM', Qwen3Config),
                           'mistral': ('MistralForCausalLM', MistralConfig)}[family]
    cls = getattr(module, class_name)
    name = str(checkpoint) if checkpoint else config['name']
    revision = None if checkpoint else config.get('revision')
    cfg = cfg_cls.from_pretrained(name, revision=revision, attn_implementation='eager')
    if family == 'llama2':
        cfg.fp32_residual_stream = config.get('fp32_residual_stream', False)
    cfg.fp32_attention_scores = config.get('fp32_attention_scores', False)
    model = cls.from_pretrained(name, config=cfg, revision=revision,
                               torch_dtype=getattr(torch, config['dtype']), low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=None if checkpoint else (config.get('tokenizer_revision') or revision))
    if measurement:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    else:
        if family in ('llama2', 'mistral'):
            tokenizer.pad_token_id = 0
        if tokenizer.pad_token_id is None:
            raise ValueError('The Qwen3 tokenizer must define its original padding token')
        tokenizer.padding_side = 'left'
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer
