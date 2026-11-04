"""Pinned HF Mistral architecture; optional fp32 eager attention scores.

The attention function follows the pinned Transformers Mistral eager interface.
GQA, RoPE, sliding-window masks and generation/cache handling stay upstream.
"""
import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM as HFMistralForCausalLM, repeat_kv,
)


def fp32_attention_forward(module, query, key, value, attention_mask, scaling,
                           dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key.transpose(2, 3).float()) * scaling
        if attention_mask is not None:
            mask = attention_mask[:, :, :, :key.shape[-2]]
            # An fp16 finite minimum is not a hard mask for large fp32 scores.
            # Preserve causal/window exclusions after promoting score arithmetic.
            mask = mask.float().masked_fill(mask == torch.finfo(mask.dtype).min,
                                             torch.finfo(torch.float32).min)
            scores = scores + mask
        probabilities = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = nn.functional.dropout(probabilities, p=dropout, training=module.training)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return output, probabilities


ALL_ATTENTION_FUNCTIONS.register('dlzo_mistral_fp32', fp32_attention_forward)


class MistralForCausalLM(HFMistralForCausalLM):
    def __init__(self, config):
        if getattr(config, 'fp32_attention_scores', False):
            config._attn_implementation = 'dlzo_mistral_fp32'
        super().__init__(config)
