import logging
import re
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def ensure_finite_zo_probe(loss_plus: torch.Tensor, loss_minus: torch.Tensor, coeff: torch.Tensor, *, log_prefix: str, step=None, seed=None):
    bad = []
    if not torch.isfinite(loss_plus):
        bad.append(f"loss_plus={loss_plus}")
    if not torch.isfinite(loss_minus):
        bad.append(f"loss_minus={loss_minus}")
    if not torch.isfinite(coeff):
        bad.append(f"coeff={coeff}")
    if bad:
        step_msg = f" step={step}" if step is not None else ""
        seed_msg = f" seed={seed}" if seed is not None else ""
        raise FloatingPointError(
            f"[{log_prefix}] Non-finite ZO probe detected{step_msg}{seed_msg}: " + ", ".join(bad)
        )


def forward_wrap_with_option_len(
    self,
    input_ids=None,
    labels=None,
    option_len=None,
    num_options=None,
    return_dict=None,
    **kwargs,
):
    outputs = self.original_forward(
        input_ids=input_ids,
        return_dict=return_dict,
        **kwargs,
    )

    if labels is None:
        return outputs

    logits = outputs.logits if return_dict is not False else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    # Keep model outputs in their original dtype, but do loss math in fp32 for stability.
    # The downstream eval path already casts logits to float before scoring; align training here.
    shift_logits_for_loss = shift_logits.float()
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()

    pad_id = getattr(self.config, "pad_token_id", None)
    if pad_id is not None:
        shift_labels[shift_labels == pad_id] = -100

    if isinstance(option_len, torch.Tensor):
        option_len = option_len.tolist()
    if isinstance(num_options, torch.Tensor):
        num_options = num_options.tolist()

    for i, L in enumerate(option_len):
        shift_labels[i, :-int(L)] = -100

    loss_fct = CrossEntropyLoss(ignore_index=-100)

    if num_options is not None:
        log_probs = F.log_softmax(shift_logits_for_loss, dim=-1)
        mask = shift_labels != -100
        shift_labels[~mask] = 0
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)

        if any(x != num_options[0] for x in num_options):
            loss = shift_logits_for_loss.new_zeros(())
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + int(num_options[start_id])
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0)
                _labels = labels[start_id:end_id][0].unsqueeze(0)
                loss = loss + loss_fct(_logits, _labels)
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            nopt = int(num_options[0])
            selected_log_probs = selected_log_probs.view(-1, nopt)
            labels = labels.view(-1, nopt)[:, 0]
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(
            shift_logits_for_loss.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )

    if return_dict is False:
        output = (logits,) + outputs[1:]
        return (loss,) + output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def _cfg_get(obj, name: str, default=None):
    value = getattr(obj, name, default)
    if isinstance(value, dict):
        return default
    return value


def get_tune_block(task_config, primary: str):
    if task_config is None:
        return None

    tune = getattr(getattr(task_config, "task", None), "tune", None)
    if tune is None:
        return None

    block = getattr(tune, primary, None)
    if block is not None:
        return block

    fallback = "zo" if primary == "fo" else "fo"
    return getattr(tune, fallback, None)


def get_tune_name(task_config) -> str:
    if task_config is None:
        return ""
    tune = getattr(getattr(task_config, "task", None), "tune", None)
    if tune is None:
        return ""
    ds = getattr(tune, "tune_dataset", None)
    if ds is None:
        return ""
    name = getattr(ds, "name", "")
    return str(name).lower()


def get_lora_cfg(task_config):
    model_cfg = getattr(task_config, "model", None)
    peft_cfg = getattr(model_cfg, "peft", None) if model_cfg is not None else None
    lora_cfg = getattr(peft_cfg, "lora", None) if peft_cfg is not None else None
    if isinstance(lora_cfg, dict):
        return None
    return lora_cfg


def get_lora_scope(task_config) -> str:
    lora_cfg = get_lora_cfg(task_config)
    return str(_cfg_get(lora_cfg, "attach_scope", "full")).lower() if lora_cfg is not None else "full"


def get_lora_layer_idx(task_config) -> Optional[int]:
    lora_cfg = get_lora_cfg(task_config)
    layer_idx = _cfg_get(lora_cfg, "layer_idx", None) if lora_cfg is not None else None
    return int(layer_idx) if layer_idx is not None else None


def infer_lora_scope_from_model(model: torch.nn.Module) -> Tuple[str, Optional[int]]:
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if not trainable_names:
        return "unknown", None

    layer_ids = set()
    unmatched = 0
    for name in trainable_names:
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m is None:
            unmatched += 1
            continue
        layer_ids.add(int(m.group(1)))

    if unmatched == 0 and len(layer_ids) == 1:
        return "single_layer", next(iter(layer_ids))
    return "full", None


def ensure_lora_scope(task_config, expected_scope: Optional[str], model: Optional[torch.nn.Module] = None):
    if expected_scope is None:
        return None
    scope = get_lora_scope(task_config)
    layer_idx = get_lora_layer_idx(task_config)
    normalized_expected = "single_layer" if expected_scope in {"layer", "single_layer"} else expected_scope
    normalized_scope = "single_layer" if scope in {"layer", "single_layer"} else scope

    if normalized_scope == normalized_expected:
        return layer_idx

    inferred_scope = None
    inferred_layer_idx = None
    if model is not None:
        inferred_scope, inferred_layer_idx = infer_lora_scope_from_model(model)
        if inferred_scope == normalized_expected:
            logger.warning(
                f"LoRA config scope mismatch: config says {scope!r}, but attached model looks like "
                f"{inferred_scope!r} (layer_idx={inferred_layer_idx}). Proceeding with model-derived scope."
            )
            return inferred_layer_idx if inferred_scope == "single_layer" else layer_idx

    if normalized_scope != normalized_expected:
        raise ValueError(
            f"Expected LoRA attach_scope={normalized_expected!r}, but got {scope!r}."
        )
    return layer_idx


def assert_lora_model(model: torch.nn.Module):
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise ValueError("Expected a LoRA model with trainable adapter parameters, but found no trainable tensors.")
    if not any("lora_" in name for name in trainable_names):
        raise ValueError(
            "Expected LoRA adapter parameters to be trainable, but no trainable parameter name contains 'lora_'."
        )


def attach_mezo_classification_forward_if_needed(model: torch.nn.Module, task_config):
    mezo_classification_tasks = {
        "sst2_mezo", "rte_mezo", "cb_mezo", "boolq_mezo",
        "wsc_mezo", "wic_mezo", "multirc_mezo",
    }
    if get_tune_name(task_config) not in mezo_classification_tasks:
        return
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
    model.forward = forward_wrap_with_option_len.__get__(model, type(model))


def maybe_cast_trainable_fp32(model: torch.nn.Module, log_prefix: str):
    for n, p in model.named_parameters():
        if p.requires_grad and p.is_floating_point() and p.data.dtype != torch.float32:
            p.data = p.data.float()
            logger.info(f"[{log_prefix}] cast to fp32: {n}")


def collect_trainable_params(model: torch.nn.Module, cast_trainable_fp32: bool, log_prefix: str):
    if cast_trainable_fp32:
        maybe_cast_trainable_fp32(model, log_prefix)
    trainable = [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]
    if not trainable:
        raise ValueError(f"[{log_prefix}] No floating-point trainable parameters found after LoRA attachment.")
    return trainable


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def log_trainable_sample(model: torch.nn.Module, max_items: int = 30):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n in names[:max_items]:
        logger.info(f"  [trainable] {n}")
    if len(names) > max_items:
        logger.info(f"  ... (+{len(names) - max_items} more trainable tensors)")


def ensure_optimizer_has_train_eval(opt):
    if opt is None:
        return
    if not hasattr(opt, "train"):
        opt.train = types.MethodType(lambda self, mode=True: self, opt)
    if not hasattr(opt, "eval"):
        opt.eval = types.MethodType(lambda self: self, opt)
