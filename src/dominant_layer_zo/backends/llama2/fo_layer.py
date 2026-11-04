# external_code/tee_train/trainers/fo_layer_trainer/fo_layer_trainer.py
import types
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


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
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        shift_labels[~mask] = 0
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)

        if any(x != num_options[0] for x in num_options):
            loss = 0.0
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
            shift_logits.view(-1, self.config.vocab_size),
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


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    """
    Safely unwrap common wrappers (DDP / PEFT / HF base_model_prefix).

    IMPORTANT: Do NOT unwrap by calling transformers' `.base_model` property directly.
    On some models (e.g. LlamaModel) `.base_model` can return `self`, which causes
    infinite recursion if you keep unwrapping.
    """
    seen = set()

    while True:
        mid = id(m)
        if mid in seen:
            # cycle detected; stop unwrapping
            return m
        seen.add(mid)

        # 1) DDP / DataParallel
        if hasattr(m, "module"):
            m = m.module
            continue

        # 2) PEFT wrappers (if ever used)
        if hasattr(m, "get_base_model"):
            try:
                base = m.get_base_model()
                if isinstance(base, torch.nn.Module) and base is not m:
                    m = base
                    continue
            except Exception:
                pass

        # 3) HF PreTrainedModel-style base model via base_model_prefix (SAFE)
        prefix = getattr(m, "base_model_prefix", None)
        if isinstance(prefix, str) and prefix:
            base = getattr(m, prefix, None)
            if isinstance(base, torch.nn.Module) and base is not m:
                m = base
                continue

        # 4) Last resort: some wrappers store base_model as a real attribute (not a property)
        base_attr = getattr(m, "__dict__", {}).get("base_model", None)
        if isinstance(base_attr, torch.nn.Module) and base_attr is not m:
            m = base_attr
            continue

        return m


def _find_llama_layers(model: torch.nn.Module):
    core = _unwrap_model(model)

    # LlamaForCausalLM -> has .model.layers
    if hasattr(core, "model") and hasattr(core.model, "layers"):
        return core.model.layers, "model.model.layers"

    # LlamaModel -> has .layers directly
    if hasattr(core, "layers"):
        return core.layers, "layers"

    return None, "unknown"


def _get_fo_cfg(task_config):
    """
    Supports either:
      - task.tune.fo (recommended naming for FO)
      - task.tune.zo (fallback, so your existing YAML can be reused)
    """
    if task_config is None:
        return None

    tune = getattr(getattr(task_config, "task", None), "tune", None)
    if tune is None:
        return None

    fo = getattr(tune, "fo", None)
    if fo is not None:
        return fo

    # Fall back to the ZO configuration block.
    zo = getattr(tune, "zo", None)
    return zo

def _get_tune_name(task_config) -> str:
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


def _freeze_all_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _set_trainable(module: torch.nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad = trainable


def _maybe_cast_trainable_fp32(model: torch.nn.Module):
    """
    Cast ONLY trainable tensors to fp32.
    Helpful if model is bf16/fp16 but you want stable updates for a small subset.
    """
    for n, p in model.named_parameters():
        if p.requires_grad and p.data.dtype != torch.float32:
            p.data = p.data.float()
            logger.info(f"[fo_layer] cast to fp32: {n}")


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _log_trainable_sample(model: torch.nn.Module, max_items: int = 30):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n in names[:max_items]:
        logger.info(f"  [trainable] {n}")
    if len(names) > max_items:
        logger.info(f"  ... (+{len(names) - max_items} more trainable tensors)")


def _apply_one_layer_freeze(model: torch.nn.Module, layer_idx: int, part: str):
    """
    Freeze everything except selected layer/part.
      part:
        - "layer": entire transformer block
        - "mlp": only MLP submodule
        - "attn": only attention submodule
    """
    _freeze_all_params(model)

    layers, hint = _find_llama_layers(model)
    if layers is None:
        raise RuntimeError(
            f"[fo_layer] Could not locate Llama layers list on model. hint={hint} type={type(model)}"
        )

    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"[fo_layer] layer_idx={layer_idx} out of range, model has {len(layers)} layers ({hint}).")

    layer = layers[layer_idx]
    part = (part or "layer").lower()

    if part == "layer":
        _set_trainable(layer, True)
    elif part == "mlp":
        if not hasattr(layer, "mlp"):
            raise RuntimeError(f"[fo_layer] layer has no .mlp attribute at idx={layer_idx}")
        _set_trainable(layer.mlp, True)
    elif part in {"attn", "attention"}:
        # HF LlamaDecoderLayer uses .self_attn
        if hasattr(layer, "self_attn"):
            _set_trainable(layer.self_attn, True)
        elif hasattr(layer, "attn"):
            _set_trainable(layer.attn, True)
        else:
            raise RuntimeError(f"[fo_layer] layer has no attention attribute (.self_attn/.attn) at idx={layer_idx}")
    else:
        raise ValueError(f"[fo_layer] Unsupported part='{part}'. Use: layer | mlp | attn")

def _ensure_optimizer_has_train_eval(opt):
    """
    Provide no-op train/eval hooks for Transformers Trainer and Accelerate.
    """
    if opt is None:
        return

    # no-op train(mode=True) to match common signatures
    if not hasattr(opt, "train"):
        opt.train = types.MethodType(lambda self, mode=True: self, opt)

    # no-op eval()
    if not hasattr(opt, "eval"):
        opt.eval = types.MethodType(lambda self: self, opt)

class MyTrainer(Trainer):
    """
    First-order (backprop) trainer that trains ONLY one transformer layer (or a part of it)
    and freezes everything else.
    """

    def __init__(self, *args, **kwargs):
        # Consume task_config before forwarding arguments to the base Trainer.
        self.task_config = kwargs.pop("task_config", None)
        super().__init__(*args, **kwargs)

        fo_cfg = _get_fo_cfg(self.task_config)
        if fo_cfg is None:
            raise RuntimeError(
                "[fo_layer] Could not find task.tune.fo or task.tune.zo in config. "
                "Add one of them with layer_idx/part."
            )

        layer_idx = int(getattr(fo_cfg, "layer_idx", 0))
        part = getattr(fo_cfg, "part", "layer")
        cast_fp32 = bool(getattr(fo_cfg, "cast_trainable_fp32", False))
        mezo_classification_tasks = {
            "sst2_mezo", "rte_mezo", "cb_mezo", "boolq_mezo",
            "wsc_mezo", "wic_mezo", "multirc_mezo",
        }

        # Safety: if Trainer args LR is 0 (common in ZO configs), optionally adopt fo_cfg.lr
        # so you can reuse your YAML template.
        cfg_lr = getattr(fo_cfg, "lr", None)
        try:
            cfg_lr = float(cfg_lr) if cfg_lr is not None else None
        except Exception:
            cfg_lr = None

        if getattr(self.args, "learning_rate", 0.0) == 0.0 and cfg_lr is not None and cfg_lr > 0:
            logger.info(f"[fo_layer] TrainingArguments.learning_rate is 0. Using cfg lr={cfg_lr} from tune.(fo|zo).lr")
            self.args.learning_rate = cfg_lr

        logger.info(f"[fo_layer] Freezing all params except layer_idx={layer_idx}, part={part}")
        _apply_one_layer_freeze(self.model, layer_idx=layer_idx, part=part)

        if _get_tune_name(self.task_config) in mezo_classification_tasks:
            if not hasattr(self.model, "original_forward"):
                self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if cast_fp32:
            logger.info("[fo_layer] cast_trainable_fp32=True: casting trainable tensors to fp32")
            _maybe_cast_trainable_fp32(self.model)

        # Enable input gradients so checkpointed blocks build a gradient graph
        # when embeddings and preceding blocks are frozen.
        if getattr(self.args, "gradient_checkpointing", False) and hasattr(self.model, "enable_input_require_grads"):
            logger.info("[fo_layer] gradient_checkpointing=True: enabling input require grads")
            self.model.enable_input_require_grads()

        trainable, total = _count_params(self.model)
        logger.info(f"[fo_layer] Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total,1):.6f}%)")
        _log_trainable_sample(self.model)


    def create_optimizer(self):
        super().create_optimizer()
        _ensure_optimizer_has_train_eval(self.optimizer)
        return self.optimizer
