import types
import logging
from typing import Tuple

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


def _get_fo_cfg(task_config):
    """
    Supports either:
      - task.tune.fo (recommended naming for FO)
      - task.tune.zo (fallback, so your existing YAML templates can be reused)
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


def _set_all_trainable(model: torch.nn.Module):
    """Ensure every parameter is trainable (full finetune)."""
    for p in model.parameters():
        p.requires_grad = True


def _maybe_cast_trainable_fp32(model: torch.nn.Module):
    """
    Cast ONLY trainable tensors to fp32.
    If you full-finetune and enable this while model is bf16/fp16,
    it will cast the *entire model* to fp32 (big memory hit).
    """
    for n, p in model.named_parameters():
        if p.requires_grad and p.data.dtype != torch.float32:
            p.data = p.data.float()
            logger.info(f"[fo_full] cast to fp32: {n}")


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


def _ensure_optimizer_has_train_eval(opt):
    """
    Provide no-op train/eval hooks for Transformers Trainer and Accelerate.
    """
    if opt is None:
        return

    if not hasattr(opt, "train"):
        opt.train = types.MethodType(lambda self, mode=True: self, opt)

    if not hasattr(opt, "eval"):
        opt.eval = types.MethodType(lambda self: self, opt)


class MyTrainer(Trainer):
    """
    First-order (standard backprop) trainer that finetunes the WHOLE model.
    (No freezing / layer selection.)
    """

    def __init__(self, *args, **kwargs):
        # Consume task_config before forwarding arguments to the base Trainer.
        self.task_config = kwargs.pop("task_config", None)
        super().__init__(*args, **kwargs)

        fo_cfg = _get_fo_cfg(self.task_config)
        mezo_classification_tasks = {
            "sst2_mezo", "rte_mezo", "cb_mezo", "boolq_mezo",
            "wsc_mezo", "wic_mezo", "multirc_mezo",
        }
        tune_name = _get_tune_name(self.task_config)

        # Optional behavior flags from config (nice to keep symmetry with fo_layer_trainer)
        cast_fp32 = False
        cfg_lr = None
        if fo_cfg is not None:
            cast_fp32 = bool(getattr(fo_cfg, "cast_trainable_fp32", False))

            cfg_lr = getattr(fo_cfg, "lr", None)
            try:
                cfg_lr = float(cfg_lr) if cfg_lr is not None else None
            except Exception:
                cfg_lr = None

        # If TrainingArguments lr is 0 (common in ZO templates), optionally adopt fo_cfg.lr
        if getattr(self.args, "learning_rate", 0.0) == 0.0 and cfg_lr is not None and cfg_lr > 0:
            logger.info(f"[fo_full] TrainingArguments.learning_rate is 0. Using cfg lr={cfg_lr} from tune.(fo|zo).lr")
            self.args.learning_rate = cfg_lr

        # Full finetune: ensure everything is trainable
        logger.info("[fo_full] Full-model finetune: setting requires_grad=True for all parameters")
        _set_all_trainable(self.model)

        if tune_name in mezo_classification_tasks:
            if not hasattr(self.model, "original_forward"):
                self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        # Optional fp32 trainable parameters increase memory use for fp16/bf16 models.
        if cast_fp32:
            logger.info("[fo_full] cast_trainable_fp32=True: casting ALL trainable tensors to fp32 (full model!)")
            _maybe_cast_trainable_fp32(self.model)

        # Gradient checkpointing common gotchas
        if getattr(self.args, "gradient_checkpointing", False):
            # Llama-like models typically require use_cache=False when checkpointing
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                if self.model.config.use_cache:
                    logger.info("[fo_full] gradient_checkpointing=True: setting model.config.use_cache=False")
                    self.model.config.use_cache = False

            # Safe to enable if available (won't hurt full finetune)
            if hasattr(self.model, "enable_input_require_grads"):
                logger.info("[fo_full] gradient_checkpointing=True: enabling input require grads")
                self.model.enable_input_require_grads()

        trainable, total = _count_params(self.model)
        logger.info(f"[fo_full] Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total,1):.6f}%)")
        _log_trainable_sample(self.model)

    def create_optimizer(self):
        super().create_optimizer()
        _ensure_optimizer_has_train_eval(self.optimizer)
        return self.optimizer
