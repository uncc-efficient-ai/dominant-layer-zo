import logging

from transformers import Trainer

from .lora_common import (
    assert_lora_model,
    attach_mezo_classification_forward_if_needed,
    collect_trainable_params,
    count_params,
    ensure_lora_scope,
    ensure_optimizer_has_train_eval,
    get_lora_layer_idx,
    get_tune_block,
    log_trainable_sample,
)

logger = logging.getLogger(__name__)


class MyTrainer(Trainer):
    """First-order trainer for LoRA attached only inside one selected transformer layer."""

    def __init__(self, *args, task_config=None, **kwargs):
        self.task_config = kwargs.pop("task_config", task_config)
        super().__init__(*args, **kwargs)

        fo_cfg = get_tune_block(self.task_config, "fo")
        cast_fp32 = bool(getattr(fo_cfg, "cast_trainable_fp32", False)) if fo_cfg is not None else False
        cfg_lr = getattr(fo_cfg, "lr", None) if fo_cfg is not None else None
        try:
            cfg_lr = float(cfg_lr) if cfg_lr is not None else None
        except Exception:
            cfg_lr = None

        if getattr(self.args, "learning_rate", 0.0) == 0.0 and cfg_lr is not None and cfg_lr > 0:
            logger.info(f"[fo_lora_layer] TrainingArguments.learning_rate is 0. Using cfg lr={cfg_lr}.")
            self.args.learning_rate = cfg_lr

        ensure_lora_scope(self.task_config, "single_layer", model=self.model)
        assert_lora_model(self.model)
        attach_mezo_classification_forward_if_needed(self.model, self.task_config)
        collect_trainable_params(self.model, cast_fp32, "fo_lora_layer")

        if getattr(self.args, "gradient_checkpointing", False):
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache") and self.model.config.use_cache:
                logger.info("[fo_lora_layer] gradient_checkpointing=True: setting model.config.use_cache=False")
                self.model.config.use_cache = False
            if hasattr(self.model, "enable_input_require_grads"):
                logger.info("[fo_lora_layer] gradient_checkpointing=True: enabling input require grads")
                self.model.enable_input_require_grads()

        trainable, total = count_params(self.model)
        layer_idx = get_lora_layer_idx(self.task_config)
        logger.info(
            f"[fo_lora_layer] layer_idx={layer_idx} trainable params: "
            f"{trainable:,} / {total:,} ({100.0 * trainable / max(total,1):.6f}%)"
        )
        log_trainable_sample(self.model)

    def create_optimizer(self):
        super().create_optimizer()
        ensure_optimizer_has_train_eval(self.optimizer)
        return self.optimizer
