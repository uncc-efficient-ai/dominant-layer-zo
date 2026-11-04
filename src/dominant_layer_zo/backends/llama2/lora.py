import logging
import math
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


SUPPORTED_LLAMA_LORA_TARGETS = {
    "q_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def find_module(root_module: nn.Module, key: str) -> Tuple[nn.Module, str, nn.Module]:
    """Find a module by its dotted path and return (parent, child_name, child)."""
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    child_name = sub_keys[-1]
    child_module = getattr(parent_module, child_name)
    return parent_module, child_name, child_module


class MezoLoRALinear(nn.Linear):
    """LoRA dense layer closely following the official MeZO LoRA implementation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.r = int(r)
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive for MezoLoRALinear.")
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(self.r)
        self.fan_in_fan_out = bool(fan_in_fan_out)
        self.merge_weights = bool(merge_weights)
        self.merged = False
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.r)))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.reset_lora_parameters()
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
    ) -> "MezoLoRALinear":
        replacement = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            merge_weights=merge_weights,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        replacement.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            replacement.bias.data.copy_(linear.bias.data)
        return replacement

    def _transpose_if_needed(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.transpose(0, 1) if self.fan_in_fan_out else weight

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.merge_weights:
            return self

        if mode and self.merged:
            self.weight.data -= self._transpose_if_needed(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
        elif (not mode) and (not self.merged):
            self.weight.data += self._transpose_if_needed(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self._transpose_if_needed(self.weight), bias=self.bias)
        if not self.merged:
            delta = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
            delta = delta @ self.lora_B.transpose(0, 1)
            result = result + delta * self.scaling
        return result


def attach_mezo_llama_lora(
    model: nn.Module,
    *,
    target_module_paths: Iterable[str],
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    fan_in_fan_out: bool = False,
) -> nn.Module:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if str(model_type).lower() != "llama":
        raise ValueError(
            "Built-in custom MeZO-style LoRA is currently implemented only for LLaMA models."
        )

    target_module_paths = sorted(set(target_module_paths))
    if not target_module_paths:
        raise ValueError("No target module paths were resolved for custom MeZO-style LoRA attachment.")

    unsupported = sorted({
        module_name.rsplit(".", 1)[-1]
        for module_name in target_module_paths
        if module_name.rsplit(".", 1)[-1] not in SUPPORTED_LLAMA_LORA_TARGETS
    })
    if unsupported:
        raise ValueError(
            "Built-in custom MeZO-style LLaMA LoRA currently supports only "
            f"{sorted(SUPPORTED_LLAMA_LORA_TARGETS)}, "
            f"but got: {unsupported}"
        )

    injected: List[str] = []
    for module_name in target_module_paths:
        parent_module, child_name, child_module = find_module(model, module_name)
        if isinstance(child_module, MezoLoRALinear):
            injected.append(module_name)
            continue
        if not isinstance(child_module, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear at target module {module_name!r}, got {type(child_module)!r}."
            )
        replacement = MezoLoRALinear.from_linear(
            child_module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            merge_weights=False,
        )
        setattr(parent_module, child_name, replacement)
        injected.append(module_name)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_names:
        raise ValueError("Custom MeZO-style LoRA attachment produced no trainable adapter parameters.")

    target_leaves = sorted({module_name.rsplit(".", 1)[-1] for module_name in injected})
    model._custom_lora_impl = "mezo_llama_custom"
    model._custom_lora_target_modules = tuple(injected)
    logger.info(
        "[peft] Injected built-in custom MeZO-style LoRA into %d LLaMA linear modules.",
        len(injected),
    )
    logger.info("[peft] Custom LoRA target leaves: %s", target_leaves)
    logger.info("[peft] Custom LoRA target modules: %s", injected)
    return model
