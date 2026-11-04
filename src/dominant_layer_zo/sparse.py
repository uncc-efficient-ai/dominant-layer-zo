"""Magnitude-masked Gaussian finite differences shared by all model families."""
import math
from dataclasses import dataclass

import torch
from .methods.seeds import sample_seed
from .methods.numerics import ParameterSnapshot, objective_value, central_coefficient


@dataclass
class SparseMeZOConfig:
    selected_ratio: float = .25

    def __post_init__(self):
        if not math.isfinite(self.selected_ratio) or not 0 < self.selected_ratio <= 1:
            raise ValueError("SparseMeZO selected_ratio must be in (0, 1]")

    @property
    def mask_ratio(self):
        return 1 - self.selected_ratio


@dataclass
class SparseParamGroup:
    name: str
    param: torch.Tensor
    threshold: torch.Tensor
    initial_selected: int
    dense_entries: int


def configure_sparse_mezo_groups(named_parameters, config, trainer_label="SparseMeZO"):
    groups = []
    excluded = 0
    for name, param in named_parameters:
        if not param.requires_grad or not param.is_floating_point():
            continue
        if "norm" in name.lower():
            param.requires_grad_(False)
            excluded += param.numel()
            continue
        magnitude = param.detach().abs().flatten()
        k = max(int(config.selected_ratio * param.numel()), 1)
        threshold = magnitude.kthvalue(k).values.detach().clone()
        selected = int((magnitude < threshold).sum())
        groups.append(SparseParamGroup(name, param, threshold, selected, param.numel()))
    if not groups:
        raise ValueError(f"{trainer_label} selected no eligible parameter tensors")
    eligible = sum(g.dense_entries for g in groups)
    selected = sum(g.initial_selected for g in groups)
    return groups, {"dense_entries": eligible, "initial_selected_entries": selected,
                    "excluded_entries": excluded, "eligible_tensors": len(groups),
                    "initial_selected_fraction_of_eligible": selected / eligible}


def build_step_masks(groups):
    masks = {id(g.param): g.param.detach().abs() < g.threshold for g in groups}
    return masks, sum(int(mask.sum()) for mask in masks.values())


class SparseMeZOStep:
    def __init__(self, parameters, selected_ratio=.25):
        self.config = SparseMeZOConfig(selected_ratio)
        self.groups, self.metadata = configure_sparse_mezo_groups(parameters, self.config)
        self.parameters = [(g.name, g.param) for g in self.groups]

    @torch.no_grad()
    def step(self, loss_fn, *, seed, epsilon, lr):
        if not all(math.isfinite(x) and x > 0 for x in (epsilon, lr)):
            raise ValueError("SparseMeZO epsilon/lr must be finite and positive")
        masks, selected = build_step_masks(self.groups)

        def noise(name, param):
            z = torch.normal(mean=0., std=1., size=param.shape, device=param.device, dtype=param.dtype)
            return z.masked_fill_(~masks[id(param)], 0)

        with ParameterSnapshot(self.parameters) as weights:
            weights.write(noise, seed, epsilon)
            positive = objective_value(loss_fn)
            weights.write(noise, seed, -epsilon)
            negative = objective_value(loss_fn)
            coefficient = central_coefficient(positive, negative, epsilon)
            weights.write(noise, seed, -lr * coefficient)
            weights.commit()
        return positive, {"zo/loss_plus": positive.item(), "zo/loss_minus": negative.item(),
                          "zo/coeff": coefficient, "zo/sparse_selected_entries": selected}


class SparseMeZOTrainerMixin:
    """Use the family's ZO objective and selected parameter scope."""
    def __init__(self, *args, task_config=None, **kwargs):
        super().__init__(*args, task_config=task_config, **kwargs)
        ratio = task_config.task.tune.zo.sparse_selected_ratio
        self.sparse_stepper = SparseMeZOStep(self.model.named_parameters(), ratio)
        self.sparse_config = self.sparse_stepper.config
        self.sparse_param_groups = self.sparse_stepper.groups
        self.trainable = [g.param for g in self.sparse_param_groups]
        self.sparse_dense_entries = sum(g.dense_entries for g in self.sparse_param_groups)
        self.sparse_initial_selected_entries = sum(g.initial_selected for g in self.sparse_param_groups)
        self.trainable_dim = self.sparse_initial_selected_entries
        self.has_fp16_trainable = any(p.dtype == torch.float16 for p in self.trainable)

    def _get_learning_rate(self):
        return self.lr

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        model.eval()
        loss, info = self.sparse_stepper.step(
            lambda: self.compute_loss_from_batch(model, inputs),
            seed=sample_seed(self), epsilon=self.eps, lr=self.lr)
        self._last_finite_loss = loss
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(info)
        return loss


def sparse_metadata(trainer):
    groups = trainer.sparse_param_groups
    eligible = sum(g.dense_entries for g in groups)
    initial = sum(g.initial_selected for g in groups)
    return {
        "code_reference": "0c7a92858943752117264307ece4b03209109b37",
        "selected_ratio": trainer.sparse_config.selected_ratio,
        "masked_ratio": trainer.sparse_config.mask_ratio,
        "exclude_keywords": ["norm"],
        "threshold_policy": "fixed initial per-tensor kth absolute weight; strict less-than",
        "mask_policy": "rebuilt once per step and reused for both probes and update",
        "eligible_tensors": len(groups), "eligible_entries": eligible,
        "initial_selected_entries": initial, "initial_selected_fraction": initial / eligible,
        "ratio_one_policy": "strictly below maximum absolute weight",
        "update_dtype": "float32", "restoration": "exact_cpu_snapshot",
        "cpu_snapshot_bytes": sum(g.param.numel() * g.param.element_size() for g in groups),
    }
