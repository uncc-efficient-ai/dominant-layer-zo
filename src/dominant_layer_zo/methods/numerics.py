"""Finite ZO objectives and transactional parameter perturbations."""
import math
from contextlib import AbstractContextManager

import torch


def objective_value(loss_fn):
    value = loss_fn().detach().float()
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise FloatingPointError('ZO objective must be a finite scalar')
    return value.reshape(())


def central_coefficient(positive, negative, epsilon):
    value = (positive.item() - negative.item()) / (2 * epsilon)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ZO finite-difference coefficient')
    return value


class ParameterSnapshot(AbstractContextManager):
    """Restore exact weights on failure; each probe starts at the same weights."""
    @torch.no_grad()
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.original = [p.detach().to('cpu', copy=True) for _, p in self.parameters]
        self.committed = False

    @torch.no_grad()
    def write(self, noise_fn, seed, scale):
        if not math.isfinite(scale):
            raise FloatingPointError('Nonfinite ZO step scale')
        torch.manual_seed(seed)
        for (name, param), base in zip(self.parameters, self.original):
            direction = noise_fn(name, param).float()
            value = base.to(device=param.device, copy=True).float()
            value.add_(direction, alpha=scale)
            if (not torch.isfinite(value).all() or
                    value.abs().max() > torch.finfo(param.dtype).max):
                raise FloatingPointError(f'Unrepresentable ZO parameters: {name}')
            param.copy_(value)

    def commit(self):
        self.committed = True

    @torch.no_grad()
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None or not self.committed:
            for (_, param), base in zip(self.parameters, self.original):
                param.copy_(base)
        self.original.clear()
        return False
