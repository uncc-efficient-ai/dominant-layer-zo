"""Activation-guided Gaussian finite differences."""
import math
from contextlib import contextmanager

import torch
from .seeds import sample_seed
from .numerics import ParameterSnapshot, objective_value, central_coefficient


@torch.no_grad()
def activation_basis(activation, rank, power_iterations):
    x = activation.detach().reshape(-1, activation.shape[-1]).float()
    if not x.numel() or not torch.isfinite(x).all():
        raise FloatingPointError("AGZO requires nonempty finite activations")
    # Uniform scaling preserves the subspace and bounds covariance products.
    magnitude = x.abs().max()
    if magnitude > 1e10:
        x = x / magnitude
    rank = min(rank, x.shape[1])
    q = torch.randn(x.shape[1], rank, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(q, mode="reduced").Q
    for _ in range(power_iterations):
        z = x.T @ (x @ q)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite AGZO power iteration")
        q = torch.linalg.qr(z, mode="reduced").Q
    basis = q.T
    return (basis / (basis.norm(p=2, dim=1, keepdim=True) + 1e-12)).T


class AGZOStep:
    def __init__(self, model, parameters, *, rank=1, power_iterations=5):
        if type(rank) is not int or rank < 1:
            raise ValueError("AGZO rank must be positive")
        if type(power_iterations) is not int or power_iterations < 1:
            raise ValueError("AGZO power_iterations must be positive")
        self.model, self.parameters = model, list(parameters)
        if not self.parameters:
            raise ValueError("AGZO requires trainable parameters")
        self.rank, self.power_iterations = rank, power_iterations
        self.bases = {}

    @contextmanager
    def collect(self):
        names = {id(p): name for name, p in self.parameters}
        handles = []
        self.bases.clear()
        try:
            for module in self.model.modules():
                if not isinstance(module, torch.nn.Linear) or id(module.weight) not in names:
                    continue
                name = names[id(module.weight)]

                def hook(mod, inputs, output, name=name):
                    rank = min(self.rank, *mod.weight.shape)
                    basis = activation_basis(inputs[0], rank, self.power_iterations)
                    self.bases[name] = basis.to(mod.weight.dtype)

                handles.append(module.register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def noise(self, name, param):
        basis = self.bases.get(name)
        shape = param.shape if basis is None else (param.shape[0], basis.shape[1])
        z = torch.normal(mean=0., std=1., size=shape, device=param.device, dtype=param.dtype)
        return z if basis is None else (z @ basis.T) / math.sqrt(basis.shape[1])

    @torch.no_grad()
    def step(self, loss_fn, *, seed, epsilon, lr):
        if not all(math.isfinite(x) and x > 0 for x in (epsilon, lr)):
            raise ValueError("AGZO epsilon/lr must be finite and positive")
        self.model.eval()
        try:
            with self.collect():
                objective_value(loss_fn)
            if not self.bases:
                raise ValueError("AGZO captured no selected linear activation")
            with ParameterSnapshot(self.parameters) as weights:
                weights.write(self.noise, seed, epsilon)
                positive = objective_value(loss_fn)
                weights.write(self.noise, seed, -epsilon)
                negative = objective_value(loss_fn)
                coefficient = central_coefficient(positive, negative, epsilon)
                weights.write(self.noise, seed, -lr * coefficient)
                weights.commit()
            return positive, {"agzo_coefficient": coefficient,
                              "agzo_basis_count": len(self.bases), "agzo_forwards": 3}
        finally:
            self.bases.clear()


class AGZOTrainerMixin:
    """Use the family's ZO objective and selected parameter scope."""
    def __init__(self, *args, task_config=None, **kwargs):
        super().__init__(*args, task_config=task_config, **kwargs)
        opts = task_config.task.tune.zo.agzo
        self.agzo_stepper = AGZOStep(
            self.model, [(n, p) for n, p in self.model.named_parameters() if p.requires_grad],
            rank=opts.rank, power_iterations=opts.power_iterations)

    def _get_learning_rate(self):
        return self.lr

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        loss, info = self.agzo_stepper.step(
            lambda: self.compute_loss_from_batch(model, inputs),
            seed=sample_seed(self), epsilon=self.eps, lr=self.lr)
        self._last_finite_loss = loss
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(info)
        return loss
