"""Serial ES from Qiu et al. (2025), Algorithm 2, with explicit CPU restoration.

Normalized rewards use population standard deviation (ddof=0). Alpha absorbs
1/sigma. Seeds are independent per named tensor; there are no antithetic probes.
CPU snapshots retain the parameter dtype and restore each population member to
the exact same starting point, avoiding accumulated fp16 restoration drift.
"""
import math
import random
import numpy as np
import torch
from dominant_layer_zo.methods.scope import generator_for


def normalized_rewards(rewards):
    values = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("ES requires at least two finite scalar rewards")
    if np.all(values == values[0]):
        return np.zeros_like(values)
    return (values - values.mean()) / (values.std(ddof=0) + 1e-8)


class EvolutionStrategies:
    def __init__(self, parameters, *, population_size=16, sigma=.001, alpha=.0005, seed=0):
        self.parameters = list(parameters)
        if not self.parameters or type(population_size) is not int or population_size < 2:
            raise ValueError("ES requires selected parameters and population_size >= 2")
        if not all(math.isfinite(x) and x > 0 for x in (sigma, alpha)):
            raise ValueError("ES sigma/alpha must be finite and positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("ES seed must be a nonnegative integer")
        for _, p in self.parameters:
            if p.dtype not in (torch.float16, torch.float32):
                raise ValueError("ES supports float16/float32, not raw bf16")
        self.population_size, self.sigma, self.alpha = population_size, sigma, alpha
        self.rng = random.Random(seed)

    def noise(self, name, param, seed):
        return torch.randn(param.shape, device=param.device, dtype=param.dtype,
                           generator=generator_for(param, seed, name))

    @torch.no_grad()
    def step(self, reward_fn):
        seeds = [self.rng.randrange(2**63 - 1) for _ in range(self.population_size)]
        base = [p.detach().to(device="cpu", copy=True) for _, p in self.parameters]

        def restore():
            for (_, param), value in zip(self.parameters, base):
                param.copy_(value)

        rewards = []
        try:
            for seed in seeds:
                try:
                    for name, param in self.parameters:
                        param.add_(self.noise(name, param, seed), alpha=self.sigma)
                    reward = float(reward_fn())
                    if not math.isfinite(reward):
                        raise FloatingPointError("Nonfinite ES reward")
                    rewards.append(reward)
                finally:
                    restore()
            scores = normalized_rewards(rewards)
            update_squared_norm = 0.0
            changed_entries = 0
            for name, param in self.parameters:
                # One tensor at a time; fp32 accumulation is not a master model.
                update = torch.zeros_like(param, dtype=torch.float32)
                for seed, score in zip(seeds, scores):
                    update.add_(self.noise(name, param, seed).float(), alpha=float(score))
                updated = param.float() + (self.alpha / self.population_size) * update
                cast = updated.to(param.dtype)
                if not torch.isfinite(cast).all():
                    raise FloatingPointError("ES update overflows parameter dtype")
                delta = cast.float() - param.float()
                update_squared_norm += float(delta.square().sum())
                changed_entries += int(torch.count_nonzero(cast != param))
                param.copy_(cast)
                del delta
        except BaseException:
            restore()
            raise
        return {"population_seeds": seeds, "rewards": rewards,
                "normalized_rewards": scores.tolist(),
                "mean_reward": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "actual_update_l2": math.sqrt(update_squared_norm),
                "changed_parameter_entries": changed_entries,
                "objective_evaluations": self.population_size}
