"""Perturbation seed stream, independent of data sampling."""
import numpy as np


def sample_seed(trainer):
    if not hasattr(trainer, '_perturbation_rng'):
        trainer._perturbation_rng = np.random.RandomState(trainer.seed)
    return int(trainer._perturbation_rng.randint(1_000_000_000))
