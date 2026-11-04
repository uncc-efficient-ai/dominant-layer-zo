"""Overflow, exact restoration, and failed-step rollback contracts."""
import unittest
from unittest.mock import patch
import torch

from dominant_layer_zo.methods.agzo import AGZOStep, activation_basis
from dominant_layer_zo.methods.numerics import ParameterSnapshot, central_coefficient
from dominant_layer_zo.sparse import SparseMeZOStep


class NumericsTests(unittest.TestCase):
    def test_large_coefficient_is_scaled_before_half_conversion(self):
        for method in ('agzo', 'sparse'):
            with self.subTest(method=method):
                torch.manual_seed(17)
                model = torch.nn.Linear(4, 4).half()
                base = {n: p.detach().clone() for n, p in model.named_parameters()}
                if method == 'agzo':
                    stepper = AGZOStep(model, model.named_parameters())
                    values = iter([0., 50., -50.])
                else:
                    stepper = SparseMeZOStep(model.named_parameters(), .75)
                    values = iter([50., -50.])
                def loss():
                    model(torch.ones(2, 4).half())
                    return torch.tensor(next(values), dtype=torch.float16)
                _, info = stepper.step(loss, seed=7, epsilon=.0005, lr=1e-6)
                coefficient = info.get('agzo_coefficient', info.get('zo/coeff'))
                self.assertEqual(coefficient, 100000.)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(any(not torch.equal(p, base[n]) for n, p in model.named_parameters()))

    def test_failed_late_parameter_update_rolls_back_every_parameter(self):
        a = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16))
        b = torch.nn.Parameter(torch.tensor([65000.], dtype=torch.float16))
        before = [p.detach().clone() for p in (a, b)]
        with self.assertRaises(FloatingPointError):
            with ParameterSnapshot([('a', a), ('b', b)]) as weights:
                weights.write(lambda name, p: torch.ones_like(p), seed=7, scale=1000.)
                weights.commit()
        for p, expected in zip((a, b), before):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)

    def test_sparse_zero_and_nonfinite_objective_restore_exactly(self):
        p = torch.nn.Parameter(torch.tensor([.01, .03, .08, .1], dtype=torch.float16))
        stepper = SparseMeZOStep([('weight', p)], .75)
        initial = p.detach().clone()
        stepper.step(lambda: torch.tensor(0.), seed=2, epsilon=.001, lr=1e-6)
        torch.testing.assert_close(p, initial, rtol=0, atol=0)
        calls = iter([torch.tensor(1.), torch.tensor(float('nan'))])
        with self.assertRaises(FloatingPointError):
            stepper.step(lambda: next(calls), seed=3, epsilon=.001, lr=1e-6)
        torch.testing.assert_close(p, initial, rtol=0, atol=0)

    def test_fp32_scalar_division_and_large_activation_basis(self):
        coefficient = central_coefficient(torch.tensor(1e38), torch.tensor(-1e38), 1e-5)
        self.assertGreater(coefficient, 1e42)
        basis = activation_basis(torch.diag(torch.tensor([1e25, 1e23, 1e21])), 1, 5)
        self.assertTrue(torch.isfinite(basis).all())
        self.assertGreater(abs(basis[0, 0].item()), .9999)


if __name__ == '__main__':
    unittest.main()
