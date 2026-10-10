"""Reproducible quadratic-spline noise for opt-in native STORM sampling.

Scrambled SciPy Halton respects the configured seed even at low dimensions.
Exact interpolation is vectorized through a precomputed spline basis; it differs
from the historical adaptive smoothing spline. Optional antithetic pairs cancel
the empirical mean. Banks are prepared once and selected by the optimizer step.
"""

import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from scipy.special import ndtri
from scipy.stats import qmc


class SeededKnotSamples:
    def __init__(self, horizon, d_action, seed=0, tensor_args=None, knot_scale=4,
                 fixed_samples=True, antithetic=False, bank_size=1,
                 filter_coeffs=None):
        if horizon < 1 or d_action < 1 or knot_scale < 1:
            raise ValueError('horizon, d_action and knot_scale must be positive')
        if filter_coeffs is not None:
            raise ValueError('seeded-knot uses spline interpolation; filter_coeffs must be None')
        self.horizon = int(horizon)
        self.d_action = int(d_action)
        self.n_knots = self.horizon // int(knot_scale)
        if self.n_knots <= 2:
            raise ValueError('Quadratic spline requires horizon // knot_scale >= 3')
        self.seed = int(seed)
        self.tensor_args = dict(tensor_args or {'device': 'cpu', 'dtype': torch.float32})
        self.fixed_samples = bool(fixed_samples)
        self.antithetic = bool(antithetic)
        self.bank_size = int(bank_size)
        if self.bank_size < 1:
            raise ValueError('bank_size must be positive')
        knots = np.linspace(0.0, 1.0, self.n_knots)
        basis = make_interp_spline(knots, np.eye(self.n_knots), k=2)(
            np.linspace(0.0, 1.0, self.horizon))
        self.basis = torch.as_tensor(basis, **self.tensor_args)
        self.samples = None
        # Shape is part of the cache key: ancillary controller calls may request
        # a different count than the rollout population.
        self.bank = {}

    def get_samples(self, sample_shape, base_seed=None, seed=None, **kwargs):
        if len(sample_shape) not in (1, 2) or int(sample_shape[0]) < 0:
            raise ValueError('sample_shape must contain a nonnegative particle count')
        n = int(sample_shape[0])
        # generate_noise() also requests (count, horizon) or (count, 1).
        requested_horizon = self.horizon if len(sample_shape) == 1 else int(sample_shape[1])
        if requested_horizon < 1 or requested_horizon > self.horizon:
            raise ValueError('requested sample horizon must be in [1, horizon]')
        # OLGaussianMPC.generate_noise uses the older `seed` spelling.
        if base_seed is None:
            base_seed = seed
        index = 0 if base_seed is None else (int(base_seed) - self.seed) % self.bank_size
        key = (n, index)
        cache = self.fixed_samples or self.bank_size > 1
        if cache and key in self.bank:
            self.samples = self.bank[key]
            return self.samples[:, :requested_horizon]
        draw_seed = self.seed if self.fixed_samples or base_seed is None else int(base_seed)
        if self.bank_size > 1:
            draw_seed = self.seed + 104729 * index
        sampler = qmc.Halton(d=self.d_action * self.n_knots, scramble=True, seed=draw_seed)
        draw_count = n // 2 if self.antithetic else n
        normal = ndtri(np.clip(sampler.random(draw_count), 1.0e-7, 1.0 - 1.0e-7))
        if self.antithetic:
            normal = np.concatenate((normal, -normal, np.zeros((n % 2, normal.shape[1]))), axis=0)
        coefficients = torch.as_tensor(normal, **self.tensor_args).reshape(n, self.d_action, self.n_knots)
        self.samples = (coefficients @ self.basis.T).transpose(1, 2).contiguous()
        if cache:
            self.bank[key] = self.samples
        return self.samples[:, :requested_horizon]

    def prepare(self, sample_shape):
        """Precompute banks before entering the repeated optimization loop."""
        if self.fixed_samples or self.bank_size > 1:
            for index in range(self.bank_size):
                self.get_samples(sample_shape, base_seed=self.seed + index)
