"""Collocation point sampling: Latin hypercube and RAD residual-based resampling."""

import numpy as np
import torch
from scipy.stats import qmc


def latin_hypercube(n, bounds, seed=None):
    """n Latin-hypercube samples in the box `bounds` = [(lo, hi), ...] -> (n, d)."""
    bounds = np.asarray(bounds, dtype=np.float64)
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    u = sampler.random(n)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def uniform(n, bounds, rng=None):
    rng = rng or np.random.default_rng()
    bounds = np.asarray(bounds, dtype=np.float64)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, len(bounds)))


def rad_resample(residual_fn, bounds, n_points, n_candidates=20000, k=1.0, c=1.0,
                 rng=None):
    """Residual-based Adaptive Distribution sampling (Wu et al., arXiv:2207.10289).

    Draws candidates uniformly, then samples collocation points with probability
    p ~ |r|^k / mean(|r|^k) + c, which concentrates points where the PDE residual
    is large while keeping global coverage (c > 0).

    residual_fn: maps an (n, d) float numpy array to per-point |residual| (n,).
    """
    rng = rng or np.random.default_rng()
    candidates = uniform(n_candidates, bounds, rng)
    r = np.abs(residual_fn(candidates)).reshape(-1)
    p = r**k / max(r.mean() ** k, 1e-12) + c
    p = p / p.sum()
    idx = rng.choice(n_candidates, size=n_points, replace=False, p=p)
    return candidates[idx]


def to_tensor(x, device="cpu", requires_grad=False):
    t = torch.as_tensor(np.asarray(x), dtype=torch.get_default_dtype(), device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t
