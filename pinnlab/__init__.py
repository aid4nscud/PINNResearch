"""pinnlab: a small PyTorch library implementing the modern (2025/2026) PINN
training stack — Fourier/periodic embeddings, PirateNet blocks with random
weight factorization, self-adaptive loss balancing, causal weighting, RAD
adaptive sampling, and Adam -> SOAP -> float64 L-BFGS optimization."""

import random

import numpy as np
import torch

from . import fdm, losses, nn, operators, references, sampling, viz
from .optim import SOAP
from .trainer import Phase, default_phases, train

__all__ = ["fdm", "losses", "nn", "operators", "references", "sampling", "viz",
           "SOAP", "Phase", "default_phases", "train", "set_seed", "rel_l2"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rel_l2(pred, ref):
    """Relative L2 error ||pred - ref|| / ||ref|| over flattened arrays."""
    pred, ref = np.asarray(pred).ravel(), np.asarray(ref).ravel()
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
