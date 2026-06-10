"""Differential operators via autograd, for building PDE residuals."""

import torch


def grad(u, x):
    """First derivatives du/dx, shape (N, d) for u of shape (N, 1) or (N,)."""
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]


def derivative(u, x, dim, order=1):
    """n-th derivative of u with respect to x[:, dim], shape (N, 1)."""
    du = u
    for _ in range(order):
        du = grad(du, x)[:, dim : dim + 1]
    return du


def laplacian(u, x, dims):
    """Sum of second derivatives over the given input dims, shape (N, 1)."""
    first = grad(u, x)
    out = 0.0
    for d in dims:
        out = out + grad(first[:, d : d + 1], x)[:, d : d + 1]
    return out
