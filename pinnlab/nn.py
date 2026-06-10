"""Network architectures for PINNs.

Implements the modern PINN architecture stack:
- Random Fourier feature embeddings (Wang, Wang & Perdikaris, arXiv:2012.10047)
- Exact periodic embeddings for periodic BCs (jaxpi-style)
- Random weight factorization, W = diag(exp(s)) V (arXiv:2210.01274)
- Modified MLP with U/V gating (Wang, Teng & Perdikaris, arXiv:2001.04536)
- PirateNet adaptive residual blocks, alpha-gated skips initialized to identity
  (Wang et al., arXiv:2402.00326)
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding: x -> [cos(xB), sin(xB)], B ~ N(0, sigma^2)."""

    def __init__(self, in_dim, n_frequencies=64, sigma=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_frequencies) * (2.0 * math.pi * sigma)
        self.register_buffer("B", B)
        self.out_dim = 2 * n_frequencies

    def forward(self, x):
        z = x @ self.B.to(x.dtype)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class PeriodicEmbedding(nn.Module):
    """Exact periodic embedding for selected input dimensions.

    Periodic dims are mapped to [cos(2*pi*k*x/P), sin(2*pi*k*x/P)] for k=1..m,
    which makes the network exactly P-periodic in those dims (hard periodic BC).
    Remaining dims are passed through unchanged.
    """

    def __init__(self, in_dim, periodic_dims, periods, n_harmonics=10):
        super().__init__()
        self.in_dim = in_dim
        self.periodic_dims = list(periodic_dims)
        self.periods = list(periods)
        self.m = n_harmonics
        self.other_dims = [i for i in range(in_dim) if i not in self.periodic_dims]
        self.out_dim = 2 * self.m * len(self.periodic_dims) + len(self.other_dims)

    def forward(self, x):
        feats = []
        for dim, P in zip(self.periodic_dims, self.periods):
            xi = x[:, dim : dim + 1]
            k = torch.arange(1, self.m + 1, dtype=x.dtype, device=x.device)
            z = 2.0 * math.pi * xi * k / P
            feats.extend([torch.cos(z), torch.sin(z)])
        for dim in self.other_dims:
            feats.append(x[:, dim : dim + 1])
        return torch.cat(feats, dim=-1)


class RWFLinear(nn.Module):
    """Linear layer with random weight factorization: W = diag(exp(s)) V.

    s ~ N(mu, sigma); V is set to W_init / exp(s) so the initial function matches
    a Glorot-initialized dense layer while the exp(s) reparameterization gives
    each neuron its own effective learning rate (arXiv:2210.01274).
    """

    def __init__(self, in_dim, out_dim, mu=0.5, sigma=0.1):
        super().__init__()
        W = torch.empty(out_dim, in_dim)
        nn.init.xavier_normal_(W)
        s = mu + sigma * torch.randn(out_dim)
        self.s = nn.Parameter(s)
        self.V = nn.Parameter(W / torch.exp(s).unsqueeze(1))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        W = torch.exp(self.s).unsqueeze(1) * self.V
        return x @ W.t() + self.bias


def _make_linear(in_dim, out_dim, rwf=True):
    if rwf:
        return RWFLinear(in_dim, out_dim)
    layer = nn.Linear(in_dim, out_dim)
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """Plain MLP. `layers` includes input and output dims, e.g. [2, 64, 64, 1]."""

    def __init__(self, layers, activation=torch.tanh, embedding=None, rwf=False,
                 output_transform=None):
        super().__init__()
        self.embedding = embedding
        in_dim = embedding.out_dim if embedding is not None else layers[0]
        dims = [in_dim] + list(layers[1:])
        self.linears = nn.ModuleList(
            [_make_linear(dims[i], dims[i + 1], rwf) for i in range(len(dims) - 1)]
        )
        self.activation = activation
        self.output_transform = output_transform

    def forward(self, x):
        h = self.embedding(x) if self.embedding is not None else x
        for layer in self.linears[:-1]:
            h = self.activation(layer(h))
        out = self.linears[-1](h)
        if self.output_transform is not None:
            out = self.output_transform(x, out)
        return out


class ModifiedMLP(nn.Module):
    """Modified MLP: two encoder streams U, V gate every hidden layer.

    h_{l+1} = (1 - f_l) * U + f_l * V with f_l = act(W_l h_l)  (arXiv:2001.04536).
    """

    def __init__(self, in_dim, width, depth, out_dim, activation=torch.tanh,
                 embedding=None, rwf=True, output_transform=None):
        super().__init__()
        self.embedding = embedding
        d_in = embedding.out_dim if embedding is not None else in_dim
        self.enc_u = _make_linear(d_in, width, rwf)
        self.enc_v = _make_linear(d_in, width, rwf)
        self.input_layer = _make_linear(d_in, width, rwf)
        self.hidden = nn.ModuleList(
            [_make_linear(width, width, rwf) for _ in range(depth - 1)]
        )
        self.head = _make_linear(width, out_dim, rwf)
        self.activation = activation
        self.output_transform = output_transform

    def forward(self, x):
        g = self.embedding(x) if self.embedding is not None else x
        act = self.activation
        U, V = act(self.enc_u(g)), act(self.enc_v(g))
        h = act(self.input_layer(g))
        h = (1 - h) * U + h * V
        for layer in self.hidden:
            f = act(layer(h))
            h = (1 - f) * U + f * V
        out = self.head(h)
        if self.output_transform is not None:
            out = self.output_transform(x, out)
        return out


class PirateBlock(nn.Module):
    """PirateNet adaptive residual block (arXiv:2402.00326, eqs. 12-18).

    Three gated dense layers and a trainable skip gate alpha initialized to 0,
    so each block starts as the identity and the network deepens as needed.
    """

    def __init__(self, width, activation=torch.tanh, rwf=True):
        super().__init__()
        self.f = _make_linear(width, width, rwf)
        self.g = _make_linear(width, width, rwf)
        self.h = _make_linear(width, width, rwf)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = activation

    def forward(self, x, U, V):
        act = self.activation
        f = act(self.f(x))
        z1 = f * U + (1 - f) * V
        g = act(self.g(z1))
        z2 = g * U + (1 - g) * V
        h = act(self.h(z2))
        return self.alpha * h + (1 - self.alpha) * x


class PirateNet(nn.Module):
    """PirateNet: embedding -> dense -> adaptive residual blocks -> linear head."""

    def __init__(self, in_dim, width, n_blocks, out_dim, activation=torch.tanh,
                 embedding=None, rwf=True, output_transform=None):
        super().__init__()
        self.embedding = embedding
        d_in = embedding.out_dim if embedding is not None else in_dim
        self.enc_u = _make_linear(d_in, width, rwf)
        self.enc_v = _make_linear(d_in, width, rwf)
        self.input_layer = _make_linear(d_in, width, rwf)
        self.blocks = nn.ModuleList(
            [PirateBlock(width, activation, rwf) for _ in range(n_blocks)]
        )
        self.head = _make_linear(width, out_dim, rwf)
        self.activation = activation
        self.output_transform = output_transform

    def forward(self, x):
        g = self.embedding(x) if self.embedding is not None else x
        act = self.activation
        U, V = act(self.enc_u(g)), act(self.enc_v(g))
        h = act(self.input_layer(g))
        for block in self.blocks:
            h = block(h, U, V)
        out = self.head(h)
        if self.output_transform is not None:
            out = self.output_transform(x, out)
        return out
