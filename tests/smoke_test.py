"""Sanity tests for pinnlab. Run with:  python tests/smoke_test.py"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pinnlab
from pinnlab import nn as pnn
from pinnlab import operators as ops


def test_networks():
    pinnlab.set_seed(0)
    x = torch.randn(7, 2)
    ff = pnn.FourierFeatures(2, n_frequencies=16, sigma=1.0)
    assert ff(x).shape == (7, 32)

    emb = pnn.PeriodicEmbedding(2, periodic_dims=[0], periods=[2.0], n_harmonics=4)
    z = emb(x)
    assert z.shape == (7, 9)
    # exact periodicity in dim 0
    x_shift = x.clone()
    x_shift[:, 0] += 2.0
    assert torch.allclose(emb(x), emb(x_shift), atol=1e-5)

    for net in [
        pnn.MLP([2, 32, 32, 1]),
        pnn.ModifiedMLP(2, 32, 3, 1, embedding=ff, rwf=True),
        pnn.PirateNet(2, 32, 2, 1, embedding=ff, rwf=True),
    ]:
        assert net(x).shape == (7, 1)

    # PirateNet blocks start as identity (alpha=0): gradients still flow.
    net = pnn.PirateNet(2, 32, 2, 1, embedding=ff)
    net(x).sum().backward()
    assert all(p.grad is not None for n, p in net.named_parameters()
               if "blocks" not in n)
    print("networks ok")


def test_operators():
    x = torch.randn(50, 2, dtype=torch.float64, requires_grad=True)
    u = (x[:, 0:1] ** 3) * torch.sin(x[:, 1:2])
    lap = ops.laplacian(u, x, dims=[0, 1])
    expected = 6 * x[:, 0:1] * torch.sin(x[:, 1:2]) - u
    assert torch.allclose(lap, expected, atol=1e-8)
    d10 = ops.derivative(u, x, dim=1, order=1)
    assert torch.allclose(d10, x[:, 0:1] ** 3 * torch.cos(x[:, 1:2]), atol=1e-8)
    print("operators ok")


def test_soap_beats_adam_on_pinn_loss():
    # 1D Poisson PINN with hard BCs: u'' = -25 sin(5x). SOAP should beat Adam
    # by a wide margin at equal step budget (arXiv:2502.00604).
    import math

    X = torch.linspace(-1, 1, 256).reshape(-1, 1)

    def loss_fn(net):
        x = X.clone().requires_grad_(True)
        s5, s_m5 = math.sin(5.0), math.sin(-5.0)
        u = (s_m5 + (x + 1) / 2 * (s5 - s_m5)) + (1 - x**2) * net(x)
        u_xx = ops.derivative(u, x, 0, order=2)
        return ((u_xx + 25 * torch.sin(5 * x)) ** 2).mean()

    def run(make_opt, steps=800):
        pinnlab.set_seed(3)
        net = pnn.MLP([1, 64, 64, 1])
        opt = make_opt(net.parameters())
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(net)
            loss.backward()
            opt.step()
        return loss.item()

    adam_loss = run(lambda p: torch.optim.Adam(p, lr=1e-3))
    soap_loss = run(lambda p: pinnlab.SOAP(p, lr=3e-3, precondition_frequency=10))
    print(f"  adam={adam_loss:.3e} soap={soap_loss:.3e}")
    assert soap_loss < adam_loss * 0.2, (adam_loss, soap_loss)
    print("soap ok")


def test_losses():
    from pinnlab.losses import CausalWeighter, GradNormBalancer

    cw = CausalWeighter(0.0, 1.0, n_bins=8, eps_schedule=(1.0,))
    t = torch.rand(200, 1)
    res_sq = torch.rand(200, 1) + 5.0 * (t > 0.5).double().float()
    loss = cw.loss(res_sq, t)
    assert loss.item() > 0 and cw.min_weight < 1.0

    p = torch.nn.Parameter(torch.randn(4))
    losses = {"a": (p**2).sum(), "b": 100 * (p**2).sum()}
    bal = GradNormBalancer(["a", "b"], update_every=1)
    bal.maybe_update(0, losses, [p])
    assert bal.weights["b"] < bal.weights["a"]  # bigger gradient -> smaller weight
    print("losses ok")


def test_sampling():
    from pinnlab.sampling import latin_hypercube, rad_resample

    pts = latin_hypercube(100, [(-1, 1), (0, 1)], seed=0)
    assert pts.shape == (100, 2)
    assert pts[:, 0].min() >= -1 and pts[:, 0].max() <= 1

    # residual concentrated near x=0 -> samples should concentrate there
    res = lambda X: np.exp(-50 * X[:, 0] ** 2)
    pts = rad_resample(res, [(-1, 1), (0, 1)], 500, n_candidates=5000, k=2,
                       c=0.0, rng=np.random.default_rng(0))
    assert np.mean(np.abs(pts[:, 0]) < 0.2) > 0.5
    print("sampling ok")


def test_fdm():
    from pinnlab.fdm import sample_sensors, solve_heat2d

    # No source, insulated, uniform IC -> stays uniform.
    x, y, t, T = solve_heat2d(alpha=0.1, nx=21, ny=21, n_snapshots=5, T0=0.3)
    assert np.allclose(T, 0.3, atol=1e-12)

    # Right-wall Dirichlet heating: monotone in x, approaches 1 everywhere.
    x, y, t, T = solve_heat2d(
        alpha=1.0, nx=41, ny=41, t_end=2.0, n_snapshots=11,
        bc={"right": ("dirichlet", 1.0)})
    assert np.all(np.diff(T[-1, 20, :]) >= -1e-9)  # monotone along x
    assert T[-1].min() > 0.9  # near steady state
    assert np.allclose(T[-1, 10, :], T[-1, 30, :], atol=1e-9)  # y-invariant

    # Cooling balances a source.
    q = lambda X, Y: np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    x, y, t, T = solve_heat2d(alpha=0.01, nx=41, ny=41, t_end=1.0,
                              n_snapshots=6, source=q, h_cool=2.0)
    assert 0 < T[-1].max() < q(0.5, 0.5) / 2.0 + 1.0

    pts, vals = sample_sensors(x, y, t, T, [(0.5, 0.5), (0.1, 0.1)],
                               times=[0.5, 1.0])
    assert pts.shape == (4, 3) and vals.shape == (4, 1)
    assert vals.max() <= T.max() + 1e-9
    print("fdm ok")


def test_references():
    from pinnlab.references import allen_cahn_etdrk4, burgers_cole_hopf

    nu = 0.01 / np.pi
    x = np.linspace(-1, 1, 201)
    U = burgers_cole_hopf(x, np.array([0.0, 0.25, 0.5]), nu)
    assert np.allclose(U[0], -np.sin(np.pi * x), atol=1e-12)
    assert np.allclose(U[:, 0], 0, atol=1e-6) and np.allclose(U[:, -1], 0, atol=1e-6)
    assert np.abs(U).max() <= 1.0 + 1e-6
    # odd symmetry u(-x) = -u(x)
    assert np.allclose(U[1], -U[1][::-1], atol=1e-10)
    # residual check on the t=0.25 solution via finite differences
    h = 1e-4
    xs = np.linspace(-0.5, 0.5, 11)
    u = lambda xx, tt: burgers_cole_hopf(np.atleast_1d(xx), np.array([tt]), nu)[0]
    u_t = (u(xs, 0.25 + h) - u(xs, 0.25 - h)) / (2 * h)
    u_x = (u(xs + h, 0.25) - u(xs - h, 0.25)) / (2 * h)
    u_xx = (u(xs + h, 0.25) - 2 * u(xs, 0.25) + u(xs - h, 0.25)) / h**2
    res = u_t + u(xs, 0.25) * u_x - nu * u_xx
    assert np.abs(res).max() < 1e-3, np.abs(res).max()

    x, t, U = allen_cahn_etdrk4(n_modes=256, dt=5e-4, n_snapshots=21)
    assert np.allclose(U[0], x**2 * np.cos(np.pi * x), atol=1e-12)
    assert np.abs(U).max() < 1.01  # solution stays in [-1, 1]
    assert np.abs(U[-1]).mean() > 0.5  # phase separation has happened by t=1
    print("references ok")


def test_trainer_smoke():
    # Fit a tiny regression through all three phases.
    pinnlab.set_seed(0)
    net = pnn.MLP([1, 16, 1])
    X = torch.linspace(-1, 1, 64).reshape(-1, 1)
    Y = torch.sin(2 * X)

    def make_loss():
        Xl = X.to(torch.get_default_dtype())
        Yl = Y.to(torch.get_default_dtype())

        def loss_fn(step):
            loss = ((net(Xl) - Yl) ** 2).mean()
            return loss, {"mse": loss.detach()}

        return loss_fn

    history = pinnlab.train(
        [net], make_loss,
        [pinnlab.Phase("adam", 200, lr=1e-2),
         pinnlab.Phase("soap", 100, lr=1e-2),
         pinnlab.Phase("lbfgs", 200, float64=True)],
        log_every=1000, print_fn=lambda *a: None)
    assert history[-1]["loss"] < 1e-6, history[-1]["loss"]
    torch.set_default_dtype(torch.float32)  # reset for other tests
    print(f"trainer ok (final mse {history[-1]['loss']:.1e})")


if __name__ == "__main__":
    test_networks()
    test_operators()
    test_soap_beats_adam_on_pinn_loss()
    test_losses()
    test_sampling()
    test_fdm()
    test_references()
    test_trainer_smoke()
    print("\nall smoke tests passed")
