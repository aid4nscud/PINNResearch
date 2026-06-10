"""2D transient heat conduction (modern port of the original repo problem).

    T_t = alpha (T_xx + T_yy),   (x, y) in [0,1]^2, t in [0, 1]
    T = 1 on the right wall (Dirichlet), zero-flux Neumann elsewhere, T(x,y,0)=0

The original TF/DeepXDE version applied alpha to only one spatial term and
relied on hand-tuned loss weights; here the diffusion is isotropic, the IC is
hard-constrained via T = t * N(x,y,t), and the boundary terms are balanced
automatically with grad-norm weights. Validated against the in-repo FDM solver.

Run:  python problems/forward/heat2d.py [--alpha 1.0]
      python problems/forward/heat2d.py --quick
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pinnlab
from pinnlab import nn as pnn
from pinnlab import operators as ops
from pinnlab.fdm import solve_heat2d
from pinnlab.losses import GradNormBalancer
from pinnlab.sampling import latin_hypercube, to_tensor

BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # (x, y, t)


def build_net(args):
    embedding = pnn.FourierFeatures(3, n_frequencies=args.n_freq, sigma=args.ff_sigma)

    def transform(x, raw):
        return x[:, 2:3] * raw  # hard IC: T(x, y, 0) = 0

    return pnn.PirateNet(3, args.width, args.blocks, 1, embedding=embedding,
                         rwf=True, output_transform=transform)


def residual(net, X, alpha):
    T = net(X)
    dT = ops.grad(T, X)
    T_t = dT[:, 2:3]
    T_xx = ops.grad(dT[:, 0:1], X)[:, 0:1]
    T_yy = ops.grad(dT[:, 1:2], X)[:, 1:2]
    return T_t - alpha * (T_xx + T_yy)


def boundary_points(n, rng):
    """Random boundary points per side, shape (n, 3) each: left/right/bottom/top."""
    def side(fixed_dim, fixed_val):
        pts = rng.uniform(0, 1, size=(n, 3))
        pts[:, fixed_dim] = fixed_val
        return pts

    return side(0, 0.0), side(0, 1.0), side(1, 0.0), side(1, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--adam", type=int, default=5000)
    p.add_argument("--soap", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=2000)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--soap-lr", type=float, default=1e-3)
    p.add_argument("--n-colloc", type=int, default=4096)
    p.add_argument("--n-bc", type=int, default=256, help="points per side")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--ff-sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/heat2d")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.adam, args.soap, args.lbfgs = 300, 200, 100

    pinnlab.set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    net = build_net(args)
    balancer = GradNormBalancer(["res", "bc_d", "bc_n"], update_every=250)
    rng = np.random.default_rng(args.seed)

    def make_loss():
        X = to_tensor(latin_hypercube(args.n_colloc, BOUNDS, seed=args.seed),
                      requires_grad=True)
        left, right, bottom, top = [to_tensor(s, requires_grad=True)
                                    for s in boundary_points(args.n_bc, rng)]

        def loss_fn(step):
            loss_res = (residual(net, X, args.alpha) ** 2).mean()
            loss_d = ((net(right) - 1.0) ** 2).mean()
            flux = [ops.grad(net(left), left)[:, 0:1],
                    ops.grad(net(bottom), bottom)[:, 1:2],
                    ops.grad(net(top), top)[:, 1:2]]
            loss_n = sum((f**2).mean() for f in flux) / 3
            losses = {"res": loss_res, "bc_d": loss_d, "bc_n": loss_n}
            if step > 0:
                balancer.maybe_update(step, losses, list(net.parameters()))
            parts = {**{k: v.detach() for k, v in losses.items()},
                     "w_d": balancer.weights["bc_d"]}
            return balancer.total(losses), parts

        return loss_fn

    phases = pinnlab.default_phases(args.adam, args.soap, args.lbfgs,
                                    adam_lr=args.adam_lr, soap_lr=args.soap_lr)
    history = pinnlab.train([net], make_loss, phases, log_every=500)

    # ---- Validation against FDM ----
    nx = 41 if args.quick else 101
    x, y, t_snap, T_fdm = solve_heat2d(
        args.alpha, nx=nx, ny=nx, n_snapshots=11,
        bc={"right": ("dirichlet", 1.0)})
    Xg, Yg = np.meshgrid(x, y)
    T_pinn = np.empty_like(T_fdm)
    with torch.no_grad():
        for k, tk in enumerate(t_snap):
            pts = np.stack([Xg.ravel(), Yg.ravel(),
                            np.full(Xg.size, tk)], axis=1)
            T_pinn[k] = net(to_tensor(pts)).cpu().numpy().reshape(Xg.shape)
    err = pinnlab.rel_l2(T_pinn, T_fdm)
    print(f"\nRelative L2 error vs FDM ({nx}x{nx}, 11 snapshots): {err:.3e}")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"rel_l2_vs_fdm": err, "args": vars(args)}, f, indent=2)

    k = len(t_snap) // 2
    pinnlab.viz.save_field_grid(
        [T_fdm[k], T_pinn[k], np.abs(T_pinn[k] - T_fdm[k])],
        [f"FDM, t={t_snap[k]:.1f}", f"PINN, t={t_snap[k]:.1f}", "|error|"],
        os.path.join(args.outdir, "fields_mid.png"),
        share_clim_groups=[(0, 1)])
    pinnlab.viz.save_field_grid(
        [T_fdm[-1], T_pinn[-1], np.abs(T_pinn[-1] - T_fdm[-1])],
        ["FDM, t=1", "PINN, t=1", "|error|"],
        os.path.join(args.outdir, "fields_final.png"),
        share_clim_groups=[(0, 1)])
    pinnlab.viz.save_loss_history(history, os.path.join(args.outdir, "loss.png"),
                                  keys=["loss", "res", "bc_d", "bc_n"])
    if args.animate:
        pinnlab.viz.save_animation_2d([T_fdm, T_pinn], ["FDM", "PINN"], t_snap,
                                      os.path.join(args.outdir, "evolution.mp4"))
    print(f"Outputs written to {args.outdir}/")
    return err


if __name__ == "__main__":
    main()
