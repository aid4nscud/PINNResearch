"""1D viscous Burgers benchmark.

    u_t + u u_x = nu u_xx,   x in [-1, 1], t in [0, 1], nu = 0.01/pi
    u(x, 0) = -sin(pi x),    u(-1, t) = u(1, t) = 0

Modern stack: hard-constrained IC/BC (the ansatz u = -sin(pi x) + t (1-x^2) N
satisfies both exactly, leaving a single residual loss with nothing to
balance), Fourier features, PirateNet, causal weighting, RAD resampling, and
Adam -> SOAP -> fp64 L-BFGS. Validated against the exact Cole-Hopf solution.

Run:  python problems/forward/burgers1d.py            (full, ~15-30 min CPU)
      python problems/forward/burgers1d.py --quick    (smoke test)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pinnlab
from pinnlab import nn as pnn
from pinnlab import operators as ops
from pinnlab.losses import CausalWeighter
from pinnlab.references import burgers_cole_hopf
from pinnlab.sampling import latin_hypercube, rad_resample, to_tensor

NU = 0.01 / math.pi
BOUNDS = [(-1.0, 1.0), (0.0, 1.0)]  # (x, t)


def build_net(args):
    embedding = pnn.FourierFeatures(2, n_frequencies=args.n_freq, sigma=args.ff_sigma)

    def transform(x, raw):
        xs, ts = x[:, 0:1], x[:, 1:2]
        return -torch.sin(math.pi * xs) + ts * (1 - xs**2) * raw

    return pnn.PirateNet(2, args.width, args.blocks, 1, embedding=embedding,
                         rwf=True, output_transform=transform)


def residual(net, X):
    u = net(X)
    du = ops.grad(u, X)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = ops.grad(u_x, X)[:, 0:1]
    return u_t + u * u_x - NU * u_xx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adam", type=int, default=5000)
    p.add_argument("--soap", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=3000)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--soap-lr", type=float, default=1e-3)
    p.add_argument("--n-colloc", type=int, default=4096)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--ff-sigma", type=float, default=1.0)
    p.add_argument("--causal-bins", type=int, default=32)
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--resample-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/burgers1d")
    p.add_argument("--quick", action="store_true", help="tiny budget smoke test")
    args = p.parse_args()
    if args.quick:
        args.adam, args.soap, args.lbfgs = 300, 200, 100

    pinnlab.set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    net = build_net(args)
    causal = None if args.no_causal else CausalWeighter(0.0, 1.0, args.causal_bins)
    rng = np.random.default_rng(args.seed)

    def make_loss():
        state = {"X": to_tensor(latin_hypercube(args.n_colloc, BOUNDS,
                                                seed=args.seed),
                                requires_grad=True)}

        def rad_residual(np_pts):
            X = to_tensor(np_pts, requires_grad=True)
            return residual(net, X).detach().cpu().numpy()

        def loss_fn(step):
            if (step > 0 and args.resample_every > 0
                    and step % args.resample_every == 0):
                state["X"] = to_tensor(
                    rad_resample(rad_residual, BOUNDS, args.n_colloc, rng=rng),
                    requires_grad=True)
            X = state["X"]
            res_sq = residual(net, X) ** 2
            if causal is not None and step >= 0:
                loss = causal.loss(res_sq, X[:, 1:2])
                return loss, {"res": res_sq.mean().detach(),
                              "min_w": causal.min_weight, "eps": causal.eps}
            loss = res_sq.mean()
            return loss, {"res": loss.detach()}

        return loss_fn

    phases = pinnlab.default_phases(args.adam, args.soap, args.lbfgs,
                                    adam_lr=args.adam_lr, soap_lr=args.soap_lr)
    history = pinnlab.train([net], make_loss, phases, log_every=500)

    # ---- Evaluation against the exact Cole-Hopf solution ----
    x_eval = np.linspace(-1, 1, 256)
    t_eval = np.linspace(0, 1, 101)
    U_ref = burgers_cole_hopf(x_eval, t_eval, NU)
    Xg, Tg = np.meshgrid(x_eval, t_eval)
    pts = np.stack([Xg.ravel(), Tg.ravel()], axis=1)
    with torch.no_grad():
        U_pred = net(to_tensor(pts)).cpu().numpy().reshape(U_ref.shape)
    err = pinnlab.rel_l2(U_pred, U_ref)
    print(f"\nRelative L2 error vs Cole-Hopf exact solution: {err:.3e}")

    metrics = {"rel_l2": err, "nu": NU, "args": vars(args)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pinnlab.viz.save_spacetime_1d(
        x_eval, t_eval, [U_ref, U_pred, np.abs(U_pred - U_ref)],
        ["exact (Cole-Hopf)", "PINN", "|error|"],
        os.path.join(args.outdir, "spacetime.png"))
    slice_ts = [0.25, 0.5, 0.75, 1.0]
    idx = [np.argmin(np.abs(t_eval - ti)) for ti in slice_ts]
    pinnlab.viz.save_slices_1d(x_eval, slice_ts, [U_pred[i] for i in idx],
                               [U_ref[i] for i in idx],
                               os.path.join(args.outdir, "slices.png"))
    pinnlab.viz.save_loss_history(history, os.path.join(args.outdir, "loss.png"),
                                  keys=["loss"])
    print(f"Outputs written to {args.outdir}/")
    return err


if __name__ == "__main__":
    main()
