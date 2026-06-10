"""1D Allen-Cahn benchmark — a classic hard case for vanilla PINNs.

    u_t = d u_xx + 5(u - u^3),  d = 1e-4,  x in [-1, 1], t in [0, 1]
    u(x, 0) = x^2 cos(pi x),    periodic BCs

Modern stack: exact periodicity via a hard periodic embedding, PirateNet,
soft IC balanced against the residual with grad-norm weights, causal
weighting (this problem is the poster child for the "PINN learns late times
first" failure mode), RAD resampling, Adam -> SOAP -> fp64 L-BFGS.
Validated against an ETDRK4 spectral reference computed on the fly.

Run:  python problems/forward/allen_cahn1d.py            (full, ~20-40 min CPU)
      python problems/forward/allen_cahn1d.py --quick    (smoke test)
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
from pinnlab.losses import CausalWeighter, GradNormBalancer
from pinnlab.references import allen_cahn_etdrk4
from pinnlab.sampling import latin_hypercube, rad_resample, to_tensor

D = 1e-4
BOUNDS = [(-1.0, 1.0), (0.0, 1.0)]  # (x, t)


def build_net(args):
    embedding = pnn.PeriodicEmbedding(2, periodic_dims=[0], periods=[2.0],
                                      n_harmonics=args.harmonics)
    return pnn.PirateNet(2, args.width, args.blocks, 1, embedding=embedding,
                         rwf=True)


def residual(net, X):
    u = net(X)
    du = ops.grad(u, X)
    u_t = du[:, 1:2]
    u_xx = ops.grad(du[:, 0:1], X)[:, 0:1]
    return u_t - D * u_xx - 5 * (u - u**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adam", type=int, default=5000)
    p.add_argument("--soap", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=2000)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--soap-lr", type=float, default=1e-3)
    p.add_argument("--n-colloc", type=int, default=6144)
    p.add_argument("--n-ic", type=int, default=512)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--harmonics", type=int, default=10)
    p.add_argument("--causal-bins", type=int, default=32)
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--resample-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/allen_cahn1d")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.adam, args.soap, args.lbfgs = 300, 200, 100

    pinnlab.set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    net = build_net(args)
    causal = None if args.no_causal else CausalWeighter(0.0, 1.0, args.causal_bins)
    balancer = GradNormBalancer(["ic", "res"], update_every=250)
    rng = np.random.default_rng(args.seed)

    x_ic = np.linspace(-1, 1, args.n_ic)
    u_ic = x_ic**2 * np.cos(math.pi * x_ic)

    # Collocation points persist across phases; make_loss only re-casts dtype.
    state = {"X_np": latin_hypercube(args.n_colloc, BOUNDS, seed=args.seed)}

    def make_loss():
        state["X"] = to_tensor(state["X_np"], requires_grad=True)
        X_ic = to_tensor(np.stack([x_ic, np.zeros_like(x_ic)], axis=1))
        U_ic = to_tensor(u_ic.reshape(-1, 1))

        def rad_residual(np_pts):
            X = to_tensor(np_pts, requires_grad=True)
            return residual(net, X).detach().cpu().numpy()

        def loss_fn(step):
            if (step > 0 and args.resample_every > 0
                    and step % args.resample_every == 0):
                state["X_np"] = rad_resample(rad_residual, BOUNDS,
                                             args.n_colloc, rng=rng)
                state["X"] = to_tensor(state["X_np"], requires_grad=True)
            X = state["X"]
            res_sq = residual(net, X) ** 2
            if causal is not None and step >= 0:
                loss_res = causal.loss(res_sq, X[:, 1:2])
            else:
                loss_res = res_sq.mean()
            loss_ic = ((net(X_ic) - U_ic) ** 2).mean()
            losses = {"ic": loss_ic, "res": loss_res}
            if step > 0:
                balancer.maybe_update(step, losses, list(net.parameters()))
            parts = {"ic": loss_ic.detach(), "res": res_sq.mean().detach(),
                     "w_ic": balancer.weights["ic"]}
            if causal is not None:
                parts["min_w"] = causal.min_weight
            return balancer.total(losses), parts

        return loss_fn

    phases = pinnlab.default_phases(args.adam, args.soap, args.lbfgs,
                                    adam_lr=args.adam_lr, soap_lr=args.soap_lr)
    history = pinnlab.train([net], make_loss, phases, log_every=500)

    # ---- Evaluation against the ETDRK4 spectral reference ----
    x_ref, t_ref, U_ref = allen_cahn_etdrk4(
        n_modes=512, dt=1e-4 if not args.quick else 5e-4, n_snapshots=101)
    Xg, Tg = np.meshgrid(x_ref, t_ref)
    pts = np.stack([Xg.ravel(), Tg.ravel()], axis=1)
    with torch.no_grad():
        preds = [net(to_tensor(chunk)).cpu().numpy()
                 for chunk in np.array_split(pts, 8)]
    U_pred = np.vstack(preds).reshape(U_ref.shape)
    err = pinnlab.rel_l2(U_pred, U_ref)
    print(f"\nRelative L2 error vs ETDRK4 spectral reference: {err:.3e}")

    torch.save(net.state_dict(), os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"rel_l2": err,
                   "max_abs_err": float(np.abs(U_pred - U_ref).max()),
                   "args": vars(args)}, f, indent=2)

    pinnlab.viz.save_spacetime_1d(
        x_ref, t_ref, [U_ref, U_pred, np.abs(U_pred - U_ref)],
        ["ETDRK4 reference", "PINN", "|error|"],
        os.path.join(args.outdir, "spacetime.png"))
    slice_ts = [0.25, 0.5, 0.75, 1.0]
    idx = [np.argmin(np.abs(t_ref - ti)) for ti in slice_ts]
    pinnlab.viz.save_slices_1d(x_ref, slice_ts, [U_pred[i] for i in idx],
                               [U_ref[i] for i in idx],
                               os.path.join(args.outdir, "slices.png"))
    pinnlab.viz.save_loss_history(history, os.path.join(args.outdir, "loss.png"),
                                  keys=["loss", "ic", "res"])
    print(f"Outputs written to {args.outdir}/")
    return err


if __name__ == "__main__":
    main()
