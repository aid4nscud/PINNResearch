"""Sparse-sensor thermal hotspot tomography (flagship inverse problem).

A physics-informed digital twin for a 2D "board/module" (think battery pack or
chip floorplan): from a handful of noisy point temperature sensors, jointly
reconstruct the full transient temperature field AND the hidden heat-source
map q(x, y) — including an anomalous hotspot that no sensor sits on.

    T_t = alpha (T_xx + T_yy) + q(x, y) - h (T - T_inf)
    zero-flux Neumann walls, T(x, y, 0) = 0 (hard-constrained)

Ground truth comes from the in-repo FDM solver: three nominal sources
("cells/cores") plus one anomalous hotspot. Sensors are a 4x4 grid of virtual
thermocouples with Gaussian noise (literature-standard 1-10% sweep). Two
networks are trained jointly: a PirateNet for T(x,y,t) and an MLP with a
softplus output for q(x,y) >= 0. This problem setup mirrors the active
"temperature/heat-source field inversion" literature (PINN-TFI, arXiv:2201.06880;
source inversion with separate u/f networks, arXiv:2512.07755; E-PINN,
arXiv:2209.10195).

Run:  python problems/inverse/hotspot_tomography.py             (~20-40 min CPU)
      python problems/inverse/hotspot_tomography.py --quick     (smoke test)
      python problems/inverse/hotspot_tomography.py --data-only (no-physics baseline)
      python problems/inverse/hotspot_tomography.py --unknown-alpha
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
from pinnlab.fdm import sample_sensors, solve_heat2d
from pinnlab.losses import GradNormBalancer
from pinnlab.sampling import latin_hypercube, to_tensor

ALPHA = 0.01
H_COOL = 2.0
T_INF = 0.0
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # (x, y, t)

# (x0, y0, sigma, amplitude): three nominal sources + one anomalous hotspot.
SOURCES = [
    (0.25, 0.70, 0.06, 6.0),
    (0.30, 0.25, 0.06, 5.0),
    (0.75, 0.25, 0.06, 4.0),
    (0.70, 0.65, 0.05, 8.0),  # anomaly: hottest, off the sensor grid
]


def q_true_fn(X, Y):
    q = np.zeros_like(X)
    for x0, y0, sig, amp in SOURCES:
        q += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sig**2))
    return q


def build_T_net(args):
    embedding = pnn.FourierFeatures(3, n_frequencies=args.n_freq, sigma=args.ff_sigma)

    def transform(x, raw):
        return x[:, 2:3] * raw  # hard IC: T(x, y, 0) = 0 (known ambient start)

    return pnn.PirateNet(3, args.width, args.blocks, 1, embedding=embedding,
                         rwf=True, output_transform=transform)


def build_q_net(args):
    def transform(x, raw):
        return args.q_scale * torch.nn.functional.softplus(raw)  # q >= 0

    return pnn.MLP([2, 64, 64, 64, 1], output_transform=transform)


def residual(T_net, q_net, X, log_alpha=None):
    T = T_net(X)
    dT = ops.grad(T, X)
    T_t = dT[:, 2:3]
    T_xx = ops.grad(dT[:, 0:1], X)[:, 0:1]
    T_yy = ops.grad(dT[:, 1:2], X)[:, 1:2]
    q = q_net(X[:, 0:2])
    alpha = torch.exp(log_alpha) if log_alpha is not None else ALPHA
    return T_t - alpha * (T_xx + T_yy) - q + H_COOL * (T - T_INF)


def boundary_points(n, rng):
    sides = []
    for fixed_dim, fixed_val in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)]:
        pts = rng.uniform(0, 1, size=(n, 3))
        pts[:, fixed_dim] = fixed_val
        sides.append(pts)
    return sides


def locate_peaks(q, x, y, threshold=0.2, size=9):
    """Local maxima of a field above threshold*max, as (x, y, value) rows."""
    from scipy.ndimage import maximum_filter

    is_peak = (q == maximum_filter(q, size=size)) & (q > threshold * q.max())
    jj, ii = np.where(is_peak)
    return np.array([[x[i], y[j], q[j, i]] for j, i in zip(jj, ii)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adam", type=int, default=6000)
    p.add_argument("--soap", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=2000)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--soap-lr", type=float, default=1e-3)
    p.add_argument("--n-colloc", type=int, default=4096)
    p.add_argument("--n-bc", type=int, default=128, help="points per side")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--ff-sigma", type=float, default=1.0)
    p.add_argument("--q-scale", type=float, default=10.0,
                   help="output scale of the source network")
    p.add_argument("--q-reg", type=float, default=1e-4,
                   help="L1 penalty on q to suppress ghost sources")
    p.add_argument("--sensors", type=int, default=4, help="sensors per axis")
    p.add_argument("--n-times", type=int, default=40, help="readings per sensor")
    p.add_argument("--noise", type=float, default=0.05,
                   help="noise std as a fraction of sensor-signal std")
    p.add_argument("--unknown-alpha", action="store_true",
                   help="also recover the diffusivity alpha")
    p.add_argument("--data-only", action="store_true",
                   help="baseline: fit sensors with no physics loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=None)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.adam, args.soap, args.lbfgs = 300, 200, 100
    if args.outdir is None:
        args.outdir = ("outputs/hotspot_data_only" if args.data_only
                       else "outputs/hotspot_tomography")

    pinnlab.set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Ground truth and noisy sensor data ----
    nx = 61 if args.quick else 101
    x, y, t_snap, T_true = solve_heat2d(
        ALPHA, nx=nx, ny=nx, n_snapshots=51, source=q_true_fn,
        h_cool=H_COOL, T_inf=T_INF)
    grid = np.linspace(0, 1, args.sensors + 2)[1:-1]
    sensor_xy = np.array([[xi, yi] for xi in grid for yi in grid])
    times = np.linspace(0, 1, args.n_times + 1)[1:]
    obs_X, obs_T = sample_sensors(x, y, t_snap, T_true, sensor_xy, times)
    noise_std = args.noise * obs_T.std()
    obs_T_noisy = obs_T + noise_std * rng.standard_normal(obs_T.shape)
    print(f"{len(sensor_xy)} sensors x {args.n_times} readings, "
          f"noise std {noise_std:.4f} ({args.noise:.0%} of signal std), "
          f"T range [0, {T_true.max():.2f}]")

    # ---- Networks ----
    T_net = build_T_net(args)
    q_net = build_q_net(args)
    log_alpha = (torch.nn.Parameter(torch.tensor(np.log(0.05)))
                 if args.unknown_alpha else None)
    modules = [T_net] if args.data_only else [T_net, q_net]
    if log_alpha is not None:
        modules.append(log_alpha)
    term_names = (["data"] if args.data_only else ["data", "res", "bc"])
    balancer = GradNormBalancer(term_names, update_every=250)

    def make_loss():
        Xd = to_tensor(obs_X)
        Td = to_tensor(obs_T_noisy)
        Xc = to_tensor(latin_hypercube(args.n_colloc, BOUNDS, seed=args.seed),
                       requires_grad=True)
        bc_pts = [to_tensor(s, requires_grad=True)
                  for s in boundary_points(args.n_bc, rng)]
        bc_dims = [0, 0, 1, 1]

        def loss_fn(step):
            losses = {"data": ((T_net(Xd) - Td) ** 2).mean()}
            if not args.data_only:
                losses["res"] = (residual(T_net, q_net, Xc, log_alpha) ** 2).mean()
                flux = [ops.grad(T_net(pts), pts)[:, d : d + 1]
                        for pts, d in zip(bc_pts, bc_dims)]
                losses["bc"] = sum((f**2).mean() for f in flux) / 4
            params = [pp for m in modules
                      for pp in (m.parameters() if isinstance(m, torch.nn.Module)
                                 else [m])]
            if step > 0:
                balancer.maybe_update(step, losses, params)
            total = balancer.total(losses)
            parts = {k: v.detach() for k, v in losses.items()}
            if not args.data_only and args.q_reg > 0:
                total = total + args.q_reg * q_net(Xc[:, 0:2].detach()).abs().mean()
            if log_alpha is not None:
                parts["alpha"] = torch.exp(log_alpha).detach()
            return total, parts

        return loss_fn

    phases = pinnlab.default_phases(args.adam, args.soap, args.lbfgs,
                                    adam_lr=args.adam_lr, soap_lr=args.soap_lr)
    history = pinnlab.train(modules, make_loss, phases, log_every=500)

    # ---- Evaluation ----
    Xg, Yg = np.meshgrid(x, y)
    metrics = {"noise": args.noise, "n_sensors": len(sensor_xy),
               "data_only": args.data_only, "args": vars(args)}

    T_pred = np.empty_like(T_true)
    with torch.no_grad():
        for k, tk in enumerate(t_snap):
            pts = np.stack([Xg.ravel(), Yg.ravel(),
                            np.full(Xg.size, tk)], axis=1)
            T_pred[k] = T_net(to_tensor(pts)).cpu().numpy().reshape(Xg.shape)
    metrics["rel_l2_T"] = pinnlab.rel_l2(T_pred, T_true)
    print(f"\nTemperature field rel. L2 error: {metrics['rel_l2_T']:.3e}")

    if not args.data_only:
        q_true = q_true_fn(Xg, Yg)
        with torch.no_grad():
            q_pred = q_net(to_tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1))
                           ).cpu().numpy().reshape(Xg.shape)
        metrics["rel_l2_q"] = pinnlab.rel_l2(q_pred, q_true)
        print(f"Hidden source field rel. L2 error: {metrics['rel_l2_q']:.3e}")

        peaks = locate_peaks(q_pred, x, y)
        loc_errors, amp_ratios = [], []
        for x0, y0, _, amp in SOURCES:
            if len(peaks) == 0:
                loc_errors.append(float("nan"))
                amp_ratios.append(0.0)
                continue
            d = np.hypot(peaks[:, 0] - x0, peaks[:, 1] - y0)
            j = int(np.argmin(d))
            loc_errors.append(float(d[j]))
            amp_ratios.append(float(peaks[j, 2] / amp))
        metrics["n_detected_peaks"] = len(peaks)
        metrics["localization_errors"] = loc_errors
        metrics["amplitude_ratios"] = amp_ratios
        print(f"Detected {len(peaks)} source peaks "
              f"(true: {len(SOURCES)}, incl. the anomaly)")
        for i, ((x0, y0, _, amp), le, ar) in enumerate(
                zip(SOURCES, loc_errors, amp_ratios)):
            tag = " <- anomaly" if i == len(SOURCES) - 1 else ""
            print(f"  source @({x0:.2f},{y0:.2f}) A={amp}: "
                  f"localized within {le:.3f}, amplitude x{ar:.2f}{tag}")
        if log_alpha is not None:
            metrics["alpha_recovered"] = float(torch.exp(log_alpha))
            print(f"Recovered alpha = {metrics['alpha_recovered']:.4f} "
                  f"(true {ALPHA})")

        pinnlab.viz.save_field_grid(
            [q_true, q_pred, np.abs(q_pred - q_true)],
            ["true source q(x,y)", "recovered q(x,y)", "|error|"],
            os.path.join(args.outdir, "source_recovery.png"),
            points=sensor_xy, share_clim_groups=[(0, 1)],
            suptitle=f"{len(sensor_xy)} sensors, {args.noise:.0%} noise")

    k = len(t_snap) - 1
    pinnlab.viz.save_field_grid(
        [T_true[k], T_pred[k], np.abs(T_pred[k] - T_true[k])],
        ["true T, t=1", "reconstructed T, t=1", "|error|"],
        os.path.join(args.outdir, "temperature_final.png"),
        points=sensor_xy, share_clim_groups=[(0, 1)])
    pinnlab.viz.save_loss_history(history, os.path.join(args.outdir, "loss.png"),
                                  keys=[k for k in ("loss",) + tuple(term_names)])
    if args.animate:
        pinnlab.viz.save_animation_2d(
            [T_true, T_pred], ["true T", "reconstructed T"], t_snap,
            os.path.join(args.outdir, "evolution.mp4"), points=sensor_xy)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Outputs written to {args.outdir}/")
    return metrics


if __name__ == "__main__":
    main()
