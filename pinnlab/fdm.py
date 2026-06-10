"""Vectorized finite-difference ground-truth solver for 2D transient heat problems.

Solves   dT/dt = alpha * (T_xx + T_yy) + q(x, y) - h_cool * (T - T_inf)
on a rectangle with per-side Dirichlet or zero-flux Neumann BCs, explicit Euler.

Grid convention: T has shape (ny, nx) with T[j, i] = T(y[j], x[i]), matching
numpy.meshgrid(x, y) with the default 'xy' indexing — so T.flatten() pairs
correctly with X.flatten(), Y.flatten(). (The old TF-era generator saved the
transpose of this, which silently scrambled the inverse-problem data.)
"""

import numpy as np


def solve_heat2d(alpha, nx=101, ny=101, lx=1.0, ly=1.0, t_end=1.0, n_snapshots=51,
                 source=None, h_cool=0.0, T_inf=0.0, T0=0.0,
                 bc=None, safety=0.25):
    """Returns (x, y, t_snap, T_snap) with T_snap of shape (n_snapshots, ny, nx).

    bc: dict side -> ("dirichlet", value) or ("neumann", 0.0) for sides
        "left", "right", "bottom", "top". Default: all zero-flux Neumann.
    source: None, or callable q(X, Y) -> (ny, nx), or callable q(X, Y, t).
    """
    bc = bc or {}
    for side in ("left", "right", "bottom", "top"):
        bc.setdefault(side, ("neumann", 0.0))

    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    dt_stable = safety * min(dx, dy) ** 2 / max(alpha, 1e-12)
    if h_cool > 0:
        dt_stable = min(dt_stable, 0.5 / h_cool)
    steps_per_snap = max(1, int(np.ceil((t_end / (n_snapshots - 1)) / dt_stable)))
    dt = t_end / ((n_snapshots - 1) * steps_per_snap)

    T = np.full((ny, nx), float(T0)) if np.isscalar(T0) else np.array(T0, dtype=float)
    t_snap = np.linspace(0, t_end, n_snapshots)
    T_snap = np.empty((n_snapshots, ny, nx))
    T_snap[0] = T

    q_static = None
    q_transient = source is not None and source.__code__.co_argcount >= 3
    if source is not None and not q_transient:
        q_static = source(X, Y)

    def apply_bcs(T):
        for side, (kind, value) in bc.items():
            if side == "left":
                T[:, 0] = value if kind == "dirichlet" else T[:, 1]
            elif side == "right":
                T[:, -1] = value if kind == "dirichlet" else T[:, -2]
            elif side == "bottom":
                T[0, :] = value if kind == "dirichlet" else T[1, :]
            elif side == "top":
                T[-1, :] = value if kind == "dirichlet" else T[-2, :]
        return T

    T = apply_bcs(T)
    t = 0.0
    for k in range(1, n_snapshots):
        for _ in range(steps_per_snap):
            lap = np.zeros_like(T)
            lap[1:-1, 1:-1] = (
                (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2
                + (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy**2
            )
            rhs = alpha * lap
            if q_static is not None:
                rhs = rhs + q_static
            elif q_transient:
                rhs = rhs + source(X, Y, t)
            if h_cool > 0:
                rhs = rhs - h_cool * (T - T_inf)
            T = apply_bcs(T + dt * rhs)
            t += dt
        T_snap[k] = T

    return x, y, t_snap, T_snap


def sample_sensors(x, y, t_snap, T_snap, sensor_xy, times):
    """Readings of the FDM field at sensor locations and times.

    sensor_xy: (S, 2) array of (x, y); times: (M,) array.
    Returns (S*M, 3) inputs [x, y, t] and (S*M, 1) temperatures.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((t_snap, y, x), T_snap)
    sensor_xy = np.asarray(sensor_xy)
    times = np.asarray(times)
    S, M = len(sensor_xy), len(times)
    pts = np.empty((S * M, 3))
    pts[:, 0] = np.repeat(sensor_xy[:, 0], M)
    pts[:, 1] = np.repeat(sensor_xy[:, 1], M)
    pts[:, 2] = np.tile(times, S)
    vals = interp(pts[:, [2, 1, 0]]).reshape(-1, 1)
    return pts, vals
