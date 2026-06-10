"""Self-contained reference solutions for benchmark problems."""

import numpy as np


def burgers_cole_hopf(x, t, nu, n_quad=128):
    """Exact solution of u_t + u u_x = nu u_xx on [-1,1], u(x,0) = -sin(pi x),
    u(+-1, t) = 0, via the Cole-Hopf transform evaluated with Gauss-Hermite
    quadrature (Basdevant et al. 1986).

    x: (Nx,), t: (Nt,). Returns U of shape (Nt, Nx).
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    z, w = np.polynomial.hermite.hermgauss(n_quad)
    U = np.empty((len(t), len(x)))

    def f(y):
        # Hopf-Cole potential of the initial condition: exp(-cos(pi y)/(2 pi nu))
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))

    for i, ti in enumerate(t):
        if ti <= 0:
            U[i] = -np.sin(np.pi * x)
            continue
        s = np.sqrt(4 * nu * ti)
        # eta = x - s*z, integral weights e^{-z^2} are absorbed by hermgauss
        eta = x[:, None] - s * z[None, :]
        F = f(eta)
        num = -(np.sin(np.pi * eta) * F * w[None, :]).sum(axis=1)
        den = (F * w[None, :]).sum(axis=1)
        U[i] = num / den
    return U


def allen_cahn_etdrk4(n_modes=512, dt=1e-4, t_end=1.0, n_snapshots=201,
                      d=1e-4, a=5.0):
    """Spectral reference for u_t = d u_xx + a(u - u^3) on x in [-1,1], periodic,
    u(x,0) = x^2 cos(pi x), using the ETDRK4 scheme (Kassam & Trefethen 2005).

    Returns (x, t_snap, U) with U of shape (n_snapshots, n_modes).
    """
    N = n_modes
    x = -1 + 2.0 * np.arange(N) / N  # periodic grid on [-1, 1)
    u = x**2 * np.cos(np.pi * x)
    v = np.fft.fft(u)

    k = np.fft.fftfreq(N, d=2.0 / N) * 2 * np.pi  # wavenumbers for period 2
    L = -d * k**2 + a  # linear part of d u_xx + a u - a u^3
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2)

    # ETDRK4 coefficients via complex contour integral (Kassam-Trefethen).
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean(
        (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    def nonlinear(v):
        u = np.real(np.fft.ifft(v))
        return np.fft.fft(-a * u**3)

    n_steps = int(round(t_end / dt))
    snap_every = max(1, n_steps // (n_snapshots - 1))
    n_steps = snap_every * (n_snapshots - 1)

    t_snap = np.linspace(0, t_end, n_snapshots)
    U = np.empty((n_snapshots, N))
    U[0] = u
    idx = 1
    for step in range(1, n_steps + 1):
        Nv = nonlinear(v)
        a1 = E2 * v + Q * Nv
        Na = nonlinear(a1)
        b1 = E2 * v + Q * Na
        Nb = nonlinear(b1)
        c1 = E2 * a1 + Q * (2 * Nb - Nv)
        Nc = nonlinear(c1)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % snap_every == 0:
            U[idx] = np.real(np.fft.ifft(v))
            idx += 1
    return x, t_snap, U
