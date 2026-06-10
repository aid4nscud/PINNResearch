"""Plotting helpers. Animations fall back to GIF (Pillow) when ffmpeg is absent."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def save_loss_history(history, path, keys=None):
    """Log-scale training curves from the trainer's history list of dicts."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = keys or sorted({k for h in history for k in h
                           if k not in ("step", "phase")})
    for key in keys:
        steps = [h["step"] for h in history if key in h]
        vals = [h[key] for h in history if key in h]
        if vals and all(np.isscalar(v) for v in vals):
            ax.plot(steps, np.abs(vals) + 1e-30, label=key, lw=1.2)
    for boundary in _phase_boundaries(history):
        ax.axvline(boundary, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _phase_boundaries(history):
    bounds, prev = [], None
    for h in history:
        if prev is not None and h["phase"] != prev:
            bounds.append(h["step"])
        prev = h["phase"]
    return bounds


def save_field_grid(fields, titles, path, extent=(0, 1, 0, 1), cmap="inferno",
                    points=None, suptitle=None, share_clim_groups=None):
    """Row of 2D heatmaps. `share_clim_groups`: list of index tuples that share
    color limits (e.g. [(0, 1)] to put truth and prediction on the same scale)."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    clims = {}
    for group in share_clim_groups or []:
        vmin = min(fields[i].min() for i in group)
        vmax = max(fields[i].max() for i in group)
        for i in group:
            clims[i] = (vmin, vmax)
    for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        vmin, vmax = clims.get(i, (field.min(), field.max()))
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], s=18, c="cyan",
                       edgecolors="k", linewidths=0.5, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spacetime_1d(x, t, fields, titles, path, cmap="jet"):
    """(t, x) heatmaps for 1D time-dependent problems; fields shape (Nt, Nx)."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    extent = (t[0], t[-1], x[0], x[-1])
    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap,
                       aspect="auto", interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_slices_1d(x, t_slices, sol_by_slice, ref_by_slice, path):
    """Solution vs reference at a few time slices."""
    n = len(t_slices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ti, u, uref in zip(axes, t_slices, sol_by_slice, ref_by_slice):
        ax.plot(x, uref, "k-", lw=2, label="reference")
        ax.plot(x, u, "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {ti:.2f}", fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_animation_2d(frames_list, titles, t_data, path, extent=(0, 1, 0, 1),
                      cmap="inferno", points=None):
    """Side-by-side animated heatmaps; frames_list[i] has shape (Nt, ny, nx).
    Saves .mp4 if ffmpeg is available, else .gif via Pillow."""
    n = len(frames_list)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    vmin = min(f.min() for f in frames_list)
    vmax = max(f.max() for f in frames_list)
    ims = []
    for ax, frames, title in zip(axes, frames_list, titles):
        im = ax.imshow(frames[0], origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], s=18, c="cyan",
                       edgecolors="k", linewidths=0.5, zorder=3)
        ax.set_title(title, fontsize=10)
        ims.append(im)
    fig.colorbar(ims[-1], ax=list(axes), fraction=0.03)

    def update(k):
        for im, frames in zip(ims, frames_list):
            im.set_array(frames[k])
        fig.suptitle(f"t = {t_data[k]:.2f}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(t_data), interval=60)
    if animation.FFMpegWriter.isAvailable():
        ani.save(path if path.endswith(".mp4") else path + ".mp4", writer="ffmpeg")
    else:
        gif = os.path.splitext(path)[0] + ".gif"
        ani.save(gif, writer=animation.PillowWriter(fps=15))
    plt.close(fig)
