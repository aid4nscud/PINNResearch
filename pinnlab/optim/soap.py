"""SOAP: Shampoo with Adam in the Preconditioner's eigenbasis.

Compact PyTorch implementation of SOAP (Vyas et al., arXiv:2409.11321),
which recent work shows is a step-change optimizer for PINNs, resolving
inter-term gradient conflicts (Wang et al., arXiv:2502.00604).

For each matrix parameter W with gradient G:
- maintain Kronecker covariance factors L = EMA[G G^T], R = EMA[G^T G];
- maintain Adam moments, with the second moment kept in the eigenbasis
  (Q_L, Q_R) of L and R;
- step: project G into the eigenbasis, do an Adam update there, rotate back;
- refresh Q_L, Q_R every `precondition_frequency` steps by one round of
  orthogonal (power) iteration, re-sorting the second moment to follow the
  permutation of eigenvectors.

Vector/scalar parameters fall back to plain Adam.
"""

import torch
from torch.optim import Optimizer


def _eigh_basis(M):
    jitter = 1e-30 * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    _, Q = torch.linalg.eigh(M + jitter)
    return Q.flip(-1)  # descending eigenvalue order


def _power_iter_refresh(M, Q, exp_avg_sq, side):
    """One orthogonal-iteration step; permutes exp_avg_sq rows/cols to track
    the re-sorted eigenvector ordering (as in the reference SOAP code)."""
    Q_new, _ = torch.linalg.qr(M @ Q)
    est_eig = torch.diagonal(Q_new.T @ M @ Q_new)
    order = torch.argsort(est_eig, descending=True)
    Q_new = Q_new[:, order]
    if side == "left":
        exp_avg_sq.copy_(exp_avg_sq[order, :])
    else:
        exp_avg_sq.copy_(exp_avg_sq[:, order])
    return Q_new


class SOAP(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, weight_decay=0.0, precondition_frequency=10):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            lr = group["lr"]
            freq = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if G.dim() == 2 and min(G.shape) > 1:
                        m, n = G.shape
                        state["L"] = torch.zeros(m, m, dtype=G.dtype, device=G.device)
                        state["R"] = torch.zeros(n, n, dtype=G.dtype, device=G.device)

                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t

                if "L" not in state:
                    # Plain Adam for vectors/scalars.
                    exp_avg.mul_(beta1).add_(G, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(G, G, value=1 - beta2)
                    denom = (exp_avg_sq / bias2).sqrt_().add_(eps)
                    update = (exp_avg / bias1) / denom
                else:
                    L, R = state["L"], state["R"]
                    L.mul_(shampoo_beta).add_(G @ G.T, alpha=1 - shampoo_beta)
                    R.mul_(shampoo_beta).add_(G.T @ G, alpha=1 - shampoo_beta)
                    if t == 1:
                        state["QL"], state["QR"] = _eigh_basis(L), _eigh_basis(R)
                    QL, QR = state["QL"], state["QR"]

                    G_rot = QL.T @ G @ QR
                    # First moment in the original space; second in the eigenbasis.
                    exp_avg.mul_(beta1).add_(G, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)

                    m_rot = QL.T @ exp_avg @ QR
                    dir_rot = (m_rot / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
                    update = QL @ dir_rot @ QR.T

                    if t % freq == 0:
                        state["QL"] = _power_iter_refresh(L, QL, exp_avg_sq, "left")
                        state["QR"] = _power_iter_refresh(R, QR, exp_avg_sq, "right")

                if group["weight_decay"] > 0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr)

        return loss
