"""Self-adaptive loss weighting schemes.

- GradNormBalancer: global loss weights from gradient-norm ratios
  (Wang, Teng & Perdikaris, arXiv:2001.04536; defaults per the Expert's Guide,
  arXiv:2308.08468).
- CausalWeighter: respect temporal causality in the residual loss
  (Wang, Sankaran & Perdikaris, arXiv:2203.07404), with tolerance annealing.
"""

import torch


class GradNormBalancer:
    """lambda_i <- EMA of  sum_j ||grad L_j|| / ||grad L_i||, updated every
    `update_every` steps. Weights are kept detached scalars."""

    def __init__(self, names, alpha=0.9, update_every=250):
        self.weights = {name: 1.0 for name in names}
        self.alpha = alpha
        self.update_every = update_every

    def maybe_update(self, step, losses, params):
        if step % self.update_every != 0:
            return
        params = [p for p in params if p.requires_grad]
        norms = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True,
                                        allow_unused=True)
            sq = sum(g.pow(2).sum() for g in grads if g is not None)
            norms[name] = torch.sqrt(sq).item() if sq != 0 else 0.0
        total = sum(norms.values())
        for name in self.weights:
            if norms[name] > 0:
                lam_hat = total / norms[name]
                self.weights[name] = (
                    self.alpha * self.weights[name] + (1 - self.alpha) * lam_hat
                )

    def total(self, losses):
        return sum(self.weights[name] * loss for name, loss in losses.items())


class CausalWeighter:
    """Causal residual weighting over time bins.

    The residual loss is binned in time; bin i gets weight
    w_i = exp(-eps * sum_{j<i} L_j), stop-gradient, so later times only
    contribute once earlier times are resolved. When all weights exceed
    `threshold`, eps is annealed up through `eps_schedule`.
    """

    def __init__(self, t_min, t_max, n_bins=32,
                 eps_schedule=(0.01, 0.1, 1.0, 10.0, 100.0), threshold=0.99):
        self.t_min, self.t_max = t_min, t_max
        self.n_bins = n_bins
        self.eps_schedule = list(eps_schedule)
        self.eps_idx = 0
        self.threshold = threshold
        self.min_weight = 0.0

    @property
    def eps(self):
        return self.eps_schedule[self.eps_idx]

    def loss(self, res_sq, t):
        """Weighted mean of squared residuals `res_sq` (N,1) at times `t` (N,1)."""
        bins = ((t.detach().squeeze(-1) - self.t_min)
                / (self.t_max - self.t_min) * self.n_bins).long()
        bins = bins.clamp(0, self.n_bins - 1)
        per_bin = torch.zeros(self.n_bins, dtype=res_sq.dtype, device=res_sq.device)
        counts = torch.zeros_like(per_bin)
        per_bin.scatter_add_(0, bins, res_sq.squeeze(-1))
        counts.scatter_add_(0, bins, torch.ones_like(res_sq.squeeze(-1)))
        per_bin = per_bin / counts.clamp(min=1)

        with torch.no_grad():
            cum = torch.cat([torch.zeros(1, device=per_bin.device,
                                         dtype=per_bin.dtype),
                             torch.cumsum(per_bin, 0)[:-1]])
            w = torch.exp(-self.eps * cum)
            self.min_weight = w.min().item()
            if (self.min_weight > self.threshold
                    and self.eps_idx < len(self.eps_schedule) - 1):
                self.eps_idx += 1
        return (w * per_bin).mean()
