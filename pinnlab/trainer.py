"""Multi-phase PINN trainer: Adam (warmup + cosine decay) -> SOAP -> fp64 L-BFGS.

The phase structure follows the post-2023 consensus that the optimizer matters
more than any single architecture trick (Rathore et al., arXiv:2402.01868;
SOAP for PINNs, arXiv:2502.00604; fp64 quasi-Newton polish, arXiv:2501.16371).

Problems supply `make_loss() -> loss_fn(step) -> (total_loss, parts_dict)`.
`make_loss` is called once per phase so tensors can be rebuilt in the current
default dtype (we switch to float64 for the quasi-Newton polish, where fp32
hits a round-off floor and fake-converges). `loss_fn` receives the global step,
or step=-1 inside L-BFGS closures: dynamic behavior (RAD resampling, weight
updates) must key off `step > 0` so the L-BFGS objective stays deterministic.
"""

import math
import time
from dataclasses import dataclass, field

import torch

from .optim import SOAP


@dataclass
class Phase:
    optimizer: str  # "adam" | "soap" | "lbfgs"
    steps: int
    lr: float = 1e-3
    warmup: int = 0
    final_lr_frac: float = 0.01  # cosine decay floor, as a fraction of lr
    float64: bool = False
    kwargs: dict = field(default_factory=dict)


def default_phases(adam=5000, soap=3000, lbfgs=3000, adam_lr=1e-3, soap_lr=3e-4):
    phases = []
    if adam > 0:
        phases.append(Phase("adam", adam, lr=adam_lr, warmup=min(500, adam // 5)))
    if soap > 0:
        phases.append(Phase("soap", soap, lr=soap_lr))
    if lbfgs > 0:
        phases.append(Phase("lbfgs", lbfgs, float64=True))
    return phases


def _collect_params(modules):
    params, seen = [], set()
    for m in modules:
        ps = m.parameters() if isinstance(m, torch.nn.Module) else [m]
        for p in ps:
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _lr_schedule(phase):
    def factor(step):
        if phase.warmup > 0 and step < phase.warmup:
            return (step + 1) / phase.warmup
        if phase.steps <= phase.warmup:
            return 1.0
        progress = (step - phase.warmup) / (phase.steps - phase.warmup)
        lo = phase.final_lr_frac
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def train(modules, make_loss, phases, log_every=500, print_fn=print):
    """Run the optimization phases. Returns a history list of logged dicts."""
    history = []
    global_step = 0
    t0 = time.time()

    for phase in phases:
        if phase.float64 and torch.get_default_dtype() != torch.float64:
            torch.set_default_dtype(torch.float64)
            for m in modules:
                if isinstance(m, torch.nn.Module):
                    m.double()
                else:
                    m.data = m.data.double()
        loss_fn = make_loss()
        params = _collect_params(modules)

        def log(step_in_phase, loss_val, parts):
            entry = {"step": global_step, "phase": phase.optimizer,
                     "loss": loss_val,
                     **{k: (v.item() if torch.is_tensor(v) else v)
                        for k, v in parts.items()}}
            history.append(entry)
            parts_str = "  ".join(f"{k}={entry[k]:.3e}" for k in parts)
            print_fn(f"[{phase.optimizer:5s} {step_in_phase:6d}] "
                     f"loss={loss_val:.4e}  {parts_str}  "
                     f"({time.time() - t0:.0f}s)")

        if phase.optimizer in ("adam", "soap"):
            if phase.optimizer == "adam":
                opt = torch.optim.Adam(params, lr=phase.lr, **phase.kwargs)
            else:
                opt = SOAP(params, lr=phase.lr, **phase.kwargs)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_schedule(phase))
            for i in range(phase.steps):
                opt.zero_grad(set_to_none=True)
                loss, parts = loss_fn(global_step)
                loss.backward()
                opt.step()
                sched.step()
                global_step += 1
                if i % log_every == 0 or i == phase.steps - 1:
                    log(i, loss.item(), parts)
                if not torch.isfinite(loss):
                    print_fn(f"Non-finite loss in {phase.optimizer}; stopping phase.")
                    break

        elif phase.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                params, lr=1.0, max_iter=phase.steps, history_size=100,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-14, tolerance_change=1e-16, **phase.kwargs)
            n_evals = [0]

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, parts = loss_fn(-1)
                loss.backward()
                n_evals[0] += 1
                if n_evals[0] % log_every == 0:
                    log(n_evals[0], loss.item(), parts)
                return loss

            opt.step(closure)
            loss, parts = loss_fn(-1)
            global_step += n_evals[0]
            log(n_evals[0], loss.item(), parts)
        else:
            raise ValueError(f"unknown optimizer {phase.optimizer!r}")

    return history
