# PINNResearch

Physics-Informed Neural Networks (PINNs) in PyTorch, implementing the
modern (2025/2026) training stack, with benchmark problems validated against
exact/spectral references and a flagship real-world-style inverse problem:
**sparse-sensor thermal hotspot tomography**.

> This repo was rewritten in June 2026 from a 2023-era DeepXDE + TensorFlow
> codebase. Why and what changed is documented in
> [docs/RESEARCH.md](docs/RESEARCH.md) (framework survey, training-recipe
> literature, and application research with citations).

## What's inside

```
pinnlab/                  small library (~1.5k lines, raw PyTorch)
  nn.py                   Fourier features, exact periodic embeddings,
                          modified MLP, PirateNet (alpha-gated residual blocks),
                          random weight factorization
  optim/soap.py           SOAP optimizer (Adam in Shampoo's eigenbasis)
  losses.py               grad-norm self-adaptive weights, causal weighting
  sampling.py             Latin hypercube, RAD residual-based resampling
  trainer.py              Adam (warmup+cosine) -> SOAP -> float64 L-BFGS
  operators.py            autograd grad/laplacian helpers
  fdm.py                  vectorized FDM ground-truth solver (2D heat + source + cooling)
  references.py           exact Cole-Hopf Burgers; ETDRK4 spectral Allen-Cahn
  viz.py                  heatmaps, space-time plots, animations (mp4/gif)

problems/
  forward/burgers1d.py            benchmark vs exact solution (hard IC/BC, causal, RAD)
  forward/allen_cahn1d.py         benchmark vs spectral reference (periodic embedding,
                                  grad-norm balancing, causal)
  forward/heat2d.py               2D heat conduction validated against FDM
  inverse/hotspot_tomography.py   FLAGSHIP: recover a hidden heat-source map +
                                  full temperature field from 16 noisy sensors

tests/smoke_test.py       component sanity tests (run: python tests/smoke_test.py)
```

## The flagship: thermal hotspot tomography

A 2D "board/module" (battery pack or chip floorplan abstraction) obeys

```
T_t = alpha (T_xx + T_yy) + q(x, y) - h (T - T_inf)
```

with three nominal heat sources and **one anomalous hotspot**. Given only a
4x4 grid of noisy virtual thermocouples (literature-standard 1-10% noise),
two networks are trained jointly against the physics: a PirateNet for
T(x, y, t) and an MLP (softplus output) for the hidden source map q(x, y).
The PINN reconstructs the full temperature field *and* localizes all four
sources — including the anomaly, which sits between sensors. A `--data-only`
baseline (same network, no physics) shows why the physics loss is what makes
sparse-sensor reconstruction work. This setup mirrors active research in
battery thermal monitoring, chip power-map inversion, and thermal tomography
(citations in [docs/RESEARCH.md](docs/RESEARCH.md)).

```bash
python problems/inverse/hotspot_tomography.py             # full run
python problems/inverse/hotspot_tomography.py --noise 0.10 --sensors 3
python problems/inverse/hotspot_tomography.py --unknown-alpha   # also recover diffusivity
python problems/inverse/hotspot_tomography.py --data-only       # no-physics baseline
```

## Results (CPU, default budgets)

| Problem | Validation | Rel. L2 error |
|---|---|---|
| Burgers 1D | exact Cole-Hopf solution | _see outputs/burgers1d/metrics.json_ |
| Allen-Cahn 1D | ETDRK4 spectral reference | _see outputs/allen_cahn1d/metrics.json_ |
| Heat 2D | FDM (101x101) | _see outputs/heat2d/metrics.json_ |
| Hotspot tomography (T field) | FDM ground truth | _see outputs/hotspot_tomography/metrics.json_ |
| Hotspot tomography (hidden q) | true source map | _see outputs/hotspot_tomography/metrics.json_ |

Each script writes `metrics.json`, diagnostic figures, and training curves to
`outputs/<problem>/`. All runs are CPU-feasible (minutes to ~half an hour);
`--quick` runs a tiny-budget smoke test.

## The modern PINN recipe implemented here

1. **Hard-constrained ICs/BCs** (ansatz transforms, exact periodic embeddings)
2. **Random Fourier features** against spectral bias
3. **PirateNet** residual blocks + **random weight factorization**
4. **Adam -> SOAP -> float64 L-BFGS** (the post-2023 lesson: optimizers matter
   more than architecture; fp32 quasi-Newton fake-converges)
5. **Grad-norm self-adaptive loss balancing** (no hand-tuned loss weights)
6. **Causal weighting** for time-dependent PDEs
7. **RAD residual-based adaptive resampling**

Each item has a one-line citation in [docs/RESEARCH.md](docs/RESEARCH.md).

## Setup

```bash
pip install -r requirements.txt   # torch, numpy, scipy, matplotlib
python tests/smoke_test.py        # ~1 min sanity check, incl. SOAP-vs-Adam A/B
```
