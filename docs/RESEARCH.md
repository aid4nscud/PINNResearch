# Research notes: modernizing this repo (June 2026)

This repo was originally a 2023-era DeepXDE + TensorFlow codebase (plain tanh
MLPs, Adam -> L-BFGS, hand-tuned loss weights, one-point-at-a-time residual
refinement). Before rewriting it we surveyed the 2024-2026 PINN literature.
Summary of findings and the decisions they drove:

## 1. Framework: raw PyTorch

- TensorFlow is effectively legacy for PINNs: no major new PINN library or
  paper targets it. NVIDIA migrated Modulus (TF-based SimNet) to PyTorch and
  rebranded it [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo); DeepXDE's
  modern feature work targets PyTorch/Paddle.
- The methods frontier is split PyTorch/JAX. The leading methods group
  (Perdikaris lab) publishes in JAX via
  [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi); the PyTorch
  ecosystem has [PINA](https://github.com/mathLab/PINA) and PhysicsNeMo.
- For research flexibility a **thin custom PyTorch training loop** beats any
  framework: the entire modern recipe (causal weights, self-adaptive
  balancing, PirateNet blocks, SOAP, fp64 quasi-Newton polish) is loss-loop
  surgery that `Model.train()`-style abstractions obstruct. Hence `pinnlab`,
  a ~1.5k-line package mirroring jaxpi's structure in PyTorch.

## 2. Training recipe (implemented in `pinnlab`)

In priority order, per the Expert's Guide ([arXiv:2308.08468](https://arxiv.org/abs/2308.08468))
and 2024-2025 follow-ups:

1. **Non-dimensionalize; hard-constrain ICs/BCs where possible**
   (approximate-distance trial solutions, [arXiv:2104.08426](https://arxiv.org/abs/2104.08426);
   exact periodic embeddings). Eliminates loss terms and the balancing problem.
2. **Random Fourier features** for spectral bias
   ([arXiv:2012.10047](https://arxiv.org/abs/2012.10047)).
3. **PirateNet** adaptive residual blocks (alpha-gated skips initialized to
   identity) with **random weight factorization**
   ([arXiv:2402.00326](https://arxiv.org/abs/2402.00326),
   [arXiv:2210.01274](https://arxiv.org/abs/2210.01274)). Supersedes the plain
   modified MLP.
4. **Optimizers matter more than any architecture trick** — the headline
   post-2023 result. Adam(warmup+cosine) -> **SOAP** (quasi-Newton/Shampoo
   family; 2-10x error reductions on PINNs,
   [arXiv:2502.00604](https://arxiv.org/abs/2502.00604)) -> **float64 L-BFGS
   polish** (fp32 L-BFGS hits a round-off floor and fake-converges:
   [arXiv:2402.01868](https://arxiv.org/abs/2402.01868),
   [arXiv:2501.16371](https://arxiv.org/abs/2501.16371),
   [arXiv:2505.10949](https://arxiv.org/abs/2505.10949)).
5. **Grad-norm self-adaptive loss balancing**
   ([arXiv:2001.04536](https://arxiv.org/abs/2001.04536)).
6. **Causal (temporal) weighting** for time-dependent PDEs — fixes the "PINN
   learns late times first" failure
   ([arXiv:2203.07404](https://arxiv.org/abs/2203.07404)).
7. **RAD residual-based resampling** — strictly better and cheaper than the
   old one-point-at-a-time RAR
   ([arXiv:2207.10289](https://arxiv.org/abs/2207.10289)).

Deliberately skipped: KAN-based PINNs (comparable accuracy, worse robustness
and speed per [arXiv:2406.02917](https://arxiv.org/abs/2406.02917)); separable
PINNs (pays off only for 3D+ at >1e6 collocation points); heavyweight
frameworks; PDE foundation models (different, many-query problem class —
complementary, not a replacement for sparse-data inverse PINNs).

Benchmark context: "good" relative L2 today is <=1e-4 on 1D Burgers and 1D
Allen-Cahn (SOTA with big budgets/GPUs: SOAP 4.0e-5 / 3.5e-6; fp64 SSBroyden
reaches 7.6e-8 on Burgers). This repo's CPU-scale runs land in the
1e-3 to 1e-4 band, with the full recipe in place to push further on GPU.

## 3. Flagship application: sparse-sensor thermal hotspot tomography

Joint reconstruction of the temperature field and a hidden heat-source map
from a few noisy point sensors is the shared mathematical core of at least
four active application verticals:

- **Battery thermal monitoring / runaway early warning** — internal state from
  surface sensors (J. Electrochem. Soc. 2025 PINN core-temperature estimation;
  UL9540A:2025 pushes digital-twin validation).
- **Chip / heat-source-system thermal field inversion** — a named subfield
  (PINN-TFI, [arXiv:2201.06880](https://arxiv.org/abs/2201.06880); ICHMT 2025
  heat-source field inversion; blind power-map identification on real CPUs).
- **Additive manufacturing** — Goldak heat-source parameter inversion.
- **Thermal tomography / ablation monitoring** (Tomography 2024: physics loss
  lifts reconstruction SSIM 0.45 -> 0.91 at 10% noise vs a data-only baseline).

Literature-standard setups use 4-30 point sensors and 1-10% Gaussian noise,
with the hidden source either a second neural network (research-grade; e.g.
separate u/f networks in [arXiv:2512.07755](https://arxiv.org/html/2512.07755v1))
or parametric Gaussians. A fully free q(x,y,t) is severely ill-posed; published
transient works fix the source's space-time structure (we use a static q(x,y)
with transient T, plus a softplus non-negativity constraint and a small L1
penalty). Our demo (`problems/inverse/hotspot_tomography.py`) follows the
published protocol: 4x4 sensor grid, noise sweep, dual networks, a data-only
baseline, and source-localization metrics including an off-sensor anomaly.
