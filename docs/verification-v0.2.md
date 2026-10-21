# Verification · version 0.2 · 21 October 2026

## Executed checks

- Integrated test suite: **168 passed, 1 skipped**. The skipped test is the
  opt-in full stochastic accuracy run; its underlying default validation
  script was executed separately, with all **9 of 9 fits passing**.
- Normal mean and known-noise Bayesian regression match independent
  closed-form posterior/evidence calculations. Exact HMM smoothing and
  evidence match exhaustive short-path enumeration. Existing tests for tiny
  noise, missing observations, deterministic transitions and joint samples pass.
- Direct distribution syntax, optional return, old explicit `draw`, named
  observations, stable site names, constructor shadowing and series mutation
  guards are tested. HalfNormal density, scale validation and support are tested.
- VI backend tests cover deterministic seeds, Pyro parameter-store isolation,
  torch RNG/thread restoration on success and error, positive samples, guide
  moments/quantiles, detached old results, scalar-shape validation, fixed-index
  gradients, unsupported models and internal sample-name collisions.
- Genuine SVI fits use a full-covariance guide, without exact initialisation.
  Independent numerical quadrature validates the unknown-noise case. All
  settings, datasets, errors, guide parameters and loss traces are saved in
  `outputs/vi-validation.json`; the predeclared thresholds and results are in
  `docs/vi-validation-design.md`.
- The quadrature reference passed the unchanged grid-refinement and tail
  checks after adaptive refinement to 16,385 points. The largest interval
  endpoint change was 3.05e-6; estimated mass outside the original range was
  4.45e-127. This is a numerical reference, not an analytical exact posterior.
- Independent read-only Codex review found no actionable correctness defects
  in the v0.2 extension. Its test snapshot preceded three final name-collision
  regression tests; all three pass in the integrated count above.
- The exact three-model demo and the unknown-noise VI demo both execute.
- The VI comparison figure was rendered from saved measurements and visually
  checked for units, legends, approximation labels, joint-contour labelling
  and clipping. No VI runs were selected or omitted for the figure.
- The version 0.2 wheel builds. A fresh environment installed that wheel with
  NumPy only: exact inference runs without Pyro/PyTorch installed or imported,
  and requesting VI gives the optional-extra installation command.

Tested environment: Python 3.13.2, NumPy 2.5.4, Pyro 1.9.2, torch 2.14.1.
The exact dependency resolution is recorded in `uv.lock`. Matplotlib is only
used for optional plots, and pytest is a development dependency.

## Accuracy evidence

Maximum relative error in marginal posterior SD across three fixed optimisation
seeds was 1.54% for a Normal mean, 1.08% for known-noise regression and 8.39%
for unknown-noise regression. Covariance, correlation and interval checks also
passed their predeclared limits. These are measurements on three synthetic
fixed datasets, not coverage, calibration or scalability guarantees.

VI's Monte Carlo ELBO is labelled separately from evidence. All six Gaussian
ELBO estimates were within one reported Monte Carlo standard error of the
analytically computed ELBO for the learned guide. One estimate exceeded exact
evidence by sampling fluctuation; the report retains it and its standard error.

## Current boundaries

VI is approximate and currently continuous-only. HMM state inference remains
exact with supplied parameters; HMM parameter learning, categorical VI, MCMC,
GPU and minibatching are not implemented. Model functions require readable
Python source. Finite scalar graphs, a restricted Python subset and dense
factorisations are deliberate prototype constraints.

## Release replay

The deliverable source tree was replayed from a separate staging directory,
with the import path checked to ensure it used that copy. All 168 tests passed
(1 opt-in accuracy wrapper skipped), and both exact and unknown-noise demos
executed successfully. Replay logs are included in `outputs/archive-*.txt`.
The packaged code matches the reviewed and numerically validated project.
