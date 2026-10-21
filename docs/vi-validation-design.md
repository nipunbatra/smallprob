# VI validation plan (declared before running VI)

These checks use the same models and observations for the exact Gaussian engine
and genuinely optimised full-rank VI. No exact posterior is used to initialise
VI. All generated datasets are synthetic, with fixed seeds. This is a small
prototype validation suite, not a guarantee for arbitrary models or datasets.

## Fixed cases and settings

- Normal mean: 12 observations, observation SD 0.8, prior `Normal(0, 5)`,
  synthetic mean 1.25, data seed 101.
- Known-noise regression: 24 equally spaced x values from 0.5 to 2.5,
  observation SD 0.6, independent `Normal(0, 5)` coefficient priors,
  synthetic intercept 1.2 and slope -0.7, data seed 102. The uncentred inputs
  deliberately make the posterior coefficients strongly correlated.
- Unknown-noise regression: 32 equally spaced x values from -1.5 to 1.5,
  independent `Normal(0, 5)` coefficient priors and `HalfNormal(1)` noise prior,
  synthetic intercept 1.2, slope -0.7 and noise SD 0.6, data seed 103.
- Optimisation seeds: 11, 29, 47 for each case. Default settings: 3,000 steps,
  32 particles, initial learning rate 0.03. The engine's learning-rate schedule
  is recorded in the output. All comparisons use the entire run, including
  optimisation variability; the report does not select a favourable seed.

## Predeclared acceptance thresholds

For every run, compared with its reference:

| Quantity | Maximum allowed error |
| --- | --- |
| Posterior mean, divided by reference marginal SD | 0.35 |
| Marginal SD, relative error | 0.20 |
| Full covariance, relative Frobenius norm | 0.25 |
| Correlation matrix, maximum absolute entry error | 0.15 |
| 95% marginal interval endpoint, divided by reference SD | 0.45 |

Unknown-noise VI has a restricted variational family even at its optimum:
coefficients and log noise are jointly Gaussian under the guide. Its reference
is a non-Gaussian posterior. These tolerances apply to that approximation too;
passing does not make VI exact. True generating parameters and whether they
lie in the reported intervals are descriptive, not coverage guarantees.

Gaussian posterior KL divergence is evaluated analytically from learned guide
moments and the exact reference. Consequently `log_evidence - KL(q || p)` is
an independent deterministic ELBO reference for those cases. The backend's
held-out Monte Carlo ELBO and its difference from this reference are reported;
Monte Carlo fluctuation above the evidence is not interpreted as a valid bound.

## Independent numerical reference for unknown noise

For each fixed sigma, independently compute the Gaussian regression posterior
by solving its two-dimensional precision system. Analytically integrate the
coefficients out of the likelihood. Integrate the remaining one-dimensional
posterior over `u = log(sigma)` using the trapezoid rule, including the `sigma`
Jacobian. The HalfNormal prior uses its actual normalisation constant.

Use 4,097 points over `[-8, 3]`, refine to 8,193 points, and compare against an
expanded `[-10, 4]` grid at at least the refined resolution. Require mean/SD
relative changes below `1e-5` and interval absolute changes below `1e-5`.
Require posterior mass outside the original range, measured on the expanded
grid, below `1e-8`. Record the absolute evidence, means, SDs, full covariance,
intervals and all convergence measurements. This is a **numerical quadrature
reference**, not an analytic exact posterior.

Reference-only check before the first VI run: the initial 4,097/8,193 grids
gave a maximum interval change of `1.25774e-5`, just outside the declared
`1e-5` threshold. The script therefore refines the grid until the original
threshold is met (up to four attempts) and records every attempt. This changes
numerical resolution, not the declared comparison tolerances.

## Execution and interpretation

Run `python -m examples.vi_validation` after installing `smallprob[vi]`.
It writes `outputs/vi-validation.json`, prints numerical errors, and returns a
nonzero exit status when a declared comparison fails. Use `--steps`,
`--particles`, `--learning-rate` or `--seeds` for experiments; changed settings
are saved visibly and do not rewrite the thresholds. The full default command
must be executed for a release claim. Fast tests check the independent
reference itself; the full stochastic accuracy test is opt-in via
`SMALLPROB_RUN_VI_ACCURACY=1 python -m pytest tests/test_vi_accuracy.py -q`.

## Measured default run, 21 October 2026

The full default command ran all nine fits with the original acceptance
thresholds. Every comparison passed. The optimisation runs totalled 28.9
seconds on the development machine, including the first backend import;
reference construction and report writing are additional. Timings are
observations from this machine, not performance guarantees.

Maximum errors across the three seeds in each case:

| Case | Mean / reference SD | SD relative | Covariance relative | Correlation absolute | Interval / reference SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal mean | 0.04428 | 0.01544 | 0.03065 | 0 | 0.05601 |
| Known-noise regression | 0.02041 | 0.01075 | 0.01755 | 0.00061 | 0.03185 |
| Unknown-noise regression | 0.08381 | 0.08388 | 0.09344 | 0.03035 | 0.36428 |

The exact known-noise coefficient correlation is `-0.92768`. The unknown-noise
reference noise mean is `0.68322`, SD `0.09159`, and 95% interval
`[0.53195, 0.88957]`. VI noise means for seeds 11, 29, 47 are respectively
`0.67740`, `0.68447`, and `0.68173`. All 10,000 sampled noise values per run
were positive. The unknown-noise approximation is visibly less accurate than
the Gaussian cases, particularly in its interval tails.

The final numerical reference used 16,385 points; its largest endpoint change
on refinement was `3.05e-6`, and estimated mass beyond the original log-sigma
range was `4.45e-127`. Both passed the original reference checks.

All six Gaussian Monte Carlo ELBO errors were within one reported Monte Carlo
standard error of the independently calculated guide ELBO. One stochastic
estimate exceeded exact log evidence by `0.00019` (reported standard error
`0.00046`). This sampling fluctuation is why the reported estimate is labelled
approximate rather than treated as a guaranteed numerical lower bound.

Complete parameters, datasets, means, intervals, covariance matrices,
diagnostics, convergence attempts, and per-seed errors are saved in
`outputs/vi-validation.json`. The script is the reproducible source of these
numbers; later changes require rerunning it before reusing this result.

No test here establishes calibration across datasets, scalability, multimodal
accuracy, robustness to extreme prior scales, or HMM parameter learning.

## Version 0.3 vector-declaration replay

The two regression models now declare all observations with
`y = Normal(intercept + slope * x, noise)` (with `sigma` replacing `noise`
when unknown). All nine default fits were rerun and passed the unchanged
thresholds. Their posterior means, SDs, covariances and intervals match the
saved version 0.2 values exactly. The summed fit time in this replay was
29.6 seconds. The Normal-mean reference still uses its explicit repetition.
