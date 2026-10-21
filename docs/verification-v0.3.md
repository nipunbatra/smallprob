# Verification · version 0.3 · 21 October 2026

- Integrated source suite: **215 passed, 1 skipped**. The opt-in stochastic
  accuracy wrapper is skipped by default; the complete nine-fit validation
  command was executed separately and all **9 of 9 fits passed**.
- The 47 broadcasting tests compare the new declaration with explicit
  comprehensions: graph grouping, deterministic joint simulation, joint log
  density, exact posterior moments/evidence and joint samples agree.
- An independent Bayesian regression calculation checks exact moments,
  evidence and covariance. The empirical covariance comparison accounts for
  its finite-sample standard error.
- Known- and unknown-noise VI models produce identical learned guide means,
  covariance, ELBO, loss history and posterior samples to their comprehension
  counterparts with the same optimisation seed.
- The full numerical validation now uses vector declarations for both
  regression cases. Across all nine fits, saved posterior means, SDs,
  covariance matrices and intervals are identical to the version 0.2 report
  (maximum absolute difference zero). No thresholds changed. Fit time summed
  to 29.6 seconds in this run; this is not a performance guarantee.
- Shape checks cover lists, tuples, NumPy vectors, scalar/singleton/empty
  inputs, scalar and singleton broadcasting, per-observation SDs, incompatible
  lengths, rejected matrices and missing observations. Chaining, input
  snapshots, scalar indexing of vector expressions and mutation guards pass.
- Existing categorical probability-vector semantics and HMM table indexing
  remain intact. HMM results still match exhaustive short-path enumeration.
- Independent read-only review found no actionable correctness defects within
  the broadcasting extension. The inference kernels were not modified.
- Both original demos and the new vector-regression example execute.
- The version 0.3 wheel builds and its code matches the tested source. Installed
  into an isolated NumPy-only environment, vector simulation and exact
  inference work without Pyro/PyTorch; requesting VI gives an actionable
  optional-extra installation instruction.

The base dependency remains NumPy. VI remains optional. The dependency lock
records the tested environment; no new runtime dependencies were introduced.
The source is in the continuing project `/Users/nipun/git/smallprob`.

## Semantics and boundaries

A vector Normal declaration expands to conditionally independent scalar sites
with shared parents. It does not create a multivariate Normal covariance model.
Scalar inputs stay scalar; one-element and empty vectors retain series shape.
Only one observation dimension is supported. Categorical and HalfNormal
constructors retain their previous meanings.

Arithmetic involving random variables/series is symbolic and elementwise.
Entirely fixed Python expressions keep Python semantics, including list
repetition; use NumPy arrays for purely deterministic vector arithmetic.
Model functions require readable Python source. VI intervals remain approximate.
See `docs/vi-validation-design.md` for numerical references and thresholds, and
`docs/verification-v0.2.md` for the previous backend validation record.

## Release replay

The staged deliverable was replayed from a separate directory with its import
path verified. Both the exact demo and vector-regression VI demo executed;
logs are included in `outputs/archive-*.txt`. The source, examples and tests
in the archive were hash-checked against the tested project before packaging.
