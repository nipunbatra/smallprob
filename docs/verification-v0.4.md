# Verification · version 0.4 · 21 October 2026

## Executed checks

- Integrated source suite: **305 passed, 3 skipped**. The skips are the CUDA
  frontend test on a machine without CUDA and two opt-in accuracy wrappers.
  Both accuracy scripts were executed separately; see below.
- The notebook executed all **13 code cells** with no errors or warning outputs.
  It covers Normal observations, NumPy/Torch regression, unknown-scale Student-t
  regression with VI, and exact HMM state inference. Model definitions were
  traced from actual notebook cells, not imported substitutes.
- All four notebook diagrams and three data/inference plots were rendered and
  visually reviewed. The self-contained HTML preview was inspected in a browser;
  mathematical equations, SVG diagrams and PNG plots display without network
  dependencies. No clipping or mathematical/model mismatches were found.
- All 11 README Python snippets executed, including CPU/MPS, HMM and SVG/PNG
  export. Documentation links resolve and code blocks parse.
- The 0.4.0 wheel builds and its Python files match the tested source byte for
  byte. In an isolated environment without Torch/Pyro, NumPy exact inference,
  simulation, StudentT generation and model mathematics work. Requesting VI
  gives an actionable optional-extra installation instruction.

## Numerical references

The original nine Normal, known-noise and unknown-noise regression fits pass
all predeclared limits. Posterior mean, SD, covariance, interval and Monte Carlo
ELBO fields are identical to the pre-backend report; see
`outputs/backend-legacy-regression.json` and `outputs/vi-validation.json`.
The unknown-noise reference is independently converged numerical quadrature,
not an analytic exact posterior.

Three Student-t location fits are checked against an independently coded
Normal-prior/Student-t-likelihood numerical integral. Its grid refinement,
range expansion and omitted tail mass checks pass. A full MPS regression fit
runs 3,000 steps with 32 particles in float32 and is compared with an independent
Gaussian formula and the NumPy exact engine. All four new fits pass the original
limits: mean error below 0.35 reference SD, SD relative error below 20%, and
95% interval endpoint error below 0.45 reference SD. See
`outputs/backend-validation.json` for actual values, seeds, reference checks
and timings. These are small-model correctness checks, not evidence of a GPU
speed advantage.

Torch exact Gaussian tests forbid NumPy QR/solve to verify native computation.
Native Torch posterior sampling, output type/device, MPS execution, RNG
restoration, finite/support checks and unsupported-device errors are exercised.
Existing independent Gaussian references and exhaustive short HMM path
comparisons continue to pass.

## Review and numerical stability

An independent read-only code review identified cancellation in Torch's
Student-t log-density normalization at very large degrees of freedom. The
shared Torch/Pyro density now uses a differentiable gamma-ratio expansion and
stable tail calculations on the selected device. Sampling remains the native
Torch distribution sampler. Focused tests compare densities and gradients for
14 degrees-of-freedom values from 1e-10 to 1e12 at three residuals, including
branch boundaries, against stored high-precision references on CPU float32/64
and actual MPS float32. Tests also forbid the unstable original Torch log_prob
while calling simulation, joint density and real VI.

At df=1e7 and zero residual, the corrected float32 log density is approximately
-0.91893852, within float32 rounding of the reference; the previous Torch result
was zero. Parameter derivatives remain finite and agree with the reference.

## Semantics and boundaries

The base dependencies are NumPy and array-api-compat. Only NumPy and Torch are
implemented as numerical backends in this version. Torch/Pyro and Matplotlib
remain optional extras. The lock file records the tested resolution.

A vector Normal or StudentT declaration represents conditionally independent
scalar observations with shared parents, not a multivariate event. Only one
observation dimension is supported. Normal's Python scale parameter is its SD;
mathematical diagrams show variance. StudentT scale is not its SD. Display
aliases do not change inference names, and observed shading is a presentation
choice; conditioning uses actual observations supplied to infer.

Fixed tensor inputs are detached and snapshotted on the host during graph
construction and validation. Torch simulation, joint density, Gaussian
conditioning, VI optimization and posterior sampling use the selected device.
Small VI summary statistics are computed from detached host parameters and
returned in the selected array type. No gradients flow into original input
tensors. MPS requires float32. CUDA paths are implemented but no CUDA hardware
was available to validate them. HMM exact inference remains NumPy/CPU, with
supplied transition and emission parameters. VI remains an approximation.

Historical verification records are retained in `verification-v0.2.md` and
`verification-v0.3.md`. The continuing source project is
`/Users/nipun/git/smallprob`.

## Release replay

The staged source archive was replayed from a separate directory with its import
path verified. The exact demo and all 13 notebook cells executed successfully.
Replay logs are included in `outputs/archive-*.txt`. The release contains the
reviewed executed notebook from the continuing project; source, examples, tests,
documentation and notebook bytes were checked against that project before
packaging. Archive CRC verification passed.
