# Verification · version 0.5 · 21 October 2026

## Executed checks

- Integrated source suite including showcase checks: **399 passed, 3 skipped**. The skips are the CUDA
  frontend check on a machine without CUDA and two opt-in legacy accuracy
  wrappers. The earlier full accuracy runs are retained in the version 0.4
  record; they were not all repeated for this release.
- Both notebooks execute with no cell errors: **13 code cells** in the
  introductory notebook and **11** in the new Pyro gallery. Model declarations
  are traced from notebook cells, rather than imported substitutes.
- The gallery's two VI fits each run 4,000 steps with seed 11. The validation
  script and executed notebook report the same posterior summaries.
- Rendered diagrams and plots were inspected. In particular, the long HMM
  explicitly marks omitted time steps, and Eight Schools retains every distinct
  school effect and its edge to the corresponding observation. Scale aliases fit
  inside their nodes; the schools legend does not obscure observations.
- Two independent read-only code reviews covered the HMM helper, probability
  roundoff handling, numerical references and display changes. Neither reported
  actionable correctness findings. The full local suite includes rendering and
  VI tests that the review environment could not execute.

## HMM equivalence

The declarative constructor is compared with the existing loop notation for
graph structure, seeded simulation, joint density, smoothed state probabilities,
log evidence and posterior paths. Short chains are also checked against
independent exhaustive path enumeration, including partial observations.
Validation covers invalid dimensions, every transition row (even unreachable
rows), one-step chains, scalar/per-state emission SDs, state names and native
Torch simulation.

Probabilities with representational float32 rounding are accepted using a
dtype-aware tolerance, then normalized consistently across the categorical law.
Meaningfully unnormalized weights remain errors. Float32 HMM tables are checked
against normalized float64 references and explicit Torch float64 execution.

## Pyro examples and numerical references

The sources and every adaptation are recorded in `pyro-examples.md` and the
notebook. The weighing scale preserves its original probability law. Interaction
regression uses synthetic data and a HalfNormal(1) noise prior in place of
Uniform(0, 10). Eight Schools retains the published effects and standard errors
but uses HalfNormal(5) in place of HalfCauchy(25), with VI instead of NUTS.

Separately written Pyro models of these adapted laws agree with Smallprob's
joint densities to a maximum absolute difference of **7.11e-15** at the tested
assignments. The weighing-scale posterior is analytically Normal with mean
**9.14** and SD **0.60**. Known-noise interaction regression agrees with an
independent Gaussian reference to less than **8.0e-15 reference SD** in its
posterior means.

Unknown-noise interaction regression and Eight Schools are compared with
independent one-dimensional quadrature after analytically integrating Gaussian
variables. These numerical references are not analytic exact posteriors.

| Fit | Maximum mean error / reference SD | Maximum SD relative error |
| --- | ---: | ---: |
| Interaction regression VI | 0.126 | 9.54% |
| Eight Schools VI, mu and tau | 0.083 | 7.43% |

Both fits meet the predeclared moment limits of 0.35 reference SD and 20% SD
relative error. Eight Schools effect moments are checked for reference stability;
the reported VI comparison above is for its two population parameters. It does
not certify every effect marginal, posterior tails or convergence on other data.

Quadrature is checked by grid refinement and range expansion, including school
effect moments. Maximum moment change under expansion is **7.03e-8**, below
the **1e-6** limit. Estimated mass outside the original range is at most
**2.15e-8**, below **1e-7**. This mass estimate uses the expanded grid and is not a
rigorous bound beyond it. A deliberately truncated reference fails the checks.
Actual values, seeds, bounds and limitations are saved in
`outputs/pyro-gallery-validation.json`.

## Scope

The HMM helper currently supports fixed Normal emissions and supplied initial
probabilities and transitions. It infers states, not these parameters. State
names are display labels; sampled states are integer indices. Exact HMM
inference remains NumPy/CPU; Torch HMM simulation is supported. A Torch-backed
chain requests exact inference with `backend="numpy"`.

Only NumPy and Torch are implemented array backends. Torch/Pyro and Matplotlib
remain optional extras. CUDA hardware was unavailable. VI is approximate.
These three selected examples do not imply general Pyro compatibility or a
port of Pyro's richer learned-parameter HMM tutorial.

Earlier backend, Student-t precision and full VI accuracy evidence is retained
in `verification-v0.4.md`, with still earlier records in `verification-v0.2.md`
and `verification-v0.3.md`. The continuing project is
`/Users/nipun/git/smallprob`.

## Release checks

The 0.5.0 wheel builds and its Python files match the tested source byte for
byte. Installing it in a separate NumPy-only environment, with neither Torch
nor Pyro, passes HMM generation, exact inference, joint posterior path sampling
and mathematical rendering checks.

The staged archive was replayed from a separate directory with its source
import path verified. The exact demo, declarative HMM, weighing-scale posterior
and all three Pyro joint-density comparisons pass. The log is retained in
`outputs/archive-v0.5-smoke.txt`. Source, examples, tests, documentation and
executed notebook bytes are checked against the continuing project during
packaging; archive CRC verification also passes.

## Personal showcase

Seven paired Smallprob/Pyro definitions are generated from executable source.
All 21 joint-density comparisons (three complete generated assignments per
model) pass, with maximum absolute error **1.42e-14** in float64. Eleven new
tests cover the comparison data, displayed declarations, asset provenance and
deterministic builds. See `website/assets/comparison-validation.json`.

All seven downloadable Python scripts were executed from a separate temporary
directory with the installed library, including their full VI fits; each exits
successfully and saves its model SVG. Six plots reuse executed notebook outputs
whose SHA256 hashes are recorded in the comparison report. The normal-mean
example presents its analytic numerical result without a plot.

Browser checks cover all seven selectors and loaded diagrams at a 390-pixel
viewport without horizontal page overflow, desktop presentation, code copying,
history navigation and keyboard navigation. All local links and assets resolve.
A separate read-only review found no actionable issues in the showcase and its
workflow. Public Pages publication is gated by `ENABLE_PUBLIC_PAGES == 'true'`;
this gate is distinct from the repository's visibility.
