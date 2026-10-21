# Smallprob 0.5 design

The model describes a generative process. Inference is a separate choice made
when observations are supplied. Ordinary Python is the surface language;
a finite graph of scalar random variables is the boundary between that
language and the numerical backends.

```python
from smallprob import model, Normal, HalfNormal

@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)

m = regression([-1., 0., 1.])
data = m.simulate(seed=7, fixed={"intercept": 1., "slope": 2., "sigma": .5})
p = m.infer(y=data["y"][0], method="vi", seed=7)
p.mean("slope"), p.sd("slope"), p.interval("slope")
p.sample(draws=1000, seed=8)["slope"]
```

No model-level `return`, sample names, plates or tensor imports are required.
Explicit `draw(Normal(...))` remains compatible. NumPy and `array-api-compat`
are the base runtime dependencies. Optional extras provide Torch (`torch`),
Pyro/PyTorch (`vi`), diagram rendering (`viz`) and notebook execution (`notebook`).

Normal and StudentT parameters may be scalars or one-dimensional vectors. At declaration,
vector expressions are lowered elementwise to scalar sites with the same
names/order as an explicit comprehension. Shared parents remain shared.
Scalars and length-one vectors broadcast; other vector lengths must match.
Empty/singleton vectors preserve series shape. Matrices are rejected rather
than flattened, and categorical probability vectors retain their original
single-variable meaning. Shape analysis uses structure and fixed inputs,
never sampled values of latent variables. The numerical inference engines
retain scalar graph semantics; observed likelihood expressions and inference
particles can be batched internally.

## Frontend and graph

`syntax.py` reads a decorated function's source, lowers direct distribution
declarations to the existing draw pipeline, infers stable site names, and
rewrites indexing to symbolic lookup where necessary. Shorthand requires
unshadowed direct imports of `Normal`, `HalfNormal`, `StudentT` and `Categorical`.
Constructors outside declaration positions remain distribution objects;
explicit `draw(..., name="...")` handles other supported spelling patterns.

Running the compiled function with fixed inputs builds a graph; it does not
sample. Fixed-input branches and loops work. Random-dependent control flow,
implicit coercion of symbolic values, and unsupported dynamic structures
raise clear errors. Functions must have readable Python source, including
IPython's cached notebook cell source. The model factory retains the original
source and bound fixed-input snapshots for display and records the array
backend before detaching tensor inputs for tracing.

`hmm.py` offers a second, source-independent frontend for fixed-parameter HMMs:

```python
from smallprob import HMM, Normal

chain = HMM(
    states=["low", "high"],
    initial=[.5, .5],
    transition=[[.9, .1], [.15, .85]],
    emissions=Normal([-1.5, 1.5], [.4, .8]),
    steps=30,
)
```

It builds an ordinary `BoundModel` directly from `Graph`, `Node` and expression
objects: all categorical `z` sites in time order, followed by the corresponding
Normal `y` sites. There is no generated Python source and no separate HMM
inference implementation. Initial support determines the state count; every
transition row is validated, including unreachable rows and one-step models.
Normal mean and SD may each be scalar or have one entry per state. Parameters
are fixed snapshots. The optional `states` names become `hmm_state_labels`
presentation metadata; categorical values remain zero-based integers. Ordinary
`@model` loops remain supported for custom chain structures.

`core.py` contains scalar distributions and expression nodes for constants,
references, arithmetic, powers and indexing. `Graph` records nodes, groups
and series. Scalars are named `slope`; a list is named `y[0]`, `y[1]`, etc.
Even a one-element list preserves its series shape. Mutable inputs are
snapshotted. Rebinding or mutating random-variable names/series is restricted
so observations retain their original meaning.
Tracked random series support symbolic elementwise arithmetic, so a vector
declaration can feed another declaration without list repetition/concatenation.

`HalfNormal(sd)` is a distinct distribution, not a Normal subclass. It has
nonnegative support, density twice `Normal(0, sd)` there, and the underlying
Normal's scale parameter. This prevents the exact Gaussian engine from
mistaking it for an ordinary Gaussian.

`StudentT(df, loc, scale)` is also a separate family. Degrees of freedom and
scale are positive; location is finite. Location is not a mean for `df <= 1`,
and scale is not SD: variance is `scale**2 * df / (df - 2)` only for `df > 2`.
The NumPy implementation samples with `standard_t` and uses a stable gamma-ratio
log density. Exact Gaussian inference rejects StudentT; VI supports it through
the same continuous graph interface.

`Categorical` accepts probability vectors normalized to within input-precision
roundoff and divides by their accepted sum consistently for simulation,
density and inference. The tolerance is `max(1e-8, 4 * eps)` for floating input
dtype, or `1e-8` for integer input. Negative/nonfinite values and materially
unnormalized weights are rejected; this is not a general weight-normalization
API. HMM validation preserves probability-array dtype until this check.

## Array boundary and devices

`arrays.py` supplies an `ArrayBackend` around `array-api-compat` namespaces.
The implemented choices are NumPy and Torch. Lists and NumPy inputs default to
NumPy; Torch model inputs or observations select Torch. Explicit `backend`,
`device` and `dtype` options are accepted by simulation, inference and joint
density. Torch device choices are CPU, available CUDA devices and Apple MPS.
MPS requires float32; CPU/CUDA support float32 or float64. Unsupported
combinations fail explicitly instead of falling back.

The `HMM` constructor follows the same policy for its original `initial` and
`transition` inputs and accepts explicit `backend`, `device` and `dtype`.
Tensors supplied only inside a preconstructed `Normal` emission object have
already become host snapshots, so those do not select Torch automatically.
A Torch HMM can simulate natively, but inference requires the explicit
`backend="numpy"` override because its exact engine remains on CPU.

Fixed input tensors are detached and copied to the host for source-based
tracing, graph constants and scalar validation. This keeps a bound model stable
and supports gradient-enabled/device tensors without implicit NumPy conversion,
but does not preserve differentiation into external caller tensors. Model
parameters are reconstructed on the selected device by each numerical backend.
Gaussian conditioning, Torch simulation, joint density and VI use native
device operations. HMM inference remains NumPy/CPU and rejects a Torch request.

Results preserve the selected array type/device for posterior quantities and
samples. Small VI guide statistics are calculated from detached parameters on
the host, then converted for public results; Torch posterior sampling uses
native device random draws and transformations. Sampling isolates the relevant
Torch RNG states. Device support is a capability, not a speed benchmark.

## Public results

`api.py` supplies `BoundModel.simulate`, `infer`, `describe`, `log_prob`, `math`
and `diagram`.
Observations may be named keywords or an explicit mapping, with overlaps
rejected. Mapping keys can name individual series elements. `None` indicates
a missing observation; NaN is invalid.

All sample arrays have a leading draw axis. Posterior moments include observed
nodes with zero variance. Marginal intervals are exact Gaussian quantiles for
the Gaussian engine or learned-guide quantiles for VI. Categorical/mixture
interval requests are rejected; HMM state probabilities are explicit.
Model return values do not define which sites appear in results.

Exact results expose `log_evidence` and `approximate=False`. VI exposes
`elbo`, optimisation diagnostics, and `approximate=True`, with
`log_evidence=None`. An estimated ELBO never masquerades as exact evidence.

## Numerical backends

`inference.py` recognises dependency structure, not model function names.
`method="auto"` chooses only between supported exact algorithms. Request
`method="vi"` explicitly to approximate a continuous posterior.

- **Linear Gaussian:** all nodes are Normal, with fixed conditional scales
  and means affine in preceding nodes. A dense innovation factor, complete QR
  decomposition and triangular solves give posterior moments, samples and
  evidence while preserving small residual uncertainty.
- **HMM:** one categorical chain of fixed support size, with Normal emissions
  depending on at most one state and supplied transition/emission parameters.
  Log-space forward–backward gives smoothing and evidence; backward sampling
  gives joint state trajectories. Missing observations add no likelihood.
- **VI:** `vi.py` lazily imports Pyro/PyTorch and differentiably lowers
  Normal/HalfNormal/StudentT graphs. SVI optimises a full-covariance Gaussian
  guide in unconstrained space. HalfNormal sites use exponential transforms.
  Device/dtype, vectorised ELBO particles and a decaying learning rate are
  explicit and recorded. The default NumPy-facing path uses float64 CPU Torch
  internally; Torch-facing execution uses the selected device/dtype. No exact
  posterior initialises the guide.

VI returns detached guide parameters. Sampling uses a local NumPy generator or
scoped native Torch generator according to the requested output backend;
moments and marginal quantiles are analytical properties of the learned guide,
not exact target-posterior quantities. Fitting isolates Pyro's parameter store
and restores torch RNG/thread state. A separate held-out Monte Carlo estimate
reports the ELBO and its standard error. Its finite estimate can fluctuate
above log evidence even though the mathematical ELBO cannot.

## Mathematical displays

`display.py` builds equations and directed diagrams from the actual graph,
using validated source expressions to retain readable vector notation and
fixed-input names. Display labels are independent of inference site names;
for example, `slope` can display as `m` and `intercept` as `b`. Normal notation
uses variance (`sd**2`), while StudentT/HalfNormal notation identifies scale.

`math()` exposes rich notebook LaTeX, equation source and the joint-density
factorization. `diagram()` adds observed shading, repeated-observation boxes,
family-shape glyphs and SVG/PNG export with optional Matplotlib. Shading is
presentation, not conditioning, and glyphs do not represent fitted densities.
Simple conditionally independent series are compacted; dependency chains
retain their edges. Recognized fixed HMMs get a horizontal chain layout with
shared or per-state SDs and optional state-label legends. For more than six
time steps, the first three and final step are shown around an ellipsis; full
sequence length, transition/emission equations and factorization remain
explicit. The underlying graph and inference are unchanged. Other expanded
diagrams have a 30-node limit, while mathematical equations remain available.
No external Graphviz executable or TeX installation is required.

## Validation and boundaries

Existing exact kernels remain independent numerical references. The VI suite
compares means, SDs, covariances, correlations and intervals across three seeds
for a Normal mean, a correlated known-noise regression, and an unknown-noise
regression. The last case uses independently checked numerical quadrature,
not an analytic exact posterior. Thresholds were specified before fitting.
See `docs/vi-validation-design.md` and `docs/verification.md`.

The array and distribution suite checks backend parity, actual MPS execution,
Student-t densities against independent formulas/Torch, and three Student-t
posterior fits against numerical quadrature. Version 0.5 adds HMM shorthand
graph equivalence, seeded simulation and posterior-path equivalence, and
independent exhaustive path references. Float32 probability tables are checked
on NumPy and Torch.

The [four-example notebook](notebooks/smallprob_examples.ipynb) remains the
short introduction. A separate [Pyro gallery](notebooks/smallprob_pyro_gallery.ipynb)
covers a noisy weighing scale, interaction regression and Eight Schools.
[Source/adaptation notes](docs/pyro-examples.md) distinguish preserved models
from changed priors and datasets. Direct Pyro joint-density comparisons check
the translated laws; analytic or numerical references assess inference.
These checks establish accuracy for measured cases, not uniform accuracy of a
Gaussian VI guide for heavy-tailed or hierarchical posteriors.

This release has finite scalar graphs and dense methods for small examples.
Categorical VI, HMM parameter fitting, MCMC, minibatches, additional Array API
backends and neural-network models are outside scope. A future HMM parameter backend can
marginalise discrete states with a differentiable forward recursion and then
fit continuous parameters, while retaining the same model notation.
