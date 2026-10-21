# Smallprob

Describe a generative model in ordinary Python. Generate data and infer unknowns
from the same definition. No plates, tensor dimensions or manual sample names.

Version 0.5 is a small research prototype with NumPy and PyTorch array
execution, exact linear Gaussian/HMM inference, optional variational inference,
and notebook-ready model diagrams. It adds a loop-free `HMM(...)` constructor
and three runnable examples translated from Pyro. It supports Normal,
HalfNormal, StudentT and Categorical distributions. Model definitions need no
backend imports.

The [personal showcase](docs/website.md) compares seven Smallprob and Pyro model
definitions, with diagrams, saved results and runnable downloads. The full
library lives in [nipunbatra/smallprob](https://github.com/nipunbatra/smallprob).
It is a personal prototype, not a PyPI release.

```python
from smallprob import model, Normal, HalfNormal

@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)

# Given x_data and y_data:
p = regression(x_data).infer(y=y_data, method="vi", seed=7)
print(p.mean("sigma"), p.interval("sigma"))
```

Inside `@model`, a distribution declaration introduces a random variable.
`draw(...)` remains supported, and `return` is optional.

## Run it

Requires Python 3.11 or newer. From this project directory:

```sh
uv sync --extra dev --extra vi --extra notebook --locked
.venv/bin/python -m examples.run_notebook
.venv/bin/python -m examples.export_notebook
.venv/bin/python -m examples.unknown_noise --json outputs/unknown-noise.json
.venv/bin/python -m examples.demo --json outputs/demo-results.json
.venv/bin/python -m pytest -q
```

Start with [the four-example notebook](notebooks/smallprob_examples.ipynb): a
Normal observation, linear regression, robust Student-t regression with unknown
scale, and an HMM. It includes generated data, inference, mathematical equations,
diagrams and plots. Open it in your notebook editor and select this project's
Python environment. The command above executes every cell using the calling
`.venv` interpreter and saves the results back into the notebook; it does not
depend on a separately registered global kernel. The export command creates
`outputs/smallprob-examples.html`, a self-contained preview with the saved
equations, diagrams and results; open it in a browser without a notebook server.

Then explore [the separate Pyro gallery](notebooks/smallprob_pyro_gallery.ipynb):
a noisy weighing scale, regression with a group interaction, and Eight Schools.
The [source and adaptation notes](docs/pyro-examples.md) identify the original
Pyro examples, changed priors, data provenance and numerical checks. The scale
keeps the original probability model; the other two are explicit adaptations
to supported distributions. Run and export this notebook with:

```sh
.venv/bin/python -m examples.run_notebook notebooks/smallprob_pyro_gallery.ipynb
.venv/bin/python -m examples.export_notebook notebooks/smallprob_pyro_gallery.ipynb
```

The second command creates `outputs/smallprob-pyro-gallery.html`. The original
four-example notebook remains the short introduction.

The base dependencies are NumPy and `array-api-compat`. Use `uv sync --locked`
for NumPy exact inference and simulation. Optional extras are `torch` for
PyTorch array execution, `vi` for Pyro/PyTorch inference, `viz` for Matplotlib
diagrams, and `notebook` for Matplotlib plus notebook execution tools. VI already
installs PyTorch, so it does not also require the `torch` extra. Missing optional
dependencies produce installation instructions.

Use `examples.vi_validation` for the original nine-fit numerical comparison and
`examples.backend_validation` for backend and Student-t validation. These take
longer than the small demos.

A validation figure uses the optional plotting dependency:

```sh
uv run --extra dev --extra vi --extra viz python -m examples.plot_vi_validation
```

With standard Python tools instead:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,vi,notebook]'
.venv/bin/python -m examples.run_notebook
```

`uv.lock` records the tested dependency resolution.

## Normal observations

Define models in a `.py` file or notebook cell whose source can be inspected.

```python
from smallprob import model, Normal

@model
def normal():
    mu = Normal(mean=0, sd=5)
    y = Normal(mean=mu, sd=1)

m = normal()
data = m.simulate(seed=7, fixed={"mu": 2.0})
posterior = m.infer(y=data["y"][0])

print(posterior.mean("mu"))
print(posterior.interval("mu"))
print(posterior.sample(draws=1000, seed=8)["mu"])
```

For `Normal`, `sd` means **standard deviation**. For `HalfNormal(sd)`, it is
the scale of the underlying zero-mean Normal, whose absolute value defines
the HalfNormal; the resulting HalfNormal has a smaller standard deviation.
The numerical priors here are
illustrative choices, not automatic recommendations for real datasets.

## Linear regression

```python
@model
def regression(x, noise=0.5):
    intercept = Normal(mean=0, sd=5)
    slope = Normal(mean=0, sd=5)
    y = Normal(mean=intercept + slope * x, sd=noise)

m = regression([-2, -1, 0, 1, 2])
data = m.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
p = m.infer(y=data["y"][0])
print(p.mean("slope"), p.interval("slope"))
print(p.method)  # exact-gaussian
```

Noise is a supplied input in this exact example. The engine derives the joint Gaussian
from the dependency graph and conditions it analytically, including posterior
correlations between coefficients. This is Bayesian regression, with the
explicit priors shown above.

To predict at additional inputs, include them in `x` and put `None` at those
positions in the observed `y` list. The missing outputs remain random and their
posterior includes both parameter uncertainty and observation noise.

### One declaration, multiple observations

`x` can be a Python list, tuple, one-dimensional NumPy array or Torch tensor. With a vector
`x`, the expression `y = Normal(intercept + slope * x, sigma)` creates one
observation per input. Each has its own noise draw, conditional on the shared
intercept, slope and sigma. It is equivalent to the explicit comprehension.
The graph still contains `y[0]`, `y[1]`, etc., so simulation, missing observations,
exact inference and VI use the same existing API.

- Scalar mean and scalar SD produce a scalar random variable.
- A vector mean or SD produces a series. Scalars and one-element vectors
  broadcast to the other vector's length; other vector lengths must match.
- A one-element vector remains a series of length one. An empty vector gives
  an empty series. Sample shape is always `(draws, number_of_observations)`.
- A vector SD supplies one noise scale per observation. This specifies
  conditionally independent Normal variables, not a multivariate Normal with
  a supplied covariance matrix.
- Matrices and incompatible lengths raise an error instead of being flattened.

Series can feed another declaration, for example `z = Normal(y, 0.2)`.
Arithmetic on random series is elementwise. `Categorical([0.3, 0.7])` continues
to mean one categorical variable with two possible states.

Fixed inputs retain ordinary Python semantics until combined with a symbolic
random variable. Thus the regression's `slope * x` works with a list, while
the entirely fixed expression `2 * [1, 2]` still repeats a Python list. Use a
NumPy array for purely deterministic vector arithmetic. A vector declaration
must have its own series name; nesting vector draws inside a comprehension
would require a second observation dimension and is rejected.

## Learning the noise with VI

Replace the fixed noise input with `sigma = HalfNormal(1)`, as in the opening
example, and request `method="vi"`. The prior is on the positive observation
standard deviation. Use the same `simulate` method to generate data:

```python
from examples.models import regression_unknown_noise

m = regression_unknown_noise([-2, -1, 0, 1, 2])
data = m.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8, "sigma": 0.5})
p = m.infer(y=data["y"][0], method="vi", seed=7)
print(p.mean("sigma"), p.sd("sigma"), p.interval("sigma"))
print(p.approximate)    # True
print(p.elbo)           # Monte Carlo ELBO estimate
print(p.log_evidence)   # None: ELBO is not exact evidence
print(p.diagnostics)   # family, settings and optimisation history
```

VI uses a full-covariance Gaussian in unconstrained latent space. Positive
HalfNormal sites use a log transform, so their variational marginals are
lognormal. Intervals and moments describe this learned approximation. Training
settings can be supplied as `steps=3000`, `learning_rate=0.03`, `particles=32`,
and `seed=7`. Stable loss alone does not establish accurate posterior uncertainty.

`method="auto"` retains exact inference and never silently switches to VI.
You can explicitly request VI on a known-noise model to compare it with exact
inference. The validation script does this across three seeds and checks the
unknown-noise model against a separately computed quadrature reference.

## NumPy, PyTorch and devices

The array boundary uses [`array-api-compat`](https://data-apis.org/array-api-compat/).
This version implements NumPy and PyTorch; other Array API implementations are
not yet supported. Ordinary lists and NumPy arrays default to NumPy. A bound
Torch input or Torch observation selects Torch, preserving its device and using
float64 if any input tensor is float64, otherwise float32.

```python
import torch

x = torch.linspace(-2, 2, 30, dtype=torch.float64)
m = regression(x, noise=0.5)  # known-noise regression defined above
data = m.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
p = m.infer(y=data["y"][0])
print(type(p.sample(10, seed=8)["slope"]))  # torch.Tensor
```

You can also choose execution explicitly on `simulate`, `infer` and `log_prob`:

```python
p = m.infer(y=data["y"][0], backend="torch", device="cpu", dtype="float64")
# With an available Apple GPU:
p = m.infer(y=data["y"][0], backend="torch", device="mps", dtype="float32")
# CUDA hardware uses device="cuda"; float32 and float64 are supported.
```

Gaussian conditioning, Torch simulation/joint density, and VI execute native
Torch operations on the selected device. MPS requires float32. Devices that are
unavailable raise an error; execution does not silently fall back to CPU. HMM
exact inference remains NumPy on CPU: request `backend="numpy"` for that engine.

Fixed model inputs and observations are detached and snapshotted on the host
while constructing the scalar graph and validating values. Gradients do not
flow back into externally supplied tensors. VI training still differentiates
the latent model internally. Its small fitted-guide summary statistics are
computed on the host and converted to the selected output array type; Torch
posterior sampling is native. This is device-aware inference, not an end-to-end
differentiable model of the caller's tensors. The prototype does not promise a
GPU speedup for small examples.

## Student-t observations

```python
from smallprob import StudentT

@model
def robust_regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(df=4, loc=b + m * x, scale=sigma)

robust = robust_regression([-2, -1, 0, 1, 2])
data = robust.simulate(seed=7, fixed={"b": 1.5, "m": -0.8, "sigma": 0.5})
p = robust.infer(y=data["y"][0], method="vi", seed=7)
```

StudentT has heavier tails than a Normal. Its `df` and `scale` must be positive.
`loc` is a location parameter; its mean exists only when `df > 1`.
`scale` is **not a standard deviation**: for `df > 2`, the variance is
`scale**2 * df / (df - 2)`. Its three parameters support scalar/1D broadcasting.
StudentT models require explicit VI for posterior inference; exact Gaussian
inference never treats them as Normal models. A Gaussian variational guide can
miss target tail behaviour, so use the validation examples as checks, not a
general guarantee of uncertainty accuracy.

## Equations and graphical models

Use a rich display as the last expression of a notebook cell:

```python
labels = {"intercept": "b", "slope": "m", "sigma": r"\sigma"}
m = regression_unknown_noise([-2, -1, 0, 1, 2])
m.math(labels=labels)
```

```python
m.diagram(labels=labels, observed=["y"])
```

The graph shows dependencies, repeated observations, distribution equations
and small distribution-family glyphs. Normal equations use the conventional
variance parameter: `Normal(mu, sigma)` is displayed as
$\mathcal{N}(\mu,\sigma^2)$. Student-t and HalfNormal equations explicitly use
scale. Glyphs illustrate distribution families; they are not fitted densities.
Marking `observed` changes shading only, not model conditioning.

Use `b` and `m` directly in Python, as in the robust model, or apply labels for
display without changing names used by inference. Distinct quantities should
have distinct symbols: slope `m`, intercept `b`. TeX labels such as `r"\sigma"`
are accepted. Read `m.math().latex` for equation source and
`m.math().factorization` for the joint-density factorization. Export diagrams:

```python
m.diagram(labels=labels, observed=["y"]).save("outputs/regression.svg")
m.diagram(labels=labels, observed=["y"]).save("outputs/regression.png")
```

Mathematical text needs no graphics dependency; rendering diagrams requires
the optional `viz` or `notebook` extra. Simple independent series are grouped;
recognized fixed-parameter HMMs use a horizontal chain. Long chains, including
the 30-step example below, show the first three steps, an ellipsis and the final
step, with equations for the whole sequence. This is a display abbreviation;
simulation and inference still use every state and emission. Other expanded
diagrams are limited to 30 displayed nodes; `math()` retains all equations.

## Hidden Markov models

```python
from smallprob import HMM, Normal

m = HMM(
    states=["low", "high"],
    initial=[0.5, 0.5],
    transition=[[0.9, 0.1], [0.15, 0.85]],
    emissions=Normal([-1.5, 1.5], 0.5),
    steps=30,
)
data = m.simulate(seed=7)
p = m.infer(y=data["y"][0])
print(p.probabilities("z"))              # (30, 2), smoothed state probabilities
print(p.sample(draws=10, seed=8)["z"])   # (10, 30), joint state trajectories
print(p.log_evidence)                    # log p(observed data)
m.diagram(observed=["y"])
```

No decorator, loop or `append` is needed. `HMM` builds the same scalar graph
used by the existing model and inference APIs. Explicit `@model` loops remain
valid when a more custom chain is needed.

State indices start at zero. Optional `states` contains unique display names
in that order; observations, samples and probability columns still use numeric
indices. `transition[i][j]` is the probability of moving from state `i` to state
`j`; each row must sum to one. Small dtype-level rounding errors in normalized
rows, including float32 tables, are corrected consistently. Arbitrary
unnormalized weights are rejected. Emission `y[t]` belongs to state `z[t]`, and
`steps` must be a positive integer.

The constructor accepts fixed Normal emissions. Each mean and SD is a scalar
or a vector with one entry per state. For example,
`emissions=Normal([-1.5, 1.5], [0.4, 0.8])` gives different noise levels in the
two states. Initial, transition and emission parameters are supplied, not
learned; this shorthand uses one transition matrix throughout the sequence.

Lists and NumPy tables select NumPy; Torch `initial` or `transition` tensors
select Torch, with constructor overrides `backend`, `device` and `dtype` also
available. A `Normal` object has already snapshotted its parameters, so tensors
supplied only inside `emissions` do not select Torch automatically. Torch
simulation and joint density work, while exact HMM inference remains NumPy/CPU:
use `m.infer(y=data["y"][0], backend="numpy")` for a Torch-backed model. Unsupported
execution requests raise an error instead of silently changing device.

Inference uses log-space forward–backward smoothing and backward sampling of
whole trajectories. Independently sampling each time's marginal would lose
temporal dependence. Marginal most-likely states are not a Viterbi path.

## A few consistent rules

- Calling `normal()` builds a graph; it does not sample. Direct distribution
  declarations and explicit `draw` introduce random variables during tracing.
- `simulate` returns every random variable, including latent ones. There is
  always a draw axis: scalar `mu` has shape `(draws,)`; list `y` has shape
  `(draws, n)`. With the default one draw, use `data["y"][0]` as observations.
- `fixed` in simulation clamps chosen variables to generate data at known
  values. It does not perform posterior conditioning; use `infer(observed=...)`
  for that.
- `infer()` without observations gives the prior for exact-supported models;
  explicit VI approximates the prior when observations are absent.
- A missing observation is `None`, never NaN. Exact indexed observations such as
  `{"y[2]": 1.4}` work too. Overlapping group and indexed values are rejected.
- `m.describe()` lists random variables and their parents. `m.log_prob(values)`
  evaluates the **joint** log density/mass for a complete assignment; posterior
  `log_evidence` integrates/sums out unknowns for exact inference. VI exposes
  an estimated `elbo` separately and sets `log_evidence` to None.
- Intervals are marginal, not simultaneous: exact Gaussian quantiles for the
  Gaussian engine, and approximate guide quantiles for VI. Use state
  probabilities for HMM states.
- `infer(y=data)` and `infer(observed={"y": data})` are equivalent. Use the
  mapping for variables named like options (`seed`, `steps`, `method`, etc.).
- Model `return` values are for readability; the public results contain named
  random variables, not a reconstructed Python return value.

## Deliberate limits

This is a restricted graph-building subset of Python. Direct `Normal(...)`,
`HalfNormal(...)`, `StudentT(...)`, `Categorical(...)` and explicit `draw(...)` declarations in
scalar assignments, list comprehensions and `list.append` get automatic names.
Normal and StudentT declarations also expand one-dimensional parameters automatically.
Use direct imports of the constructors without shadowing those names. Outside
`@model`, constructors still create ordinary distribution objects. Use
`draw(..., name="...")` where automatic naming is unavailable.
Fixed-input loops and branches work. Branching on an unknown random value,
arbitrary numerical library calls on symbolic values, dynamic random-sized
models and nested decorated model calls are unsupported.

Named random-variable lists have permanent observation indices. Appending the
next declared draw is supported; reordering, replacing or deleting items is
rejected. Use a separate derived list if you want to rearrange references.
Do not rebind a random variable's original name to a transformed value; assign
the derived quantity to a new name so observations keep their original meaning.
Mutable parameter inputs are snapshotted when their expressions are created,
so later edits do not change an already-declared distribution.

The exact Gaussian inference engine requires constant noise and affine conditional
means. The HMM engine requires one categorical chain with fixed support size
and Normal emissions depending on at most one state. Inference recognises
structure rather than function names. Unsupported structures raise a clear
error; there is no silent switch to an inaccurate approximation.

VI currently supports continuous Normal/HalfNormal/StudentT graphs, not
Categorical latents. HMM state inference stays exact on NumPy; HMM parameter
learning, MCMC and minibatching remain outside this version.

The Gaussian engine builds dense innovation factors and is meant for small
examples. Normal/StudentT broadcasting and ordinary Python loops expand into scalar
graph sites; multidimensional events and large-scale compilation remain future
work. Do not treat the prototype as a
production probabilistic programming system or execute untrusted model code.

## Validation and future inference

The exact engines remain independent reference implementations. Future work
can add other variational families, minibatching and hybrid HMM parameter
inference behind the same graph interface. For hybrid HMMs, marginalising the
discrete states before fitting continuous parameters can preserve the chain's
dependence structure.

See [the inference design](docs/inference.md), [the runnable model definitions](examples/models.py),
[the Pyro example translations](docs/pyro-examples.md), and
[the VI validation plan and measured results](docs/vi-validation-design.md).
