# Inference is a backend, not part of the model notation

The frontend records a finite directed graph of scalar random variables and
deterministic expressions. That intermediate representation is the boundary
between model syntax and algorithms.

Version 0.5 supports automatic one-dimensional Normal and StudentT broadcasting
at this boundary. A vector expression becomes scalar sites before any inference
backend sees it. This retains the same conditional independence, shared
parents, observation names and joint density as an explicit comprehension.
The loop-free `HMM(...)` constructor builds the same scalar graph directly;
model notation does not select a different algorithm.

## Implemented

### Linear Gaussian models

For every Normal node, represent its conditional mean as an intercept plus a
linear combination of preceding nodes. Independent innovations with fixed
standard deviations then define a joint Gaussian through an innovation factor.
Apply a complete QR decomposition to the observed rows of that factor. The
constrained coordinates give the conditional mean and marginal evidence through
triangular linear solves; the orthogonal complement gives the remaining random
innovations.

Posterior moments and joint samples use the same remaining innovation factor.
This avoids subtracting nearly equal covariance matrices or projected factors,
preserving uncertainty even when scaled observation noise is very small.

Known-noise normal mean inference and Bayesian linear regression are instances
of this engine. They are not separate special cases matched by function name.
Unobserved response nodes give posterior predictive distributions directly.

Dense factor construction is intentionally limited to small examples. A
future implementation can use Gaussian precision matrices or factorisation
without changing the model interface.

The Gaussian numerical kernel runs through `array-api-compat` on NumPy or
Torch. Its QR, linear solves, conditioning and posterior sampling operate on
the selected array device. Torch CPU/CUDA support float32 or float64; Apple MPS
requires float32. Choose with input tensors or explicit `backend`, `device`
and `dtype` options. Float32 has less precision, so parity tolerances and small
residual-variance checks account for dtype rather than assuming float64 accuracy.

### Hidden Markov chains

Specify a chain without a decorated function or loop:

```python
from smallprob import HMM, Normal

m = HMM(
    states=["low", "high"],
    initial=[.5, .5],
    transition=[[.9, .1], [.15, .85]],
    emissions=Normal([-1.5, 1.5], [.4, .8]),
    steps=30,
)
data = m.simulate(seed=7)
p = m.infer(y=data["y"][0])
p.probabilities("z")  # shape (30, 2)
```

The helper supplies one fixed transition matrix and fixed Normal emission
parameters. Mean and SD can each be shared scalars or vectors with one entry
per state. Initial and transition probabilities must be normalized; only
small input-precision rounding errors are accepted and corrected consistently.
Arbitrary unnormalized weights are rejected. Optional `states` names are
display labels: state observations, samples and probability columns keep
zero-based numeric indices. `steps` is a positive integer. Explicit `@model`
loops remain available for custom chains, including time-varying fixed tables.

Recognise a single chain of categorical variables and Normal emissions that
depend on at most one chain state. Evaluate conditional probabilities and
emission likelihoods for each state; absent observations have unit likelihood.
Run forward and backward recursions in log space. Normalised products give
smoothed state marginals and the final forward normaliser gives log evidence.

Draw posterior state trajectories backwards from the filtered distributions,
conditioning each state on its sampled successor. Then draw unobserved
emissions given the corresponding states. Supplied transitions may be
deterministic or time-varying; support size must remain constant.

This HMM kernel remains NumPy on CPU. A Torch inference request raises an
explicit error; use `backend="numpy"`. This limitation does not prevent native
Torch generative simulation or joint-density evaluation of the same graph.
Torch `initial` or `transition` inputs select Torch at construction; explicit
constructor `backend`, `device` and `dtype` options are also accepted. Tensors
supplied only inside a preconstructed `Normal` emission object have already
been snapshotted on the host and do not select the backend.

The numerical kernels are checked against independent closed-form regression
calculations and exhaustive enumeration of short HMM paths. This is correctness
evidence on supported cases, not a performance benchmark.

Reference: [Oxford Computational Statistics HMM notes](https://www.stats.ox.ac.uk/~caron/teaching/sb1b/lecturehmm.pdf).

### Variational inference

An optional Pyro/PyTorch backend lowers the same graph's deterministic
expressions and Normal/HalfNormal/StudentT distributions to differentiable
Torch tensors. SVI optimises `AutoMultivariateNormal`, a full-covariance Gaussian
in unconstrained latent space. This can represent strongly correlated
regression coefficients. HalfNormal variables use a log transform, yielding
lognormal guide marginals on their original positive scale.

The NumPy-facing default runs float64 CPU Torch internally. A selected Torch
backend runs the likelihood, guide, gradients and optimizer on its device and
dtype, including CUDA or float32 MPS when available. The model's notation does
not change. Observed likelihoods and ELBO particles are batched; the frontend
still expands vector declarations into scalar graph sites.

The defaults are 3,000 steps, 32 vectorised Monte Carlo particles per step,
and initial learning rate 0.03. Clipped Adam uses gradient clipping at 10
and exponential learning-rate decay to one tenth of its initial value.
Settings, loss history, latent ordering, transforms and learned covariance
are recorded. The guide starts from prior-based values, never an exact
posterior solution.

The fitted posterior is detached from Pyro. Moments and marginal intervals are
computed from its small guide parameters on the host, then converted to the
selected public array type/device. Posterior sampling uses NumPy or native
Torch draws and transformations. `approximate=True` labels all of these
as properties of an approximation to the target posterior. A held-out
2,048-particle Monte Carlo ELBO estimate and its standard error are reported
separately; `log_evidence` is `None`. A finite Monte Carlo estimate can exceed
exact evidence by sampling error and is not a guaranteed numerical bound.

VI supports scalar Normal/HalfNormal/StudentT graphs and fixed indexing. Categorical
nodes are rejected. `method="auto"` remains exact-only; the caller opts into
approximation with `method="vi"`. Pyro imports are lazy, so NumPy exact models
need only the base NumPy/Array API dependencies. Backend state is scoped and
relevant CPU/device RNG and thread settings are restored.

The same Normal and known-noise regression models are compared with the exact
engine across three seeds. Unknown-noise regression is checked against an
independent numerical integral over log noise, analytically integrating out
the coefficients and verifying grid refinement and tails. See the
[validation plan and results](vi-validation-design.md).

Student-t density is also checked against independent formulas and Torch
references. Three Student-t posterior fits are compared with an independent
numerical quadrature reference; the backend validation includes a complete
3,000-step MPS VI fit. See `examples/backend_validation.py` and the measured
verification record. Neither matching a finite reference case nor a smooth loss
curve guarantees a Gaussian guide captures every heavy-tailed posterior.

### Student-t parameter conventions

`StudentT(df, loc, scale)` uses positive degrees of freedom and scale. Its
location is the mean only for `df > 1`; its variance is
`scale**2 * df / (df - 2)` only for `df > 2`. The `scale` parameter must not be
reported as a standard deviation. StudentT has all-real support and heavier
tails than a Normal. Its presence excludes exact Gaussian inference; callers
must opt into VI explicitly. Degrees of freedom can be fixed or represented by
a positive random variable, subject to the same continuous graph restrictions.

### External tensor and display boundaries

Graph tracing and input validation detach fixed inputs/observations and copy
them to the host. Inference therefore does not differentiate into external
caller tensors. Differentiation within VI still operates on the selected
device's latent computation graph. Public array types describe numerical
execution/results, not end-to-end automatic differentiation through the API.

`math()` and `diagram()` describe the generative graph independently of the
chosen inference engine. Display labels do not rename observation keys.
Normal mathematical notation squares its SD to show variance; StudentT and
HalfNormal formulas identify scale. Observed shading is a display annotation,
not an inference operation, and distribution glyphs are illustrative shapes.
Recognized fixed HMMs support shared or per-state SDs and state-name legends.
A 30-step chain is drawn with the first three steps, an ellipsis and the final
step; equations and factorization describe the full sequence. This abbreviation
does not remove graph nodes or change numerical inference. Unrecognized
expanded graphs retain the 30-displayed-node limit; `math()` still provides
their equations.

VI does not guarantee accurate posterior uncertainty merely because the ELBO
stabilises. Keep exact references and posterior predictive checks in the test
strategy. MCMC is a separate possible backend, useful for cross-checking small
non-conjugate examples.

References: [Pyro's SVI introduction](https://pyro.ai/examples/svi_part_i),
[Pyro automatic guides](https://docs.pyro.ai/en/dev/infer.autoguide.html).

### Worked translations from Pyro

The separate [Pyro gallery notebook](../notebooks/smallprob_pyro_gallery.ipynb)
adds a noisy weighing scale, a regression with a group interaction and Eight
Schools. [The source notes](pyro-examples.md) record which probability laws are
preserved and which priors or data were adapted. The scale has an analytic
posterior; a fixed-noise interaction companion also has exact inference.
Unknown-noise interaction regression and the adapted hierarchical model use
explicit VI, checked against independent numerical integration. Probability-law
checks and VI approximation errors are reported separately.

From the project root:

```sh
.venv/bin/python -m examples.run_notebook notebooks/smallprob_pyro_gallery.ipynb
.venv/bin/python -m examples.export_notebook notebooks/smallprob_pyro_gallery.ipynb
```

These examples supplement the original four-example introduction; they do not
claim support for the entire Pyro tutorial catalogue.

## Next extensions and current exclusions

For hybrid HMM parameter inference, a differentiable forward recursion could
marginalise discrete states before applying VI to continuous parameters.
Arbitrary discrete structures need additional estimators or enumeration
strategies; continuous reparameterisation is not sufficient.

Version 0.5 does not implement categorical VI, MCMC, HMM parameter estimation,
minibatching, other Array API backends, full Python control flow over random
values, or multidimensional event distributions. One-dimensional Normal and
StudentT declarations expand to scalar sites; they do not introduce a
multivariate event. Dense factors and a full-covariance VI guide are intended
for small models; device support alone does not remove those scaling limits.
