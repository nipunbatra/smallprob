# Version 0.2 implementation contract

User-authorised extension: genuine VI, unknown regression noise, simpler Python
notation, validation against existing exact engines. Preserve v0.1 functionality.

## Public surface

```python
@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = [Normal(intercept + slope * xi, sigma) for xi in x]

m = regression(x)
data = m.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8, "sigma": .5})
p = m.infer(y=data["y"][0], method="vi", seed=1)
p.mean("sigma"), p.sd("sigma"), p.interval("sigma")
p.sample(draws=1000, seed=2)
p.elbo, p.diagnostics
```

Direct distribution constructors in assignment, comprehension, list/tuple and
append declaration positions lower to the EXISTING draw pipeline. Explicit
draw remains supported. Distribution constructors nested inside draw must not
be drawn twice. No return is necessary (already true). Outside @model,
constructors remain reusable distribution objects. Restrict shorthand to direct
unshadowed imports Normal/HalfNormal/Categorical; reject uncertain/shadowed
constructor binding rather than change ordinary Python silently. Existing
source readability, stable names and mutation safeguards still apply.

## Ownership

- Frontend worker: core.py, syntax.py, __init__.py, tests/test_simpler_syntax.py.
  Add HalfNormal(sd=1) with Expr .sd, dependencies/sample/log_prob validation.
  Support is x>=0; its continuous density is doubled Normal(0,sd) on that
  support. Draws nonnegative; VI exp transform yields strictly positive samples.
  Add it to Graph/builder accepted types. Do not subclass Normal if that makes
  existing isinstance-based exact inference silently treat it as Gaussian.
- VI worker: new vi.py and its own focused test_vi_backend.py if useful. Do not
  modify API/core/inference.py/project config. Root owns dispatch and wrappers.
- Validation worker: examples/vi_validation.py, tests/test_vi_accuracy.py,
  docs/vi-validation-design.md. Coordinate run settings with VI worker.
- Root: api.py, inference.py dispatch only (preserve numerical exact kernels),
  config/dependencies, docs/README/examples, plots, packaging and review.

## VI engine interface

`infer_vi(graph, observed_flat, *, steps=3000, learning_rate=.03,
particles=32, seed=0) -> VIResult` in vi.py. Use optional pyro-ppl/PyTorch via
lazy imports, with float64 CPU tensors. Pyro AutoMultivariateNormal + SVI and
Trace_ELBO, reparameterisation and automatic positive-support transformation.
Full covariance in unconstrained latent space is the explicit initial family.
Vectorise ELBO particles internally where possible. Use decaying optimisation
learning rate if it improves stability; report schedule. No exact-engine
initialisation or exact solution shortcut, so comparisons actually validate VI.

All graph nodes must be Normal or HalfNormal initially. Reject Categorical
models clearly; HMM state inference remains exact. Validate unsupported Expr
forms explicitly. Preserve differentiation through Expr arithmetic. Any index
must be fixed, not a random continuous index. Parameters and observations need
consistent finite/support checks; fail clearly on nonfinite optimisation.

`VIResult` matches exact result fields/methods:
- method = 'vi-fullrank'; log_evidence = None (never ELBO masquerading as evidence)
- means, variances: flat maps including observed variables with zero variance.
- marginals = {}.
- sample(draws, numpy_rng): flat arrays including observations, detached from
  Pyro state; original-scale positive variables transformed appropriately.
- interval(probability): returns two flat dicts of guide marginal quantiles.
- elbo: held-out Monte Carlo ELBO estimate, labelled approximate.
- diagnostics: JSON-safe backend/version, family, seed, steps, particles,
  learning rate/schedule, loss history, initial/final loss summaries, warnings,
  latent variable ordering and optional unconstrained covariance/means.

Guide marginal moments/quantiles can be computed analytically for identity
(Normal sites) and exp (HalfNormal sites) transforms once the learned Gaussian
is detached. Check actual Pyro guide latent order and transforms rather than
assuming an undocumented incompatible layout. Observations are not guide
latents. No observation case still uses VI when explicitly requested. Handle
zero latents explicitly or reject with useful guidance.

Isolate Pyro parameter store; restore global torch RNG and any temporary thread
settings. Do not leak optimisation state between models or mutate prior
Posterior results. Local NumPy RNG drives returned posterior sampling. A module
lock may guard global backend state. Do not use pyro.set_rng_seed unless all
affected global RNG states are restored.

Root API extends `infer(observed=None, *, method='auto', steps=None,
learning_rate=None, particles=None, seed=None, **observations)`; named
observations and explicit dict are both accepted without silent overlap.
method='auto' continues exact inference only. Unsupported exact models suggest
explicit method='vi' where appropriate; no silent switch to approximation.

## Validation

1. Same normal-mean model/data through VI and exact, compare posterior mean/SD,
   and ELBO to exact log evidence with Monte Carlo uncertainty acknowledged.
2. Known-noise regression with correlated coefficients: compare full posterior
   mean/SD/covariance and credible intervals across at least three fixed seeds.
   Report numerical errors and thresholds selected before observing results.
3. Unknown-noise regression (HalfNormal noise): synthetic parameter recovery
   plus an independent one-dimensional quadrature reference over log sigma.
   Given sigma, coefficients are Gaussian; integrate their exact conditional
   moments and p(y|sigma)*p(sigma)*sigma Jacobian over log sigma. This is a
   numerical reference, NOT an analytic exact posterior; verify grid refinement
   and tail mass. Include this comparison if feasible within the prototype.
4. Verify positive sigma samples, deterministic seeds, parameter-store/RNG
   isolation, no old posterior mutation, interval labelling and discrete
   rejection. Existing exact HMM/normal/regression tests must still pass.

NumPy remains sole base runtime dependency; optional `vi` extra adds pyro-ppl.
Pin the actually resolved versions in uv.lock. Documentation must distinguish
VI approximation, exact Gaussian/HMM results, and numerical quadrature.
