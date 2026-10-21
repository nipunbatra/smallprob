# Pyro examples in Smallprob

The gallery translates three examples from official Pyro material. Each model
can generate data with `.simulate()` and condition on data with `.infer()`.
The sources were inspected on 21 October 2026. These are small translations,
not an implementation of the full Pyro example catalogue.

| Example | Source | What is preserved | Adaptation and inference |
|---|---|---|---|
| Noisy weighing scale | [Pyro's older inference tutorial](https://pyro.ai/examples/intro_part_ii.html) | Normal(guess, 1) weight, Normal(weight, 0.75) measurement, guess 8.5, measurement 9.5 | Faithful probability model. Exact posterior Normal(9.14, 0.6). Pyro marks this tutorial deprecated. |
| Group, slope and interaction | [Introduction to Pyro](https://pyro.ai/examples/intro_long.html#example-from-maximum-likelihood-regression-to-bayesian-regression) | Intercept prior Normal(0, 10), three coefficient priors Normal(0, 1), two predictors and their interaction | Fixed synthetic observations replace the GDP dataset. HalfNormal(1) replaces Uniform(0, 10) for noise. Approximate VI; a fixed-noise companion uses exact inference. |
| Eight Schools | [Pyro 1.9.1 model](https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/mcmc.py), [data](https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/data.py) | Published eight effects and standard errors, Normal(0, 10) population mean, standard-normal school offsets, non-centered effect `mu + tau * eta` | HalfNormal(5) replaces HalfCauchy(25), which Smallprob does not support. Approximate VI replaces NUTS. This is a different prior and therefore a different posterior from the original example. |

The adaptations are deliberate and visible. In particular, substituting
HalfNormal for HalfCauchy is a substantive modeling choice, not a syntax change.
The models use Normal standard deviations in code.

## 1. How much does the object weigh?

```python
from smallprob import model, Normal

@model
def weighing_scale(guess=8.5):
    weight = Normal(guess, 1)
    measurement = Normal(weight, 0.75)

scale = weighing_scale()
synthetic = scale.simulate(draws=5, seed=7)
posterior = scale.infer(measurement=9.5)
posterior.mean("weight"), posterior.sd("weight")  # 9.14, 0.60
scale.diagram(observed=["measurement"])
```

## 2. Does the slope differ between two groups?

```python
from examples.pyro_gallery import interaction_regression, interaction_data

group, x, y, truth = interaction_data()
model = interaction_regression(group, x)
posterior = model.infer(y=y, method="vi", seed=11, steps=4000)
posterior.mean("interaction"), posterior.interval("interaction")
model.diagram(observed=["y"])
```

The underlying declaration is
`y = Normal(intercept + group_effect * group + slope * x + interaction * group * x, sigma)`.
Group 0 has slope `slope`; group 1 has slope `slope + interaction`.
For a known noise level use `interaction_known_noise(group, x, noise=0.5)`;
then `.infer(y=y)` is exact. The synthetic truth is available for illustrating
the model, but it is not used by inference.

## 3. What can eight schools learn from one another?

```python
import numpy as np
from smallprob import HalfNormal
from examples.pyro_gallery import eight_schools, SCHOOL_EFFECTS

schools = eight_schools()
synthetic = schools.simulate(draws=2, seed=7)
posterior = schools.infer(y=SCHOOL_EFFECTS, method="vi", seed=11, steps=4000)
draws = posterior.sample(draws=4000, seed=12)
effects = draws["mu"][:, None] + draws["tau"][:, None] * draws["eta"]
effects.mean(axis=0)
schools.diagram(observed=["y"])
```

The definition is only four random-variable declarations:

```python
@model
def eight_schools(se):
    mu = Normal(0, 10)
    tau = HalfNormal(5)
    eta = Normal(np.zeros(len(se)), 1)
    y = Normal(mu + tau * eta, se)
```

`eta` contains eight independent offsets; `mu` and `tau` are shared. The
population mean and amount of between-school variation are learned together.
The derived effects are computed from joint posterior draws, preserving their
dependence. A full-rank Gaussian guide does not guarantee accurate tail
uncertainty in this hierarchical model.

## Reproduce the checks

```sh
uv sync --extra dev --extra vi --extra viz --extra notebook
uv run --no-sync python -m examples.pyro_gallery
uv run --no-sync python -m pytest -q tests/test_pyro_gallery.py
```

`outputs/pyro-gallery-validation.json` records direct joint-density comparisons
against separately written Pyro models of the translated probability laws,
the scale's analytic posterior, an independent Gaussian reference for the
known-noise interaction model, and one-dimensional numerical references for
the two VI examples. The references analytically integrate the Gaussian
variables and numerically integrate the positive scale. Refining each grid
and separately expanding its range checks the reported means and standard
deviations, including the eight derived school effects. The fixed acceptance
limits are an absolute moment change below 1e-6 and estimated posterior mass
outside the original range below 1e-7. The latter is measured under the wider
numerical grid; it is not a rigorous bound on probability beyond that grid.

The report separates probability-model checks from VI approximation errors.
A passing gallery check means the models have the declared probability law,
exact answers agree with references, references are numerically resolved and
VI produces finite output. It does not mean VI has converged or that all
posterior uncertainty is accurate. Read the measured mean and SD errors.
The separate `vi_accuracy_passed` field uses the prototype's existing moment
limits: mean error at most 0.35 reference standard deviations and relative SD
error at most 20%. The limits are declared before fitting and are not relaxed
for the hierarchical example.

Not yet covered: Bernoulli/logistic regression, learned mixture weights,
Dirichlet priors, VAEs, amortized inference, general discrete enumeration,
NUTS and arbitrary PyTorch modules inside a model. These need additional
distributions, transformations or inference support; they are not hidden
behind the simple syntax.
