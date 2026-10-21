"""Two groups, two slopes: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python interaction-regression-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Adaptation: fixed synthetic data replace GDP data, and HalfNormal(1) replaces Uniform(0, 10) for noise SD. The prior change changes the model.
- Both code panes use the adapted prior. The plot shows 90% guide bands for mean responses, excluding new observation noise.
- Against independent numerical integration, the saved fit's largest mean error is 0.126 reference SD and largest SD error is 9.5%. These checks do not establish accurate tails.
- The saved result uses guide="full" (Pyro-backed VI). Choose guide="diagonal" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations.

Pyro's regression tutorial: https://pyro.ai/examples/intro_long.html#example-from-maximum-likelihood-regression-to-bayesian-regression
Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import pyro
import pyro.distributions as dist

def interaction_regression(group, x, y=None):
    intercept = pyro.sample("intercept", dist.Normal(0., 10.))
    group_effect = pyro.sample("group_effect", dist.Normal(0., 1.))
    slope = pyro.sample("slope", dist.Normal(0., 1.))
    interaction = pyro.sample("interaction", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    mean = intercept + group_effect * group + slope * x + interaction * group * x
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.Normal(mean, sigma), obs=y)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

x = torch.linspace(-1.5, 1.5, 20).repeat(2)
group = torch.tensor([0., 1.]).repeat_interleave(20)
truth = {name: torch.tensor(value) for name, value in {
    "intercept": 1.2, "group_effect": 0.7,
    "slope": -0.6, "interaction": 0.9, "sigma": 0.5}.items()}
generator = pyro.poutine.condition(interaction_regression, data=truth)
pyro.set_rng_seed(103)
data = Predictive(generator, num_samples=1)(group, x)
y = data["y"][0]

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(11)
guide = AutoMultivariateNormal(interaction_regression)
steps = 4000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(interaction_regression, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(group, x, y=y)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=4000)(group, x, y=y)
group_0_slope = draws["slope"].reshape(-1)
group_1_slope = group_0_slope + draws["interaction"].reshape(-1)
print(group_0_slope.mean().item(), group_1_slope.mean().item())
samples = draws["interaction"].reshape(-1)
print(samples.mean().item(), torch.quantile(samples, torch.tensor([0.025, 0.975])))
