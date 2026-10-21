"""A line through observations: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python linear-regression-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Thirty synthetic observations, generated with intercept 1.5, slope −0.8 and noise SD 0.35.
- The displayed fit uses NumPy. Torch inputs select Torch execution; exact Gaussian inference also supports available Torch devices.

Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import pyro
import pyro.distributions as dist

def regression(x, y=None):
    intercept = pyro.sample("intercept", dist.Normal(0., 5.))
    slope = pyro.sample("slope", dist.Normal(0., 5.))
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.Normal(intercept + slope * x, 0.35), obs=y)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

x = torch.linspace(-2, 2, 30)
truth = {"intercept": torch.tensor(1.5), "slope": torch.tensor(-0.8)}
generator = pyro.poutine.condition(regression, data=truth)
data = Predictive(generator, num_samples=1)(x)
y = data["y"][0]

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(7)
guide = AutoMultivariateNormal(regression)
steps = 3000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(regression, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(x, y=y)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=4000)(x, y=y)
slopes = draws["slope"].reshape(-1)
intercepts = draws["intercept"].reshape(-1)
print(slopes.mean().item(), torch.quantile(slopes, torch.tensor([0.025, 0.975])))
x_new = torch.linspace(-3, 3, 50)
mean_response = intercepts[:, None] + slopes[:, None] * x_new
print(torch.quantile(mean_response, torch.tensor([0.05, 0.95]), dim=0))
# A band for the mean response; add observation noise for predictive draws.
