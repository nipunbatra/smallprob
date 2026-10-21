"""One unknown mean: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python normal-mean-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Normal(mean, sd) uses standard deviation in Python; the mathematical diagram writes variance.
- The observation y = 2 is an illustrative input. The posterior calculation is analytic.

Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import pyro
import pyro.distributions as dist

def normal_observation(y=None):
    mu = pyro.sample("mu", dist.Normal(0., 3.))
    pyro.sample("y", dist.Normal(mu, 1.), obs=y)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

generated = Predictive(normal_observation, num_samples=5)()
y = torch.tensor(2.0)  # Illustrative observation.

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(7)
guide = AutoMultivariateNormal(normal_observation)
steps = 3000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(normal_observation, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(y=y)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=4000)(y=y)
samples = draws["mu"].reshape(-1)
print(samples.mean().item(), samples.std().item())
