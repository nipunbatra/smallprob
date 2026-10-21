"""How much does it weigh?: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python weighing-scale-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Same probability model as the original scale example: prior SD 1 and measurement SD 0.75.
- Pyro labels this older tutorial deprecated. The analytic posterior is Normal with mean 9.14 and SD 0.60.

Pyro's scale example: https://pyro.ai/examples/intro_part_ii.html
Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import pyro
import pyro.distributions as dist

def weighing_scale(guess=8.5, measurement=None):
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    pyro.sample("measurement", dist.Normal(weight, 0.75), obs=measurement)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

generated = Predictive(weighing_scale, num_samples=5)()
measurement = torch.tensor(9.5)  # Illustrative reading.

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(7)
guide = AutoMultivariateNormal(weighing_scale)
steps = 3000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(weighing_scale, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(measurement=measurement)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=4000)(measurement=measurement)
samples = draws["weight"].reshape(-1)
print(samples.mean().item(), samples.std().item())
