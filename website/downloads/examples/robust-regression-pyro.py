"""A line with heavier tails: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python robust-regression-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Synthetic Normal observations with two deliberately added outliers. This one dataset illustrates the model, not a general robustness guarantee.
- StudentT scale is not its SD. With df = 4, variance is twice scale squared.
- VI intervals describe the fitted guide; they are not exact posterior intervals. This VI implementation uses Pyro and PyTorch internally.
- The saved result uses guide="full" (Pyro-backed VI). Choose guide="diagonal" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations.

Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import pyro
import pyro.distributions as dist

def robust_regression(x, y=None):
    b = pyro.sample("b", dist.Normal(0., 5.))
    m = pyro.sample("m", dist.Normal(0., 5.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.StudentT(4., b + m * x, sigma), obs=y)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

import numpy as np

x_np = np.linspace(-2, 2, 30)
x = torch.as_tensor(x_np)
generated = Predictive(robust_regression, num_samples=1)(x)
y_np = 1.5 - 0.8 * x_np + np.random.default_rng(7).normal(0, 0.35, len(x_np))
y_np[[5, 24]] += [3.0, -3.0]
y = torch.as_tensor(y_np)

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(7)
guide = AutoMultivariateNormal(robust_regression)
steps = 3000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(robust_regression, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(x, y=y)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=4000)(x, y=y)
samples = draws["m"].reshape(-1)
print(samples.mean().item(), samples.std().item())
print(torch.quantile(samples, torch.tensor([0.025, 0.975])))
