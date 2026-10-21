"""Eight schools, shared information: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python eight-schools-pyro.py

Approximate VI · full-covariance Gaussian guide.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- The eight observations and standard errors are the published Eight Schools data, not synthetic data.
- Adaptation: HalfNormal(5) replaces Pyro's HalfCauchy(25) prior on tau, and VI replaces NUTS. Both code panes use the adapted law; its posterior differs from the original example.
- School effects come from joint draws of mu, tau and eta. Against numerical integration, hyperparameter mean errors are at most 0.083 reference SD and SD errors at most 7.4%; intervals remain approximate.
- The saved result uses guide="full" (Pyro-backed VI). Choose guide="diagonal" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations.

Pyro's Eight Schools model: https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/mcmc.py
Published observations: https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/data.py
Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import torch
import pyro
import pyro.distributions as dist

def eight_schools(se, y=None):
    mu = pyro.sample("mu", dist.Normal(0., 10.))
    tau = pyro.sample("tau", dist.HalfNormal(5.))
    with pyro.plate("schools", len(se)):
        eta = pyro.sample("eta", dist.Normal(torch.zeros_like(se), 1.))
        pyro.sample("y", dist.Normal(mu + tau * eta, se), obs=y)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

se = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])
y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
generated = Predictive(eight_schools, num_samples=2)(se)
# Fit the published observations, not the prior-predictive draw.

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed(11)
guide = AutoMultivariateNormal(eight_schools)
steps = 4000
optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.1 ** (1 / steps)})
svi = SVI(eight_schools, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step(se, y=y)

pyro.set_rng_seed(12)
draws = Predictive(guide, num_samples=6000)(se, y=y)
effects = draws["mu"].reshape(-1, 1) + draws["tau"].reshape(-1, 1) * draws["eta"]
print(effects.mean(dim=0))
