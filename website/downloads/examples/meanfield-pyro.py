"""Unknown-noise regression with a diagonal Normal guide. Synthetic data."""
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

# Standalone script settings: use double precision for this small example.
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
pyro.set_rng_seed(7)
pyro.clear_param_store()

def regression(x, y=None):
    b = pyro.sample("b", dist.Normal(0., 5.))
    m = pyro.sample("m", dist.Normal(0., 5.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    with pyro.plate("observations", len(x)):
        return pyro.sample("y", dist.Normal(b + m * x, sigma), obs=y)

x = torch.linspace(-1.5, 1.5, 32)
fixed = {"b": torch.tensor(1.2), "m": torch.tensor(-0.7), "sigma": torch.tensor(0.6)}
y = pyro.poutine.condition(regression, data=fixed)(x)
guide = AutoDiagonalNormal(regression)
steps = 4000
optimizer = ClippedAdam({"lr": .03, "lrd": .1**(1/steps)})
svi = SVI(regression, guide, optimizer, loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    svi.step(x, y)
draws = Predictive(guide, num_samples=2000)(x, y)
print(draws["m"].mean(), draws["m"].quantile(torch.tensor([.025, .975])))
print("Noise scale:", draws["sigma"].mean())
