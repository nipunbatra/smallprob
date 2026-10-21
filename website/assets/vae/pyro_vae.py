# Synthetic 8x8 crossing bars. Offline, CPU, independently trained VAE.
import torch
torch.set_num_threads(1)

import numpy as np

def synthetic_bars(n, seed):
    """Independent synthetic images: crossing bars with 1% random pixel flips."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 8, 8), dtype=np.float32)
    rows, columns = rng.integers(1, 7, size=(2, n))
    images[np.arange(n), rows, :] = 1
    images[np.arange(n), :, columns] = 1
    flips = rng.random(images.shape) < .01
    return np.where(flips, 1 - images, images).reshape(n, 64)

train = synthetic_bars(1024, seed=23)
test = synthetic_bars(256, seed=24)


import torch
from torch import nn
from torch.nn import functional as F
import pyro
import pyro.distributions as dist

torch.manual_seed(7)

encoder = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 4))
decoder = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 64))

def model(x):
    pyro.module("decoder", decoder)
    with pyro.plate("images", len(x)):
        z = pyro.sample("z", dist.Normal(x.new_zeros(2), x.new_ones(2)).to_event(1))
        pyro.sample("x", dist.Bernoulli(logits=decoder(z)).to_event(1), obs=x)

def guide(x):
    pyro.module("encoder", encoder)
    location, raw_scale = encoder(x).chunk(2, dim=-1)
    scale = F.softplus(raw_scale) + 1e-4
    with pyro.plate("images", len(x)):
        pyro.sample("z", dist.Normal(location, scale).to_event(1))


from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam

x_train = torch.as_tensor(train, dtype=torch.float32)
x_test = torch.as_tensor(test, dtype=torch.float32)
pyro.clear_param_store()
torch.manual_seed(11)
svi = SVI(model, guide, Adam({"lr": 0.003}), TraceMeanField_ELBO())
for epoch in range(200):
    for indices in torch.randperm(len(x_train)).split(128):
        svi.step(x_train[indices])

# The loss sums over images; divide to get nats per image.
torch.manual_seed(21)
evaluation = TraceMeanField_ELBO(num_particles=128)
test_negative_elbo = evaluation.loss(model, guide, x_test) / len(x_test)


torch.manual_seed(29)
with torch.no_grad():
    location, raw_scale = encoder(x_test[:12]).chunk(2, dim=-1)
    scale = F.softplus(raw_scale) + 1e-4
    z = dist.Normal(location, scale).sample((128,))
    reconstructed = decoder(z).sigmoid().mean(0)


torch.manual_seed(31)
with torch.no_grad():
    z = torch.randn(12, 2)
    pixel_probabilities = decoder(z).sigmoid()
    generated = dist.Bernoulli(pixel_probabilities).sample()

print({"test_negative_elbo": test_negative_elbo})
print("generated", tuple(generated.shape))
