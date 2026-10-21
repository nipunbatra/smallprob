"""Train a small Bernoulli VAE on synthetic 8x8 crossing bars, entirely offline.

Run ``python -m examples.vae`` for Smallprob or append ``--pyro`` to also run
the corresponding Pyro model + guide + SVI implementation. The Pyro comparison
uses the same data, architecture and ELBO, but is trained independently.
"""

from __future__ import annotations

import argparse
import json
import textwrap

import numpy as np

from smallprob.neural import VAE


def synthetic_bars(n, seed):
    """Independent synthetic images: crossing bars with 1% random pixel flips."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 8, 8), dtype=np.float32)
    rows, columns = rng.integers(1, 7, size=(2, n))
    images[np.arange(n), rows, :] = 1
    images[np.arange(n), :, columns] = 1
    flips = rng.random(images.shape) < .01
    return np.where(flips, 1 - images, images).reshape(n, 64)


PYRO_MODEL_CODE = textwrap.dedent('''\
import torch
from torch import nn
from torch.nn import functional as F
import pyro
import pyro.distributions as dist

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
''')

PYRO_FIT_CODE = textwrap.dedent('''\
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
''')

PYRO_RECONSTRUCT_CODE = textwrap.dedent('''\
with torch.no_grad():
    location, raw_scale = encoder(x_test[:12]).chunk(2, dim=-1)
    scale = F.softplus(raw_scale) + 1e-4
    z = dist.Normal(location, scale).sample((128,))
    reconstructed = decoder(z).sigmoid().mean(0)
''')

PYRO_GENERATE_CODE = textwrap.dedent('''\
with torch.no_grad():
    z = torch.randn(12, 2)
    pixel_probabilities = decoder(z).sigmoid()
    generated = dist.Bernoulli(pixel_probabilities).sample()
''')


def pyro_components(seed=7):
    """Build the exact displayed Pyro code while preserving caller Torch RNG."""
    from smallprob.arrays import ArrayBackend
    namespace = {}
    with ArrayBackend("torch", "cpu", "float32").random(seed):
        exec(PYRO_MODEL_CODE, namespace)
    return namespace


def train_smallprob(train, test, *, epochs=200):
    vae = VAE(input_dim=64, latent_dim=2, hidden_dim=128, seed=7)
    before = vae.evaluate(test, particles=128, seed=21)
    history = vae.fit(train, validation_data=test, epochs=epochs, batch_size=128,
                      learning_rate=.003, validation_particles=8, seed=11)
    after = vae.evaluate(test, particles=128, seed=21)
    return vae, history, {"before": before, "after": after}


def train_pyro(train, test, *, epochs=200):
    """Fit independently using Pyro SVI; no Smallprob neural helper is involved."""
    import torch
    import pyro
    from pyro.infer import SVI, TraceMeanField_ELBO
    from pyro.optim import Adam
    from smallprob.arrays import ArrayBackend, TORCH_RNG_LOCK

    backend = ArrayBackend("torch", "cpu", "float32")
    namespace = pyro_components()
    model, guide = namespace["model"], namespace["guide"]
    x_train = torch.as_tensor(train, dtype=torch.float32)
    x_test = torch.as_tensor(test, dtype=torch.float32)
    history = []
    with TORCH_RNG_LOCK, backend.random(11), pyro.get_param_store().scope():
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            svi = SVI(model, guide, Adam({"lr": .003}), TraceMeanField_ELBO())
            evaluation = TraceMeanField_ELBO(num_particles=128)
            with backend.random(21), torch.no_grad():
                before = evaluation.loss(model, guide, x_test) / len(x_test)
            for epoch in range(1, epochs + 1):
                for indices in torch.randperm(len(x_train)).split(128):
                    svi.step(x_train[indices])
                if epoch % 10 == 0 or epoch == epochs:
                    with backend.random(21), torch.no_grad():
                        value = TraceMeanField_ELBO(num_particles=8).loss(model, guide, x_test) / len(x_test)
                    history.append({"epoch": epoch, "validation_negative_elbo": value})
            with backend.random(21), torch.no_grad():
                after = evaluation.loss(model, guide, x_test) / len(x_test)
        finally:
            torch.set_num_threads(previous_threads)
    return namespace, history, {"before": before, "after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyro", action="store_true", help="also fit the independent Pyro implementation")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    train, test = synthetic_bars(1024, 23), synthetic_bars(256, 24)
    vae, history, metrics = train_smallprob(train, test, epochs=args.epochs)
    print(json.dumps({"synthetic": True, "train_images": len(train), "test_images": len(test),
                      "epochs": args.epochs, "smallprob": metrics,
                      "generated_shape": list(vae.generate(12, seed=31).shape),
                      "reconstruction_shape": list(vae.reconstruct(test[:12], seed=29).shape)}, indent=2))
    if args.pyro:
        _, _, metrics = train_pyro(train, test, epochs=args.epochs)
        print(json.dumps({"pyro": metrics}, indent=2))


if __name__ == "__main__":
    main()
