"""A focused Bernoulli VAE with amortized inference, using optional PyTorch.

This is a separate neural API, not an extension of the finite ``@model`` graph.
Importing this module does not import PyTorch. The built-in networks have one
SiLU hidden layer; the encoder supplies a diagonal Normal distribution and the
decoder supplies Bernoulli logits. See ``docs/neural.md`` for the exact ELBO.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import chain
import math
import numbers

from .arrays import ArrayBackend, TORCH_RNG_LOCK
from .core import InferenceError, ModelError


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError("VAE needs PyTorch: pip install 'smallprob[torch]'.") from exc
    return torch


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ModelError(f"{name} must be a positive integer.")
    return int(value)


def _seed(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Integral)
                              or not 0 <= value < 2**63):
        raise ModelError("seed must be None or an integer in [0, 2**63).")
    return None if value is None else int(value)


class VAE:
    """A small fully connected VAE for binary observations of shape ``(N, D)``.

    ``VAE(input_dim=64, latent_dim=2, hidden_dim=64)`` fixes the architecture,
    standard Normal prior, Bernoulli likelihood, and diagonal Normal guide.
    Initialization and stochastic methods isolate the selected Torch RNG.
    CPU is the default; ``device='cuda'`` or ``'mps'`` selects an available GPU
    explicitly. Results are detached Torch tensors on that device, except for
    :meth:`loss`, which retains gradients for an optional custom training loop.
    """

    def __init__(self, input_dim, latent_dim=2, hidden_dim=64, *, device="cpu",
                 dtype="float32", seed=0):
        self.input_dim = _integer(input_dim, "input_dim")
        self.latent_dim = _integer(latent_dim, "latent_dim")
        self.hidden_dim = _integer(hidden_dim, "hidden_dim")
        seed = _seed(seed)
        torch = _torch()
        dtype_name = str(dtype).removeprefix("torch.")
        self._backend = ArrayBackend("torch", str(device), dtype_name)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = getattr(torch, dtype_name)
        nn = torch.nn
        options = {"device": self.device, "dtype": self.dtype}
        with self._backend.random(seed):
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim, **options), nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.latent_dim, **options),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim, **options), nn.SiLU(),
                nn.Linear(self.hidden_dim, self.input_dim, **options),
            )
        # Torch normalizes an unindexed accelerator to its concrete device.
        self.device = next(self.parameters()).device

    def parameters(self):
        """Iterate over both networks' trainable Torch parameters."""
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def _observations(self, x):
        torch = _torch()
        if isinstance(x, torch.Tensor) and x.device != self.device:
            raise ModelError(f"Input tensor is on {x.device}; move it explicitly to {self.device}.")
        try:
            values = torch.as_tensor(x, device=self.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ModelError("x must be a rectangular numeric array of shape (N, input_dim).") from exc
        if values.ndim != 2 or values.shape[1] != self.input_dim or values.shape[0] == 0:
            raise ModelError(f"x must have shape (N, {self.input_dim}) with N > 0.")
        if values.is_complex() or not bool(torch.isfinite(values).all()):
            raise ModelError("x must contain finite binary values (0 or 1).")
        if not bool(((values == 0) | (values == 1)).all()):
            raise ModelError("Bernoulli observations must be binary (0 or 1); binarize explicitly.")
        return values.detach().to(dtype=self.dtype)

    def _encode(self, x):
        torch = _torch()
        location, raw_scale = self.encoder(x).chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        return location, scale

    def _terms(self, x, particles):
        torch = _torch()
        location, scale = self._encode(x)
        noise = torch.randn((particles, len(x), self.latent_dim), device=self.device, dtype=self.dtype)
        latent = location.unsqueeze(0) + scale.unsqueeze(0) * noise
        logits = self.decoder(latent)
        reconstruction = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, x.unsqueeze(0).expand_as(logits), reduction="none",
        ).sum(-1).mean(0)
        # KL[Normal(location, diag(scale**2)) || Normal(0, I)], per observation.
        kl = 0.5 * (location.square() + scale.square() - 1 - 2 * scale.log()).sum(-1)
        return reconstruction, kl

    @contextmanager
    def _evaluation(self):
        torch = _torch()
        modes = self.encoder.training, self.decoder.training
        self.encoder.eval()
        self.decoder.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.encoder.train(modes[0])
            self.decoder.train(modes[1])

    def encode(self, x):
        """Return detached ``(location, scale)`` of ``q(z | x)``; shape ``(N, Z)``."""
        x = self._observations(x)
        with self._evaluation():
            return self._encode(x)

    def loss(self, x, *, particles=1, seed=0, reduction="mean"):
        """Differentiable negative ELBO, with exact KL and sampled reconstruction.

        Sum over the feature/event dimension; ``reduction`` controls only the
        observation dimension and is ``'mean'``, ``'sum'``, or ``'none'``.
        The finite Monte Carlo value is not a guaranteed numerical bound.
        """
        particles, seed = _integer(particles, "particles"), _seed(seed)
        if reduction not in {"mean", "sum", "none"}:
            raise ModelError("reduction must be 'mean', 'sum', or 'none'.")
        x = self._observations(x)
        with self._backend.random(seed):
            reconstruction, kl = self._terms(x, particles)
            result = reconstruction + kl
        return result if reduction == "none" else getattr(result, reduction)()

    def evaluate(self, x, *, particles=32, batch_size=256, seed=0):
        """Report negative ELBO, reconstruction NLL and KL in nats per image.

        Batch totals are weighted by their actual size, including the final
        partial batch. Different batch sizes can change Monte Carlo draws.
        """
        particles = _integer(particles, "particles")
        batch_size, seed = _integer(batch_size, "batch_size"), _seed(seed)
        x = self._observations(x)
        reconstruction_sum, kl_sum = 0.0, 0.0
        with self._backend.random(seed), self._evaluation():
            for batch in x.split(batch_size):
                reconstruction, kl = self._terms(batch, particles)
                reconstruction_sum += float(reconstruction.sum())
                kl_sum += float(kl.sum())
        result = {"negative_elbo": (reconstruction_sum + kl_sum) / len(x),
                  "reconstruction_nll": reconstruction_sum / len(x),
                  "kl": kl_sum / len(x)}
        if not all(math.isfinite(value) for value in result.values()):
            raise InferenceError("VAE evaluation became nonfinite; reduce the learning rate or check the parameters.")
        return result

    def fit(self, x, *, validation_data=None, epochs=50, batch_size=128,
            learning_rate=0.003, particles=1, validation_particles=16, seed=0,
            threads=1):
        """Optimize the mean negative ELBO with Adam and shuffled minibatches.

        Return records for epoch zero and every completed epoch. Each record's
        ``train_negative_elbo`` is an evaluation of the current networks, not
        an average of changing minibatch objectives. Validation never updates
        parameters. Each call continues the networks with a fresh optimizer;
        it does not reinitialize them. ``threads`` temporarily limits CPU
        threads and is restored even if training fails.
        """
        torch = _torch()
        epochs, batch_size = _integer(epochs, "epochs"), _integer(batch_size, "batch_size")
        particles = _integer(particles, "particles")
        validation_particles = _integer(validation_particles, "validation_particles")
        threads, seed = _integer(threads, "threads"), _seed(seed)
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, numbers.Real)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ModelError("learning_rate must be finite and positive.")
        x = self._observations(x)
        validation = None if validation_data is None else self._observations(validation_data)
        history = []

        def record(epoch):
            values = self.evaluate(x, particles=validation_particles, batch_size=batch_size, seed=seed)
            row = {"epoch": epoch, "train_negative_elbo": values["negative_elbo"]}
            if validation is not None:
                values = self.evaluate(validation, particles=validation_particles, batch_size=batch_size, seed=seed)
                row["validation_negative_elbo"] = values["negative_elbo"]
            history.append(row)

        with TORCH_RNG_LOCK, self._backend.random(seed):
            previous_threads = torch.get_num_threads()
            modes = self.encoder.training, self.decoder.training
            try:
                torch.set_num_threads(threads)
                self.encoder.train()
                self.decoder.train()
                optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
                record(0)
                for epoch in range(1, epochs + 1):
                    order = torch.randperm(len(x), device=self.device)
                    for indices in order.split(batch_size):
                        optimizer.zero_grad(set_to_none=True)
                        reconstruction, kl = self._terms(x[indices], particles)
                        objective = (reconstruction + kl).mean()
                        if not bool(torch.isfinite(objective)):
                            raise InferenceError("VAE loss became nonfinite; reduce learning_rate.")
                        objective.backward()
                        try:
                            torch.nn.utils.clip_grad_norm_(list(self.parameters()), 10.0, error_if_nonfinite=True)
                        except RuntimeError as exc:
                            raise InferenceError("VAE gradient became nonfinite; reduce learning_rate.") from exc
                        optimizer.step()
                    record(epoch)
            finally:
                torch.set_num_threads(previous_threads)
                self.encoder.train(modes[0])
                self.decoder.train(modes[1])
        return history

    def reconstruct(self, x, *, particles=32, seed=0):
        """Return pixel probabilities averaged over ``q(z | x)`` draws.

        This is a Monte Carlo reconstruction under the learned approximate
        posterior, not a binary image or a draw from the prior.
        """
        torch = _torch()
        particles, seed = _integer(particles, "particles"), _seed(seed)
        x = self._observations(x)
        with self._backend.random(seed), self._evaluation():
            location, scale = self._encode(x)
            noise = torch.randn((particles, len(x), self.latent_dim), device=self.device, dtype=self.dtype)
            return self.decoder(location + scale * noise).sigmoid().mean(0)

    def generate(self, n=16, *, probabilities=False, seed=0):
        """Draw ``z ~ Normal(0, I)`` then binary observations from the decoder.

        ``probabilities=True`` returns Bernoulli means at those prior latent
        draws, rather than sampling the final pixels. Output shape is ``(n,D)``.
        """
        torch = _torch()
        n, seed = _integer(n, "n"), _seed(seed)
        if not isinstance(probabilities, bool):
            raise ModelError("probabilities must be True or False.")
        with self._backend.random(seed), self._evaluation():
            latent = torch.randn((n, self.latent_dim), device=self.device, dtype=self.dtype)
            values = self.decoder(latent).sigmoid()
            return values if probabilities else torch.bernoulli(values)
