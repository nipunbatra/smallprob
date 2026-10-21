"""Independent ELBO identities and end-to-end checks for the focused VAE API."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smallprob import ModelError
from smallprob.neural import VAE


def binary_data(n=64):
    rng = np.random.default_rng(14)
    patterns = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    return patterns[rng.integers(0, 2, n)]


def constant_networks(vae, location, scale, logits):
    """Set a constant q and p without using the implementation's ELBO code."""
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.zero_()
        vae.encoder[-1].bias[:vae.latent_dim].copy_(torch.tensor(location, dtype=vae.dtype))
        raw_scale = torch.log(torch.expm1(torch.tensor(scale, dtype=vae.dtype) - 1e-4))
        vae.encoder[-1].bias[vae.latent_dim:].copy_(raw_scale)
        vae.decoder[-1].bias.copy_(torch.tensor(logits, dtype=vae.dtype))


def test_neural_import_does_not_load_torch():
    code = "import sys; import smallprob.neural; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": "src"})


def test_elbo_matches_bernoulli_and_diagonal_normal_kl_in_nats():
    vae = VAE(3, latent_dim=2, dtype="float64")
    location, scale, logits = [.4, -1.2], [.7, 1.4], [-2., .5, 1.3]
    constant_networks(vae, location, scale, logits)
    data = np.array([[0, 1, 0], [1, 0, 1]], dtype=float)
    probability = 1 / (1 + np.exp(-np.array(logits)))
    reconstruction = -(data * np.log(probability) + (1 - data) * np.log1p(-probability)).sum(1)
    kl = .5 * sum(mu**2 + sd**2 - 1 - math.log(sd**2) for mu, sd in zip(location, scale))
    expected = reconstruction + kl
    np.testing.assert_allclose(vae.loss(data, particles=7, reduction="none").detach(), expected, atol=1e-12)
    assert float(vae.loss(data, reduction="mean").detach()) == pytest.approx(expected.mean())
    assert float(vae.loss(data, reduction="sum").detach()) == pytest.approx(expected.sum())
    metrics = vae.evaluate(data, particles=3, batch_size=1)
    assert metrics["reconstruction_nll"] == pytest.approx(reconstruction.mean())
    assert metrics["kl"] == pytest.approx(kl)
    assert metrics["negative_elbo"] == pytest.approx(expected.mean())


def test_final_partial_batch_is_weighted_by_observations():
    vae = VAE(4, latent_dim=1, dtype="float64")
    constant_networks(vae, [0.], [1.], [-2., -1., .5, 1.])
    data = np.vstack([np.zeros((4, 4)), np.ones((1, 4))])
    assert vae.evaluate(data, batch_size=2) == vae.evaluate(data, batch_size=5)


def test_gradients_reach_encoder_and_decoder():
    vae = VAE(4, hidden_dim=8)
    vae.loss(binary_data(), particles=3, seed=9).backward()
    for module in (vae.encoder, vae.decoder):
        assert all(parameter.grad is not None for parameter in module.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        assert sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()) > 0


def test_reproducibility_rng_modes_and_threads_are_preserved():
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    vae = VAE(4, hidden_dim=8, seed=9)
    replica = VAE(4, hidden_dim=8, seed=9)
    vae.encoder.eval()
    vae.decoder.eval()
    data = binary_data()
    history = vae.fit(data, validation_data=data[:13], epochs=3, batch_size=17, seed=11)
    repeated = replica.fit(data, validation_data=data[:13], epochs=3, batch_size=17, seed=11)
    assert history == repeated
    assert torch.equal(vae.generate(5, seed=17), replica.generate(5, seed=17))
    assert torch.equal(vae.reconstruct(data[:3], seed=2), replica.reconstruct(data[:3], seed=2))
    assert all(torch.equal(a, b) for a, b in zip(vae.parameters(), replica.parameters()))
    assert torch.equal(state, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert not vae.encoder.training and not vae.decoder.training
    assert [row["epoch"] for row in history] == [0, 1, 2, 3]


def test_training_improves_heldout_elbo_and_updates_both_networks():
    vae = VAE(4, latent_dim=2, hidden_dim=16, seed=3)
    before = [parameter.detach().clone() for parameter in vae.parameters()]
    # Independent draws of the same synthetic distribution, not training rows.
    train = binary_data(192)
    test = np.random.default_rng(8).choice(2, size=48)
    test = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]], dtype=np.float32)[test]
    history = vae.fit(train, validation_data=test, epochs=35, batch_size=32,
                      learning_rate=.01, seed=6)
    assert history[-1]["validation_negative_elbo"] < history[0]["validation_negative_elbo"] - .6
    assert all(not torch.equal(old, new) for old, new in zip(before, vae.parameters()))
    assert len(history) == 36


def test_outputs_are_probabilities_or_binary_samples_with_explicit_shapes():
    vae = VAE(4, latent_dim=3, hidden_dim=8, dtype=torch.float64)
    data = binary_data(5)
    location, scale = vae.encode(data)
    assert location.shape == scale.shape == (5, 3)
    assert (scale > 0).all()
    for values in (location, scale, vae.reconstruct(data), vae.generate(5), vae.generate(5, probabilities=True)):
        assert values.dtype == torch.float64 and values.device.type == "cpu"
        assert not values.requires_grad
    assert vae.generate(5).shape == (5, 4)
    assert ((vae.generate(5) == 0) | (vae.generate(5) == 1)).all()
    probabilities = vae.generate(5, probabilities=True)
    assert ((probabilities > 0) & (probabilities < 1)).all()


@pytest.mark.parametrize("bad", [np.zeros((0, 4)), np.ones(4), np.ones((3, 5)),
                                  [[1, 0], [1]], [[0, 1, np.nan, 0]],
                                  [[0, 1, .5, 0]], [[0, 1, 2, 0]],
                                  np.ones((1, 4), dtype=complex) * 1j])
def test_observation_validation(bad):
    vae = VAE(4)
    with pytest.raises(ModelError):
        vae.fit(bad, epochs=1)


@pytest.mark.parametrize("options", [{"input_dim": 0}, {"latent_dim": True}, {"hidden_dim": 2.5},
                                     {"dtype": "float16"}, {"device": "meta"}, {"seed": -1}])
def test_constructor_validation(options):
    with pytest.raises(ModelError):
        VAE(**({"input_dim": 4} | options))


@pytest.mark.parametrize("options", [{"epochs": 0}, {"batch_size": 0}, {"particles": False},
                                     {"validation_particles": 0}, {"learning_rate": 0},
                                     {"learning_rate": float("nan")}, {"threads": 0}, {"seed": True}])
def test_fit_option_validation(options):
    with pytest.raises(ModelError):
        VAE(4).fit(binary_data(), **options)


def test_unavailable_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelError, match="CUDA.*not available"):
        VAE(4, device="cuda")


def test_nonfinite_training_restores_caller_state():
    vae = VAE(4)
    with torch.no_grad():
        vae.decoder[-1].bias.fill_(float("inf"))
    state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    with pytest.raises(ModelError, match="nonfinite"):
        vae.fit(binary_data(), epochs=1)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == threads


def test_elbo_and_gradients_agree_with_independent_pyro_mean_field_loss():
    pyro = pytest.importorskip("pyro")
    from pyro.infer import TraceMeanField_ELBO
    from smallprob.arrays import ArrayBackend
    from examples.vae import pyro_components, synthetic_bars

    vae = VAE(64, latent_dim=2, hidden_dim=128, seed=7)
    direct = pyro_components(seed=7)
    reference_parameters = list(direct["encoder"].parameters()) + list(direct["decoder"].parameters())
    assert all(torch.equal(a, b) for a, b in zip(vae.parameters(), reference_parameters))
    data = torch.tensor(synthetic_bars(8, 29))
    native_loss = vae.loss(data, particles=1, seed=41)
    with pyro.get_param_store().scope(), ArrayBackend("torch", "cpu", "float32").random(41):
        reference_loss = TraceMeanField_ELBO().differentiable_loss(direct["model"], direct["guide"], data) / len(data)
        reference_loss.backward()
    native_loss.backward()
    assert float(native_loss.detach()) == pytest.approx(float(reference_loss.detach()), abs=1e-5)
    for native, reference in zip(vae.parameters(), reference_parameters):
        torch.testing.assert_close(native.grad, reference.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_execution_preserves_rng_and_rejects_implicit_tensor_moves():
    cpu_state, mps_state = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    vae = VAE(4, hidden_dim=8, device="mps")
    history = vae.fit(binary_data(16), epochs=2, batch_size=8)
    assert math.isfinite(history[-1]["train_negative_elbo"])
    assert vae.generate(3).device.type == "mps"
    assert vae.reconstruct(binary_data(3)).device.type == "mps"
    with pytest.raises(ModelError, match="move it explicitly"):
        vae.encode(torch.tensor(binary_data(3)))
    vae.encode(torch.tensor(binary_data(3), device="mps"))
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(mps_state, torch.mps.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_uses_the_requested_device():
    vae = VAE(4, hidden_dim=8, device="cuda")
    vae.fit(binary_data(16), epochs=2, batch_size=8)
    assert vae.generate(3).device.type == "cuda"
