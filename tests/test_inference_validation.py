"""Generation and inference must apply identical symbolic parameter rules."""

from statistics import NormalDist

import numpy as np
import pytest

from smallprob import Categorical, InferenceError, ModelError, Normal, draw, model


@model
def transition_chain(transition):
    z = [draw(Categorical([0.5, 0.5]))]
    z.append(draw(Categorical(transition[z[0]])))
    return z


@model
def state_emission(means, scales):
    state = draw(Categorical([0.5, 0.5]))
    y = draw(Normal(mean=means[state], sd=scales[state]))
    return y


@model
def precise_measurement(noise, coefficient=1.0):
    x = draw(Normal())
    y = draw(Normal(coefficient * x, noise))
    return y


@pytest.mark.parametrize(
    "transition",
    [
        [[True, False], [False, True]],
        np.array([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
        # This is outside the shared 1e-8 absolute tolerance but inside the
        # looser relative tolerance that the original inference path used.
        [[0.50000005, 0.5], [0.5, 0.50000005]],
    ],
    ids=["boolean-rows", "complex-rows", "unnormalised-rows"],
)
def test_invalid_symbolic_transitions_rejected_by_simulation_and_inference(transition):
    bound = transition_chain(transition)
    with pytest.raises(ModelError):
        bound.simulate(seed=0)
    with pytest.raises(InferenceError, match="Transition probabilities"):
        bound.infer()


@pytest.mark.parametrize(
    "means,scales",
    [
        ([True, False], [1.0, 1.0]),
        (np.array([1.0 + 0j, 2.0 + 0j]), [1.0, 1.0]),
        ([1.0, 2.0], [True, True]),
        ([1.0, 2.0], np.array([1.0 + 0j, 2.0 + 0j])),
    ],
    ids=["boolean-means", "complex-means", "boolean-scales", "complex-scales"],
)
def test_invalid_symbolic_emissions_rejected_by_simulation_and_inference(means, scales):
    bound = state_emission(means, scales)
    with pytest.raises(ModelError):
        bound.simulate(seed=0)
    with pytest.raises(InferenceError, match="Normal parameters"):
        bound.infer()


def test_normalised_symbolic_transition_rows_are_supported():
    bound = transition_chain([[0.4, 0.6], [0.7, 0.3]])
    assert bound.simulate(draws=2, seed=0)["z"].shape == (2, 2)
    np.testing.assert_allclose(bound.infer().probabilities("z"), [[0.5, 0.5], [0.55, 0.45]])


@pytest.mark.parametrize("noise", [1e-4, 1e-8, 1e-12])
def test_tiny_gaussian_posterior_uncertainty_matches_samples_and_intervals(noise):
    posterior = precise_measurement(noise).infer(observed={"y": 0.0})
    expected_sd = noise / np.sqrt(1.0 + noise ** 2)
    assert posterior.sd("x") == pytest.approx(expected_sd, rel=1e-10, abs=0)
    q = NormalDist().inv_cdf(0.975)
    np.testing.assert_allclose(
        posterior.interval("x"), [-q * expected_sd, q * expected_sd], rtol=1e-10, atol=0,
    )
    samples = posterior.sample(draws=30000, seed=17)
    assert samples["x"].std() == pytest.approx(expected_sd, rel=0.02, abs=0)
    np.testing.assert_array_equal(samples["y"], np.zeros(30000))


@pytest.mark.parametrize("coefficient", [100.1, -100.1, 0.1, 1e-10, 1e8])
def test_scaled_tight_noise_gaussian_matches_analytic_uncertainty(coefficient):
    noise = 1e-16
    posterior = precise_measurement(noise, coefficient).infer(observed={"y": 0.0})
    predictive_sd = np.hypot(coefficient, noise)
    expected_sd = noise / predictive_sd
    assert posterior.sd("x") == pytest.approx(expected_sd, rel=1e-10, abs=0)
    q = NormalDist().inv_cdf(0.975)
    np.testing.assert_allclose(
        posterior.interval("x"), [-q * expected_sd, q * expected_sd], rtol=1e-10, atol=0,
    )
    assert posterior.log_evidence == pytest.approx(-np.log(predictive_sd) - 0.5 * np.log(2 * np.pi))
    samples = posterior.sample(draws=30000, seed=17)
    assert samples["x"].std() == pytest.approx(expected_sd, rel=0.02, abs=0)
