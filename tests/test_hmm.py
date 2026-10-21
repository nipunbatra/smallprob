"""Tiny HMM ground truth is exhaustive enumeration, not forward-backward."""

from itertools import product

import numpy as np
import pytest

from smallprob import Categorical, InferenceError, Normal, draw, model


@model
def hmm(T, initial, transition, means, scales):
    z = [draw(Categorical(initial))]
    for t in range(1, T):
        z.append(draw(Categorical(transition[z[t - 1]])))
    y = [draw(Normal(mean=means[z[t]], sd=scales[z[t]])) for t in range(T)]
    return y


INITIAL = [0.6, 0.4]
TRANSITION = [[0.85, 0.15], [0.25, 0.75]]
MEANS = [-1.0, 1.2]
SCALES = [0.8, 0.6]


def enumerate_paths(observed, known_states=None):
    known_states = known_states or [None] * len(observed)
    paths = list(product(range(2), repeat=len(observed)))
    weights = []
    for path in paths:
        weight = INITIAL[path[0]]
        for t in range(1, len(path)):
            weight *= TRANSITION[path[t - 1]][path[t]]
        for t, state in enumerate(path):
            if known_states[t] is not None and state != known_states[t]:
                weight = 0.0
            if observed[t] is not None:
                residual = (observed[t] - MEANS[state]) / SCALES[state]
                weight *= np.exp(-0.5 * residual**2) / (SCALES[state] * np.sqrt(2 * np.pi))
        weights.append(weight)
    evidence = sum(weights)
    probabilities = np.array(weights) / evidence
    marginals = np.array(
        [[sum(p for path, p in zip(paths, probabilities) if path[t] == k) for k in range(2)]
         for t in range(len(observed))]
    )
    return paths, probabilities, marginals, np.log(evidence)


@pytest.mark.parametrize(
    "observed,states",
    [([-0.8, 0.4, 1.0], None), ([-0.8, None, 1.0], None), ([-0.8, None, 1.0], [None, 0, None])],
)
def test_hmm_smoothing_and_evidence_match_enumeration(observed, states):
    m = hmm(3, INITIAL, TRANSITION, MEANS, SCALES)
    observations = {"y": observed}
    if states is not None:
        observations["z"] = states
    posterior = m.infer(observed=observations)
    _, _, marginals, evidence = enumerate_paths(observed, states)
    assert posterior.method == "exact-hmm"
    np.testing.assert_allclose(posterior.probabilities("z"), marginals, atol=1e-12)
    assert posterior.log_evidence == pytest.approx(evidence, abs=1e-12)
    for t, value in enumerate(observed):
        if value is None:
            expected_mean = marginals[t] @ np.array(MEANS)
            expected_variance = marginals[t] @ (np.square(MEANS) + np.square(SCALES)) - expected_mean**2
            assert posterior.mean("y")[t] == pytest.approx(expected_mean)
            assert posterior.sd("y")[t] == pytest.approx(np.sqrt(expected_variance))


def test_posterior_samples_are_joint_state_paths():
    observed = [-0.2, None, 0.5]
    m = hmm(3, INITIAL, TRANSITION, MEANS, SCALES)
    posterior = m.infer(observed={"y": observed})
    paths, probabilities, _, _ = enumerate_paths(observed)
    samples = posterior.sample(draws=12000, seed=81)
    assert samples["z"].shape == (12000, 3)
    frequencies = [np.mean(np.all(samples["z"] == path, axis=1)) for path in paths]
    np.testing.assert_allclose(frequencies, probabilities, atol=0.018)
    # An unobserved emission is drawn conditional on its sampled state.
    for state in range(2):
        emissions = samples["y"][samples["z"][:, 1] == state, 1]
        assert emissions.mean() == pytest.approx(MEANS[state], abs=0.05)
        assert emissions.std() == pytest.approx(SCALES[state], abs=0.05)
    np.testing.assert_array_equal(samples["y"][:, 0], np.full(12000, observed[0]))
    np.testing.assert_array_equal(samples["y"][:, 2], np.full(12000, observed[2]))


def test_one_time_step_and_one_state():
    m = hmm(1, [1.0], [[1.0]], [2.0], [0.4])
    posterior = m.infer(observed={"y": [2.2]})
    np.testing.assert_array_equal(posterior.probabilities("z"), [[1.0]])
    assert posterior.sample(draws=3, seed=4)["z"].shape == (3, 1)
    expected = -0.5 * ((2.2 - 2.0) / 0.4) ** 2 - np.log(0.4 * np.sqrt(2 * np.pi))
    assert posterior.log_evidence == pytest.approx(expected)


def test_zero_transitions_never_appear_in_joint_samples():
    m = hmm(4, [0.5, 0.5], [[0.0, 1.0], [1.0, 0.0]], MEANS, SCALES)
    posterior = m.infer(observed={})
    samples = posterior.sample(draws=1000, seed=8)
    np.testing.assert_array_equal(samples["z"][:, 1:], 1 - samples["z"][:, :-1])
    assert posterior.log_evidence == pytest.approx(0.0)
    np.testing.assert_allclose(posterior.probabilities("z"), 0.5)


def test_impossible_state_evidence_raises():
    m = hmm(2, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], MEANS, SCALES)
    with pytest.raises(InferenceError):
        m.infer(observed={"z": [0, 1]})


def test_hmm_simulation_is_reproducible():
    m = hmm(3, INITIAL, TRANSITION, MEANS, SCALES)
    first = m.simulate(draws=5, seed=12)
    second = m.simulate(draws=5, seed=12)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_mixture_intervals_are_not_mislabelled_as_gaussian_intervals():
    posterior = hmm(1, INITIAL, TRANSITION, MEANS, SCALES).infer(observed={})
    with pytest.raises(InferenceError):
        posterior.interval("z")
    with pytest.raises(InferenceError):
        posterior.interval("y")


def test_very_small_hmm_likelihood_remains_finite_in_log_space():
    m = hmm(80, [1.0], [[1.0]], [0.0], [1.0])
    posterior = m.infer(observed={"y": [40.0] * 80})
    expected = 80 * (-0.5 * 40.0**2 - 0.5 * np.log(2 * np.pi))
    assert np.isfinite(posterior.log_evidence)
    assert posterior.log_evidence == pytest.approx(expected, abs=1e-8)
    np.testing.assert_array_equal(posterior.probabilities("z"), np.ones((80, 1)))
