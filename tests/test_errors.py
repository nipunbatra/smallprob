"""Reject ambiguous data and unsupported inference instead of guessing."""

import numpy as np
import pytest

from smallprob import Categorical, ModelError, Normal, UnsupportedModelError, draw, model


@model
def normal_data(n=2, noise=1.0):
    mu = draw(Normal(0.0, 1.0))
    y = [draw(Normal(mu, noise)) for _ in range(n)]
    return y


@model
def categorical_data(probs):
    z = draw(Categorical(probs))
    return z


@model
def nonlinear():
    x = draw(Normal(0.0, 1.0))
    y = draw(Normal(x * x, 1.0))
    return y


@model
def random_noise():
    scale = draw(Normal(0.0, 1.0))
    y = draw(Normal(0.0, scale * scale + 0.2))
    return y


@model
def random_branch():
    x = draw(Normal(0.0, 1.0))
    if x:
        y = draw(Normal(1.0, 1.0))
    else:
        y = draw(Normal(-1.0, 1.0))
    return y


@pytest.mark.parametrize("sd", [0.0, -1.0, np.nan, np.inf])
def test_invalid_normal_scale_rejected(sd):
    with pytest.raises(ModelError):
        normal_data(noise=sd).simulate(seed=1)


@pytest.mark.parametrize("probs", [[], [0.2, 0.2], [-0.1, 1.1], [np.nan, 1.0], [[0.5, 0.5]]])
def test_bad_categorical_probabilities_are_not_silently_normalised(probs):
    with pytest.raises(ModelError):
        categorical_data(probs).simulate(seed=1)


@pytest.mark.parametrize("value", [-1, 2, 0.5, np.nan])
def test_invalid_category_observation_rejected(value):
    with pytest.raises(ModelError):
        categorical_data([0.4, 0.6]).infer(observed={"z": value})


@pytest.mark.parametrize("observed", [
    {"unknown": 0.0},
    {"y": [1.0]},
    {"y": [[1.0, 2.0]]},
    {"y": [1.0, 2.0], "y[0]": 1.0},
    {"y": [1.0, np.nan]},
    {"mu": [1.0]},
])
def test_invalid_observation_mapping_rejected(observed):
    with pytest.raises(ModelError):
        normal_data().infer(observed=observed)


def test_fixed_values_do_not_accept_missing_entries():
    with pytest.raises(ModelError):
        normal_data().simulate(seed=1, fixed={"y": [1.0, None]})


def test_nonlinear_model_can_simulate_but_exact_inference_is_rejected():
    m = nonlinear()
    assert m.simulate(draws=5, seed=1)["y"].shape == (5,)
    with pytest.raises(UnsupportedModelError):
        m.infer(observed={"y": 1.0})


def test_exact_inference_rejects_vi_optimisation_options():
    with pytest.raises(ModelError, match="vi"):
        normal_data().infer(observed={"y": [1.0, 2.0]}, steps=10)


def test_learning_noise_requires_explicit_vi_instead_of_exact():
    m = random_noise()
    assert m.simulate(draws=5, seed=1)["y"].shape == (5,)
    with pytest.raises(UnsupportedModelError):
        m.infer(observed={"y": 1.0})


def test_random_dependent_control_flow_is_rejected():
    with pytest.raises((ModelError, TypeError)):
        random_branch()


@pytest.mark.parametrize("draws", [0, -1, 1.5, True])
def test_draw_count_must_be_positive_integer(draws):
    m = normal_data()
    with pytest.raises(ModelError):
        m.simulate(draws=draws)
    with pytest.raises(ModelError):
        m.infer(observed={}).sample(draws=draws)
