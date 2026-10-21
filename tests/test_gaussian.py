"""Reference checks against independently derived Gaussian posteriors."""

from statistics import NormalDist

import numpy as np
import pytest

from smallprob import Normal, draw, model


@model
def normal_mean(n, prior_mean=1.5, prior_sd=2.0, noise=0.7):
    mu = draw(Normal(mean=prior_mean, sd=prior_sd))
    y = [draw(Normal(mean=mu, sd=noise)) for _ in range(n)]
    return y


@model
def regression(x, noise=0.5):
    intercept = draw(Normal(mean=1.0, sd=2.0))
    slope = draw(Normal(mean=-0.5, sd=1.5))
    y = [draw(Normal(mean=intercept + slope * xi, sd=noise)) for xi in x]
    return y


def test_normal_posterior_matches_precision_formula():
    observed = np.array([2.0, 2.5, 3.1, 1.8])
    posterior = normal_mean(len(observed)).infer(observed={"y": observed})
    variance = 1 / (1 / 2.0**2 + len(observed) / 0.7**2)
    mean = variance * (1.5 / 2.0**2 + observed.sum() / 0.7**2)
    assert posterior.method == "exact-gaussian"
    assert posterior.mean("mu") == pytest.approx(mean, abs=1e-12)
    assert posterior.sd("mu") == pytest.approx(np.sqrt(variance), abs=1e-12)
    z = NormalDist().inv_cdf(0.975)
    np.testing.assert_allclose(
        posterior.interval("mu", probability=0.95),
        [mean - z * np.sqrt(variance), mean + z * np.sqrt(variance)],
    )
    np.testing.assert_allclose(posterior.mean("y"), observed)
    np.testing.assert_allclose(posterior.sd("y"), 0)


def test_normal_evidence_includes_correlated_prior_predictive():
    y = np.array([0.2, 1.7, 2.1])
    posterior = normal_mean(len(y)).infer(observed={"y": y})
    covariance = 4.0 * np.ones((len(y), len(y))) + 0.49 * np.eye(len(y))
    residual = y - 1.5
    expected = -0.5 * (
        len(y) * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ np.linalg.solve(covariance, residual)
    )
    assert posterior.log_evidence == pytest.approx(expected, abs=1e-11)


def test_regression_matches_bayesian_linear_algebra():
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    y = np.array([-2.6, 0.0, 1.2, 3.9, 6.0])
    design = np.column_stack([np.ones(len(x)), x])
    prior_precision = np.diag([1 / 2.0**2, 1 / 1.5**2])
    prior_mean = np.array([1.0, -0.5])
    covariance = np.linalg.inv(prior_precision + design.T @ design / 0.5**2)
    mean = covariance @ (prior_precision @ prior_mean + design.T @ y / 0.5**2)
    posterior = regression(x).infer(observed={"y": y})
    np.testing.assert_allclose(
        [posterior.mean("intercept"), posterior.mean("slope")], mean, atol=1e-12
    )
    np.testing.assert_allclose(
        [posterior.sd("intercept"), posterior.sd("slope")],
        np.sqrt(np.diag(covariance)),
        atol=1e-12,
    )
    # Joint posterior draws must preserve coefficient covariance.
    samples = posterior.sample(draws=12000, seed=17)
    empirical = np.cov(np.column_stack([samples["intercept"], samples["slope"]]), rowvar=False)
    np.testing.assert_allclose(empirical, covariance, atol=0.0025)
    np.testing.assert_allclose(samples["y"], np.broadcast_to(y, (12000, len(y))))


def test_missing_normal_observations_are_integrated_out():
    partial = normal_mean(3).infer(observed={"y": [2.0, None, 3.0]})
    complete = normal_mean(2).infer(observed={"y": [2.0, 3.0]})
    assert partial.mean("mu") == pytest.approx(complete.mean("mu"))
    assert partial.sd("mu") == pytest.approx(complete.sd("mu"))
    assert partial.log_evidence == pytest.approx(complete.log_evidence)
    assert partial.mean("y")[1] == pytest.approx(partial.mean("mu"))
    assert partial.sd("y")[1] == pytest.approx(np.sqrt(0.7**2 + partial.sd("mu") ** 2))


def test_empty_data_returns_prior_and_zero_log_evidence():
    posterior = normal_mean(0).infer(observed={})
    assert posterior.mean("mu") == pytest.approx(1.5)
    assert posterior.sd("mu") == pytest.approx(2.0)
    assert posterior.log_evidence == pytest.approx(0.0)


def test_simulation_fixed_parameters_and_reproducibility():
    m = regression([0.0, 2.0])
    first = m.simulate(draws=3000, seed=23, fixed={"intercept": 3.0, "slope": 2.0})
    second = m.simulate(draws=3000, seed=23, fixed={"intercept": 3.0, "slope": 2.0})
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["y"].shape == (3000, 2)
    np.testing.assert_array_equal(first["intercept"], np.full(3000, 3.0))
    np.testing.assert_allclose(first["y"].mean(axis=0), [3.0, 7.0], atol=0.035)
    np.testing.assert_allclose(first["y"].std(axis=0), 0.5, atol=0.035)


def test_posterior_sampling_reproducibility_and_singleton_series():
    m = normal_mean(1)
    assert m.simulate(draws=1, seed=0)["y"].shape == (1, 1)
    posterior = m.infer(observed={"y": [2.0]})
    first = posterior.sample(draws=4, seed=0)
    second = posterior.sample(draws=4, seed=0)
    assert first["mu"].shape == (4,)
    assert first["y"].shape == (4, 1)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_reusing_model_does_not_mutate_an_existing_posterior():
    m = normal_mean(2)
    first = m.infer(observed={"y": [1.0, 1.0]})
    mean = first.mean("mu")
    second = m.infer(observed={"y": [8.0, 8.0]})
    assert second.mean("mu") > mean
    assert first.mean("mu") == mean
