"""Independent references are cheap; the nine-fit VI accuracy suite is opt-in."""

import os
from statistics import NormalDist

import numpy as np
import pytest

from examples.vi_validation import (
    check_quadrature_convergence,
    gaussian_regression_reference,
    quadrature_regression_reference,
    run_validation,
)


def test_conjugate_reference_matches_direct_marginal_gaussian():
    x = np.linspace(0.5, 2.5, 9)
    design = np.column_stack([np.ones_like(x), x])
    y = np.array([1.1, -0.2, 0.3, 0.1, 0.8, -0.3, 0.1, -0.4, -1.0])
    sigma, prior_sd = 0.6, 5.0
    reference = gaussian_regression_reference(design, y, sigma, prior_sd)
    marginal_covariance = prior_sd**2 * design @ design.T + sigma**2 * np.eye(len(y))
    expected_mean = prior_sd**2 * design.T @ np.linalg.solve(marginal_covariance, y)
    expected_covariance = prior_sd**2 * np.eye(2) - prior_sd**4 * design.T @ np.linalg.solve(marginal_covariance, design)
    expected_evidence = -0.5 * (len(y) * np.log(2 * np.pi)
                                + np.linalg.slogdet(marginal_covariance)[1]
                                + y @ np.linalg.solve(marginal_covariance, y))
    np.testing.assert_allclose(reference["mean"], expected_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(reference["covariance"], expected_covariance, rtol=1e-10, atol=1e-11)
    assert reference["log_evidence"] == pytest.approx(expected_evidence, abs=1e-11)


def test_quadrature_without_observations_recovers_normal_halfnormal_priors():
    # Includes evidence normalisation and the log-sigma Jacobian; omitting
    # either would fail even when posterior weights were renormalised.
    reference = quadrature_regression_reference(np.empty((0, 2)), np.empty(0),
                                                bounds=(-16, 4), points=16385)
    expected_mean = [0, 0, np.sqrt(2 / np.pi)]
    expected_variance = [25, 25, 1 - 2 / np.pi]
    np.testing.assert_allclose(reference["mean"], expected_mean, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(reference["covariance"], np.diag(expected_variance), atol=1e-7, rtol=1e-7)
    expected_sigma_interval = [NormalDist().inv_cdf((1 + p) / 2) for p in (0.025, 0.975)]
    np.testing.assert_allclose(reference["interval95"][-1], expected_sigma_interval, atol=2e-6, rtol=1e-6)
    assert abs(reference["log_evidence"]) < 1e-7


def test_unknown_noise_numerical_reference_converges():
    x = np.linspace(-1.5, 1.5, 32)
    y = 1.2 - 0.7 * x + np.random.default_rng(103).normal(0, 0.6, len(x))
    reference = check_quadrature_convergence(np.column_stack([np.ones_like(x), x]), y)
    assert reference["convergence"]["passed"], reference["convergence"]
    assert reference["kind"].startswith("numerical")
    assert 0 < reference["interval95"][-1, 0] < reference["mean"][-1] < reference["interval95"][-1, 1]
    assert np.linalg.eigvalsh(reference["covariance"]).min() > 0


@pytest.mark.skipif(os.environ.get("SMALLPROB_RUN_VI_ACCURACY") != "1",
                    reason="Set SMALLPROB_RUN_VI_ACCURACY=1 for the nine-fit stochastic accuracy suite.")
def test_vi_accuracy_against_exact_and_numerical_references():
    pytest.importorskip("pyro")
    report = run_validation(progress=False)
    failures = {
        case["name"]: [{"seed": run["seed"], "errors": run["errors"]} for run in case["runs"] if not run["passed"]]
        for case in report["cases"] if not case["passed"]
    }
    assert report["passed"], failures
