"""Reproducible VI checks with independent Gaussian and quadrature references.

Run with ``python -m examples.vi_validation``. The thresholds are declared in
docs/vi-validation-design.md and are never fitted to the observed results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

from smallprob import HalfNormal, Normal, model


THRESHOLDS = {
    "max_mean_error_in_reference_sd": 0.35,
    "max_sd_relative_error": 0.20,
    "covariance_relative_frobenius_error": 0.25,
    "max_correlation_absolute_error": 0.15,
    "max_interval_error_in_reference_sd": 0.45,
}
DEFAULT_SEEDS = (11, 29, 47)


@model
def normal_mean(n, noise=0.8):
    mu = Normal(0, 5)
    y = [Normal(mu, noise) for _ in range(n)]


@model
def known_noise_regression(x, noise=0.6):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    y = Normal(intercept + slope * x, noise)


@model
def unknown_noise_regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)


def gaussian_regression_reference(design, y, sigma, prior_sd=5.0):
    """Independent conjugate calculation, without the Smallprob graph/engine."""
    design, y = np.asarray(design, dtype=float), np.asarray(y, dtype=float)
    n, p = design.shape
    prior_precision = np.eye(p) / prior_sd**2
    precision = prior_precision + design.T @ design / sigma**2
    covariance = np.linalg.solve(precision, np.eye(p))
    mean = np.linalg.solve(precision, design.T @ y / sigma**2)
    residual = y - design @ mean
    minimum_quadratic = residual @ residual / sigma**2 + mean @ prior_precision @ mean
    log_determinant = 2 * n * np.log(sigma) + 2 * p * np.log(prior_sd) + np.linalg.slogdet(precision)[1]
    log_evidence = -0.5 * (n * np.log(2 * np.pi) + log_determinant + minimum_quadratic)
    sd = np.sqrt(np.diag(covariance))
    quantile = NormalDist().inv_cdf(0.975)
    return {
        "kind": "analytic Gaussian posterior",
        "mean": mean,
        "sd": sd,
        "covariance": covariance,
        "interval95": np.stack([mean - quantile * sd, mean + quantile * sd], axis=1),
        "log_evidence": float(log_evidence),
    }


def _normal_cdf(values):
    values = np.asarray(values)
    return np.fromiter((0.5 * math.erfc(-float(v) / math.sqrt(2)) for v in values.flat),
                       dtype=float, count=values.size).reshape(values.shape)


def quadrature_regression_reference(design, y, *, prior_sd=5.0, noise_prior_sd=1.0,
                                    bounds=(-8.0, 3.0), points=4097):
    """Integrate log sigma numerically after analytically marginalising weights.

    The result is a numerical reference, not an analytically exact posterior.
    This deliberately uses only NumPy and independent conjugate mathematics.
    """
    design, y = np.asarray(design, dtype=float), np.asarray(y, dtype=float)
    if design.ndim != 2 or y.shape != (design.shape[0],):
        raise ValueError("Expected a design matrix and a matching observation vector.")
    if points < 3 or not bounds[0] < bounds[1] or prior_sd <= 0 or noise_prior_sd <= 0:
        raise ValueError("Invalid quadrature grid or prior scale.")
    n, p = design.shape
    log_sigma = np.linspace(*bounds, points)
    sigma = np.exp(log_sigma)
    sigma_squared = sigma**2
    prior_precision = np.eye(p) / prior_sd**2
    precision = prior_precision[None] + (design.T @ design)[None] / sigma_squared[:, None, None]
    covariance = np.linalg.inv(precision)
    mean = np.einsum("kij,kj->ki", covariance, (design.T @ y)[None] / sigma_squared[:, None])
    residual = y[None] - mean @ design.T
    quadratic = np.sum(residual**2, axis=1) / sigma_squared + np.sum(mean**2, axis=1) / prior_sd**2
    logdet = 2 * n * log_sigma + 2 * p * np.log(prior_sd) + np.linalg.slogdet(precision)[1]
    log_marginal = -0.5 * (n * np.log(2 * np.pi) + logdet + quadratic)
    # HalfNormal(s) density times d sigma / d log_sigma = sigma.
    log_prior_and_jacobian = (0.5 * np.log(2 / np.pi) - np.log(noise_prior_sd)
                              - 0.5 * (sigma / noise_prior_sd)**2 + log_sigma)
    log_density = log_marginal + log_prior_and_jacobian
    scaled_density = np.exp(log_density - log_density.max())
    spacing = log_sigma[1] - log_sigma[0]
    trapezoid_weights = np.full(points, spacing)
    trapezoid_weights[[0, -1]] *= 0.5
    normaliser = scaled_density @ trapezoid_weights
    weights = scaled_density * trapezoid_weights / normaliser
    conditional_means = np.column_stack([mean, sigma])
    posterior_mean = weights @ conditional_means
    centred_means = conditional_means - posterior_mean
    posterior_covariance = (centred_means * weights[:, None]).T @ centred_means
    posterior_covariance[:p, :p] += np.einsum("k,kij->ij", weights, covariance)
    posterior_sd = np.sqrt(np.diag(posterior_covariance))
    intervals = np.empty((p + 1, 2))
    for j in range(p):
        conditional_sd = np.sqrt(covariance[:, j, j])
        for endpoint, probability in enumerate((0.025, 0.975)):
            lower = float(np.min(mean[:, j] - 12 * conditional_sd))
            upper = float(np.max(mean[:, j] + 12 * conditional_sd))
            for _ in range(55):
                midpoint = (lower + upper) / 2
                cdf = weights @ _normal_cdf((midpoint - mean[:, j]) / conditional_sd)
                if cdf < probability:
                    lower = midpoint
                else:
                    upper = midpoint
            intervals[j, endpoint] = (lower + upper) / 2
    cumulative = np.concatenate([[0.0], np.cumsum((scaled_density[:-1] + scaled_density[1:]) * spacing / 2)])
    cumulative /= cumulative[-1]
    intervals[-1] = np.exp(np.interp([0.025, 0.975], cumulative, log_sigma))
    return {
        "kind": "numerical log-sigma quadrature; coefficients analytically integrated out",
        "mean": posterior_mean,
        "sd": posterior_sd,
        "covariance": posterior_covariance,
        "interval95": intervals,
        "log_evidence": float(log_density.max() + np.log(normaliser)),
        "grid": {"log_sigma_bounds": list(bounds), "points": points, "rule": "trapezoid"},
        "_log_sigma": log_sigma,
        "_weights": weights,
    }


def check_quadrature_convergence(design, y):
    # Tighten the numerical reference, never the accuracy thresholds, if the
    # initial grid does not resolve interval endpoints sufficiently well.
    attempts = []
    for coarse_points in (4097, 8193, 16385, 32769):
        refined_points = 2 * coarse_points - 1
        coarse = quadrature_regression_reference(design, y, points=coarse_points)
        refined = quadrature_regression_reference(design, y, points=refined_points)
        expanded = quadrature_regression_reference(design, y, bounds=(-10.0, 4.0),
                                                    points=3 * coarse_points - 2)
        measurements = _quadrature_convergence_measurements(coarse, refined, expanded)
        attempts.append({"coarse_points": coarse_points, "refined_points": refined_points,
                         **measurements})
        if measurements["passed"]:
            break
    measurements["refinement_attempts"] = attempts
    refined["convergence"] = measurements
    return refined


def _quadrature_convergence_measurements(coarse, refined, expanded):
    measurements = {}
    for label, other in (("grid_refinement", coarse), ("range_expansion", expanded)):
        measurements[label] = {
            "mean_max_relative_change": float(np.max(np.abs(other["mean"] - refined["mean"]) / np.maximum(1, np.abs(refined["mean"])))),
            "sd_max_relative_change": float(np.max(np.abs(other["sd"] / refined["sd"] - 1))),
            "interval_max_absolute_change": float(np.max(np.abs(other["interval95"] - refined["interval95"]))),
            "log_evidence_absolute_change": float(abs(other["log_evidence"] - refined["log_evidence"])),
        }
    outside = (expanded["_log_sigma"] < -8) | (expanded["_log_sigma"] > 3)
    measurements["posterior_mass_outside_original_range"] = float(expanded["_weights"][outside].sum())
    measurements["comparison_threshold"] = 1e-5
    measurements["tail_mass_threshold"] = 1e-8
    measurements["passed"] = bool(
        all(v < 1e-5 for label in ("grid_refinement", "range_expansion") for v in measurements[label].values())
        and measurements["posterior_mass_outside_original_range"] < 1e-8
    )
    return measurements


def _guide_covariance(posterior, names):
    """Analytic original-scale covariance of a Normal/lognormal guide."""
    diagnostics = posterior.diagnostics
    order = diagnostics["latent_order"]
    indices = [order.index(name) for name in names]
    unconstrained = np.asarray(diagnostics["unconstrained_covariance"])[np.ix_(indices, indices)]
    covariance = unconstrained.copy()
    for i, name in enumerate(names):
        for j, other in enumerate(names):
            if name == "sigma" and other == "sigma":
                covariance[i, j] = posterior.sd(name)**2
            elif name == "sigma":
                covariance[i, j] *= posterior.mean(name)
            elif other == "sigma":
                covariance[i, j] *= posterior.mean(other)
    return covariance


def _correlation(covariance):
    sd = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sd, sd)


def compare_posterior(posterior, names, reference):
    means = np.asarray([posterior.mean(name) for name in names])
    sds = np.asarray([posterior.sd(name) for name in names])
    covariance = _guide_covariance(posterior, names)
    intervals = np.asarray([posterior.interval(name) for name in names])
    errors = {
        "max_mean_error_in_reference_sd": float(np.max(np.abs(means - reference["mean"]) / reference["sd"])),
        "max_sd_relative_error": float(np.max(np.abs(sds / reference["sd"] - 1))),
        "covariance_relative_frobenius_error": float(np.linalg.norm(covariance - reference["covariance"]) / np.linalg.norm(reference["covariance"])),
        "max_correlation_absolute_error": float(np.max(np.abs(_correlation(covariance) - _correlation(reference["covariance"])))),
        "max_interval_error_in_reference_sd": float(np.max(np.abs(intervals - reference["interval95"]) / reference["sd"][:, None])),
    }
    checks = {key: errors[key] <= threshold for key, threshold in THRESHOLDS.items()}
    result = {
        "mean": means,
        "sd": sds,
        "covariance": covariance,
        "interval95": intervals,
        "errors": errors,
        "checks": checks,
        "passed": all(checks.values()),
        "elbo_monte_carlo": posterior.elbo,
        "log_evidence": posterior.log_evidence,
    }
    if reference["kind"] == "analytic Gaussian posterior":
        delta = means - reference["mean"]
        ref_cov = reference["covariance"]
        kl = 0.5 * (np.trace(np.linalg.solve(ref_cov, covariance))
                    + delta @ np.linalg.solve(ref_cov, delta) - len(names)
                    + np.linalg.slogdet(ref_cov)[1] - np.linalg.slogdet(covariance)[1])
        exact_elbo = reference["log_evidence"] - kl
        result.update({"gaussian_kl_q_to_reference": float(kl), "elbo_analytic_for_learned_guide": float(exact_elbo),
                       "elbo_monte_carlo_error": float(posterior.elbo - exact_elbo)})
    return result


def validation_cases():
    normal_y = np.random.default_rng(101).normal(1.25, 0.8, 12)
    x_known = np.linspace(0.5, 2.5, 24)
    known_y = 1.2 - 0.7 * x_known + np.random.default_rng(102).normal(0, 0.6, len(x_known))
    x_unknown = np.linspace(-1.5, 1.5, 32)
    unknown_y = 1.2 - 0.7 * x_unknown + np.random.default_rng(103).normal(0, 0.6, len(x_unknown))
    design_unknown = np.column_stack([np.ones_like(x_unknown), x_unknown])
    return [
        {"name": "normal_mean", "model": normal_mean(len(normal_y)), "y": normal_y,
         "names": ["mu"], "truth": [1.25], "data_seed": 101,
         "reference": gaussian_regression_reference(np.ones((len(normal_y), 1)), normal_y, 0.8)},
        {"name": "known_noise_correlated_regression", "model": known_noise_regression(x_known), "y": known_y,
         "x": x_known, "names": ["intercept", "slope"], "truth": [1.2, -0.7], "data_seed": 102,
         "reference": gaussian_regression_reference(np.column_stack([np.ones_like(x_known), x_known]), known_y, 0.6)},
        {"name": "unknown_noise_regression", "model": unknown_noise_regression(x_unknown), "y": unknown_y,
         "x": x_unknown, "names": ["intercept", "slope", "sigma"], "truth": [1.2, -0.7, 0.6], "data_seed": 103,
         "reference": check_quadrature_convergence(design_unknown, unknown_y)},
    ]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_validation(*, steps=3000, particles=32, learning_rate=0.03, seeds=DEFAULT_SEEDS, progress=True):
    report = {
        "purpose": "Synthetic prototype validation; VI is approximate.",
        "thresholds_predeclared": dict(THRESHOLDS),
        "settings": {"steps": steps, "particles": particles, "learning_rate": learning_rate, "seeds": list(seeds)},
        "cases": [],
    }
    for case in validation_cases():
        reference, names = case["reference"], case["names"]
        record = {key: case[key] for key in ("name", "names", "truth", "data_seed", "y")}
        if "x" in case:
            record["x"] = case["x"]
        record["reference"] = reference
        if reference["kind"] == "analytic Gaussian posterior":
            exact = case["model"].infer(y=case["y"])
            for key, values in (("mean", [exact.mean(name) for name in names]),
                                ("sd", [exact.sd(name) for name in names]),
                                ("interval95", [exact.interval(name) for name in names])):
                np.testing.assert_allclose(values, reference[key], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(exact.log_evidence, reference["log_evidence"], atol=1e-10, rtol=1e-10)
            record["exact_engine_agrees_with_independent_reference"] = True
        record["runs"] = []
        for seed in seeds:
            start = time.monotonic()
            posterior = case["model"].infer(y=case["y"], method="vi", seed=seed, steps=steps,
                                            particles=particles, learning_rate=learning_rate)
            run = compare_posterior(posterior, names, reference)
            run.update({"seed": seed, "wall_seconds": time.monotonic() - start,
                        "diagnostics": posterior.diagnostics})
            intervals = run["interval95"]
            run["synthetic_truth_inside_interval95"] = ((np.asarray(case["truth"]) >= intervals[:, 0])
                                                          & (np.asarray(case["truth"]) <= intervals[:, 1])).tolist()
            if "sigma" in names:
                samples = posterior.sample(draws=10000, seed=seed + 1000)["sigma"]
                run["sigma_samples_all_positive"] = bool(np.all(samples > 0))
                run["sigma_sample_minimum"] = float(samples.min())
                run["passed"] = run["passed"] and run["sigma_samples_all_positive"]
            record["runs"].append(run)
            if progress:
                errors = run["errors"]
                print(f"{case['name']} seed={seed}: {'PASS' if run['passed'] else 'FAIL'}; "
                      f"mean/SD={errors['max_mean_error_in_reference_sd']:.4f}, "
                      f"SD relative={errors['max_sd_relative_error']:.4f}, "
                      f"covariance relative={errors['covariance_relative_frobenius_error']:.4f}, "
                      f"interval/SD={errors['max_interval_error_in_reference_sd']:.4f}; "
                      f"{run['wall_seconds']:.1f}s", flush=True)
        record["passed"] = all(run["passed"] for run in record["runs"]) and reference.get("convergence", {}).get("passed", True)
        report["cases"].append(record)
    report["passed"] = all(case["passed"] for case in report["cases"])
    return _jsonable(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("outputs/vi-validation.json"))
    args = parser.parse_args()
    report = run_validation(steps=args.steps, particles=args.particles,
                            learning_rate=args.learning_rate, seeds=args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: report saved to {args.output.resolve()}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
