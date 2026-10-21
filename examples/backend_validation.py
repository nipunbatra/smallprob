"""Measured backend validation, separate from the fast test suite.

Student-t likelihood VI is checked against independently normalised numerical
quadrature. One full MPS fit is checked against exact Gaussian inference when
Apple GPU execution is available. No speedup or convergence claim is inferred
from merely completing an optimisation run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from smallprob import Normal, StudentT, model
from examples.vi_validation import THRESHOLDS, known_noise_regression, gaussian_regression_reference


ACCURACY_THRESHOLDS = {key: THRESHOLDS[key] for key in (
    "max_mean_error_in_reference_sd", "max_sd_relative_error", "max_interval_error_in_reference_sd")}
STUDENT_OBSERVATIONS = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 4.5])
SEEDS = (11, 29, 47)


@model
def student_mean(n, df=4., scale=.7, prior_sd=2.):
    mu = Normal(0, prior_sd)
    y = [StudentT(df, mu, scale) for _ in range(n)]


def student_quadrature(y, *, df=4., scale=.7, prior_sd=2., bounds=(-16., 16.), points=128001):
    """Direct Student-t density formula; no Smallprob/Pyro likelihood calls."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or df <= 0 or scale <= 0 or prior_sd <= 0 or points < 3:
        raise ValueError("Invalid Student-t reference inputs.")
    mu = np.linspace(*bounds, points)
    log_prior = -.5 * (mu / prior_sd)**2 - np.log(prior_sd) - .5 * np.log(2 * np.pi)
    normalisation = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                     - .5 * np.log(df * np.pi) - np.log(scale))
    residual = (y[None, :] - mu[:, None]) / scale
    log_likelihood = np.sum(normalisation - (df + 1) / 2 * np.log1p(residual**2 / df), axis=1)
    log_density = log_prior + log_likelihood
    maximum = log_density.max()
    density = np.exp(log_density - maximum)
    spacing = mu[1] - mu[0]
    weights = np.full(points, spacing)
    weights[[0, -1]] *= .5
    normaliser = density @ weights
    weights *= density / normaliser
    mean = float(weights @ mu)
    variance = float(weights @ ((mu - mean)**2))
    cumulative = np.concatenate([[0.], np.cumsum((density[:-1] + density[1:]) * spacing / 2)])
    cumulative /= cumulative[-1]
    return {
        "kind": "independent numerical quadrature of Normal prior times direct Student-t density",
        "mean": [mean], "sd": [np.sqrt(variance)],
        "interval95": [np.interp([.025, .975], cumulative, mu).tolist()],
        "log_evidence": float(maximum + np.log(normaliser)),
        "grid": {"bounds": list(bounds), "points": points, "rule": "trapezoid"},
        "_grid": mu, "_weights": weights,
    }


def checked_student_reference(y):
    coarse = student_quadrature(y)
    refined = student_quadrature(y, points=256001)
    expanded = student_quadrature(y, bounds=(-24., 24.), points=384001)
    measurements = {}
    for label, other in (("grid_refinement", coarse), ("range_expansion", expanded)):
        measurements[label] = {key: float(np.max(np.abs(np.asarray(refined[key]) - other[key])))
                               for key in ("mean", "sd", "interval95", "log_evidence")}
    outside = np.abs(expanded["_grid"]) > 16
    measurements["posterior_mass_outside_original_range"] = float(expanded["_weights"][outside].sum())
    measurements["comparison_threshold"] = 1e-5
    measurements["tail_mass_threshold"] = 1e-8
    measurements["passed"] = bool(all(value < 1e-5 for label in ("grid_refinement", "range_expansion")
                                       for value in measurements[label].values())
                                    and measurements["posterior_mass_outside_original_range"] < 1e-8)
    refined["convergence"] = measurements
    return {key: value for key, value in refined.items() if not key.startswith("_")}


def compare(posterior, names, reference):
    mean = np.array([float(posterior.mean(name)) for name in names])
    sd = np.array([float(posterior.sd(name)) for name in names])
    interval = np.array([[float(endpoint) for endpoint in posterior.interval(name)] for name in names])
    reference_mean, reference_sd = np.asarray(reference["mean"]), np.asarray(reference["sd"])
    errors = {
        "max_mean_error_in_reference_sd": float(np.max(np.abs(mean - reference_mean) / reference_sd)),
        "max_sd_relative_error": float(np.max(np.abs(sd / reference_sd - 1))),
        "max_interval_error_in_reference_sd": float(np.max(np.abs(interval - reference["interval95"]) / reference_sd[:, None])),
    }
    checks = {key: value <= ACCURACY_THRESHOLDS[key] for key, value in errors.items()}
    return {"mean": mean.tolist(), "sd": sd.tolist(), "interval95": interval.tolist(),
            "errors": errors, "checks": checks, "passed": all(checks.values())}


def run_validation(*, steps=3000, particles=32, seeds=SEEDS, progress=True):
    import torch
    report = {
        "purpose": "Independent Student-t posterior calibration and measured native-device VI checks",
        "thresholds_predeclared": ACCURACY_THRESHOLDS,
        "settings": {"steps": steps, "particles": particles, "learning_rate": .03, "seeds": list(seeds)},
        "torch_version": torch.__version__,
        "performance_note": "Wall times describe these small models on this machine; they do not imply a GPU speedup.",
        "cases": [],
    }
    reference = checked_student_reference(STUDENT_OBSERVATIONS)
    student = {"name": "student_t_location", "names": ["mu"], "y": STUDENT_OBSERVATIONS.tolist(),
               "likelihood": {"df": 4., "scale": .7}, "prior": {"mean": 0., "sd": 2.},
               "reference": reference, "runs": []}
    for seed in seeds:
        start = time.perf_counter()
        p = student_mean(len(STUDENT_OBSERVATIONS)).infer(y=STUDENT_OBSERVATIONS, method="vi",
                steps=steps, particles=particles, seed=seed, backend="torch", device="cpu", dtype="float64")
        elapsed = time.perf_counter() - start
        run = {"seed": seed, "device": p.diagnostics["device"], "dtype": p.diagnostics["dtype"],
               "wall_seconds": elapsed, **compare(p, ["mu"], reference), "elbo": p.elbo,
               "elbo_standard_error": p.diagnostics["elbo_standard_error"]}
        student["runs"].append(run)
        if progress:
            print(f"StudentT CPU seed={seed}: {'PASS' if run['passed'] else 'FAIL'} in {elapsed:.2f}s; {run['errors']}", flush=True)
    student["passed"] = reference["convergence"]["passed"] and all(run["passed"] for run in student["runs"])
    report["cases"].append(student)

    mps_case = {"name": "known_noise_regression_mps", "names": ["intercept", "slope"],
                "available": torch.backends.mps.is_available(), "runs": []}
    if mps_case["available"]:
        x = np.linspace(.5, 2.5, 24)
        y = 1.2 - .7 * x + np.random.default_rng(102).normal(0, .6, len(x))
        reference = gaussian_regression_reference(np.column_stack([np.ones_like(x), x]), y, .6)
        exact = known_noise_regression(x).infer(y=y)
        exact_agrees = (np.allclose([exact.mean(name) for name in mps_case["names"]], reference["mean"], rtol=1e-10, atol=1e-10)
                        and np.allclose([exact.sd(name) for name in mps_case["names"]], reference["sd"], rtol=1e-10, atol=1e-10))
        mps_case.update({"x": x.tolist(), "y": y.tolist(), "data_seed": 102,
                         "reference": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                       for key, value in reference.items()},
                         "exact_engine_agrees_with_independent_reference": bool(exact_agrees)})
        if progress:
            print(f"Running {steps}-step MPS regression accuracy fit...", flush=True)
        start = time.perf_counter()
        p = known_noise_regression(torch.tensor(x, device="mps", dtype=torch.float32)).infer(
            y=torch.tensor(y, device="mps", dtype=torch.float32), method="vi", steps=steps,
            particles=particles, seed=11)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        samples = p.sample(1000, seed=12)
        native_samples = all(value.device.type == "mps" and bool(torch.all(torch.isfinite(value))) for value in samples.values())
        run = {"seed": 11, "device": p.diagnostics["device"], "dtype": p.diagnostics["dtype"],
               "wall_seconds": elapsed, **compare(p, mps_case["names"], reference),
               "native_finite_samples": native_samples, "elbo": p.elbo,
               "elbo_standard_error": p.diagnostics["elbo_standard_error"]}
        run["passed"] = run["passed"] and native_samples
        mps_case["runs"].append(run)
        mps_case["passed"] = bool(exact_agrees) and run["passed"]
        if progress:
            print(f"MPS regression: {'PASS' if run['passed'] else 'FAIL'} in {elapsed:.2f}s; {run['errors']}", flush=True)
    else:
        mps_case.update({"passed": None, "skip_reason": "MPS unavailable on this machine; no GPU accuracy claim."})
    report["cases"].append(mps_case)
    report["passed"] = student["passed"] and mps_case["passed"] is not False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/backend-validation.json"))
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output.resolve()}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
