"""Run all displayed Pyro workflows and check shared-data inference references.

This is deliberately separate from the fast, deterministic static build.
Run: python -m examples.validate_site_workflows
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from examples.site_models import bound_cases, MODEL_FUNCTIONS, PYRO_MODELS
from examples.site_workflows import execute_pyro_workflow, workflows

ROOT = Path(__file__).resolve().parents[1]


def shared_observations():
    models = bound_cases()
    x = models["linear-regression"][1][0]
    y_robust = 1.5 - .8 * x + np.random.default_rng(7).normal(0, .35, len(x))
    y_robust[[5, 24]] += [3., -3.]
    return {
        "normal-mean": 2.,
        "linear-regression": models["linear-regression"][0].simulate(
            seed=7, fixed={"intercept": 1.5, "slope": -.8})["y"][0],
        "robust-regression": y_robust,
        "hidden-markov-model": np.linspace(-.9, .9, 30) + .2 * np.sin(np.arange(30)),
        "weighing-scale": 9.5,
        "interaction-regression": models["interaction-regression"][0].simulate(
            seed=103, fixed={"intercept": 1.2, "group_effect": .7,
                            "slope": -.6, "interaction": .9, "sigma": .5})["y"][0],
        "eight-schools": np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
    }


def gaussian_check(namespace, exact, names):
    """Thresholds allow optimization and 4,000-draw Monte Carlo error."""
    checks = []
    for name in names:
        samples = namespace["draws"][name].detach().cpu().numpy().reshape(-1)
        mean, sd = float(samples.mean()), float(samples.std(ddof=1))
        exact_mean, exact_sd = float(exact.mean(name)), float(exact.sd(name))
        mean_error = abs(mean - exact_mean) / exact_sd
        sd_error = abs(sd - exact_sd) / exact_sd
        checks.append({"site": name, "pyro_guide_sample_mean": mean,
                       "pyro_guide_sample_sd": sd, "exact_mean": exact_mean, "exact_sd": exact_sd,
                       "mean_error_in_exact_sd": mean_error, "relative_sd_error": sd_error,
                       "passed": mean_error < .15 and sd_error < .15})
    return checks


def validate(destination=None):
    models, observations = bound_cases(), shared_observations()
    cases = []
    for case_id in MODEL_FUNCTIONS:
        start = time.monotonic()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            namespace = execute_pyro_workflow(case_id, observed=observations[case_id])
        checks = []
        if case_id in {"normal-mean", "linear-regression", "weighing-scale"}:
            observation = "measurement" if case_id == "weighing-scale" else "y"
            exact = models[case_id][0].infer(**{observation: observations[case_id]})
            names = {"normal-mean": ["mu"], "linear-regression": ["intercept", "slope"],
                     "weighing-scale": ["weight"]}[case_id]
            checks = gaussian_check(namespace, exact, names)
        elif case_id == "hidden-markov-model":
            exact = models[case_id][0].infer(y=observations[case_id]).probabilities("z")
            actual = namespace["probabilities"].detach().cpu().numpy()
            error = float(np.max(np.abs(exact - actual)))
            checks = [{"max_state_probability_error": error, "passed": error < 1e-12}]
        else:
            # The existing notebook validates these non-conjugate fits against
            # quadrature. Here we execute the direct Pyro recipe and check all
            # guide draws and its final objective are finite, not VI accuracy.
            finite = np.isfinite(namespace["loss"]) and all(
                np.all(np.isfinite(value.detach().cpu().numpy()))
                for value in namespace["draws"].values())
            checks = [{"execution_and_finiteness_only": True, "passed": bool(finite)}]
        case = {"id": case_id, "steps": namespace.get("steps", 0),
                "shared_observations": np.asarray(observations[case_id]).tolist(),
                "workflow_sha256": hashlib.sha256(json.dumps(workflows(case_id), sort_keys=True).encode()).hexdigest(),
                "pyro_model_sha256": hashlib.sha256(PYRO_MODELS[case_id].encode()).hexdigest(),
                "checks": checks, "passed": all(check["passed"] for check in checks)}
        cases.append(case)
        print(f"{case_id}: {'PASS' if case['passed'] else 'FAIL'} ({time.monotonic() - start:.1f}s)", flush=True)
    report = {
        "scope": "All seven displayed direct-Pyro generation, fit and summary recipes executed with common observations. Three Gaussian cases compare Pyro VI against exact Smallprob inference; HMM enumeration compares exact marginals. Other Pyro recipes receive execution/finiteness checks only.",
        "gaussian_tolerances": {"mean_error_in_exact_sd": .15, "relative_sd_error": .15},
        "hmm_absolute_tolerance": 1e-12,
        "cases": cases, "passed": all(case["passed"] for case in cases),
    }
    if destination:
        Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    validate(ROOT / "website/assets/workflow-validation.json")
