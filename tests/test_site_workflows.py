"""Inference checks for the actual Pyro code displayed in the showcase."""
import contextlib
import io
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyro")

from examples.site_models import bound_cases, PYRO_MODELS
from examples.site_workflows import execute_pyro_workflow, workflows
from examples.validate_site_workflows import gaussian_check, shared_observations


def test_saved_workflow_evidence_matches_the_published_code():
    evidence_path = Path(__file__).resolve().parents[1] / "website/assets/workflow-validation.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["passed"]
    assert {case["id"] for case in evidence["cases"]} == set(PYRO_MODELS)
    for case in evidence["cases"]:
        assert case["workflow_sha256"] == hashlib.sha256(json.dumps(workflows(case["id"]), sort_keys=True).encode()).hexdigest()
        assert case["pyro_model_sha256"] == hashlib.sha256(PYRO_MODELS[case["id"]].encode()).hexdigest()


def test_displayed_pyro_hmm_recovers_nontrivial_exact_smoothed_marginals():
    # Emissions near zero deliberately make state uncertainty nontrivial.
    y = shared_observations()["hidden-markov-model"]
    with contextlib.redirect_stdout(io.StringIO()):
        result = execute_pyro_workflow("hidden-markov-model", observed=y)
    actual = result["probabilities"].numpy()
    expected = bound_cases()["hidden-markov-model"][0].infer(y=y).probabilities("z")
    assert np.any((expected > .05) & (expected < .95))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case_id,names", [("normal-mean", ["mu"]),
                                         ("linear-regression", ["intercept", "slope"])])
def test_displayed_pyro_vi_agrees_with_exact_gaussian_reference(case_id, names):
    observed = shared_observations()[case_id]
    with contextlib.redirect_stdout(io.StringIO()):
        result = execute_pyro_workflow(case_id, observed=observed)
    exact = bound_cases()[case_id][0].infer(y=observed)
    checks = gaussian_check(result, exact, names)
    assert all(check["passed"] for check in checks), checks
