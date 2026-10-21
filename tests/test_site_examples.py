"""Validate executable comparisons and the reproducible static case bundle."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from examples.site_models import MODEL_FUNCTIONS, bound_cases, comparison_checks, regression, smallprob_code


def test_all_seven_displayed_pyro_models_match_generated_joint_assignments():
    pytest.importorskip("pyro")
    checks = comparison_checks()
    assert len(checks) == 21
    assert {check["case"] for check in checks} == set(MODEL_FUNCTIONS)
    assert all(check["passed"] for check in checks), checks


@pytest.mark.parametrize("case_id", list(MODEL_FUNCTIONS))
def test_displayed_smallprob_definition_is_executable_and_matches_checked_model(case_id, tmp_path):
    path = tmp_path / f"{case_id.replace('-', '_')}.py"
    path.write_text(smallprob_code(case_id))
    spec = importlib.util.spec_from_file_location("site_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected, args = bound_cases()[case_id]
    actual = module.chain if case_id == "hidden-markov-model" else getattr(module, MODEL_FUNCTIONS[case_id].__name__)(*args)
    draws = expected.simulate(draws=3, seed=19)
    reconstructed = actual.simulate(draws=3, seed=19)
    assert actual.graph.groups == expected.graph.groups
    for name in expected.graph.groups:
        np.testing.assert_array_equal(draws[name], reconstructed[name])


def test_downloaded_robust_data_recipe_matches_executed_notebook_data():
    x = np.linspace(-2, 2, 30)
    from_model = regression(x).simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})["y"][0]
    from_recipe = 1.5 - .8 * x + np.random.default_rng(7).normal(0, .35, len(x))
    np.testing.assert_allclose(from_recipe, from_model, atol=1e-15)


def test_site_build_preserves_executed_plot_provenance_and_runnable_downloads(tmp_path):
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")
    pytest.importorskip("nbformat")
    from examples.build_site_data import ROOT, build

    cases = build(tmp_path)
    assert len(cases) == 7
    for case in cases:
        assert set(case) == {"id", "title", "eyebrow", "description", "method", "smallprob", "pyro",
                             "run", "result", "notes", "sources", "diagram", "plot", "download",
                             "pyro_download", "smallprob_method", "pyro_method", "workflows", "workflow_note"}
        assert case["result"].strip()
        assert (tmp_path / case["diagram"]).read_text().startswith("<?xml")
        if case["plot"]:
            plot_bytes = (tmp_path / case["plot"]).read_bytes()
            assert plot_bytes.startswith(b"\x89PNG") or b"<svg " in plot_bytes
        source = (tmp_path / case["download"]).read_text()
        tree = ast.parse(source)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name.startswith("examples") for name in imports)
        assert case["smallprob"] in source and case["run"] in source
        compile(source, case["download"], "exec")
        pyro_source = (tmp_path / case["pyro_download"]).read_text()
        compile(pyro_source, case["pyro_download"], "exec")
        assert case["pyro"] in pyro_source
        assert [step["id"] for step in case["workflows"]] == ["generate", "fit", "summarize"]
        for step in case["workflows"]:
            assert step["smallprob"] in source
            assert step["pyro"] in pyro_source
    evidence = json.loads((tmp_path / "assets/comparison-validation.json").read_text())
    assert evidence["passed"]
    for name, digest in evidence["notebook_sha256"].items():
        assert hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest() == digest


def test_static_case_build_is_deterministic(tmp_path):
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")
    pytest.importorskip("nbformat")
    from examples.build_site_data import build

    build(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    build(tmp_path)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
