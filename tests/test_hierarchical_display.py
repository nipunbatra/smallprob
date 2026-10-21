"""Keep local random effects distinct when children use different sites."""

import numpy as np
import pytest

from smallprob import Normal, model
from smallprob.core import Expr
from examples.pyro_gallery import eight_schools, interaction_regression


@pytest.mark.parametrize("observed", [None, "y", ["y[2]"], ["eta[0]", "y[2]"]])
def test_local_effect_graph_retains_every_true_dependency(observed):
    bound = eight_schools()
    original = [(node.name, set(node.distribution.dependencies())) for node in bound.graph.nodes]
    diagram = bound.diagram(observed=observed)
    assert len(diagram.nodes) == 18
    assert all(node.size is None for node in diagram.nodes)
    expected_edges = {(f"eta[{i}]", f"y[{i}]") for i in range(8)}
    expected_edges |= {(parent, f"y[{i}]") for parent in ("mu", "tau") for i in range(8)}
    assert set(diagram.edges) == expected_edges
    assert diagram.hierarchy == (("mu", "tau"), tuple(f"eta[{i}]" for i in range(8)),
                                 tuple(f"y[{i}]" for i in range(8)))
    marked = {node.key for node in diagram.nodes if node.observed}
    expected_marked = {f"y[{i}]" for i in range(8)} if observed == "y" else set(observed or [])
    assert marked == expected_marked
    assert [(node.name, set(node.distribution.dependencies())) for node in bound.graph.nodes] == original


def test_math_retains_school_specific_effects_and_standard_errors():
    mathematics = eight_schools().math()
    assert len(mathematics.equations) == 18
    assert r"\eta_{1}" in mathematics.equations[10]
    assert r"\left(15\right)^2" in mathematics.equations[10]
    assert r"\eta_{8}" in mathematics.equations[17]
    assert r"\left(18\right)^2" in mathematics.equations[17]
    assert r"\prod_{i=" not in mathematics.factorization


def test_simple_vector_regression_keeps_compact_observation_plate():
    diagram = interaction_regression(np.array([0., 1., 0., 1.]),
                                     np.array([-1., -.5, .5, 1.])).diagram(observed="y")
    assert len(diagram.nodes) == 6
    assert diagram.nodes[-1].size == 4
    assert diagram.hierarchy is None
    assert set(diagram.edges) == {(name, "y") for name in
                                 ("intercept", "group_effect", "slope", "interaction", "sigma")}


def test_misaligned_children_keep_exact_edges_without_special_layout():
    bound = eight_schools()
    node = next(node for node in bound.graph.nodes if node.name == "y[2]")
    node.distribution = Normal(Expr("ref", ("mu",)) + Expr("ref", ("tau",)) * Expr("ref", ("eta[1]",)), 16)
    diagram = bound.diagram()
    assert diagram.hierarchy is None
    assert ("eta[1]", "y[2]") in diagram.edges
    assert ("eta[2]", "y[2]") not in diagram.edges


@model
def mixed_local_and_shared():
    local = Normal([0., 0., 0.], 1)
    shared = Normal(0, 1)
    y = Normal(local + shared, 1)


def test_hierarchy_recognition_does_not_depend_on_school_variable_names():
    diagram = mixed_local_and_shared().diagram(observed="y")
    assert len(diagram.nodes) == 7
    assert diagram.hierarchy == (("shared",), ("local[0]", "local[1]", "local[2]"),
                                 ("y[0]", "y[1]", "y[2]"))


def test_hierarchy_render_labels_and_shading_remain_local(tmp_path):
    pytest.importorskip("matplotlib")
    diagram = eight_schools().diagram(observed=["y[2]"])
    figure = diagram._figure()
    try:
        graph_text = [text.get_text() for text in figure.axes[0].texts]
        assert "$\\eta_{1}$" in graph_text and "$\\eta_{8}$" in graph_text
        assert "$y_{1}$" in graph_text and "$y_{8}$" in graph_text
        assert "each local effect has its own observation" in graph_text
    finally:
        figure.clear()
    assert diagram.save(tmp_path / "schools.svg").read_text().startswith("<?xml")
