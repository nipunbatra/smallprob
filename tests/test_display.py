"""Semantic checks for presentation without changing model meaning."""

from pathlib import Path

import numpy as np
import pytest

from smallprob import Categorical, HalfNormal, Normal, model
from smallprob.core import Graph, ModelError, Node
from smallprob.api import BoundModel
from smallprob.display import ModelDiagram, ModelMath


@model
def visual_regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)


@model
def repeated(n):
    mu = Normal(0, 2)
    y = [Normal(mu, 1) for _ in range(n)]


@model
def dependent():
    z = []
    z.append(Categorical([.5, .5]))
    z.append(Categorical([[.8, .2], [.1, .9]][z[-1]]))


def with_metadata(factory, **inputs):
    """Metadata can be supplied independently of frontend wrapper integration."""
    instance = factory(**inputs)
    instance.source = factory.source
    instance.inputs = inputs
    return instance


def test_source_math_keeps_input_symbols_and_variance_convention():
    bound = with_metadata(visual_regression, x=np.asarray([3., 5., 7.]))
    mathematics = ModelMath(bound, labels={"slope": "m", "intercept": "b"})
    assert len(mathematics.equations) == 4
    assert "x_{i}" in mathematics.equations[-1]
    assert "b + m" in mathematics.equations[-1]
    assert r"\left(\sigma\right)^2" in mathematics.equations[-1]
    assert "3, 5, 7" not in mathematics.latex
    assert r"\prod_{i=1}^{3}" in mathematics.factorization
    assert mathematics._repr_latex_().startswith("$$")


def test_diagram_shows_shared_parents_and_explicit_observed_plate():
    bound = with_metadata(visual_regression, x=[1., 2., 3.])
    diagram = ModelDiagram(bound, labels={"slope": "m", "intercept": "b"}, observed=["y"])
    assert set(diagram.edges) == {("intercept", "y"), ("slope", "y"), ("sigma", "y")}
    assert len(diagram.nodes) == 4
    assert diagram.nodes[-1].size == 3
    assert diagram.nodes[-1].observed
    assert all(not node.observed for node in diagram.nodes[:-1])
    assert all(not node.observed for node in ModelDiagram(bound).nodes)
    assert bound.graph.groups["slope"] == ["slope"]


def test_comprehension_iid_compaction_and_empty_series():
    bound = with_metadata(repeated, n=4)
    diagram = ModelDiagram(bound)
    assert len(diagram.nodes) == 2
    assert diagram.nodes[-1].size == 4
    assert diagram.edges == [("mu", "y")]
    assert len(ModelDiagram(with_metadata(repeated, n=0)).nodes) == 1


def test_partial_observed_series_is_not_marked_fully_observed():
    diagram = ModelDiagram(with_metadata(repeated, n=3), observed=["y[0]"])
    assert len(diagram.nodes) == 4
    assert [node.observed for node in diagram.nodes] == [False, True, False, False]


def test_markov_dependency_is_not_compacted_into_independent_plate():
    diagram = ModelDiagram(dependent())
    assert len(diagram.nodes) == 2
    assert all(node.size is None for node in diagram.nodes)
    assert diagram.edges == [("z[0]", "z[1]")]


def test_unknown_metadata_falls_back_to_exact_scalar_equations():
    graph = Graph([Node("a", Normal(0, 2)), Node("y[0]", Normal(3, 1)), Node("y[1]", Normal(5, 1))], {"a": ["a"], "y": ["y[0]", "y[1]"]}, {"y"})
    bound = BoundModel(graph, "fallback")
    mathematics = ModelMath(bound)
    assert len(mathematics.equations) == 3
    assert r"\left(3," in mathematics.equations[1]
    assert r"\left(5," in mathematics.equations[2]


def test_custom_tex_labels_and_input_labels():
    mathematics = ModelMath(with_metadata(visual_regression, x=[1., 2.]), labels={"x": "t", "slope": r"\beta", "intercept": r"\alpha"})
    assert r"\alpha + \beta\,t_{i}" in mathematics.equations[-1]


@pytest.mark.parametrize("options", [{"labels": {"missing": "x"}}, {"labels": {"slope": ""}}, {"observed": ["missing"]}])
def test_invalid_presentation_names_raise(options):
    with pytest.raises(ModelError):
        ModelDiagram(visual_regression([1., 2.]), **options)


def test_svg_png_are_rich_and_saved(tmp_path):
    pytest.importorskip("matplotlib")
    diagram = ModelDiagram(with_metadata(visual_regression, x=[1., 2., 3.]), labels={"slope": "m", "intercept": "b"}, observed="y")
    svg = diagram._repr_svg_()
    assert "<svg" in svg and "<title>" in svg and "visual_regression" in svg
    assert diagram.save(tmp_path / "model.svg").read_text().startswith("<?xml")
    assert diagram.save(tmp_path / "model.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ModelError, match=".svg and .png"):
        diagram.save(tmp_path / "model.pdf")


def test_student_t_uses_scale_not_variance_or_sd():
    from smallprob import StudentT
    graph = Graph([Node("y", StudentT(df=4, loc=1, scale=2))], {"y": ["y"]}, set())
    mathematics = ModelMath(BoundModel(graph, "student"))
    assert mathematics.equations == (r"y\sim t_{4}\!\left(1,\;2\right)",)
    assert "^2" not in mathematics.equations[0]


@model
def negative_mean(x):
    slope = Normal(0, 1)
    y = Normal(-(slope + x), 1)


@model
def transformed_input(x):
    x = x + 10
    y = Normal(x, 1)


def test_source_preserves_unary_minus_precedence():
    mathematics = ModelMath(with_metadata(negative_mean, x=np.asarray([1., 2.])))
    assert r"-\left(\mathit{slope} + x_{i}\right)" in mathematics.equations[-1]


def test_reassigned_inputs_use_graph_values_not_original_names():
    mathematics = ModelMath(with_metadata(transformed_input, x=np.asarray([1., 2.])))
    assert len(mathematics.equations) == 2
    assert "x_{i}" not in mathematics.latex
    assert r"\left(11," in mathematics.equations[0]


def test_hmm_keeps_chain_emission_edges_and_displays_fixed_parameters():
    from examples.diagram_demo import hidden_markov_model
    bound = hidden_markov_model(4)
    diagram = bound.diagram(observed="y")
    assert diagram.hmm is not None
    assert diagram.hmm.category_count == 2
    assert diagram.hmm.states == ("z[0]", "z[1]", "z[2]", "z[3]")
    assert set(diagram.edges) == {(f"z[{i}]", f"y[{i}]") for i in range(4)} | {(f"z[{i}]", f"z[{i+1}]") for i in range(3)}
    assert all(node.size is None for node in diagram.nodes)
    assert all(node.observed == node.key.startswith("y[") for node in diagram.nodes)
    assert r"A_{1,:}=\left[0.15, 0.85\right]" in diagram.hmm.equations
    assert r"\mu=\left[-1.5, 1.5\right]" in diagram.hmm.equations
    assert r"\mathrm{states:}\ 0,\ldots,1" in bound.math().equations[-1]


def test_hmm_summary_does_not_claim_time_varying_parameters_are_constant():
    from examples.diagram_demo import hidden_markov_model
    bound = hidden_markov_model(4)
    by_name = {node.name: node for node in bound.graph.nodes}
    by_name["y[2]"].distribution = Normal(by_name["y[2]"].distribution.mean, .8)
    assert ModelDiagram(bound).hmm is None
    assert len(bound.math().equations) == 8


def test_hmm_summary_respects_explicit_site_labels():
    from examples.diagram_demo import hidden_markov_model
    mathematics = hidden_markov_model(4).math(labels={"z[0]": "s"})
    assert mathematics.equations[0].startswith(r"s\sim")
