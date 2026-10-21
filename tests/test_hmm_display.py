"""A compact HMM picture must retain the full chain's mathematics."""

import numpy as np
import pytest

from smallprob import Normal
from smallprob.core import Categorical, lookup, Expr
from smallprob.hmm import HMM


def chain(steps=30, sd=(.3, .8), states=("quiet", "active")):
    return HMM(initial=[.6, .4], transition=[[.9, .1], [.2, .8]],
               emissions=Normal([-1., 2.], sd), steps=steps, states=states)


@pytest.mark.parametrize("steps", [2, 4, 30])
def test_chain_summary_uses_full_horizon_and_all_fixed_parameters(steps):
    bound = chain(steps)
    diagram = bound.diagram(observed="y")
    assert diagram.hmm is not None
    assert len(diagram.hmm.states) == len(diagram.hmm.emissions) == steps
    assert diagram.hmm.state_labels == ("quiet", "active")
    assert r"\pi=\left[0.6, 0.4\right]" in diagram.hmm.equations
    assert r"A_{1,:}=\left[0.2, 0.8\right]" in diagram.hmm.equations
    assert r"\mu=\left[-1, 2\right]" in diagram.hmm.equations
    assert r"\sigma=\left[0.3, 0.8\right]" in diagram.hmm.equations
    assert r"\mathcal{N}(\mu_{z_t},\;\sigma_{z_t}^2)" in diagram.hmm.equations[2]
    assert f"t=1,\\ldots,{steps}" in bound.math().equations[-1]
    assert rf"\prod_{{t=2}}^{{{steps}}}" in bound.math().factorization
    # Display compression cannot introduce a shortcut transition in the graph.
    assert set(diagram.edges) == {(f"z[{i}]", f"y[{i}]") for i in range(steps)} | {
        (f"z[{i - 1}]", f"z[{i}]") for i in range(1, steps)}


def test_shared_scale_and_custom_group_labels_have_correct_math():
    bound = chain(sd=.45, states=None)
    mathematics = bound.math(labels={"z": "s", "y": "x"})
    assert r"x_t\mid s_t\sim \mathcal{N}(\mu_{s_t},\;\sigma^2)" in mathematics.equations
    assert r"\sigma=0.45" in mathematics.equations
    assert "p(s_{1:30},x_{1:30})" in mathematics.factorization
    assert bound.diagram().hmm.state_labels is None


def test_time_varying_state_scales_do_not_get_false_homogeneous_summary():
    bound = chain(4)
    node = next(node for node in bound.graph.nodes if node.name == "y[2]")
    node.distribution = Normal(node.distribution.mean,
                               lookup([.3, 1.2], Expr("ref", ("z[2]",))))
    assert bound.diagram().hmm is None
    assert len(bound.math().equations) == 8


def test_nonadjacent_transition_does_not_get_chain_summary():
    bound = chain(4)
    node = next(node for node in bound.graph.nodes if node.name == "z[3]")
    node.distribution = Categorical(lookup([[.9, .1], [.2, .8]],
                                          Expr("ref", ("z[1]",))))
    assert bound.diagram().hmm is None


def test_more_than_four_categories_preserve_rows_and_names(tmp_path):
    pytest.importorskip("matplotlib")
    count = 5
    bound = HMM(initial=np.ones(count) / count,
                transition=np.eye(count) * .8 + np.ones((count, count)) * .04,
                emissions=Normal(np.arange(count), np.linspace(.2, 1, count)),
                steps=30, states=[f"level {index}" for index in range(count)])
    diagram = bound.diagram(observed="y")
    assert diagram.hmm.category_count == count
    assert any(equation.startswith("A_{4,:}=") for equation in diagram.hmm.equations)
    svg = diagram.save(tmp_path / "five-states.svg").read_text()
    assert "state 4 = level 4" in svg


@pytest.mark.parametrize("steps", [2, 4, 30])
def test_render_small_and_long_chains_with_explicit_elision(steps, tmp_path):
    pytest.importorskip("matplotlib")
    diagram = chain(steps).diagram(observed="y")
    figure = diagram._figure()
    try:
        graph_text = [text.get_text() for text in figure.axes[0].texts]
        assert f"$z_{{{steps}}}$" in graph_text
        assert f"$y_{{{steps}}}$" in graph_text
        if steps == 30:
            assert "26 intervening time steps omitted" in graph_text
            assert graph_text.count("⋯") == 2
            assert "$z_{4}$" not in graph_text
        else:
            assert "⋯" not in graph_text
        assert any("0 = quiet" in text.get_text() and "1 = active" in text.get_text()
                   for text in figure.texts)
    finally:
        figure.clear()
    assert diagram.save(tmp_path / f"hmm-{steps}.png").read_bytes().startswith(b"\x89PNG")
    svg = diagram.save(tmp_path / f"hmm-{steps}.svg").read_text()
    assert f"{steps} time steps" in svg
    assert "state 0 = quiet" in svg
