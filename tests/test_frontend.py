"""Naming and Python semantics at the boundary of the restricted frontend."""

import numpy as np
import pytest

from smallprob import Categorical, ModelError, Normal, UnsupportedModelError, draw, model
from smallprob.core import Expr, as_expr


@model
def empty_comprehension(n):
    mu = draw(Normal(2.0, 3.0))
    y = [draw(Normal(mu, 1.0)) for _ in range(n)]
    return y


@model
def empty_append(n):
    y = []
    for _ in range(n):
        y.append(draw(Normal()))
    return y


@model
def tuple_initialised_series(n):
    x, y = [], []
    for _ in range(n):
        x.append(draw(Normal()))
        y.append(draw(Normal(x[-1], 0.5)))
    return y


@model
def explicit_names(n):
    return [draw(Normal(), name=f"measurement[{i}]") for i in range(n)]


@model
def tuple_lookup(table):
    z = draw(Categorical([0.5, 0.5]))
    locations = table[:, 0]
    y = draw(Normal(locations[z], table[z, 1]))
    return y


@model
def identity_branch():
    x = draw(Normal())
    if x is None:
        y = draw(Normal(1, 1))
    else:
        y = draw(Normal(2, 1))
    return y


@model
def membership_branch():
    x = draw(Normal())
    if x in []:
        y = draw(Normal())
    return x


@model
def fixed_branch(x):
    if 0 < x < 1 / x:
        y = draw(Normal(1, 1))
    else:
        y = draw(Normal(2, 1))
    return y


@model
def repeated_scalar(n):
    for _ in range(n):
        x = draw(Normal())
    return None


@model
def reordered_series():
    y = [draw(Normal(0, 1)), draw(Normal(10, 1))]
    y.reverse()
    z = draw(Normal(y[0], 1))
    return z


@model
def aliased_series_mutation(action):
    y = [draw(Normal(0, 1)), draw(Normal(10, 1))]
    alias = y
    if action == "replace":
        alias[0] = alias[1]
    elif action == "delete":
        del alias[0]
    elif action == "pop":
        alias.pop()
    elif action == "extend":
        alias.extend(alias)
    return y


@model
def rebound_series():
    y = [draw(Normal(0, 1)), draw(Normal(10, 1))]
    y = y[::-1]
    z = draw(Normal(y[0], 1))
    return z


@model
def separate_derived_series():
    y = [draw(Normal(0, 1)), draw(Normal(10, 1))]
    reversed_y = y[::-1]
    z = draw(Normal(reversed_y[0], 1))
    return z


@model
def list_method_mutation(alias=False):
    y = [draw(Normal(0, 1)), draw(Normal(10, 1))]
    if alias:
        reverse = list.reverse
        reverse(y)
    else:
        list.reverse(y)
    return y


@model
def dynamically_named_series(action=None, n=2):
    y = [draw(Normal(10 * i, 1), name=f"y[{i}]") for i in range(n)]
    if action == "reverse":
        y.reverse()
    elif action == "rebind":
        y = y[::-1]
    z = draw(Normal(y[0], 1))
    return z


def test_empty_comprehension_is_still_a_series():
    m = empty_comprehension(0)
    assert m.graph.groups["y"] == []
    assert "y" in m.graph.series
    posterior = m.infer(observed={"y": []})
    assert posterior.mean("mu") == pytest.approx(2.0)
    assert posterior.mean("y").shape == (0,)
    assert m.simulate(draws=2, seed=0)["y"].shape == (2, 0)
    assert posterior.sample(draws=3, seed=0)["y"].shape == (3, 0)


def test_empty_append_is_still_a_series():
    m = empty_append(0)
    assert m.graph.groups["y"] == []
    assert m.simulate(draws=2, seed=0)["y"].shape == (2, 0)


def test_destructured_list_initialisation_retains_series_tracking():
    m = tuple_initialised_series(2)
    assert m.simulate(draws=2, seed=0)["y"].shape == (2, 2)


def test_explicit_indexed_names_pack_as_a_series():
    m = explicit_names(2)
    assert [node.name for node in m.graph.nodes] == ["measurement[0]", "measurement[1]"]
    assert m.simulate(seed=0)["measurement"].shape == (1, 2)


def test_numpy_tuple_index_and_slice_work_with_symbolic_index():
    m = tuple_lookup(np.array([[-1.0, 0.5], [2.0, 1.0]]))
    assert m.graph.nodes[1].distribution.mean.eval({"z": 1}) == 2.0
    assert m.graph.nodes[1].distribution.sd.eval({"z": 0}) == 0.5
    assert m.simulate(draws=2, seed=2)["y"].shape == (2,)


def test_model_constants_snapshot_mutable_inputs():
    array = np.array([[-1.0, 0.5], [2.0, 1.0]])
    m = tuple_lookup(array)
    expected = m.simulate(draws=3, seed=2)
    array[:] = 999
    actual = m.simulate(draws=3, seed=2)
    np.testing.assert_array_equal(actual["y"], expected["y"])
    nested = [[0.8, 0.2], [0.3, 0.7]]
    expression = as_expr(nested)
    nested[0][0] = 0.1
    assert expression.eval({})[0][0] == 0.8


def test_closure_inputs_are_late_bound_but_each_model_is_frozen():
    location = 2.0

    @model
    def local(noise=0.5):
        y = draw(Normal(location, noise))
        return y

    old_model = local()
    location = 7.0
    new_model = local()
    assert old_model.graph.nodes[0].distribution.mean.eval({}) == 2.0
    assert new_model.graph.nodes[0].distribution.mean.eval({}) == 7.0
    assert new_model.graph.nodes[0].distribution.sd.eval({}) == 0.5


def test_unreadable_source_is_rejected_with_actionable_error():
    environment = {"draw": draw, "Normal": Normal}
    exec("def dynamic():\n    x = draw(Normal())\n    return x\n", environment)
    with pytest.raises(UnsupportedModelError, match="readable Python source"):
        model(environment["dynamic"])


def test_alias_and_reassignment_of_draw_are_rejected():
    with pytest.raises(UnsupportedModelError, match="aliases"):
        @model
        def alias():
            sample = draw
            return sample(Normal())


def test_random_identity_and_membership_cannot_silently_choose_branches():
    with pytest.raises(UnsupportedModelError, match="random variables"):
        identity_branch()
    with pytest.raises(UnsupportedModelError, match="random variables"):
        membership_branch()


def test_fixed_comparison_short_circuits_normally():
    assert fixed_branch(0).graph.nodes[0].distribution.mean.eval({}) == 2.0


def test_repeated_scalar_names_are_not_silently_turned_into_series():
    with pytest.raises(ModelError, match="declared twice"):
        repeated_scalar(2)


def test_series_reordering_cannot_change_the_meaning_of_observation_names():
    with pytest.raises(UnsupportedModelError, match="indices are observation names"):
        reordered_series()


@pytest.mark.parametrize("action", ["replace", "delete", "pop", "extend"])
def test_mutating_series_through_an_alias_is_rejected(action):
    with pytest.raises(UnsupportedModelError, match="indices are observation names"):
        aliased_series_mutation(action)


def test_rebinding_series_cannot_change_the_meaning_of_observation_names():
    with pytest.raises(UnsupportedModelError, match="cannot be rebound"):
        rebound_series()
    m = separate_derived_series()
    assert m.graph.nodes[-1].distribution.mean.dependencies() == {"y[1]"}


@pytest.mark.parametrize("alias", [False, True])
def test_builtin_list_mutators_and_method_aliases_cannot_bypass_series_guards(alias):
    with pytest.raises(UnsupportedModelError, match="indices are observation names"):
        list_method_mutation(alias)


@pytest.mark.parametrize("action", ["reverse", "rebind"])
def test_dynamic_explicit_names_cannot_bypass_series_guards(action):
    with pytest.raises(UnsupportedModelError, match="indices are observation names"):
        dynamically_named_series(action)


def test_dynamic_explicit_names_are_supported_when_they_form_one_stable_series():
    m = dynamically_named_series()
    assert m.graph.nodes[-1].distribution.mean.dependencies() == {"y[0]"}
    assert m.simulate(seed=0)["y"].shape == (1, 2)


def test_empty_dynamic_name_series_reports_unresolvable_group():
    with pytest.raises(UnsupportedModelError, match="empty series with dynamic explicit names"):
        dynamically_named_series(n=0)


@pytest.mark.parametrize("value", [True, complex(1, 0), float("nan"), float("inf")])
def test_normal_parameters_reject_nonfinite_or_nonnumeric_scalars(value):
    with pytest.raises(ModelError):
        Normal(mean=value)
    with pytest.raises(ModelError):
        Normal(sd=value)


def test_array_scalar_arithmetic_remains_symbolic():
    x = Expr("ref", ("x",))
    value = np.float64(2) * x + np.float64(1)
    assert value.eval({"x": 3.0}) == 7.0
