"""Shorthand is the same graph as explicit draw, with the same boundaries."""

import math

import numpy as np
import pytest

from smallprob import Categorical, HalfNormal, ModelError, Normal, UnsupportedModelError, draw, model
from smallprob.core import Expr, Graph, Node


@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = [Normal(intercept + slope * xi, sigma) for xi in x]


@model
def explicit_regression(x):
    intercept = draw(Normal(0, 5))
    slope = draw(Normal(0, 5))
    sigma = draw(HalfNormal(1))
    y = [draw(Normal(intercept + slope * xi, sigma)) for xi in x]


@model
def known_noise(x):
    intercept: float = Normal(0, 5)
    slope = Normal(0, 5)
    y = [Normal(intercept + slope * xi, 0.5) for xi in x]


@model
def shorthand_hmm(n):
    initial = [0.4, 0.6]
    transition = [[0.8, 0.2], [0.3, 0.7]]
    means = [-2, 2]
    z = [Categorical(initial)]
    for t in range(1, n):
        z.append(Categorical(transition[z[t - 1]]))
    y = [Normal(means[z[t]], 0.5) for t in range(n)]


@model
def eager_declarations():
    a, b = Normal(0, 1), HalfNormal(2)
    y = [Normal(a, b), draw(Normal(a, b))]
    z = (Normal(a, b), Normal(a, b))


@model
def append_declarations(n):
    y = []
    for _ in range(n):
        y.append(Normal())


@model
def repeated_declaration(n):
    for _ in range(n):
        x = Normal()


@model
def guarded_series(action):
    y = [Normal(0, 1), Normal(2, 1)]
    alias = y
    if action == "reverse":
        alias.reverse()
    elif action == "replace":
        alias[0] = alias[1]
    elif action == "rebind":
        y = y[::-1]
    elif action == "extend":
        alias.extend(alias)


@model
def rebound_tuple():
    y = (Normal(0, 1), Normal(2, 1))
    y = y[::-1]


@model
def rebound_scalar():
    x = Normal()
    x = x + 1


@model
def random_condition():
    x = Normal()
    if x is None:
        y = Normal()


def test_unknown_noise_model_matches_explicit_draw_without_return():
    shorthand = regression([-1, 0, 1])
    explicit = explicit_regression([-1, 0, 1])
    assert shorthand.graph.groups == explicit.graph.groups
    assert [node.name for node in shorthand.graph.nodes] == [
        "intercept", "slope", "sigma", "y[0]", "y[1]", "y[2]"
    ]
    first, second = shorthand.simulate(draws=30, seed=4), explicit.simulate(draws=30, seed=4)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert np.all(first["sigma"] >= 0)
    assert shorthand.log_prob({name: values[0] for name, values in first.items()}) == pytest.approx(
        explicit.log_prob({name: values[0] for name, values in second.items()})
    )


def test_known_noise_shorthand_supports_existing_exact_inference():
    posterior = known_noise([-1, 0, 1]).infer({"y": [-1, 1, 3]})
    assert posterior.method == "exact-gaussian"
    assert posterior.mean("slope") == pytest.approx(2 / 1.005)
    assert posterior.sd("intercept") > 0


def test_hmm_shorthand_keeps_symbolic_indexing_and_exact_engine():
    m = shorthand_hmm(4)
    samples = m.simulate(draws=3, seed=2)
    assert samples["z"].shape == (3, 4)
    posterior = m.infer({"y": [-2, -2, 2, 2]})
    assert posterior.method == "exact-hmm"
    np.testing.assert_allclose(posterior.probabilities("z").sum(axis=1), 1)


def test_tuple_unpacking_and_mixed_explicit_shorthand_series():
    m = eager_declarations()
    assert m.graph.groups == {"a": ["a"], "b": ["b"], "y": ["y[0]", "y[1]"], "z": ["z[0]", "z[1]"]}
    assert len(m.graph.nodes) == 6  # Explicit draw's constructor is not drawn twice.
    assert m.simulate(seed=0)["y"].shape == (1, 2)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_comprehension_and_append_keep_empty_and_singleton_series(n):
    assert regression(range(n)).simulate(draws=2, seed=1)["y"].shape == (2, n)
    assert append_declarations(n).simulate(draws=2, seed=1)["y"].shape == (2, n)


def test_shorthand_keeps_repeated_scalar_guard():
    with pytest.raises(ModelError, match="declared twice"):
        repeated_declaration(2)


@pytest.mark.parametrize("action", ["reverse", "replace", "rebind", "extend"])
def test_shorthand_keeps_random_series_mutation_guards(action):
    with pytest.raises(UnsupportedModelError, match="indices are observation names"):
        guarded_series(action)


def test_tuple_and_scalar_rebinding_remain_unsafe():
    with pytest.raises(UnsupportedModelError, match="cannot be rebound"):
        rebound_tuple()
    with pytest.raises(UnsupportedModelError, match="cannot be rebound"):
        rebound_scalar()


def test_shorthand_keeps_random_control_flow_guard():
    with pytest.raises(UnsupportedModelError, match="random variables"):
        random_condition()


def test_constructor_parameter_shadowing_is_rejected():
    with pytest.raises(UnsupportedModelError, match="shadowed"):
        @model
        def shadowed(Normal):
            x = Normal()


def test_constructor_local_assignment_shadowing_is_rejected():
    with pytest.raises(UnsupportedModelError, match="shadowed"):
        @model
        def shadowed():
            Normal = float
            x = Normal(1)


def test_constructor_comprehension_shadowing_is_rejected():
    with pytest.raises(UnsupportedModelError, match="shadowed"):
        @model
        def shadowed():
            y = [Normal() for Normal in [float]]


def test_unrelated_same_named_callable_is_never_rewritten_as_a_draw():
    Normal = float
    with pytest.raises(UnsupportedModelError, match="shadowed"):
        @model
        def ordinary_function():
            x = Normal(1)


def test_constructor_closure_rebinding_is_checked_at_model_execution():
    from smallprob import Normal

    @model
    def reusable():
        x = Normal()

    assert len(reusable().graph.nodes) == 1
    Normal = float
    with pytest.raises(UnsupportedModelError, match="rebound"):
        reusable()


def test_explicit_draw_still_accepts_reusable_distributions_and_aliases():
    distribution = Normal(2, 3)
    Gaussian = Normal

    @model
    def reusable():
        x = draw(distribution)
        y = draw(Gaussian(x, 1))

    assert len(reusable().graph.nodes) == 2
    assert isinstance(distribution, Normal)


def test_ambiguous_bare_constructor_is_not_silently_discarded():
    with pytest.raises(UnsupportedModelError, match="Cannot infer"):
        @model
        def unnamed():
            return Normal()
    with pytest.raises(UnsupportedModelError, match="Cannot infer"):
        @model
        def chained():
            x = y = Normal()


def test_lazy_shorthand_is_rejected_before_it_can_escape_model_building():
    with pytest.raises(UnsupportedModelError, match="eager list comprehension"):
        @model
        def lazy():
            y = (Normal() for _ in range(2))


def test_halfnormal_density_and_support_match_folded_normal():
    half = HalfNormal(2)
    normal = Normal(0, 2)
    assert not isinstance(half, Normal)
    assert half.log_prob(-1, {}) == -math.inf
    for value in [0, 0.2, 1, 5]:
        assert half.log_prob(value, {}) == pytest.approx(math.log(2) + normal.log_prob(value, {}))
    rng = np.random.default_rng(6)
    values = np.array([half.sample({}, rng) for _ in range(20_000)])
    assert np.all(values >= 0)
    assert values.mean() == pytest.approx(2 * math.sqrt(2 / math.pi), abs=0.03)
    assert values.var() == pytest.approx(4 * (1 - 2 / math.pi), abs=0.06)


@pytest.mark.parametrize("value", [0, -1, True, complex(1, 0), float("nan"), float("inf"), [1]])
def test_halfnormal_rejects_invalid_scales(value):
    with pytest.raises(ModelError, match="HalfNormal sd"):
        HalfNormal(value)


def test_halfnormal_symbolic_scale_is_validated_when_resolved():
    half = HalfNormal(Expr("ref", ("scale",)))
    assert half.dependencies() == {"scale"}
    assert half.log_prob(1, {"scale": 2}) == pytest.approx(HalfNormal(2).log_prob(1, {}))
    with pytest.raises(ModelError, match="strictly positive"):
        half.sample({"scale": -1}, np.random.default_rng(0))
    graph = Graph([Node("scale", HalfNormal()), Node("y", half)],
                  {"scale": ["scale"], "y": ["y"]}, set())
    assert len(graph.nodes) == 2


def test_halfnormal_fixed_negative_values_cannot_be_simulated():
    with pytest.raises(ModelError, match="outside the distribution's support"):
        regression([0]).simulate(fixed={"sigma": -1}, seed=0)
