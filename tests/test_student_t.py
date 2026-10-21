"""Student-t distributions preserve their density and scalar/vector meaning."""

import math

import numpy as np
import pytest

from smallprob import HalfNormal, ModelError, Normal, StudentT, UnsupportedModelError, draw, model
from smallprob.core import Expr


@model
def robust_regression(x, df=4, noise=0.8):
    b = Normal(0.5, 2)
    m = Normal(-0.2, 1.5)
    y = StudentT(df, b + m * x, noise)


@model
def robust_regression_loop(x, df=4, noise=0.8):
    b = Normal(0.5, 2)
    m = Normal(-0.2, 1.5)
    y = [StudentT(df, b + m * xi, noise) for xi in x]


@model
def t_values(df, loc=0, scale=1):
    y = StudentT(df, loc, scale)


@model
def t_parameters():
    nu = HalfNormal(3)
    mu = Normal(0, 2)
    sigma = HalfNormal(1)
    y = StudentT(nu, mu, sigma)


@model
def explicit_t(df, x):
    y = draw(StudentT(df, x, 0.5))


def test_cauchy_density_and_location_semantics():
    distribution = StudentT(1, loc=3, scale=2)
    for value in [-40, 0, 3, 8, 100]:
        reference = -math.log(math.pi * 2) - math.log1p(((value - 3) / 2) ** 2)
        assert distribution.log_prob(value, {}) == pytest.approx(reference, abs=2e-14)
    assert not hasattr(distribution, "mean")
    assert not hasattr(distribution, "sd")
    assert "mean exists only" in StudentT.__doc__


@pytest.mark.parametrize("df", [0.2, 1.0, 2.0, 4.0, 25.0, 500.0])
def test_density_matches_independent_torch_distribution(df):
    torch = pytest.importorskip("torch")
    values = np.array([-50, -4, 0, 1.5, 3, 40], dtype=float)
    distribution = StudentT(df, 1.5, 0.7)
    reference = torch.distributions.StudentT(
        torch.tensor(df, dtype=torch.float64),
        torch.tensor(1.5, dtype=torch.float64),
        torch.tensor(0.7, dtype=torch.float64),
    ).log_prob(torch.tensor(values, dtype=torch.float64)).numpy()
    actual = [distribution.log_prob(value, {}) for value in values]
    # Independent lgamma implementations differ by a few ulps at large df.
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_extreme_finite_values_have_finite_heavy_tail_density():
    assert np.isfinite(StudentT(4, -1e308, 1e-300).log_prob(1e308, {}))
    assert np.isfinite(StudentT(np.nextafter(0.0, 1.0)).log_prob(0, {}))
    normal = Normal(1.5, 0.7)
    distribution = StudentT(1e100, 1.5, 0.7)
    for value in [-5, 1.5, 6]:
        assert distribution.log_prob(value, {}) == pytest.approx(normal.log_prob(value, {}), abs=5e-12)


@pytest.mark.parametrize("df,loc,scale,label", [
    (0, 0, 1, "df"), (-1, 0, 1, "df"), (float("inf"), 0, 1, "df"),
    (True, 0, 1, "df"), (4, float("nan"), 1, "loc"), (4, 0, 0, "scale"),
    (4, 0, -1, "scale"), (4, 0, float("inf"), "scale"),
])
def test_invalid_fixed_parameters_fail_before_sampling(df, loc, scale, label):
    with pytest.raises(ModelError, match=f"StudentT {label}"):
        StudentT(df, loc, scale)


def test_symbolic_parameters_validate_at_evaluation():
    distribution = StudentT(Expr("ref", ("nu",)), 0, Expr("ref", ("sigma",)))
    assert distribution.dependencies() == {"nu", "sigma"}
    with pytest.raises(ModelError, match="StudentT df"):
        distribution.log_prob(0, {"nu": -1, "sigma": 1})
    with pytest.raises(ModelError, match="StudentT scale"):
        distribution.sample({"nu": 4, "sigma": 0}, np.random.default_rng(1))


def test_symbolic_df_location_scale_are_graph_parents():
    graph = t_parameters().graph
    assert graph.nodes[-1].distribution.dependencies() == {"nu", "mu", "sigma"}
    samples = t_parameters().simulate(draws=5, seed=2)
    assert np.isfinite(samples["y"]).all()


@pytest.mark.parametrize("x", [[-2, 0, 1], (-2, 0, 1), np.array([-2, 0, 1])])
def test_vector_and_comprehension_have_identical_probability_law(x):
    vector, loop = robust_regression(x), robust_regression_loop(x)
    assert vector.graph.groups == loop.graph.groups
    assert [node.name for node in vector.graph.nodes] == [node.name for node in loop.graph.nodes]
    a, b = vector.simulate(draws=20, seed=12), loop.simulate(draws=20, seed=12)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    values = {"b": 1.1, "m": -0.3, "y": [1.0, -0.4, 2.0]}
    assert vector.log_prob(values) == loop.log_prob(values)


def test_three_parameter_broadcasting_preserves_each_row():
    bound = t_values([2, 4, 7], [10], [0.5, 1, 2])
    assert [node.distribution._parameters({}) for node in bound.graph.nodes] == [
        (2, 10, 0.5), (4, 10, 1), (7, 10, 2),
    ]
    assert bound.simulate(draws=4, seed=12)["y"].shape == (4, 3)
    assert t_values([2], 0, 1).simulate(seed=1)["y"].shape == (1, 1)
    assert t_values([], [0], 1).simulate(draws=3, seed=1)["y"].shape == (3, 0)


def test_broadcast_parameter_mismatch_and_nested_dimensions_are_rejected():
    with pytest.raises(ModelError, match="incompatible vector lengths"):
        t_values([2, 3], [1, 2, 3], 1)
    with pytest.raises(ModelError, match="one-dimensional"):
        t_values(4, np.ones((2, 2)), 1)
    with pytest.raises(ModelError, match="StudentT df"):
        t_values([4, 0], [0, 1], 1)


def test_explicit_draw_uses_same_broadcast_path():
    a = explicit_t(4, [0, 1]).simulate(draws=3, seed=4)
    b = t_values(4, [0, 1], 0.5).simulate(draws=3, seed=4)
    np.testing.assert_array_equal(a["y"], b["y"])


def test_sampling_matches_numpy_seeded_reference():
    distribution = StudentT(4, 2, 0.3)
    actual_rng = np.random.default_rng(100)
    reference_rng = np.random.default_rng(100)
    actual = [distribution.sample({}, actual_rng) for _ in range(8)]
    expected = 2 + 0.3 * reference_rng.standard_t(4, size=8)
    np.testing.assert_array_equal(actual, expected)


def test_exact_inference_rejects_student_t_explicitly():
    with pytest.raises(UnsupportedModelError, match="StudentT|Student.t"):
        robust_regression([0, 1, 2]).infer(y=[1, 2, 3])


def test_direct_constructor_shadowing_stays_guarded():
    with pytest.raises(UnsupportedModelError, match="shadow"):
        @model
        def bad(StudentT):
            y = StudentT(3)
