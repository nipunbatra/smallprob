"""Broadcast declarations must retain the scalar model's probability law."""

from itertools import product

import numpy as np
import pytest

from smallprob import Categorical, HalfNormal, ModelError, Normal, UnsupportedModelError, draw, model


@model
def vector_regression(x, noise=0.7):
    intercept = Normal(0.5, 2)
    slope = Normal(-0.2, 1.5)
    y = Normal(intercept + slope * x, noise)


@model
def loop_regression(x, noise=0.7):
    intercept = Normal(0.5, 2)
    slope = Normal(-0.2, 1.5)
    y = [Normal(intercept + slope * xi, noise) for xi in x]


@model
def vector_unknown_noise(x):
    intercept = Normal(0.5, 2)
    slope = Normal(-0.2, 1.5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)


@model
def loop_unknown_noise(x):
    intercept = Normal(0.5, 2)
    slope = Normal(-0.2, 1.5)
    sigma = HalfNormal(1)
    y = [Normal(intercept + slope * xi, sigma) for xi in x]


@model
def vector_normal(location, scale):
    y = Normal(location, scale)


@model
def vector_latent_scale(location, scales):
    sigma = HalfNormal(1)
    y = Normal(location, sigma * scales)


@model
def vector_chain(x):
    mu = Normal(0, 2)
    y = Normal(mu + x, 0.6)
    z = Normal(2 * y + 1, 0.4)


@model
def loop_chain(x):
    mu = Normal(0, 2)
    y = [Normal(mu + xi, 0.6) for xi in x]
    z = [Normal(2 * yi + 1, 0.4) for yi in y]


@model
def vector_selected_chain(x):
    y = Normal(x, 0.6)
    z = Normal(y[1:] + y[0], 0.4)


@model
def scalar_indexed_expression(x):
    slope = Normal()
    y = Normal((slope * x)[0], 1)


@model
def scalar_expression_baseline(x):
    slope = Normal()
    y = Normal(slope * x[0], 1)


@model
def scalar_indexed_chain(x):
    y = Normal(x, 0.6)
    z = Normal((2 * y)[0], 1)


@model
def scalar_chain_baseline(x):
    y = [Normal(xi, 0.6) for xi in x]
    z = Normal(2 * y[0], 1)


@model
def guarded_vector(action):
    y = Normal([0, 1, 2], 1)
    alias = y
    if action == "reverse":
        alias.reverse()
    elif action == "replace":
        alias[0] = alias[1]
    elif action == "pop":
        alias.pop()
    elif action == "extend":
        alias.extend(alias)
    elif action == "builtin":
        list.reverse(alias)
    elif action == "rebind":
        y = y[::-1]
    elif action == "transform":
        y = y + 1


@model
def explicit_vector(x):
    intercept = draw(Normal(0.5, 2))
    slope = draw(Normal(-0.2, 1.5))
    y = draw(Normal(intercept + slope * x, 0.7))


@model
def vector_categorical():
    state = Categorical([0.3, 0.7])


@model
def existing_hmm(n):
    transition = [[0.8, 0.2], [0.25, 0.75]]
    means = [-1.0, 1.5]
    z = [Categorical([0.3, 0.7])]
    for t in range(1, n):
        z.append(Categorical(transition[z[t - 1]]))
    y = [Normal(means[state], 0.6) for state in z]


def assert_same_samples(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])


def assert_same_exact_posterior(left, right):
    assert left.method == right.method == "exact-gaussian"
    assert left.log_evidence == pytest.approx(right.log_evidence, abs=1e-12)
    for name in left.graph.groups:
        np.testing.assert_allclose(left.mean(name), right.mean(name), atol=1e-12)
        np.testing.assert_allclose(left.sd(name), right.sd(name), atol=1e-12)
    # Identical joint draws verify off-diagonal covariance, not only marginals.
    assert_same_samples(left.sample(draws=200, seed=17), right.sample(draws=200, seed=17))


@pytest.mark.parametrize("container", [list, tuple, np.array], ids=["list", "tuple", "numpy"])
def test_vector_regression_has_the_same_joint_law_as_comprehension(container):
    x = container([-1.5, -0.2, 0.8, 2.0])
    vector, loop = vector_regression(x), loop_regression(x)
    assert vector.graph.groups == loop.graph.groups
    assert vector.graph.series == {"y"}
    assert [node.name for node in vector.graph.nodes] == [
        "intercept", "slope", "y[0]", "y[1]", "y[2]", "y[3]"
    ]
    samples = vector.simulate(draws=20, seed=12)
    assert_same_samples(samples, loop.simulate(draws=20, seed=12))
    values = {name: sample[0] for name, sample in samples.items()}
    assert vector.log_prob(values) == pytest.approx(loop.log_prob(values), abs=1e-12)
    observed = [1.0, 0.4, -0.3, -1.5]
    assert_same_exact_posterior(vector.infer(y=observed), loop.infer(y=observed))


def test_broadcast_regression_matches_independent_gaussian_reference():
    x = np.array([0.4, 1.0, 1.3, 2.5])
    y = np.array([0.2, -0.4, -0.3, -1.2])
    design = np.column_stack([np.ones(len(x)), x])
    prior_mean = np.array([0.5, -0.2])
    prior_covariance = np.diag([4, 2.25])
    precision = np.linalg.inv(prior_covariance)
    covariance = np.linalg.inv(precision + design.T @ design / 0.7**2)
    mean = covariance @ (precision @ prior_mean + design.T @ y / 0.7**2)
    evidence_covariance = design @ prior_covariance @ design.T + 0.7**2 * np.eye(len(x))
    residual = y - design @ prior_mean
    evidence = -0.5 * (
        len(x) * np.log(2 * np.pi) + np.linalg.slogdet(evidence_covariance)[1]
        + residual @ np.linalg.solve(evidence_covariance, residual)
    )
    posterior = vector_regression(x).infer(y=y)
    np.testing.assert_allclose([posterior.mean("intercept"), posterior.mean("slope")], mean, atol=1e-12)
    np.testing.assert_allclose([posterior.sd("intercept"), posterior.sd("slope")], np.sqrt(covariance.diagonal()), atol=1e-12)
    assert posterior.log_evidence == pytest.approx(evidence, abs=1e-12)
    samples = posterior.sample(draws=50_000, seed=71)
    measured_covariance = np.cov([samples["intercept"], samples["slope"]])
    # Gaussian sample covariance has this known sampling variance. Use a
    # scale-aware tolerance rather than treating Monte Carlo error as exact.
    covariance_se = np.sqrt(
        (covariance**2 + np.outer(covariance.diagonal(), covariance.diagonal()))
        / (samples.draws - 1)
    )
    np.testing.assert_array_less(np.abs(measured_covariance - covariance), 6 * covariance_se)


@pytest.mark.parametrize("x", [1.2, np.float64(1.2), np.array(1.2)])
def test_scalar_input_stays_a_scalar_random_variable(x):
    m = vector_regression(x)
    assert "y" not in m.graph.series
    assert m.graph.groups["y"] == ["y"]
    assert m.simulate(draws=3, seed=1)["y"].shape == (3,)
    posterior = m.infer(y=0.3)
    assert np.ndim(posterior.mean("y")) == 0


@pytest.mark.parametrize("x", [[], (), np.array([]), [1.0], (1.0,), np.array([1.0])])
def test_empty_and_singleton_vectors_remain_series(x):
    m = vector_regression(x)
    assert "y" in m.graph.series
    assert m.simulate(draws=3, seed=8)["y"].shape == (3, len(x))
    posterior = m.infer(y=[0.0] * len(x))
    assert posterior.mean("y").shape == (len(x),)
    assert posterior.sample(draws=2, seed=3)["y"].shape == (2, len(x))
    if not len(x):
        assert posterior.mean("intercept") == pytest.approx(0.5)
        assert posterior.sd("slope") == pytest.approx(1.5)
        assert posterior.log_evidence == pytest.approx(0)


@pytest.mark.parametrize("mean,scale", [
    ([0.0, 1.0, 2.0], 0.5),
    (2.0, [0.2, 0.5, 1.5]),
    ([0.0, 1.0, 2.0], [0.2, 0.5, 1.5]),
    ([2.0], [0.2, 0.5, 1.5]),
    ([0.0, 1.0, 2.0], [0.5]),
    ([], [0.5]),
    ([2.0], []),
])
def test_scalar_and_one_dimensional_broadcasting_matches_elementwise_density(mean, scale):
    locations, scales = np.broadcast_arrays(mean, scale)
    m = vector_normal(mean, scale)
    assert m.simulate(draws=4, seed=9)["y"].shape == (4, len(locations))
    values = np.linspace(-0.2, 0.3, len(locations))
    expected = np.sum(-0.5 * np.log(2 * np.pi) - np.log(scales) - 0.5 * ((values - locations) / scales)**2)
    assert m.log_prob({"y": values}) == pytest.approx(expected, abs=1e-12)
    posterior = m.infer()
    np.testing.assert_allclose(posterior.mean("y"), locations)
    np.testing.assert_allclose(posterior.sd("y"), scales)


def test_symbolic_scale_vector_preserves_shared_noise_variable():
    m = vector_latent_scale([0.0, 1.0, 2.0], [0.2, 0.5, 1.5])
    assert m.graph.groups["sigma"] == ["sigma"]
    samples = m.simulate(draws=10, seed=4, fixed={"sigma": 0.7})
    expected = vector_normal([0.0, 1.0, 2.0], [0.14, 0.35, 1.05]).simulate(draws=10, seed=4)
    np.testing.assert_allclose(samples["y"], expected["y"], atol=1e-15)


@pytest.mark.parametrize("mean,scale", [([1, 2], [0.3, 0.5, 0.7]), ([], [0.3, 0.5])])
def test_incompatible_vector_lengths_raise_a_model_error(mean, scale):
    with pytest.raises(ModelError, match="(?i)(shape|broadcast|length)"):
        vector_normal(mean, scale)


@pytest.mark.parametrize("mean,scale", [
    ([[0, 1], [2, 3]], 0.5),
    ([0, 1], [[0.2], [0.5]]),
    (np.ones((1, 2)), 0.5),
    (0.0, np.ones((2, 1))),
])
def test_multidimensional_batching_is_rejected_explicitly(mean, scale):
    with pytest.raises(ModelError, match="(?i)(one.dimensi|1.d|shape|rank|scalar)"):
        vector_normal(mean, scale)


def test_halfnormal_is_still_a_scalar_distribution():
    with pytest.raises(ModelError, match="scalar"):
        HalfNormal([1, 2])


def test_missing_observations_and_indexed_evidence_keep_the_same_meaning():
    x = [-1.5, -0.2, 0.8, 2.0]
    observed = [1.0, None, -0.3, None]
    posterior = vector_regression(x).infer(y=observed)
    assert_same_exact_posterior(posterior, loop_regression(x).infer(y=observed))
    assert_same_exact_posterior(
        posterior, vector_regression(x).infer({"y[0]": 1.0, "y[2]": -0.3})
    )
    assert np.all(posterior.sd("y")[[1, 3]] > 0.7)
    np.testing.assert_array_equal(posterior.sd("y")[[0, 2]], [0, 0])


@pytest.mark.parametrize("container", [list, np.array])
def test_binding_snapshots_mutable_vector_inputs(container):
    x = container([-1.5, -0.2, 0.8, 2.0])
    noise = container([0.4, 0.5, 0.6, 0.7])
    m = vector_regression(x, noise)
    expected = m.simulate(draws=5, seed=3)
    expected_evidence = m.infer(y=[1.0, 0.4, -0.3, -1.5]).log_evidence
    x[:] = [99.0] * 4
    noise[:] = [99.0] * 4
    assert_same_samples(m.simulate(draws=5, seed=3), expected)
    assert m.infer(y=[1.0, 0.4, -0.3, -1.5]).log_evidence == expected_evidence


def test_vector_random_variables_can_feed_another_vector_declaration():
    vector, loop = vector_chain([-1.0, 0.0, 2.0]), loop_chain([-1.0, 0.0, 2.0])
    assert vector.graph.groups == loop.graph.groups
    assert_same_samples(vector.simulate(draws=10, seed=2), loop.simulate(draws=10, seed=2))
    assert_same_exact_posterior(vector.infer(z=[-0.3, 1.5, 3.2]), loop.infer(z=[-0.3, 1.5, 3.2]))


def test_fixed_index_and_slice_keep_the_correct_parents_in_a_vector_chain():
    m = vector_selected_chain([0.0, 1.0, 2.0])
    posterior = m.infer(y=[-1.0, 0.5, 2.0])
    np.testing.assert_allclose(posterior.mean("z"), [-0.5, 1.0])
    np.testing.assert_allclose(posterior.sd("z"), [0.4, 0.4])


@pytest.mark.parametrize("container", [list, np.array], ids=["list", "numpy"])
def test_indexing_a_vector_expression_back_to_scalar_preserves_its_law(container):
    x = container([1.5, 3.0])
    indexed, baseline = scalar_indexed_expression(x), scalar_expression_baseline(x)
    assert indexed.graph.groups == {"slope": ["slope"], "y": ["y"]}
    assert "y" not in indexed.graph.series
    assert_same_samples(indexed.simulate(draws=10, seed=4), baseline.simulate(draws=10, seed=4))
    assert_same_exact_posterior(indexed.infer(y=0.8), baseline.infer(y=0.8))


def test_indexing_transformed_random_vector_back_to_scalar_preserves_its_law():
    indexed, baseline = scalar_indexed_chain([0.5, 2.0]), scalar_chain_baseline([0.5, 2.0])
    assert indexed.graph.groups == {"y": ["y[0]", "y[1]"], "z": ["z"]}
    assert indexed.graph.series == {"y"}
    assert_same_samples(indexed.simulate(draws=10, seed=4), baseline.simulate(draws=10, seed=4))
    assert_same_exact_posterior(indexed.infer(z=1.3), baseline.infer(z=1.3))


@pytest.mark.parametrize("action", ["reverse", "replace", "pop", "extend", "builtin", "rebind", "transform"])
def test_vector_declarations_keep_stable_observation_names(action):
    with pytest.raises(UnsupportedModelError, match="(?i)(indices are observation names|cannot be rebound)"):
        guarded_vector(action)


def test_explicit_draw_uses_the_same_vector_semantics():
    x = [-1.0, 0.0, 2.0]
    explicit, simple = explicit_vector(x), vector_regression(x)
    assert explicit.graph.groups == simple.graph.groups
    assert_same_samples(explicit.simulate(draws=8, seed=11), simple.simulate(draws=8, seed=11))
    assert_same_exact_posterior(explicit.infer(y=[1.5, 0.2, -0.5]), simple.infer(y=[1.5, 0.2, -0.5]))


def test_categorical_probabilities_remain_one_distribution_and_hmm_stays_exact():
    categorical = vector_categorical()
    assert categorical.graph.groups == {"state": ["state"]}
    assert categorical.simulate(draws=4, seed=8)["state"].shape == (4,)
    observed = [-0.4, 0.7, 1.1]
    posterior = existing_hmm(3).infer(y=observed)
    initial, transition, means = np.array([0.3, 0.7]), np.array([[0.8, 0.2], [0.25, 0.75]]), np.array([-1.0, 1.5])
    paths = np.array(list(product(range(2), repeat=3)))
    weights = np.array([
        initial[path[0]] * transition[path[0], path[1]] * transition[path[1], path[2]]
        * np.prod(np.exp(-0.5 * ((np.asarray(observed) - means[path]) / 0.6)**2) / (0.6 * np.sqrt(2 * np.pi)))
        for path in paths
    ])
    expected = np.array([[weights[paths[:, t] == state].sum() / weights.sum() for state in range(2)] for t in range(3)])
    assert posterior.method == "exact-hmm"
    np.testing.assert_allclose(posterior.probabilities("z"), expected, atol=1e-12)
    assert posterior.log_evidence == pytest.approx(np.log(weights.sum()), abs=1e-12)


@pytest.mark.parametrize("vector,loop", [
    (vector_regression, loop_regression),
    (vector_unknown_noise, loop_unknown_noise),
])
def test_vi_learns_the_identical_joint_guide_for_vector_and_comprehension(vector, loop):
    pytest.importorskip("pyro")
    pytest.importorskip("torch")
    x, y = [-1.0, -0.2, 0.8, 2.0], [0.9, 0.3, -0.3, -1.4]
    options = {"y": y, "method": "vi", "steps": 80, "particles": 8, "seed": 23}
    first, second = vector(x).infer(**options), loop(x).infer(**options)
    assert first.method == second.method == "vi-fullrank"
    assert first.elbo == second.elbo
    assert first.diagnostics["latent_order"] == second.diagnostics["latent_order"]
    np.testing.assert_array_equal(first.diagnostics["unconstrained_mean"], second.diagnostics["unconstrained_mean"])
    np.testing.assert_array_equal(first.diagnostics["unconstrained_covariance"], second.diagnostics["unconstrained_covariance"])
    assert first.diagnostics["loss_history"] == second.diagnostics["loss_history"]
    for name in first.graph.groups:
        np.testing.assert_array_equal(first.mean(name), second.mean(name))
        np.testing.assert_array_equal(first.sd(name), second.sd(name))
        np.testing.assert_array_equal(first.interval(name), second.interval(name))
    assert_same_samples(first.sample(draws=100, seed=9), second.sample(draws=100, seed=9))
