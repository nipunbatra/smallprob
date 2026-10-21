"""Backend contracts independent of the public syntax and accuracy study."""

import json
import random
from statistics import NormalDist

import numpy as np
import pytest

from smallprob.core import (
    Categorical, Expr, Graph, HalfNormal, InferenceError, Node, Normal,
    UnsupportedModelError, as_expr,
)
from smallprob.vi import _compiler, infer_vi

pytest.importorskip("pyro")
torch = pytest.importorskip("torch")


def graph(*pairs):
    return Graph([Node(name, distribution) for name, distribution in pairs],
                 {name: [name] for name, distribution in pairs}, set())


def ref(name):
    return Expr("ref", (name,))


@pytest.fixture(scope="module")
def fitted_positive():
    return infer_vi(graph(("sigma", HalfNormal(1)), ("y", Normal(0, ref("sigma")))),
                    {"y": 0.7}, steps=500, particles=16, seed=5)


def test_guide_is_detached_positive_and_moments_match_its_samples(fitted_positive):
    result = fitted_positive
    assert result.method == "vi-fullrank"
    assert result.log_evidence is None
    assert result.marginals == {}
    assert result.means["y"] == 0.7
    assert result.variances["y"] == 0
    assert result.diagnostics["dtype"] == "float64"
    assert result.diagnostics["transforms"] == {"sigma": "exp"}
    assert result.diagnostics["elbo_standard_error"] > 0
    assert np.isfinite(result.elbo)
    json.dumps(result.diagnostics, allow_nan=False)
    samples = result.sample(100_000, np.random.default_rng(7))
    assert np.all(samples["sigma"] > 0)
    assert np.all(samples["y"] == 0.7)
    np.testing.assert_allclose(samples["sigma"].mean(), result.means["sigma"], rtol=0.02)
    np.testing.assert_allclose(samples["sigma"].var(), result.variances["sigma"], rtol=0.06)
    lo, hi = result.interval(0.9)
    np.testing.assert_allclose(np.quantile(samples["sigma"], [0.05, 0.95]), [lo["sigma"], hi["sigma"]], rtol=0.02)
    assert lo["y"] == hi["y"] == 0.7
    assert lo["sigma"] < result.means["sigma"] < hi["sigma"]


def test_exactly_reproducible_and_backend_global_state_is_restored():
    import pyro

    model = graph(("mu", Normal(0, 2)), ("y", Normal(ref("mu"), 1)))
    store = pyro.get_param_store()
    with store.scope():
        marker = pyro.param("unrelated_existing_parameter", torch.tensor(17.0))
        old_threads, old_dtype = torch.get_num_threads(), torch.get_default_dtype()
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        first = infer_vi(model, {"y": 1.0}, steps=80, particles=8, seed=9)
        frozen_samples = first.sample(12, np.random.default_rng(33))
        second = infer_vi(model, {"y": 1.0}, steps=80, particles=8, seed=9)
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert torch.get_num_threads() == old_threads
        assert torch.get_default_dtype() == old_dtype
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_state[0]
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        assert after_numpy[2:] == numpy_state[2:]
        assert random.getstate() == python_state
        assert set(store.keys()) == {"unrelated_existing_parameter"}
        assert store["unrelated_existing_parameter"] is marker
        assert first.means == second.means
        assert first.diagnostics["loss_history"] == second.diagnostics["loss_history"]
        assert first.elbo == second.elbo
        infer_vi(model, {"y": -5.0}, steps=40, particles=4, seed=12)
        for name, values in frozen_samples.items():
            np.testing.assert_array_equal(first.sample(12, np.random.default_rng(33))[name], values)


def test_identity_intervals_use_learned_guide_not_normal_approximation_to_positive_site():
    result = infer_vi(graph(("mu", Normal(0, 2))), {}, steps=40, particles=4, seed=9)
    lo, hi = result.interval(0.95)
    width = NormalDist().inv_cdf(0.975) * np.sqrt(result.variances["mu"])
    np.testing.assert_allclose([lo["mu"], hi["mu"]], [result.means["mu"] - width, result.means["mu"] + width])
    assert result.diagnostics["latent_order"] == ["mu"]
    assert result.diagnostics["steps"] == 40


@pytest.mark.parametrize("bad", [0, -1, True, 1.2])
def test_reject_invalid_steps_and_particles(bad):
    model = graph(("mu", Normal()))
    with pytest.raises(ValueError, match="steps"):
        infer_vi(model, {}, steps=bad)
    with pytest.raises(ValueError, match="particles"):
        infer_vi(model, {}, particles=bad)


@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), float("inf")])
def test_reject_invalid_learning_rate(bad):
    with pytest.raises(ValueError, match="learning_rate"):
        infer_vi(graph(("mu", Normal())), {}, learning_rate=bad)


@pytest.mark.parametrize("bad", [-1, True, 1.5, 2 ** 63])
def test_reject_invalid_seed(bad):
    with pytest.raises(ValueError, match="seed"):
        infer_vi(graph(("mu", Normal())), {}, seed=bad)


def test_reject_discrete_models_and_no_latents():
    with pytest.raises(UnsupportedModelError, match="Categorical"):
        infer_vi(graph(("state", Categorical([0.4, 0.6]))), {}, steps=1)
    with pytest.raises(UnsupportedModelError, match="at least one unobserved"):
        infer_vi(graph(("mu", Normal())), {"mu": 1.0}, steps=1)


def test_reject_nonfinite_or_out_of_support_observations():
    model = graph(("sigma", HalfNormal()), ("y", Normal()))
    for value in [float("nan"), float("inf"), True, "2", complex(1, 2)]:
        with pytest.raises(InferenceError, match="finite real scalar"):
            infer_vi(model, {"y": value}, steps=1)
    with pytest.raises(InferenceError, match="nonnegative"):
        infer_vi(model, {"sigma": -0.1}, steps=1)
    with pytest.raises(InferenceError, match="unknown sites"):
        infer_vi(model, {"missing": 0.0}, steps=1)


def test_halfnormal_observation_zero_is_in_support():
    result = infer_vi(graph(("sigma", HalfNormal()), ("mu", Normal())), {"sigma": 0.0}, steps=5, particles=2)
    assert result.means["sigma"] == result.variances["sigma"] == 0.0
    assert np.isfinite(result.elbo)


@pytest.mark.parametrize("name", ["_AutoMultivariateNormal_latent", "AutoMultivariateNormal.loc", "_smallprob_observed_log_prob"])
def test_explicit_site_names_do_not_collide_with_backend_internals(name):
    result = infer_vi(graph((name, Normal()), ("y", Normal(ref(name), 1))), {"y": 0.0}, steps=5, particles=2)
    assert result.diagnostics["latent_order"] == [name]
    assert np.isfinite(result.means[name])


def test_reject_random_indices_and_unsupported_expressions():
    mu = ref("mu")
    with pytest.raises(UnsupportedModelError, match="fixed indices"):
        infer_vi(graph(("mu", Normal()), ("y", Normal(as_expr([0.0, 1.0])[mu], 1))), {"y": 0.0}, steps=1)
    with pytest.raises(UnsupportedModelError, match="Expression 'sin'"):
        infer_vi(graph(("mu", Normal()), ("y", Normal(Expr("sin", (mu,)), 1))), {"y": 0.0}, steps=1)


@pytest.mark.parametrize("parameter", ["mean", "sd"])
def test_reject_vector_valued_site_parameters_before_particle_broadcasting(parameter):
    vector = ref("mu") * np.array([1.0, 2.0])
    distribution = Normal(mean=vector) if parameter == "mean" else Normal(sd=vector)
    with pytest.raises(UnsupportedModelError, match="scalar distribution parameters"):
        infer_vi(graph(("mu", Normal()), ("y", distribution)), {"y": 0.0}, steps=1, particles=2)


def test_numeric_failure_restores_rng_store_and_threads():
    import pyro

    model = graph(("mu", Normal()), ("y", Normal(0, -ref("mu") ** 2 - 1)))
    state, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    with pyro.get_param_store().scope():
        pyro.param("keep", torch.tensor(2.0))
        with pytest.raises(InferenceError, match="optimisation step 1"):
            infer_vi(model, {"y": 1.0}, steps=1)
        assert set(pyro.get_param_store().keys()) == {"keep"}
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads


def test_fixed_index_and_array_intermediates_preserve_particle_axis_and_gradients():
    compile_expr, batch = _compiler(torch)
    mu = ref("mu")
    expression = ((mu + np.array([1.0, 2.0])) * np.array([3.0, 4.0]))[1]
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    compiled = compile_expr(expression)
    assert compiled.shape == ()
    output = compiled.evaluate({"mu": values})
    np.testing.assert_allclose(output.detach().numpy(), [12.0, 16.0, 20.0])
    output.sum().backward()
    np.testing.assert_allclose(values.grad, [4.0, 4.0, 4.0])
    # Mixed expression trees take the fallback path but retain the same axis.
    output = batch([mu, expression, mu ** 2])({"mu": values})
    np.testing.assert_allclose(output.detach().numpy(), [[1, 12, 1], [2, 16, 4], [3, 20, 9]])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_sample_count(fitted_positive, bad):
    with pytest.raises(ValueError, match="draws"):
        fitted_positive.sample(bad, np.random.default_rng(1))


@pytest.mark.parametrize("bad", [0, 1, True, float("nan"), -0.1])
def test_invalid_interval_probability(fitted_positive, bad):
    with pytest.raises(ValueError, match="probability"):
        fitted_positive.interval(bad)
