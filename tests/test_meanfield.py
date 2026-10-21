"""Independent density/gradient and posterior references for native Torch VI."""
import builtins
import math
import numpy as np
import pytest

from smallprob import model, Normal, HalfNormal, StudentT, HMM
from smallprob.core import InferenceError, UnsupportedModelError
from smallprob.meanfield import _log_joint

torch = pytest.importorskip("torch")


@model
def scalar():
    mu = Normal(0, 3)
    y = Normal(mu, 1)


@model
def line(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    y = Normal(b + m*x, .6)


@model
def positive():
    sigma = HalfNormal(1)
    y = Normal(0, sigma)


@model
def robust(x):
    b = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(4, b + x, sigma)


@model
def mixed_hierarchy():
    scale = HalfNormal(2)
    mu = Normal(.3, scale)
    tau = HalfNormal(scale)
    shift = StudentT(5, mu, tau)
    y = Normal(mu + shift, [.3, .7])


def test_transformed_density_and_gradient_match_hand_formula():
    log_joint = _log_joint(positive().graph, {"y": .7}, torch, torch.device("cpu"), torch.float64)
    u = torch.tensor([[-1.], [.2], [.8]], dtype=torch.float64, requires_grad=True)
    value = log_joint(u)
    # HalfNormal(exp(u)|1) * Normal(.7|0,exp(u)) * exp(u).
    expected = math.log(2) - math.log(2*math.pi) - .5*u[:, 0].exp()**2 - .5*.7**2*torch.exp(-2*u[:, 0])
    torch.testing.assert_close(value, expected)
    gradient = torch.autograd.grad(value.sum(), u)[0][:, 0]
    torch.testing.assert_close(gradient, -torch.exp(2*u[:, 0]) + .7**2*torch.exp(-2*u[:, 0]))


@pytest.mark.parametrize("observed_scale", [False, True])
def test_mixed_hierarchy_density_and_gradient_include_only_latent_jacobians(observed_scale):
    # Grouping densities by family must preserve dependencies across groups:
    # HalfNormal -> Normal -> HalfNormal -> StudentT -> vector Normal.
    observed = {"y[0]": .2, "y[1]": -.4}
    if observed_scale:
        observed["scale"] = 1.2
    log_joint = _log_joint(mixed_hierarchy().graph, observed, torch, torch.device("cpu"), torch.float64)
    rows = [[-.2, .4, -.7, .6], [.5, -.1, .3, -1.], [.1, .8, -.2, .5]]
    u = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    if observed_scale:
        u = u[:, 1:].detach().requires_grad_(True)
        scale = torch.full_like(u[:, 0], 1.2)
        mu, tau, shift = u[:, 0], u[:, 1].exp(), u[:, 2]
        jacobian = u[:, 1]
    else:
        scale, mu, tau, shift = u[:, 0].exp(), u[:, 1], u[:, 2].exp(), u[:, 3]
        jacobian = u[:, 0] + u[:, 2]
    expected = (
        torch.distributions.HalfNormal(torch.tensor(2., dtype=torch.float64)).log_prob(scale)
        + torch.distributions.Normal(.3, scale).log_prob(mu)
        + torch.distributions.HalfNormal(scale).log_prob(tau)
        + torch.distributions.StudentT(5., mu, tau).log_prob(shift)
        + torch.distributions.Normal(mu + shift, .3).log_prob(torch.tensor(.2, dtype=torch.float64))
        + torch.distributions.Normal(mu + shift, .7).log_prob(torch.tensor(-.4, dtype=torch.float64))
        + jacobian
    )
    actual = log_joint(u)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected_gradient = torch.autograd.grad(expected.sum(), u, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), u)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_scalar_matches_analytic_posterior(seed):
    p = scalar().infer(y=2., method="vi", guide="diagonal", steps=1800, seed=seed)
    assert p.method == "vi-meanfield" and p.approximate and p.log_evidence is None
    assert p.diagnostics["backend"] == "torch"
    assert p.mean("mu") == pytest.approx(1.8, abs=.025)
    assert p.sd("mu") == pytest.approx(math.sqrt(.9), rel=.07)
    exact = scalar().infer(y=2.)
    assert p.elbo <= exact.log_evidence + 4*p.diagnostics["elbo_standard_error"] + .002


def test_correlated_regression_matches_meanfield_optimum_not_marginal_variance():
    x = np.linspace(.5, 2.5, 12)
    design = np.column_stack([np.ones(len(x)), x])
    y = 1.2 - .7*x + np.random.default_rng(21).normal(0, .6, len(x))
    precision = design.T @ design/.6**2 + np.eye(2)/25
    covariance = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, design.T @ y/.6**2)
    p = line(x).infer(y=y, method="vi", guide="diagonal", steps=3000, seed=7)
    np.testing.assert_allclose([p.mean('b'), p.mean('m')], mean, atol=.04)
    np.testing.assert_allclose([p.sd('b'), p.sd('m')], 1/np.sqrt(np.diag(precision)), rtol=.1)
    assert p.sd('b') < .65*np.sqrt(covariance[0, 0])
    draws = p.sample(40000, seed=9)
    assert abs(np.corrcoef(draws['b'], draws['m'])[0, 1]) < .025


def test_positive_moments_intervals_and_student_t_support():
    p = robust(np.array([-1., 0., 1.])).infer(y=[-.4, .5, 1.7], method="vi", guide="diagonal", steps=600, seed=3)
    samples = p.sample(100000, seed=4)["sigma"]
    assert np.all(samples > 0)
    assert p.mean("sigma") == pytest.approx(samples.mean(), rel=.02)
    np.testing.assert_allclose(p.interval("sigma"), np.quantile(samples, [.025, .975]), rtol=.02)
    assert p.diagnostics["transforms"]["sigma"] == "exp"


def test_no_pyro_import_and_global_state_restored(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name == "pyro" or name.startswith("pyro."):
            raise AssertionError("Native VI must not import Pyro")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked)
    state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    first = scalar().infer(y=1., method="vi", guide="diagonal", steps=15, seed=9)
    second = scalar().infer(y=1., method="vi", guide="diagonal", steps=15, seed=9)
    assert torch.equal(state, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert first.diagnostics == second.diagnostics
    assert first.mean('mu') == second.mean('mu')


def test_optimizer_failure_restores_rng_and_thread_state(monkeypatch):
    def fail_after_sampling(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")
    monkeypatch.setattr(torch.optim.Adam, "step", fail_after_sampling)
    before_rng = torch.get_rng_state().clone()
    before_threads, before_dtype = torch.get_num_threads(), torch.get_default_dtype()
    with pytest.raises(InferenceError, match="step 1.*injected optimizer failure"):
        scalar().infer(y=1., method="vi", guide="diagonal", steps=10, seed=123)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.get_default_dtype() == before_dtype


def test_observing_a_prior_site_leaves_likelihood_as_a_latent():
    result = scalar().infer(mu=1.25, method="vi", guide="diagonal", steps=10, seed=7)
    assert result.diagnostics["latent_order"] == ["y"]
    assert result.mean("mu") == 1.25 and result.sd("mu") == 0
    draws = result.sample(20, seed=8)
    assert np.all(draws["mu"] == 1.25)
    assert np.all(np.isfinite(draws["y"]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
def test_mps_meanfield_runs_on_device_and_restores_device_rng():
    cpu_before = torch.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone()
    result = positive().infer(y=.7, method="vi", guide="diagonal", steps=10,
                              backend="torch", device="mps", dtype="float32", seed=7)
    assert result.diagnostics["device"] == "mps" and result.diagnostics["dtype"] == "float32"
    assert result.mean("sigma").device.type == "mps"
    assert result.mean("sigma").dtype == torch.float32
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.mps.get_rng_state(), mps_before)
    samples = result.sample(20, seed=9)["sigma"]
    assert samples.device.type == "mps" and bool(torch.isfinite(samples).all())
    assert bool((samples > 0).all())


def test_torch_result_dtype_and_sampling():
    p = scalar().infer(y=torch.tensor(2., dtype=torch.float32), method="vi", guide="diagonal", steps=10)
    assert isinstance(p.mean('mu'), torch.Tensor)
    assert p.mean('mu').dtype == torch.float32
    assert p.sample(3, seed=8)['mu'].shape == (3,)


@pytest.mark.parametrize("options", [{"steps": 0}, {"particles": True}, {"seed": -1}, {"learning_rate": float('nan')}, {"guide": "bad"}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        scalar().infer(y=1., method="vi", **{"guide":"diagonal", **options})


def test_exact_does_not_silently_accept_guide_and_discrete_is_rejected():
    with pytest.raises(InferenceError, match="only"):
        scalar().infer(y=1., guide="diagonal")
    with pytest.raises(UnsupportedModelError, match="Categorical"):
        HMM(initial=[1.], transition=[[1.]], emissions=Normal([0.], 1), steps=1).infer(y=[1.], method="vi", guide="diagonal")
