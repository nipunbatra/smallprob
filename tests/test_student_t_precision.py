"""Independent high-precision checks for native Torch Student-t densities."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("mpmath")

from smallprob import Normal, StudentT, model
from smallprob.arrays import student_t_log_prob, stable_student_t_class


def reference(value, df, loc, scale):
    # Conversion uses the actually representable input values, so backend
    # arithmetic is tested separately from float32 input rounding.
    with mp.workdps(90):
        value, df, loc, scale = map(mp.mpf, map(float, (value, df, loc, scale)))

        def density(df, loc, scale):
            return (mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2)
                    - mp.log(df * mp.pi) / 2 - mp.log(scale)
                    - (df + 1) / 2 * mp.log1p(((value - loc) / scale)**2 / df))

        gradients = [mp.diff(lambda d: density(d, loc, scale), df),
                     mp.diff(lambda m: density(df, m, scale), loc),
                     mp.diff(lambda s: density(df, loc, s), scale)]
        return float(density(df, loc, scale)), np.array(list(map(float, gradients)))


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cpu", torch.float64),
    pytest.param("mps", torch.float32, marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
])
def test_density_and_parameter_gradients_across_df(device, dtype):
    dfs = [1e-10, 1e-6, .2, 1., 4., 7.99, 8., 16., 31.99, 32., 100., 1e4, 1e7, 1e12]
    # Include branch boundaries and both small-ratio and heavy-tail regimes.
    for value in (0., .9, 20.):
        for df in dfs:
            d = torch.tensor(df, dtype=dtype, device=device, requires_grad=True)
            loc = torch.tensor(-.2 if value else 0., dtype=dtype, device=device, requires_grad=True)
            scale = torch.tensor(1.3, dtype=dtype, device=device, requires_grad=True)
            v = torch.tensor(value, dtype=dtype, device=device)
            actual = student_t_log_prob(v, d, loc, scale)
            gradients = torch.autograd.grad(actual, (d, loc, scale))
            inputs = [tensor.detach().cpu().item() for tensor in (v, d, loc, scale)]
            expected, expected_gradients = reference(*inputs)
            assert actual.device.type == device
            assert actual.dtype == dtype
            assert actual.item() == pytest.approx(expected, rel=2e-6 if dtype == torch.float32 else 3e-12,
                                                 abs=3e-6 if dtype == torch.float32 else 2e-12)
            actual_gradients = np.array([gradient.detach().cpu().item() for gradient in gradients])
            # df gradients in the Normal limit are tiny (~1/df**2). Keep
            # their absolute tolerance tiny so an erroneously zero derivative
            # cannot pass. Location/scale gradients can cancel to zero at
            # very small df, where an absolute roundoff allowance is needed.
            np.testing.assert_allclose(actual_gradients[0], expected_gradients[0],
                                       rtol=1e-4 if dtype == torch.float32 else 5e-10, atol=1e-27,
                                       err_msg=f"{device}/{dtype}, df={df}, value={value}")
            np.testing.assert_allclose(actual_gradients[1:], expected_gradients[1:],
                                       rtol=1e-4 if dtype == torch.float32 else 5e-10,
                                       atol=2e-6 if dtype == torch.float32 else 1e-13,
                                       err_msg=f"{device}/{dtype}, df={df}, value={value}")


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"))])
def test_float32_large_df_central_density_and_finite_extreme_tail(device):
    cls = stable_student_t_class(torch.distributions.StudentT)
    for df in (4., 1e7):
        d = torch.tensor(df, device=device, dtype=torch.float32, requires_grad=True)
        distribution = cls(d, torch.tensor(0., device=device), torch.tensor(1., device=device))
        density = distribution.log_prob(torch.tensor(0., device=device))
        expected = reference(0, df, 0, 1)[0]
        assert density.item() == pytest.approx(expected, abs=2e-7)
        gradient = torch.autograd.grad(density, d)[0]
        assert gradient.item() > 0
        extreme = distribution.log_prob(torch.tensor(1e20, device=device))
        assert torch.isfinite(extreme)
        assert extreme.item() == pytest.approx(reference(float(np.float32(1e20)), df, 0, 1)[0], rel=2e-6)


@model
def large_df_mean(n):
    mu = Normal(0, 2)
    y = [StudentT(1e7, mu, .7) for _ in range(n)]


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"))])
def test_simulation_density_and_vi_share_stable_distribution(monkeypatch, device):
    pytest.importorskip("pyro")
    def forbidden(*args, **kwargs):
        raise AssertionError("Cancellation-prone Torch StudentT.log_prob was used")
    monkeypatch.setattr(torch.distributions.StudentT, "log_prob", forbidden)
    m = large_df_mean(3)
    joint = m.log_prob({"mu": 0., "y": [.3, .5, .7]}, backend="torch", device=device, dtype="float32")
    assert torch.isfinite(joint)
    assert torch.all(torch.isfinite(m.simulate(5, seed=5, backend="torch", device=device, dtype="float32")["y"]))
    p = m.infer(y=[.3, .5, .7], method="vi", backend="torch", device=device, dtype="float32",
                steps=20, particles=4, seed=8)
    assert np.isfinite(p.elbo)
    assert p.diagnostics["device"].startswith(device)
    assert torch.all(torch.isfinite(p.sample(10, seed=2)["mu"]))
