"""Native-array execution, independently checked against NumPy inference."""

import json
import subprocess
import sys
import os

import numpy as np
import pytest

from smallprob import ModelError, Normal, HalfNormal, StudentT, UnsupportedModelError, model
from smallprob.arrays import ArrayBackend, infer_backend, to_numpy
from examples.models import regression, regression_unknown_noise, hmm


torch = pytest.importorskip("torch")


@model
def uncertain_mean():
    mu = Normal(0, 2)
    y = Normal(mu, [0.7, 0.7, 0.7])


@model
def half_normal():
    sigma = HalfNormal(1)


@model
def robust_line(x):
    m = Normal(0, 2)
    y = StudentT(4, m * x, .7)


def test_torch_inputs_select_native_arrays_and_snapshot():
    x = torch.tensor([0., 1., 2.], dtype=torch.float64, requires_grad=True)
    m = regression(x)
    with torch.no_grad():
        x.add_(100)
    p = m.infer(y=torch.tensor([1., 2., 3.], dtype=torch.float64))
    reference = regression([0., 1., 2.]).infer(y=[1., 2., 3.])
    assert m.backend.name == "torch"
    assert p.mean("slope").dtype == torch.float64
    assert not p.mean("slope").requires_grad
    assert float(p.mean("slope")) == pytest.approx(reference.mean("slope"), abs=1e-12)
    assert isinstance(p.sd("slope"), torch.Tensor)
    assert all(isinstance(v, torch.Tensor) for v in p.interval("slope"))
    assert all(isinstance(v, torch.Tensor) for v in p.sample(10, seed=3).values())
    assert p.sample(10, seed=3)["y"].shape == (10, 3)
    assert p.diagnostics["device"] == "cpu"
    assert p.diagnostics["array_backend"] == "torch"


@pytest.mark.parametrize("dtype,tolerance", [("float64", 2e-12), ("float32", 2e-5)])
def test_exact_native_qr_matches_gaussian_reference(monkeypatch, dtype, tolerance):
    x = np.linspace(0.5, 2.5, 16)
    y = 1.2 - 0.7 * x + np.random.default_rng(12).normal(0, .6, len(x))
    m = regression(x, noise=.6)
    reference = m.infer(y=y)
    def forbidden(*args, **kwargs):
        raise AssertionError("Torch numerical conditioning must not call NumPy QR/solve")
    monkeypatch.setattr(np.linalg, "qr", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    p = m.infer(y=y, backend="torch", dtype=dtype)
    for name in ("intercept", "slope"):
        assert float(p.mean(name)) == pytest.approx(reference.mean(name), abs=tolerance)
        assert float(p.sd(name)) == pytest.approx(reference.sd(name), abs=tolerance)
    assert p.log_evidence == pytest.approx(reference.log_evidence, abs=10 * tolerance)
    draws = p.sample(30000, seed=15)
    pairs = torch.stack([draws["intercept"], draws["slope"]])
    # The highly correlated posterior must remain joint, not marginal draws.
    assert torch.corrcoef(pairs)[0, 1].item() < -.9
    assert abs(draws["slope"].mean().item() - float(p.mean("slope"))) < .02


def test_torch_observations_select_backend_without_fixed_inputs():
    p = uncertain_mean().infer(y=torch.tensor([.9, 1., 1.1]))
    assert isinstance(p.mean("mu"), torch.Tensor)
    assert p.mean("mu").dtype == torch.float32


def test_numpy_override_is_explicit_and_returns_numpy():
    m = regression(torch.tensor([0., 1., 2.]))
    p = m.infer(y=torch.tensor([1., 2., 3.]), backend="numpy")
    assert isinstance(p.sample(4, seed=3)["y"], np.ndarray)
    assert p.diagnostics["array_backend"] == "numpy"
    assert m.simulate(2, backend="numpy")["y"].shape == (2, 3)


def test_simulation_uses_torch_distributions_preserves_rng(monkeypatch):
    monkeypatch.setattr(Normal, "sample", lambda *args: (_ for _ in ()).throw(AssertionError("NumPy sampler called")))
    before = torch.get_rng_state().clone()
    m = regression([0., 1.])
    a = m.simulate(20, seed=20, backend="torch", fixed={"slope": 1})
    b = m.simulate(20, seed=20, backend="torch", fixed={"slope": 1})
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(a["y"], b["y"])
    assert torch.equal(a["slope"], torch.ones(20, dtype=torch.float64))
    assert a["y"].shape == (20, 2)


def test_torch_simulation_support_failure_restores_rng():
    before = torch.get_rng_state().clone()
    with pytest.raises(ModelError, match="outside.*support"):
        half_normal().simulate(1, backend="torch", seed=5, fixed={"sigma": -1})
    assert torch.equal(before, torch.get_rng_state())


def test_native_log_density_matches_numpy():
    m = regression([0., 1.])
    values = {"intercept": .7, "slope": -.5, "y": [.6, .3]}
    value = m.log_prob(values, backend="torch")
    assert isinstance(value, torch.Tensor)
    assert value.item() == pytest.approx(m.log_prob(values), abs=1e-12)
    assert half_normal().log_prob({"sigma": -1}, backend="torch").item() == -np.inf


def test_native_student_t_density_and_samples():
    m = robust_line([0., 1.])
    values = {"m": .7, "y": [.6, 1.3]}
    assert m.log_prob(values, backend="torch").item() == pytest.approx(m.log_prob(values), abs=1e-12)
    data = m.simulate(100, seed=8, backend="torch", fixed={"m": 1})
    assert data["y"].shape == (100, 2)
    assert torch.all(torch.isfinite(data["y"]))


def test_torch_hmm_simulation_and_explicit_exact_boundary():
    m = hmm(5, [.5, .5], [[.9, .1], [.2, .8]], [-1., 1.])
    data = m.simulate(5, seed=8, backend="torch")
    assert data["z"].dtype == torch.int64
    assert data["y"].dtype == torch.float64
    with pytest.raises(UnsupportedModelError, match="NumPy/CPU"):
        m.infer(y=data["y"][0])
    p = m.infer(y=data["y"][0], backend="numpy", device="cpu")
    np.testing.assert_allclose(p.probabilities("z").sum(axis=1), 1)
    assert m.infer(y=data["y"][0], backend="numpy", dtype="float32").diagnostics["dtype"] == "float64"


@pytest.mark.parametrize("kwargs,message", [
    ({"backend": "jax"}, "backend"),
    ({"backend": "numpy", "device": "cuda"}, "NumPy"),
    ({"backend": "torch", "dtype": "float16"}, "dtype"),
    ({"backend": "torch", "device": "cpu:1"}, "without an index"),
    ({"backend": "torch", "device": "bogus"}, "device"),
])
def test_backend_validation(kwargs, message):
    with pytest.raises(ModelError, match=message):
        infer_backend(**kwargs)


def test_unavailable_cuda_is_not_silently_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(ModelError, match="CUDA.*not available"):
        infer_backend(backend="torch", device="cuda")


def test_numpy_float32_is_requested_computation_dtype():
    m = regression([0., 1., 2.])
    p = m.infer(y=[1., 2., 3.], dtype="float32")
    assert p.sample(3, seed=4)["y"].dtype == np.float32
    assert p.diagnostics["dtype"] == "float32"


def test_numpy_only_import_and_exact_engine_do_not_import_torch():
    code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('PyTorch intentionally unavailable')
sys.meta_path.insert(0, BlockTorch())
from examples.models import regression
from smallprob import ModelError
m = regression([0., 1.])
assert m.infer(y=[1., 2.]).method == 'exact-gaussian'
assert m.simulate(2, seed=3)['y'].shape == (2, 2)
try:
    m.simulate(backend='torch')
except ModelError as e:
    assert 'optional dependency' in str(e)
else:
    raise AssertionError('missing Torch dependency should raise')
assert 'torch' not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": "src:."}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_native_exact_and_simulation():
    x = torch.linspace(0, 2, 8, device="mps")
    m = regression(x)
    y = 1 + x
    p = m.infer(y=y)
    reference = regression(to_numpy(x)).infer(y=to_numpy(y))
    assert p.mean("slope").device.type == "mps"
    assert p.sample(3, seed=5)["y"].device.type == "mps"
    assert p.mean("slope").item() == pytest.approx(reference.mean("slope"), abs=1e-5)
    before = torch.mps.get_rng_state().clone()
    data = m.simulate(5, seed=3)
    assert data["y"].device.type == "mps"
    assert torch.equal(before, torch.mps.get_rng_state())
    with pytest.raises(ModelError, match="MPS.*float64"):
        m.infer(y=y, dtype="float64")


def test_vi_returns_selected_native_arrays():
    pytest.importorskip("pyro")
    p = uncertain_mean().infer(y=torch.tensor([.9, 1., 1.1], dtype=torch.float64),
                               method="vi", steps=30, particles=4, seed=3)
    assert p.approximate
    assert p.diagnostics["device"] == "cpu"
    assert isinstance(p.mean("mu"), torch.Tensor)
    assert p.sample(5, seed=7)["mu"].dtype == torch.float64
    assert all(isinstance(endpoint, torch.Tensor) for endpoint in p.interval("mu"))
    json.dumps(p.diagnostics)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_vi_native_sampling_with_positive_transform(monkeypatch, device):
    pytest.importorskip("pyro")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    dtype = "float32" if device == "mps" else "float64"
    m = regression_unknown_noise([-1., 0., 1.])
    p = m.infer(y=[1.4, 1., .6], method="vi", steps=20, particles=4, seed=3,
                backend="torch", device=device, dtype=dtype)
    def forbidden(*args):
        raise AssertionError("Native VI draws must not call the NumPy guide sampler")
    monkeypatch.setattr(p._result, "sample", forbidden)
    cpu_before = torch.get_rng_state().clone()
    device_before = torch.mps.get_rng_state().clone() if device == "mps" else None
    a, b = p.sample(100, seed=7), p.sample(100, seed=7)
    assert torch.equal(a["sigma"], b["sigma"])
    assert a["sigma"].device.type == device
    assert torch.all(torch.isfinite(a["sigma"])) and torch.all(a["sigma"] > 0)
    assert torch.equal(cpu_before, torch.get_rng_state())
    if device_before is not None:
        assert torch.equal(device_before, torch.mps.get_rng_state())
