"""Fast device execution checks; posterior accuracy runs are explicit opt-in."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyro")

from smallprob import InferenceError, Normal, model
from smallprob.vi import infer_vi
from examples.backend_validation import checked_student_reference, student_mean, student_quadrature, run_validation


Y = np.array([.3, .5, .7, .8, .9, 1., 1.2, 4.5])


@model
def invalid_scale():
    sigma = Normal(-1, .01)
    y = Normal(0, sigma)


@pytest.mark.parametrize("dtype", ["float32", "float64", torch.float32, torch.float64])
def test_cpu_student_vi_device_and_dtype_are_real(dtype):
    expected = str(dtype).removeprefix("torch.")
    result = infer_vi(student_mean(len(Y)).graph, {f"y[{i}]": float(value) for i, value in enumerate(Y)},
                      device="cpu", dtype=dtype, steps=20, particles=4, seed=16)
    assert result.diagnostics["device"] == "cpu"
    assert result.diagnostics["dtype"] == expected
    assert result._location.dtype == np.dtype(expected)
    assert result._factor.dtype == np.dtype(expected)
    assert np.isfinite(result.elbo)
    assert np.isfinite(result.means["mu"])
    # These twenty-step calls check execution, not approximation accuracy.
    assert any("smoke" in warning for warning in result.diagnostics["warnings"])


@pytest.mark.parametrize("options,match", [
    ({"device": "nonsense"}, "Invalid VI device"),
    ({"device": "meta"}, "supports cpu, cuda or mps"),
    ({"dtype": "float16"}, "dtype must be float32 or float64"),
    ({"dtype": torch.int64}, "dtype must be float32 or float64"),
])
def test_direct_vi_option_errors(options, match):
    with pytest.raises(InferenceError, match=match):
        infer_vi(student_mean(1).graph, {"y[0]": .7}, steps=1, particles=2, **options)


def test_unavailable_cuda_is_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(InferenceError, match="CUDA.*no CUDA device"):
        infer_vi(student_mean(1).graph, {"y[0]": .7}, device="cuda", steps=1, particles=2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_student_vi_preserves_rng_and_returns_real_device_samples():
    cpu_state, mps_state = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    p = student_mean(len(Y)).infer(y=Y, method="vi", device="mps", dtype="float32",
                                 steps=20, particles=4, seed=16)
    assert p.diagnostics["device"].startswith("mps")
    assert p.diagnostics["dtype"] == "float32"
    assert np.isfinite(p.elbo)
    assert p.sample(10, seed=3)["mu"].device.type == "mps"
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(mps_state, torch.mps.get_rng_state())
    with pytest.raises(InferenceError, match="MPS does not support float64"):
        infer_vi(student_mean(1).graph, {"y[0]": .7}, device="mps", dtype="float64", steps=1, particles=2)


def test_direct_vi_failure_restores_rng_and_threads():
    before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    with pytest.raises(InferenceError, match="VI failed"):
        infer_vi(invalid_scale().graph, {"y": 0.}, seed=16, steps=2, particles=4)
    assert torch.equal(before, torch.get_rng_state())
    assert threads == torch.get_num_threads()


def test_independent_student_reference_recovers_prior_without_observations():
    from statistics import NormalDist
    reference = student_quadrature([], points=64001)
    assert abs(reference["mean"][0]) < 1e-10
    assert reference["sd"][0] == pytest.approx(2., abs=1e-10)
    assert abs(reference["log_evidence"]) < 1e-10
    np.testing.assert_allclose(reference["interval95"][0],
                               [2 * NormalDist().inv_cdf(.025), 2 * NormalDist().inv_cdf(.975)], atol=1e-7)


def test_independent_student_reference_converges():
    reference = checked_student_reference(Y)
    assert reference["convergence"]["passed"], reference["convergence"]
    assert 0 < reference["sd"][0] < 1
    assert reference["interval95"][0][0] < reference["mean"][0] < reference["interval95"][0][1]


@pytest.mark.skipif(os.environ.get("SMALLPROB_RUN_BACKEND_ACCURACY") != "1",
                   reason="Set SMALLPROB_RUN_BACKEND_ACCURACY=1 for StudentT and full MPS VI accuracy fits.")
def test_full_backend_accuracy():
    report = run_validation(progress=False)
    assert report["passed"], report
