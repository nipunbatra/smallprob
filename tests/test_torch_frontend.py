"""Torch inputs must be copied intentionally, including grad-enabled tensors."""

import linecache

import numpy as np
import pytest

from smallprob import Normal, StudentT, model

torch = pytest.importorskip("torch")


@model
def torch_regression(x, noise=0.7):
    b = Normal(0, 2)
    m = Normal(0, 2)
    y = Normal(b + m * x, noise)


@model
def torch_robust(x):
    m = Normal(0, 2)
    y = StudentT(4, m * x, 0.7)


def test_grad_enabled_tensor_inputs_are_detached_for_graph_constants():
    tensor = torch.tensor([-1.0, 0, 2], dtype=torch.float64, requires_grad=True)
    bound = torch_regression(tensor)
    assert len(bound.graph.groups["y"]) == 3
    np.testing.assert_array_equal(bound.inputs["x"], [-1, 0, 2])
    assert bound.inputs["noise"] == 0.7
    assert "y = Normal(b + m * x, noise)" in bound.source
    with torch.no_grad():
        tensor.add_(20)
    np.testing.assert_array_equal(bound.inputs["x"], [-1, 0, 2])
    assert [node.distribution.mean.eval({"b": 1, "m": 2}) for node in bound.graph.nodes[2:]] == [-1, 1, 5]


def test_student_t_accepts_tensor_vector_and_scalar_parameters():
    tensor = torch.tensor([-1.0, 0, 2], requires_grad=True)
    bound = torch_robust(tensor)
    assert len(bound.graph.groups["y"]) == 3
    distribution = StudentT(torch.tensor(4.0, requires_grad=True), torch.tensor(2.0), torch.tensor(0.5))
    assert distribution._parameters({}) == (4, 2, 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_model_inputs_transfer_explicitly_for_trace():
    bound = torch_regression(torch.tensor([-1.0, 1], device="cuda", requires_grad=True))
    np.testing.assert_array_equal(bound.inputs["x"], [-1, 1])
    assert len(bound.graph.groups["y"]) == 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_mps_model_inputs_transfer_explicitly_for_trace():
    bound = torch_regression(torch.tensor([-1.0, 1], device="mps", requires_grad=True))
    np.testing.assert_array_equal(bound.inputs["x"], [-1, 1])
    assert len(bound.graph.groups["y"]) == 2


def test_cached_notebook_cell_source_is_readable():
    filename = "<ipython-input-smallprob-test>"
    source = "@model\ndef notebook_model(x):\n    y = Normal(x, 1)\n"
    namespace = {"model": model, "Normal": Normal}
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    try:
        exec(compile(source, filename, "exec"), namespace)
        bound = namespace["notebook_model"]([0, 1])
        assert bound.source == source
        assert len(bound.graph.groups["y"]) == 2
    finally:
        linecache.cache.pop(filename, None)
