"""A small Array API boundary for NumPy and PyTorch execution.

Graph tracing snapshots constants on the host. Numerical Gaussian conditioning
and Torch simulation operate on the chosen device; a tensor result is never a
claim that a NumPy computation ran on a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import threading

import numpy as np
from array_api_compat import array_namespace

from .core import ModelError


TORCH_RNG_LOCK = threading.RLock()


def student_t_log_prob(value, df, loc, scale):
    """Differentiable Torch t log density without large-df cancellation.

    The log gamma ratio uses its asymptotic series where subtraction of two
    large lgamma values would lose precision. For the Normal-limit tail, the
    log1p(r)/r series also preserves derivatives with respect to df. All work
    stays in the input tensor's dtype and device, including MPS float32.
    """
    import math
    import torch

    cutoff = 8.0 if df.dtype == torch.float32 else 32.0
    # Clamp the inactive branches too: torch.where evaluates both branches,
    # and an unused infinite derivative can otherwise contaminate gradients.
    direct_df = torch.where(df < 1e-8, torch.full_like(df, 1e-8),
                           torch.where(df > cutoff, torch.full_like(df, cutoff), df))
    direct = (torch.lgamma(.5 * (direct_df + 1)) - torch.lgamma(.5 * direct_df)
              - .5 * (direct_df.log() + math.log(math.pi)))
    large_df = torch.where(df >= cutoff, df, torch.full_like(df, cutoff))
    inverse = large_df.reciprocal()
    inverse_squared = inverse.square()
    correction = inverse * (-.25 + inverse_squared * (1 / 24 + inverse_squared *
                    (-1 / 20 + inverse_squared * (17 / 112 - inverse_squared * 31 / 36))))
    large = -.5 * math.log(2 * math.pi) + correction
    small_df = torch.where(df < 1e-8, df, torch.full_like(df, 1e-8))
    small = (torch.lgamma(.5 * (small_df + 1)) + .5 * small_df.log()
             - math.log(2) - .5 * math.log(math.pi) + .5 * 0.5772156649015329 * small_df)
    normalizer = torch.where(df < 1e-8, small, torch.where(df >= cutoff, large, direct))

    z = (value - loc) / scale
    # logaddexp evaluates log(1 + z**2/df) without overflowing z**2.
    absolute = z.abs().clamp_min(torch.finfo(z.dtype).tiny)
    log_ratio = 2 * absolute.log() - df.log()
    direct_tail = .5 * (df + 1) * torch.logaddexp(torch.zeros_like(log_ratio), log_ratio)

    bound = (.01 * large_df).sqrt()
    safe_z = torch.where(z > bound, bound, torch.where(z < -bound, -bound, z))
    squared = safe_z.square()
    ratio = squared * inverse
    # log1p(r)/r = 1 - r/2 + r**2/3 - ...; r <= .01 here.
    polynomial = torch.ones_like(ratio) / 9
    for power in range(7, -1, -1):
        polynomial = ((-1) ** power) / (power + 1) + ratio * polynomial
    series_tail = .5 * squared * (1 + inverse) * polynomial
    use_series = (df >= cutoff) & (z.abs() <= bound)
    tail = torch.where(use_series, series_tail, direct_tail)
    return normalizer - scale.log() - tail


@lru_cache(maxsize=4)
def stable_student_t_class(base_class):
    """Keep Torch/Pyro distribution behaviour, replacing only its log density."""
    class StableStudentT(base_class):
        def log_prob(self, value):
            if self._validate_args:
                self._validate_sample(value)
            return student_t_log_prob(value, self.df, self.loc, self.scale)

    return StableStudentT


def is_torch_array(value):
    """Recognise tensors without making PyTorch a NumPy-only dependency."""
    return type(value).__module__.split(".")[0] == "torch" and hasattr(value, "detach")


def to_numpy(value):
    """Detach and snapshot tensors, recursively, for scalar graph construction."""
    if is_torch_array(value):
        tensor = value.detach().cpu()
        # NumPy has no bfloat16; fixed constants can be represented in float32.
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return tensor.numpy().copy()
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    return value


def _tensors(value):
    if is_torch_array(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


@dataclass(frozen=True)
class ArrayBackend:
    name: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self):
        if self.name not in {"numpy", "torch"}:
            raise ModelError("backend must be 'numpy' or 'torch'. Other Array API backends are not implemented yet.")
        if self.dtype not in {"float32", "float64"}:
            raise ModelError("dtype must be 'float32' or 'float64'.")
        if self.name == "numpy":
            if self.device != "cpu":
                raise ModelError("NumPy supports device='cpu' only; use backend='torch' for accelerators.")
            return
        try:
            import torch
        except ImportError as exc:
            raise ModelError("PyTorch execution needs the optional dependency: pip install 'smallprob[torch]'.") from exc
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ModelError(f"Invalid Torch device {self.device!r}.") from exc
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ModelError("Torch devices currently supported are cpu, cuda and mps.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ModelError("CUDA was requested but is not available in this PyTorch installation.")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ModelError(f"CUDA device {device.index} does not exist.")
        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ModelError("MPS was requested but is not available in this PyTorch installation.")
            if self.dtype == "float64":
                raise ModelError("MPS does not support float64; choose dtype='float32'.")
        if device.type == "cpu" and device.index is not None:
            raise ModelError("Use device='cpu' without an index.")

    @property
    def xp(self):
        if self.name == "numpy":
            return array_namespace(np.empty(0))
        import torch
        return array_namespace(torch.empty(0, device=self.device))

    @property
    def float_dtype(self):
        return getattr(self.xp, self.dtype)

    def array(self, value, *, integer=False):
        dtype = self.xp.int64 if integer else self.float_dtype
        if self.name == "numpy":
            return self.xp.asarray(to_numpy(value), dtype=dtype)
        if isinstance(value, np.generic):
            value = value.item()
        # copy=True also makes read-only NumPy snapshots safe for Torch.
        return self.xp.asarray(value, dtype=dtype, device=self.device, copy=True)

    def stack(self, values, *, axis=0, integer=False):
        return self.xp.stack([self.array(item, integer=integer) for item in values], axis=axis)

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.float_dtype, device=self.device)

    def normal(self, shape, rng):
        if self.name == "numpy":
            return self.array(rng.standard_normal(shape))
        import torch
        with self.random(int(rng.integers(0, 2**63 - 1))):
            return torch.randn(shape, dtype=getattr(torch, self.dtype), device=self.device)

    @contextmanager
    def random(self, seed):
        """Scope global Torch distribution RNG without leaking caller state."""
        if self.name != "torch":
            yield
            return
        import torch
        device = torch.device(self.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with TORCH_RNG_LOCK, torch.random.fork_rng(devices=cuda_devices):
            mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
            try:
                if seed is None:
                    seed = int(np.random.default_rng().integers(0, 2**63 - 1))
                torch.default_generator.manual_seed(int(seed))
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(int(seed))
                elif device.type == "mps":
                    torch.mps.manual_seed(int(seed))
                yield
            finally:
                if mps_state is not None:
                    torch.mps.set_rng_state(mps_state)


def infer_backend(*values, backend=None, device=None, dtype=None):
    """Infer Torch from tensors, with explicit overrides and no device fallback."""
    tensors = [tensor for value in values for tensor in _tensors(value)]
    base = backend if isinstance(backend, ArrayBackend) else None
    if base is not None:
        backend = base.name
    if backend is None:
        backend = "torch" if tensors or (device is not None and device != "cpu") else "numpy"
    devices = {str(tensor.device) for tensor in tensors}
    if len(devices) > 1 and device is None:
        raise ModelError(f"Input tensors use multiple devices {sorted(devices)}; choose device explicitly.")
    if device is None:
        device = base.device if base else (next(iter(devices)) if backend == "torch" and devices else "cpu")
    if dtype is None:
        if base is not None:
            dtype = base.dtype
        elif tensors and backend == "torch":
            dtype = "float64" if any(str(t.dtype) == "torch.float64" for t in tensors) else "float32"
        else:
            dtype = "float32" if str(device).startswith("mps") else "float64"
    return ArrayBackend(str(backend), str(device), str(dtype))


def torch_distribution(distribution, values, backend):
    """Evaluate scalar graph parameters with native Torch arithmetic."""
    import torch
    from .core import Expr, UnsupportedModelError

    def constant(value):
        if isinstance(value, Expr):
            return evaluate(value)
        if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.dtype == object):
            return torch.stack([constant(item) for item in value])
        return backend.array(value)

    def evaluate(expression):
        if expression.op == "const":
            return constant(expression.args[0])
        if expression.op == "ref":
            return values[expression.args[0]]
        if expression.op == "neg":
            return -evaluate(expression.args[0])
        left, right = expression.args
        if expression.op == "index":
            raw = right.eval({}) if not right.dependencies() else evaluate(right).to(torch.int64)
            return evaluate(left)[raw]
        a, b = evaluate(left), evaluate(right)
        if expression.op == "add":
            return a + b
        if expression.op == "sub":
            return a - b
        if expression.op == "mul":
            return a * b
        if expression.op == "div":
            return a / b
        if expression.op == "pow":
            return a ** b
        raise UnsupportedModelError(f"Torch does not implement expression {expression.op!r}.")

    family = type(distribution).__name__
    try:
        if family == "Normal":
            return torch.distributions.Normal(evaluate(distribution.mean), evaluate(distribution.sd))
        if family == "HalfNormal":
            return torch.distributions.HalfNormal(evaluate(distribution.sd))
        if family == "StudentT":
            return stable_student_t_class(torch.distributions.StudentT)(
                evaluate(distribution.df), evaluate(distribution.loc), evaluate(distribution.scale))
        if family == "Categorical":
            probabilities = evaluate(distribution.probs)
            # Torch's Categorical silently normalises; maintain model semantics.
            if probabilities.ndim != 1 or not probabilities.numel():
                raise ModelError("Categorical probabilities must be a nonempty one-dimensional vector.")
            if not bool(torch.all(torch.isfinite(probabilities))) or bool(torch.any(probabilities < 0)):
                raise ModelError("Categorical probabilities must be finite and nonnegative.")
            tolerance = max(1e-8, 4 * torch.finfo(probabilities.dtype).eps)
            if not bool(torch.isclose(probabilities.sum(), backend.array(1), atol=tolerance, rtol=0)):
                raise ModelError("Categorical probabilities must sum to 1; normalise them explicitly.")
            return torch.distributions.Categorical(probs=probabilities)
    except ValueError as exc:
        raise ModelError(f"Invalid Torch {family} parameters: {exc}") from exc
    raise UnsupportedModelError(f"Torch execution does not implement {family}.")
