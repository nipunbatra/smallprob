"""Optional full-rank Gaussian variational inference for continuous models.

The model graph is lowered to differentiable PyTorch operations on the selected device. Pyro
optimises an AutoMultivariateNormal guide with stochastic ELBO gradients; no
exact posterior is used for initialisation or optimisation. Returned results
contain only detached NumPy quantities and remain usable without Pyro state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import numbers
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from .core import Expr, Graph, HalfNormal, InferenceError, Normal, StudentT, UnsupportedModelError

from .arrays import TORCH_RNG_LOCK, stable_student_t_class

_BACKEND_LOCK = TORCH_RNG_LOCK
_SUPPORTED_OPS = {"const", "ref", "neg", "add", "sub", "mul", "div", "pow", "index"}


@dataclass
class VIResult:
    """A detached full-rank guide; positive sites have lognormal marginals."""

    means: dict[str, float]
    variances: dict[str, float]
    elbo: float
    diagnostics: dict[str, Any]
    _latent_names: tuple[str, ...] = field(repr=False)
    _location: np.ndarray = field(repr=False)
    _factor: np.ndarray = field(repr=False)
    _positive: np.ndarray = field(repr=False)
    _observed: dict[str, float] = field(repr=False)
    method: str = field(default="vi-fullrank", init=False)
    log_evidence: None = field(default=None, init=False)
    marginals: dict = field(default_factory=dict, init=False)

    def sample(self, draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 1:
            raise ValueError("draws must be a positive integer.")
        samples = self._location + rng.standard_normal((int(draws), len(self._location))) @ self._factor.T
        with np.errstate(over="ignore", under="ignore"):
            samples[:, self._positive] = np.exp(samples[:, self._positive])
        if not np.all(np.isfinite(samples)) or np.any(samples[:, self._positive] <= 0):
            raise InferenceError("Variational samples exceeded floating point range; rescale the model.")
        result = {name: samples[:, i] for i, name in enumerate(self._latent_names)}
        result.update({name: np.full(draws, value) for name, value in self._observed.items()})
        return result

    def interval(self, probability: float = 0.95) -> tuple[dict[str, float], dict[str, float]]:
        if isinstance(probability, (bool, np.bool_)) or not isinstance(probability, numbers.Real) or not 0 < probability < 1:
            raise ValueError("probability must be between 0 and 1.")
        quantile = -NormalDist().inv_cdf((1 - float(probability)) / 2)
        scales = np.sqrt(np.sum(self._factor ** 2, axis=1))
        lo, hi = self._location - quantile * scales, self._location + quantile * scales
        with np.errstate(over="ignore", under="ignore"):
            lo[self._positive], hi[self._positive] = np.exp(lo[self._positive]), np.exp(hi[self._positive])
        if not np.all(np.isfinite(hi)) or np.any(lo[self._positive] <= 0):
            raise InferenceError("Variational interval exceeded floating point range; rescale the model.")
        lower = dict(zip(self._latent_names, lo.tolist()))
        upper = dict(zip(self._latent_names, hi.tolist()))
        lower.update(self._observed)
        upper.update(self._observed)
        return lower, upper


@dataclass
class _Compiled:
    # All lowered tensors have a leading particle axis, including constants
    # (size 1). Model array axes follow it, so fixed indexing never indexes the
    # particles or accidentally broadcasts them against model coordinates.
    shape: tuple[int, ...]
    evaluate: Callable


def _compiler(torch, *, device="cpu", dtype=None):
    dtype = torch.float64 if dtype is None else dtype
    cache: dict[int, _Compiled] = {}

    def constant(value) -> _Compiled:
        if isinstance(value, Expr):
            return compile_expr(value)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return constant(value.item())
        if isinstance(value, (list, tuple, np.ndarray)):
            if isinstance(value, np.ndarray):
                pieces = [constant(item) for item in value]
            else:
                pieces = [constant(item) for item in value]
            if not pieces:
                return _Compiled((0,), lambda values: torch.empty((1, 0), dtype=dtype, device=device))
            if len({piece.shape for piece in pieces}) != 1:
                raise UnsupportedModelError("VI requires rectangular numeric containers in expressions.")
            shape = (len(pieces),) + pieces[0].shape

            def evaluate(values):
                tensors = torch.broadcast_tensors(*(piece.evaluate(values) for piece in pieces))
                return torch.stack(tensors, dim=1)

            return _Compiled(shape, evaluate)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise UnsupportedModelError(f"VI requires real numeric constants; got {type(value).__name__}.")
        number = float(value)
        if not np.isfinite(number):
            raise InferenceError("VI expression constants must be finite.")
        tensor = torch.tensor([number], dtype=dtype, device=device)
        return _Compiled((), lambda values: tensor)

    def compile_expr(expr: Expr) -> _Compiled:
        key = id(expr)
        if key in cache:
            return cache[key]
        if expr.op not in _SUPPORTED_OPS:
            raise UnsupportedModelError(f"Expression {expr.op!r} is not supported by VI.")
        if expr.op == "const":
            result = constant(expr.args[0])
        elif expr.op == "ref":
            name = expr.args[0]
            result = _Compiled((), lambda values: values[name])
        elif expr.op == "neg":
            argument = compile_expr(expr.args[0])
            result = _Compiled(argument.shape, lambda values: -argument.evaluate(values))
        elif expr.op == "index":
            base, index_expr = expr.args
            if index_expr.dependencies():
                raise UnsupportedModelError("VI supports only fixed indices; a continuous random variable cannot be an index.")
            try:
                index = index_expr.eval({})
                if isinstance(index, (bool, np.bool_)):
                    raise TypeError("boolean indices are not supported")
                compiled_base = compile_expr(base)
                # NumPy supplies the intended model shape; indexing below keeps
                # the separate leading particle dimension intact.
                shape = np.empty(compiled_base.shape)[index].shape
                indices = index if isinstance(index, tuple) else (index,)
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise UnsupportedModelError(f"Invalid fixed VI index: {exc}") from exc
            result = _Compiled(shape, lambda values: compiled_base.evaluate(values)[(slice(None),) + indices])
        else:
            left, right = map(compile_expr, expr.args)
            try:
                shape = np.broadcast_shapes(left.shape, right.shape)
            except ValueError as exc:
                raise UnsupportedModelError("VI expression array shapes cannot be broadcast together.") from exc
            operation = {"add": torch.add, "sub": torch.sub, "mul": torch.mul,
                         "div": torch.div, "pow": torch.pow}[expr.op]

            def evaluate(values):
                lhs, rhs = left.evaluate(values), right.evaluate(values)
                lhs = lhs.reshape((lhs.shape[0],) + (1,) * (len(shape) - len(left.shape)) + left.shape)
                rhs = rhs.reshape((rhs.shape[0],) + (1,) * (len(shape) - len(right.shape)) + right.shape)
                return operation(lhs, rhs)

            result = _Compiled(shape, evaluate)
        cache[key] = result
        return result

    def batch(expressions):
        """Fuse equally structured scalar expressions across observations.

        Regression observations become vector arithmetic, while heterogeneous
        graphs fall back to stacking ordinary compiled expressions. This is an
        optimisation only: the same lowering and checks apply in both cases.
        """
        expressions = tuple(expressions)
        compiled = tuple(compile_expr(expression) for expression in expressions)
        if any(item.shape for item in compiled):
            raise UnsupportedModelError("VI distributions require scalar means and scales at every site.")
        ops = {expression.op for expression in expressions}
        if ops == {"const"} and all(not expression.dependencies() for expression in expressions):
            vector = torch.tensor([[float(expression.eval({})) for expression in expressions]], dtype=dtype, device=device)
            return lambda values: vector
        if ops == {"ref"} and len({expression.args[0] for expression in expressions}) == 1:
            name = expressions[0].args[0]
            return lambda values: values[name].unsqueeze(-1).expand(-1, len(expressions))
        if len(ops) == 1 and next(iter(ops)) in {"add", "sub", "mul", "div", "pow"}:
            left = batch([expression.args[0] for expression in expressions])
            right = batch([expression.args[1] for expression in expressions])
            operation = {"add": torch.add, "sub": torch.sub, "mul": torch.mul,
                         "div": torch.div, "pow": torch.pow}[next(iter(ops))]
            return lambda values: operation(left(values), right(values))
        if ops == {"neg"}:
            operand = batch([expression.args[0] for expression in expressions])
            return lambda values: -operand(values)
        return lambda values: torch.stack(torch.broadcast_tensors(*(item.evaluate(values) for item in compiled)), dim=-1)

    return compile_expr, batch


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@contextmanager
def _isolated_rng(torch, device, seed):
    """Seed only the selected device and restore CPU/accelerator RNG states."""
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == "mps":
                torch.mps.manual_seed(int(seed))
            elif device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _execution_options(torch, device, dtype):
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise InferenceError(f"Invalid VI device {device!r}.") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise InferenceError("VI supports cpu, cuda or mps devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA was requested, but no CUDA device is available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise InferenceError("MPS was requested, but no Apple GPU is available.")
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if dtype not in {torch.float32, torch.float64}:
        raise InferenceError("VI dtype must be float32 or float64.")
    if device.type == "mps" and dtype == torch.float64:
        raise InferenceError("MPS does not support float64; select dtype='float32'.")
    return device, dtype


def infer_vi(graph: Graph, observed_flat: dict[str, float], *, steps=3000,
             learning_rate=0.03, particles=32, seed=0, device="cpu", dtype="float64") -> VIResult:
    """Fit a full-covariance Gaussian guide in unconstrained latent space.

    HalfNormal sites use the exponential support transform, so their returned
    marginals are lognormal. The final ELBO is an independent Monte Carlo
    estimate, never a claimed log evidence or convergence guarantee.
    """
    steps, particles = _positive_integer(steps, "steps"), _positive_integer(particles, "particles")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(learning_rate, numbers.Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    names = [node.name for node in graph.nodes]
    if len(set(names)) != len(names):
        raise InferenceError("The model graph contains duplicate random-variable names.")
    unknown = set(observed_flat) - set(names)
    if unknown:
        raise InferenceError(f"Observations refer to unknown sites: {sorted(unknown)}.")
    observed = {}
    seen = set()
    for node in graph.nodes:
        if not isinstance(node.distribution, (Normal, HalfNormal, StudentT)):
            raise UnsupportedModelError("VI currently supports only Normal, HalfNormal and StudentT variables; use exact HMM inference for Categorical states.")
        missing = node.distribution.dependencies() - seen
        if missing:
            raise InferenceError(f"Site {node.name!r} refers to unavailable or future sites {sorted(missing)}.")
        seen.add(node.name)
        if node.name in observed_flat:
            value = observed_flat[node.name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
                raise InferenceError(f"Observation {node.name!r} must be a finite real scalar.")
            if isinstance(node.distribution, HalfNormal) and value < 0:
                raise InferenceError(f"Observed HalfNormal variable {node.name!r} must be nonnegative.")
            observed[node.name] = float(value)
    latent_nodes = [node for node in graph.nodes if node.name not in observed]
    if not latent_nodes:
        raise UnsupportedModelError("VI requires at least one unobserved variable to optimise; use exact inference for a fully observed Gaussian model.")
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.distributions.transforms import biject_to
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoMultivariateNormal
        from pyro.infer.autoguide.initialization import init_to_median
        from pyro.optim import ClippedAdam
    except ImportError as exc:
        raise ImportError("VI needs the optional Pyro/PyTorch backend. Install it with: pip install 'smallprob[vi]'") from exc

    device, dtype = _execution_options(torch, device, dtype)
    with _BACKEND_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            # Small dense CPU models are markedly faster with one BLAS thread.
            if device.type == "cpu":
                torch.set_num_threads(1)
            with _isolated_rng(torch, device, seed), pyro.get_param_store().scope():
                compile_expr, compile_batch = _compiler(torch, device=device, dtype=dtype)
                def parameter_expressions(distribution):
                    if isinstance(distribution, StudentT):
                        return distribution.df, distribution.loc, distribution.scale
                    if isinstance(distribution, Normal):
                        return distribution.mean, distribution.sd
                    return (distribution.sd,)

                distribution_classes = {Normal: dist.Normal, HalfNormal: dist.HalfNormal,
                                        StudentT: stable_student_t_class(dist.StudentT)}
                compiled_parameters = {}
                for node in graph.nodes:
                    parameters = tuple(compile_expr(expr) for expr in parameter_expressions(node.distribution))
                    if any(parameter.shape for parameter in parameters):
                        raise UnsupportedModelError(f"VI requires scalar distribution parameters at {node.name!r}.")
                    compiled_parameters[node.name] = parameters
                observed_tensors = {name: torch.tensor([value], dtype=dtype, device=device) for name, value in observed.items()}
                observed_groups = []
                for distribution_type in (Normal, HalfNormal, StudentT):
                    nodes = [node for node in graph.nodes if node.name in observed and isinstance(node.distribution, distribution_type)]
                    if nodes:
                        parameters = [parameter_expressions(node.distribution) for node in nodes]
                        batches = tuple(compile_batch(expressions) for expressions in zip(*parameters))
                        data = torch.tensor([[observed[node.name] for node in nodes]], dtype=dtype, device=device)
                        observed_groups.append((distribution_type, batches, data))
                factor_name = "_smallprob_observed_log_prob"
                while factor_name in names:
                    factor_name += "_"

                def distribution_for(node, values):
                    parameters = [parameter.evaluate(values).squeeze(0) for parameter in compiled_parameters[node.name]]
                    return distribution_classes[type(node.distribution)](*parameters, validate_args=True)

                def observed_log_prob(values):
                    total = torch.zeros(1, dtype=dtype, device=device)
                    for kind, batches, data in observed_groups:
                        distribution = distribution_classes[kind](*(batch(values) for batch in batches), validate_args=True)
                        total = total + distribution.log_prob(data).sum(-1)
                    return total

                def model():
                    values = dict(observed_tensors)
                    for node in latent_nodes:
                        values[node.name] = pyro.sample(node.name, distribution_for(node, values)).reshape(-1)
                    if observed_groups:
                        pyro.factor(factor_name, observed_log_prob(values).squeeze(0))
                    return values

                guide = AutoMultivariateNormal(model, init_loc_fn=init_to_median(num_samples=15), init_scale=0.25)
                # Users may explicitly name sites; avoid collisions with the
                # guide's auxiliary sample and parameter names. This is set
                # before the guide creates parameters or a prototype trace.
                guide_name = guide._pyro_name
                while any(name == f"_{guide_name}_latent" or name.startswith(guide_name + ".") for name in names):
                    guide_name += "_"
                guide._pyro_name = guide_name
                decay = 0.1 ** (1 / steps)
                optimiser = ClippedAdam({"lr": float(learning_rate), "lrd": decay, "clip_norm": 10.0})
                objective = Trace_ELBO(num_particles=particles, vectorize_particles=True, max_plate_nesting=0)
                svi = SVI(model, guide, optimiser, objective)
                history = []
                for step in range(steps):
                    try:
                        loss = float(svi.step())
                    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as exc:
                        raise InferenceError(
                            f"VI failed at optimisation step {step + 1}: invalid parameters or numerical overflow. "
                            f"Check positive scales, rescale inputs, or reduce learning_rate. Backend detail: {exc}"
                        ) from exc
                    if not np.isfinite(loss):
                        raise InferenceError(f"VI produced a nonfinite loss at step {step + 1}; rescale the model or reduce learning_rate.")
                    history.append(loss)

                with torch.no_grad():
                    posterior = guide.get_posterior()
                    location = posterior.loc.detach().cpu().numpy().copy()
                    factor = posterior.scale_tril.detach().cpu().numpy().copy()
                    if not np.all(np.isfinite(location)) or not np.all(np.isfinite(factor)):
                        raise InferenceError("VI guide parameters are nonfinite; rescale the model or reduce learning_rate.")
                    # Verify the guide's actual packed ordering and support
                    # transforms before using analytic moments or NumPy draws.
                    order, positive = [], []
                    probe = torch.arange(len(location), dtype=dtype, device=device)
                    test_points = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)
                    for site, unpacked in guide._unpack_latent(probe):
                        if unpacked.numel() != 1 or float(unpacked) != len(order):
                            raise InferenceError("Unexpected Pyro guide latent packing; this backend requires scalar sites.")
                        name = site["name"]
                        transform = biject_to(site["fn"].support)
                        is_positive = isinstance(next(node.distribution for node in latent_nodes if node.name == name), HalfNormal)
                        expected = test_points.exp() if is_positive else test_points
                        if not torch.allclose(transform(test_points), expected) or not torch.allclose(transform.inv(expected), test_points):
                            raise InferenceError(f"Unexpected Pyro support transform at {name!r}; cannot detach this guide safely.")
                        order.append(name)
                        positive.append(is_positive)
                    if set(order) != {node.name for node in latent_nodes}:
                        raise InferenceError("Pyro guide latent sites do not match the model graph.")
                    positive = np.asarray(positive, dtype=bool)
                    # Independent held-out particles give both an ELBO estimate
                    # and its Monte Carlo standard error for this fitted guide.
                    evaluation_particles = 2048
                    unconstrained = posterior.sample((evaluation_particles,))
                    values = dict(observed_tensors)
                    for i, name in enumerate(order):
                        value = unconstrained[:, i]
                        values[name] = value.exp() if positive[i] else value
                    log_joint = observed_log_prob(values)
                    for node in latent_nodes:
                        log_joint = log_joint + distribution_for(node, values).log_prob(values[node.name])
                    log_q = posterior.log_prob(unconstrained)
                    if positive.any():
                        mask = torch.as_tensor(positive, device=device)
                        log_q = log_q - unconstrained[:, mask].sum(-1)
                    estimates = log_joint - log_q
                    if not bool(torch.isfinite(estimates).all()):
                        raise InferenceError("Held-out VI ELBO exceeded numeric range; rescale the model.")
                    elbo = float(estimates.mean())
                    elbo_se = float(estimates.std(unbiased=True) / np.sqrt(evaluation_particles))
        finally:
            torch.set_num_threads(previous_threads)

    covariance = factor @ factor.T
    diagonal = np.diag(covariance)
    mean, variance = location.copy(), diagonal.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean[positive] = np.exp(location[positive] + diagonal[positive] / 2)
        variance[positive] = np.expm1(diagonal[positive]) * np.exp(2 * location[positive] + diagonal[positive])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)) or np.any(mean[positive] <= 0):
        raise InferenceError("VI marginal moments exceeded floating point range; rescale the model.")
    means, variances = dict(zip(order, mean.tolist())), dict(zip(order, variance.tolist()))
    means.update(observed)
    variances.update({name: 0.0 for name in observed})
    window = min(100, max(1, steps // 10))
    warnings = ["VI is approximate; a finite ELBO and decreasing loss do not establish convergence.",
                "The guide is Gaussian in unconstrained space and cannot represent arbitrary multimodal posteriors."]
    if steps < 500:
        warnings.append("Fewer than 500 optimisation steps requested; use this only for smoke checks, not validated inference.")
    diagnostics = {
        "backend": "pyro-ppl", "backend_version": pyro.__version__, "torch_version": torch.__version__,
        "dtype": str(dtype).removeprefix("torch."), "device": str(device), "family": "full-rank Gaussian in unconstrained space",
        "seed": int(seed), "steps": steps, "particles": particles, "learning_rate": float(learning_rate),
        "learning_rate_schedule": {"type": "exponential", "per_step_multiplier": decay, "final_factor": 0.1},
        "optimiser": "ClippedAdam", "gradient_clip_norm": 10.0, "loss_history": history,
        "loss_summary_window": window, "initial_loss_mean": float(np.mean(history[:window])),
        "final_loss_mean": float(np.mean(history[-window:])), "warnings": warnings,
        "latent_order": order, "unconstrained_mean": location.tolist(),
        "unconstrained_covariance": covariance.tolist(),
        "transforms": {name: "exp" if positive[i] else "identity" for i, name in enumerate(order)},
        "elbo_estimator": "independent held-out Monte Carlo", "elbo_particles": evaluation_particles,
        "elbo_standard_error": elbo_se, "interval_kind": "equal-tail marginal guide interval",
    }
    return VIResult(means, variances, elbo, diagnostics, tuple(order), location, factor, positive, observed)
