"""Small, exact inference engines selected from the model's dependency graph.

The Gaussian engine conditions a dense joint Gaussian. The HMM engine uses
log-space forward-backward and samples complete state trajectories backwards.
Neither engine estimates distribution parameters or silently approximates an
unsupported model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import (
    Categorical,
    Expr,
    Graph,
    InferenceError,
    Normal,
    HalfNormal,
    UnsupportedModelError,
)


@dataclass
class EngineResult:
    """Flat posterior quantities consumed by the public API."""

    method: str
    log_evidence: float
    means: dict[str, float]
    variances: dict[str, float]
    marginals: dict[str, np.ndarray]
    _sampler: Callable = field(repr=False)

    def sample(self, draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 1:
            raise ValueError("draws must be a positive integer.")
        return self._sampler(int(draws), rng)


def infer(graph: Graph, observed: dict[str, float], method: str = "auto", *, backend=None, **options) -> EngineResult:
    """Select an exact engine, or explicit VI, using the same model graph."""
    from .arrays import infer_backend
    selected = infer_backend(backend=backend)
    if method == "vi":
        guide = options.pop("guide", "full")
        if guide == "diagonal":
            from .meanfield import infer_meanfield
            return infer_meanfield(graph, dict(observed), device=selected.device, dtype=selected.dtype, **options)
        if guide != "full":
            raise ValueError("guide must be 'diagonal' (native PyTorch) or 'full' (Pyro).")
        from .vi import infer_vi
        return infer_vi(graph, dict(observed), device=selected.device, dtype=selected.dtype, **options)
    if options:
        raise InferenceError(f"Options {sorted(options)} apply only to method='vi'; exact inference has no optimisation settings.")
    unsupported_families = {type(node.distribution).__name__ for node in graph.nodes
                            if type(node.distribution).__name__ in {"HalfNormal", "StudentT"}}
    if unsupported_families:
        raise UnsupportedModelError(f"{', '.join(sorted(unsupported_families))} models need explicit method='vi'; the exact engines support linear Gaussian models and fixed-parameter HMMs.")
    allowed = {"auto", "exact", "gaussian", "hmm", "exact-gaussian", "exact-hmm"}
    if method not in allowed:
        raise ValueError(f"Unknown inference method {method!r}; use 'auto', 'gaussian' or 'hmm'.")
    names = [node.name for node in graph.nodes]
    if len(set(names)) != len(names):
        raise InferenceError("The model graph contains duplicate random-variable names.")
    unknown = set(observed) - set(names)
    if unknown:
        raise InferenceError(f"Observations refer to unknown sites: {sorted(unknown)}.")
    seen: set[str] = set()
    for node in graph.nodes:
        if not isinstance(node.distribution, (Normal, Categorical)):
            raise UnsupportedModelError(f"Unsupported distribution at {node.name!r}.")
        missing = node.distribution.dependencies() - seen
        if missing:
            raise InferenceError(
                f"Site {node.name!r} refers to unavailable or future sites {sorted(missing)}."
            )
        seen.add(node.name)
    observations = dict(observed)
    all_gaussian = all(isinstance(node.distribution, Normal) for node in graph.nodes)
    if selected.name == "torch" and (not all_gaussian or method in {"hmm", "exact-hmm"}):
        raise UnsupportedModelError("Exact HMM inference currently runs on NumPy/CPU. Request backend='numpy', device='cpu' explicitly; no GPU fallback is performed.")
    if method in {"gaussian", "exact-gaussian"} and not all_gaussian:
        raise UnsupportedModelError("The Gaussian engine requires only Normal random variables.")
    if method in {"hmm", "exact-hmm"}:
        return _infer_hmm(graph, observations)
    if all_gaussian:
        if selected.name != "numpy" or selected.dtype != "float64":
            return _infer_gaussian_array(graph, observations, selected)
        return _infer_gaussian(graph, observations)
    return _infer_hmm(graph, observations)


def _scalar(value, context: str) -> float:
    try:
        array = np.asarray(value)
        if array.ndim != 0 or np.iscomplexobj(array):
            raise ValueError
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InferenceError(f"{context} must be a finite real scalar.") from exc
    if not np.isfinite(result):
        raise InferenceError(f"{context} must be a finite real scalar.")
    return result


def _constant(expr: Expr, context: str):
    try:
        return expr.eval({})
    except Exception as exc:
        raise InferenceError(f"Could not evaluate {context}: {exc}") from exc


def _affine(expr: Expr):
    """Return offset and coefficients, or reject random-dependent nonlinearity."""
    if not expr.dependencies():
        return np.asarray(_constant(expr, "a constant expression")), {}
    if expr.op == "ref":
        return np.asarray(0.0), {expr.args[0]: np.asarray(1.0)}
    if expr.op == "neg":
        offset, coefficients = _affine(expr.args[0])
        return -offset, {name: -value for name, value in coefficients.items()}
    if expr.op in {"add", "sub"}:
        left, lc = _affine(expr.args[0])
        right, rc = _affine(expr.args[1])
        sign = 1 if expr.op == "add" else -1
        coefficients = dict(lc)
        for name, value in rc.items():
            coefficients[name] = coefficients.get(name, 0) + sign * value
        return left + sign * right, coefficients
    if expr.op == "mul":
        left, lc = _affine(expr.args[0])
        right, rc = _affine(expr.args[1])
        if lc and rc:
            raise UnsupportedModelError("A product of random quantities is not an affine Gaussian mean.")
        return left * right, (
            {name: value * right for name, value in lc.items()}
            if lc else {name: left * value for name, value in rc.items()}
        )
    if expr.op == "div":
        left, coefficients = _affine(expr.args[0])
        right, denominator_coefficients = _affine(expr.args[1])
        if denominator_coefficients:
            raise UnsupportedModelError("Division by a random quantity is not an affine Gaussian mean.")
        if np.any(np.asarray(right) == 0):
            raise InferenceError("A Gaussian mean contains division by zero.")
        return left / right, {name: value / right for name, value in coefficients.items()}
    if expr.op == "pow":
        base, exponent = expr.args
        if not exponent.dependencies():
            power = np.asarray(_constant(exponent, "an exponent"))
            if power.ndim == 0 and power == 1:
                return _affine(base)
            if power.ndim == 0 and power == 0:
                return np.asarray(1.0), {}
        raise UnsupportedModelError("Powers of random quantities are not supported by exact Gaussian inference.")
    if expr.op == "index":
        base, index = expr.args
        if index.dependencies():
            raise UnsupportedModelError("Indexing by a random variable requires a supported categorical HMM.")
        offset, coefficients = _affine(base)
        idx = _constant(index, "an index")
        try:
            return offset[idx], {name: value[idx] for name, value in coefficients.items()}
        except (IndexError, TypeError) as exc:
            raise InferenceError(f"Invalid indexing in a Gaussian mean: {exc}") from exc
    raise UnsupportedModelError(
        f"Expression {expr.op!r} is not supported by exact Gaussian inference. "
        "Use affine means and fixed positive noise scales."
    )


def _infer_gaussian(graph: Graph, observed: dict[str, float]) -> EngineResult:
    names = [node.name for node in graph.nodes]
    position = {name: i for i, name in enumerate(names)}
    count = len(names)
    prior_mean = np.zeros(count)
    # factor maps independent standard-normal noise to this topological graph.
    factor = np.zeros((count, count))
    for i, node in enumerate(graph.nodes):
        distribution = node.distribution
        if distribution.sd.dependencies():
            raise UnsupportedModelError(
                f"Noise scale at {node.name!r} depends on a random variable. "
                "The exact Gaussian engine needs fixed noise scales; request method='vi' for continuous models with unknown noise."
            )
        sd = _scalar(_constant(distribution.sd, f"noise scale at {node.name!r}"), f"Noise scale at {node.name!r}")
        if sd <= 0:
            raise InferenceError(f"Noise scale at {node.name!r} must be positive.")
        try:
            offset, coefficients = _affine(distribution.mean)
            prior_mean[i] = _scalar(offset, f"Mean offset at {node.name!r}")
            for parent, coefficient in coefficients.items():
                coefficient = _scalar(coefficient, f"Coefficient of {parent!r} at {node.name!r}")
                j = position[parent]
                prior_mean[i] += coefficient * prior_mean[j]
                factor[i] += coefficient * factor[j]
            factor[i, i] = sd
        except UnsupportedModelError as exc:
            raise UnsupportedModelError(f"At site {node.name!r}: {exc}") from exc
    if not np.all(np.isfinite(prior_mean)) or not np.all(np.isfinite(factor)):
        raise InferenceError("Gaussian moments overflowed; rescale the model's inputs and parameters.")
    mean = prior_mean.copy()
    conditional_factor = factor.copy()
    observed_indices = np.array([position[name] for name in names if name in observed], dtype=int)
    observed_values = np.array([_scalar(observed[names[i]], f"Observation {names[i]!r}") for i in observed_indices])
    evidence = 0.0
    if len(observed_indices):
        observed_count = len(observed_indices)
        try:
            # In innovation coordinates, observations constrain the row space
            # of A=F[observed]. Complete QR gives its orthogonal complement
            # directly, preserving tiny uncertainties without subtracting
            # either covariance matrices or nearly equal innovation factors.
            orthogonal, triangular = np.linalg.qr(factor[observed_indices, :].T, mode="complete")
            observed_factor = triangular[:observed_count, :]
            constrained = factor @ orthogonal[:, :observed_count]
            gain = np.linalg.solve(observed_factor, constrained.T).T
            residual = observed_values - prior_mean[observed_indices]
            standard = np.linalg.solve(observed_factor.T, residual)
        except np.linalg.LinAlgError as exc:
            raise InferenceError(
                "The observed Gaussian innovation factor is numerically singular. "
                "Rescale inputs, priors or very small noise scales."
            ) from exc
        mean += gain @ residual
        conditional_factor = factor @ orthogonal[:, observed_count:]
        conditional_factor[observed_indices, :] = 0
        mean[observed_indices] = observed_values
        evidence = -0.5 * (
            len(observed_indices) * np.log(2 * np.pi)
            + 2 * np.log(np.abs(np.diag(observed_factor))).sum()
            + standard @ standard
        )
        if not np.isfinite(evidence) or not np.all(np.isfinite(mean)):
            raise InferenceError("Gaussian conditioning exceeded numeric range; rescale the observations and model.")
    variances = np.sum(conditional_factor ** 2, axis=1)
    if not np.all(np.isfinite(variances)):
        raise InferenceError("Gaussian posterior variances exceeded numeric range; rescale the model.")

    def sample(draws, rng):
        # Reported moments and joint samples use the same free innovations.
        samples = mean + rng.standard_normal((draws, conditional_factor.shape[1])) @ conditional_factor.T
        return {name: samples[:, i] for i, name in enumerate(names)}

    return EngineResult(
        "exact-gaussian", float(evidence),
        dict(zip(names, mean.tolist())), dict(zip(names, variances.tolist())), {}, sample,
    )


def _infer_gaussian_array(graph, observed, backend):
    """Square-root Gaussian conditioning in the selected Array API namespace.

Only the symbolic affine coefficients are extracted on the host. The dense
innovation factor, QR factorisation, solves, moments and sampling stay native.
"""
    xp = backend.xp
    names = [node.name for node in graph.nodes]
    position = {name: i for i, name in enumerate(names)}
    count = len(names)
    prior_mean, factor = backend.zeros((count,)), backend.zeros((count, count))
    for i, node in enumerate(graph.nodes):
        distribution = node.distribution
        if distribution.sd.dependencies():
            raise UnsupportedModelError(
                f"Noise scale at {node.name!r} depends on a random variable. "
                "The exact Gaussian engine needs fixed noise scales; request method='vi' for continuous models with unknown noise."
            )
        sd = _scalar(_constant(distribution.sd, f"noise scale at {node.name!r}"), f"Noise scale at {node.name!r}")
        if sd <= 0:
            raise InferenceError(f"Noise scale at {node.name!r} must be positive.")
        try:
            offset, coefficients = _affine(distribution.mean)
            prior_mean[i] = _scalar(offset, f"Mean offset at {node.name!r}")
            for parent, coefficient in coefficients.items():
                coefficient = _scalar(coefficient, f"Coefficient of {parent!r} at {node.name!r}")
                j = position[parent]
                prior_mean[i] += coefficient * prior_mean[j]
                factor[i] += coefficient * factor[j]
            factor[i, i] = sd
        except UnsupportedModelError as exc:
            raise UnsupportedModelError(f"At site {node.name!r}: {exc}") from exc
    if not bool(xp.all(xp.isfinite(prior_mean))) or not bool(xp.all(xp.isfinite(factor))):
        raise InferenceError("Gaussian moments overflowed; rescale the model's inputs and parameters.")
    mean = backend.array(prior_mean)
    conditional_factor = backend.array(factor)
    observed_names = [name for name in names if name in observed]
    indices = backend.array([position[name] for name in observed_names], integer=True)
    observed_values = backend.array([observed[name] for name in observed_names])
    evidence = 0.0
    if observed_names:
        observed_count = len(observed_names)
        try:
            orthogonal, triangular = xp.linalg.qr(factor[indices, :].T, mode="complete")
            observed_factor = triangular[:observed_count, :]
            constrained = factor @ orthogonal[:, :observed_count]
            gain = xp.linalg.solve(observed_factor, constrained.T).T
            residual = observed_values - prior_mean[indices]
            standard = xp.linalg.solve(observed_factor.T, residual)
            mean += gain @ residual
            conditional_factor = factor @ orthogonal[:, observed_count:]
            conditional_factor[indices, :] = 0
            mean[indices] = observed_values
            evidence = -0.5 * (observed_count * np.log(2 * np.pi)
                               + 2 * xp.sum(xp.log(xp.abs(xp.linalg.diagonal(observed_factor))))
                               + standard @ standard)
        except (np.linalg.LinAlgError, RuntimeError, NotImplementedError) as exc:
            raise InferenceError(
                f"Gaussian conditioning failed on {backend.name}/{backend.device}/{backend.dtype}: {exc}. "
                "Check numeric scale and device operation support; no CPU fallback is performed."
            ) from exc
        if not bool(xp.isfinite(evidence)) or not bool(xp.all(xp.isfinite(mean))):
            raise InferenceError("Gaussian conditioning exceeded numeric range; rescale the observations and model.")
    variances = xp.sum(conditional_factor ** 2, axis=1)
    if not bool(xp.all(xp.isfinite(variances))):
        raise InferenceError("Gaussian posterior variances exceeded numeric range; rescale the model.")

    def sample(draws, rng):
        samples = mean + backend.normal((draws, conditional_factor.shape[1]), rng) @ conditional_factor.T
        return {name: samples[:, i] for i, name in enumerate(names)}

    return EngineResult("exact-gaussian", float(evidence),
                        {name: mean[i] for i, name in enumerate(names)},
                        {name: variances[i] for i, name in enumerate(names)}, {}, sample)


def _probabilities(distribution: Categorical, values: dict, context: str) -> np.ndarray:
    try:
        # Share dtype, support and normalisation checks with generation, so a
        # symbolic row cannot acquire different semantics during inference.
        return distribution._probabilities(values)
    except Exception as exc:
        raise InferenceError(f"Could not evaluate {context}: {exc}") from exc


def _log_probabilities(probabilities):
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


def _logsumexp(values, axis=None):
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shifted = np.where(np.isfinite(maximum), values - maximum, -np.inf)
        result = maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.item()


def _normal_parameters(distribution: Normal, values, context):
    try:
        return distribution._parameters(values)
    except Exception as exc:
        raise InferenceError(f"Could not evaluate Normal parameters {context}: {exc}") from exc


def _normal_log_prob(value, means, scales):
    with np.errstate(over="ignore"):
        return -0.5 * ((value - means) / scales) ** 2 - np.log(scales) - 0.5 * np.log(2 * np.pi)


def _infer_hmm(graph: Graph, observed: dict[str, float]) -> EngineResult:
    states = [node for node in graph.nodes if isinstance(node.distribution, Categorical)]
    emissions = [node for node in graph.nodes if isinstance(node.distribution, Normal)]
    if not states:
        raise UnsupportedModelError("The HMM engine requires at least one Categorical state.")
    state_index = {node.name: t for t, node in enumerate(states)}
    if states[0].distribution.dependencies():
        raise UnsupportedModelError("The first HMM state's probabilities must be fixed inputs.")
    initial = _probabilities(states[0].distribution, {}, "Initial state probabilities")
    k, length = len(initial), len(states)
    transitions = np.empty((max(length - 1, 0), k, k))
    for t, node in enumerate(states[1:], 1):
        previous = states[t - 1].name
        parents = node.distribution.dependencies()
        if parents - {previous}:
            raise UnsupportedModelError(
                f"State {node.name!r} must depend only on the previous state {previous!r}. "
                "This prototype supports one categorical chain with fixed transition probabilities."
            )
        for category in range(k):
            row = _probabilities(node.distribution, {previous: category}, f"Transition probabilities at {node.name!r}")
            if len(row) != k:
                raise UnsupportedModelError("All HMM states must have the same number of categories.")
            transitions[t - 1, category] = row
    log_transitions = _log_probabilities(transitions)
    log_likelihood = np.zeros((length, k))
    independent_evidence = 0.0
    # Each entry contains its state time, or None for an independent Normal.
    parameters = {}
    for node in emissions:
        dependencies = node.distribution.dependencies()
        if len(dependencies) > 1 or dependencies - state_index.keys():
            raise UnsupportedModelError(
                f"Emission {node.name!r} must depend on at most one categorical state, "
                "with supplied emission parameters and no Normal parents."
            )
        state = next(iter(dependencies)) if dependencies else None
        time = state_index[state] if state is not None else None
        if state is None:
            mean, sd = _normal_parameters(node.distribution, {}, f"at {node.name!r}")
            means, scales = np.array([mean]), np.array([sd])
        else:
            evaluated = [
                _normal_parameters(node.distribution, {state: category}, f"at {node.name!r} in state {category}")
                for category in range(k)
            ]
            means = np.array([value[0] for value in evaluated])
            scales = np.array([value[1] for value in evaluated])
        parameters[node.name] = (time, means, scales)
        if node.name in observed:
            value = _scalar(observed[node.name], f"Observation {node.name!r}")
            observed[node.name] = value
            likelihood = _normal_log_prob(value, means, scales)
            if time is None:
                independent_evidence += float(likelihood[0])
            else:
                log_likelihood[time] += likelihood
    for node in states:
        if node.name in observed:
            value = _scalar(observed[node.name], f"Observed state {node.name!r}")
            if value != int(value) or not 0 <= value < k:
                raise InferenceError(f"Observed state {node.name!r} must be an integer in [0, {k - 1}].")
            observed[node.name] = int(value)
            mask = np.arange(k) != int(value)
            log_likelihood[state_index[node.name], mask] = -np.inf
    log_alpha = np.empty((length, k))
    increments = np.empty(length)
    for time in range(length):
        prediction = _log_probabilities(initial) if time == 0 else _logsumexp(
            log_alpha[time - 1, :, None] + log_transitions[time - 1], axis=0
        )
        log_alpha[time] = prediction + log_likelihood[time]
        increments[time] = _logsumexp(log_alpha[time])
        if not np.isfinite(increments[time]):
            raise InferenceError(
                f"Evidence has zero probability (or exceeds numeric range) at state {states[time].name!r}. "
                "Check observed states, transition probabilities and emission scales."
            )
        log_alpha[time] -= increments[time]
    log_beta = np.zeros((length, k))
    for time in range(length - 2, -1, -1):
        log_beta[time] = _logsumexp(
            log_transitions[time]
            + log_likelihood[time + 1][None, :]
            + log_beta[time + 1][None, :], axis=1,
        ) - increments[time + 1]
    log_marginals = log_alpha + log_beta
    log_marginals -= _logsumexp(log_marginals, axis=1)[:, None]
    posterior_states = np.exp(log_marginals)
    evidence = float(increments.sum() + independent_evidence)
    if not np.isfinite(evidence):
        raise InferenceError("The observation likelihood exceeds numeric range; rescale the emissions.")
    means, variances, marginals = {}, {}, {}
    categories = np.arange(k)
    for time, node in enumerate(states):
        probabilities = posterior_states[time]
        marginals[node.name] = probabilities.copy()
        means[node.name] = float(probabilities @ categories)
        variances[node.name] = float(probabilities @ (categories - means[node.name]) ** 2)
    for node in emissions:
        time, emission_means, emission_scales = parameters[node.name]
        if node.name in observed:
            means[node.name], variances[node.name] = float(observed[node.name]), 0.0
        else:
            probabilities = np.array([1.0]) if time is None else posterior_states[time]
            mean = float(probabilities @ emission_means)
            means[node.name] = mean
            variances[node.name] = float(probabilities @ (emission_scales ** 2 + (emission_means - mean) ** 2))

    def sample(draws, rng):
        sampled_states = np.empty((draws, length), dtype=int)
        sampled_states[:, -1] = rng.choice(k, size=draws, p=posterior_states[-1])
        for time in range(length - 2, -1, -1):
            weights = log_alpha[time][None, :] + log_transitions[time][:, sampled_states[:, time + 1]].T
            weights -= _logsumexp(weights, axis=1)[:, None]
            cumulative = np.cumsum(np.exp(weights), axis=1)
            cumulative[:, -1] = 1.0
            sampled_states[:, time] = np.sum(rng.random(draws)[:, None] >= cumulative, axis=1)
        result = {node.name: sampled_states[:, time] for time, node in enumerate(states)}
        for node in emissions:
            time, emission_means, emission_scales = parameters[node.name]
            if node.name in observed:
                result[node.name] = np.full(draws, observed[node.name], dtype=float)
            elif time is None:
                result[node.name] = rng.normal(emission_means[0], emission_scales[0], size=draws)
            else:
                selected = sampled_states[:, time]
                result[node.name] = rng.normal(emission_means[selected], emission_scales[selected])
        return result

    result = EngineResult("exact-hmm", evidence, means, variances, marginals, sample)
    result.diagnostics = {"approximate": False, "method": "exact-hmm", "device": "cpu", "dtype": "float64"}
    return result
