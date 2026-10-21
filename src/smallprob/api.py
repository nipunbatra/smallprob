"""Small user-facing API; model construction and inference live separately."""

from __future__ import annotations

from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from .core import Categorical, Graph, InferenceError, ModelError, Normal
from .arrays import ArrayBackend, infer_backend, to_numpy


def _draw_count(draws):
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 1:
        raise ModelError("draws must be a positive integer.")
    return int(draws)


def _validate_values(graph, values):
    by_name = {node.name: node for node in graph.nodes}
    clean = {}
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ModelError(f"{name}: expected a finite numeric scalar, got {value!r}.")
        if not np.isfinite(value):
            raise ModelError(f"{name}: observations and fixed values must be finite; use None for missing observations.")
        if isinstance(by_name[name].distribution, Categorical):
            if float(value) != int(value) or value < 0:
                raise ModelError(f"{name}: a categorical value must be a nonnegative integer.")
            clean[name] = int(value)
        else:
            clean[name] = float(value)
    return clean


class Samples(dict):
    """Named arrays, always with a leading draw axis, including draws=1."""

    def __init__(self, graph, flat, draws, backend=None):
        self.draws = draws
        self.backend = infer_backend(backend=backend)
        grouped = {}
        by_name = {node.name: node for node in graph.nodes}
        for group, names in graph.groups.items():
            integer = bool(names) and all(isinstance(by_name[name].distribution, Categorical) for name in names)
            if group in graph.series:
                grouped[group] = (self.backend.stack([flat[name] for name in names], axis=1, integer=integer)
                                  if names else self.backend.zeros((draws, 0)))
            else:
                grouped[group] = self.backend.array(flat[names[0]], integer=integer)
        super().__init__(grouped)


class BoundModel:
    """A model whose fixed inputs have been supplied."""

    def __init__(self, graph: Graph, name: str, *, backend=None, device=None,
                 dtype=None, source=None, inputs=None):
        self.graph = graph
        self.name = name
        self.backend = infer_backend(backend=backend, device=device, dtype=dtype)
        self.source = source
        self.inputs = {} if inputs is None else inputs

    def _backend(self, *values, backend=None, device=None, dtype=None):
        if backend is None and self.backend.name == "torch":
            backend = self.backend
        return infer_backend(*values, backend=backend, device=device, dtype=dtype)

    def simulate(self, draws=1, *, seed=None, fixed=None, backend=None,
                 device=None, dtype=None):
        """Generate joint draws; fixed values clamp sites, without posterior inference."""
        draws = _draw_count(draws)
        selected = self._backend(fixed, backend=backend, device=device, dtype=dtype)
        fixed = _validate_values(self.graph, self.graph.expand_values(to_numpy({} if fixed is None else fixed)))
        if selected.name == "torch":
            return self._simulate_torch(draws, seed=seed, fixed=fixed, backend=selected)
        rng = np.random.default_rng(seed)
        flat = {node.name: [] for node in self.graph.nodes}
        for _ in range(draws):
            values = {}
            for node in self.graph.nodes:
                if node.name in fixed:
                    value = fixed[node.name]
                    logp = node.distribution.log_prob(value, values)
                    if not np.isfinite(logp):
                        raise ModelError(f"{node.name}: fixed value is outside the distribution's support.")
                else:
                    value = node.distribution.sample(values, rng)
                values[node.name] = value
                flat[node.name].append(value)
        return Samples(self.graph, {name: np.asarray(values) for name, values in flat.items()}, draws, selected)

    def _simulate_torch(self, draws, *, seed, fixed, backend):
        from .arrays import torch_distribution
        flat = {node.name: [] for node in self.graph.nodes}
        with backend.random(seed):
            for _ in range(draws):
                values = {}
                for node in self.graph.nodes:
                    distribution = torch_distribution(node.distribution, values, backend)
                    if node.name in fixed:
                        value = backend.array(fixed[node.name], integer=isinstance(node.distribution, Categorical))
                        try:
                            logp = distribution.log_prob(value)
                        except ValueError as exc:
                            raise ModelError(f"{node.name}: fixed value is outside the distribution's support.") from exc
                        if not bool(backend.xp.isfinite(logp)):
                            raise ModelError(f"{node.name}: fixed value is outside the distribution's support.")
                    else:
                        value = distribution.sample()
                    values[node.name] = value
                    flat[node.name].append(value)
        integer_names = {node.name for node in self.graph.nodes if isinstance(node.distribution, Categorical)}
        return Samples(self.graph, {name: backend.stack(values, integer=name in integer_names)
                                   for name, values in flat.items()}, draws, backend)

    def infer(self, observed=None, *, method="auto", steps=None,
              learning_rate=None, particles=None, seed=None, backend=None,
              device=None, dtype=None, guide=None, **observations):
        """Infer with exact methods, or explicitly request approximate VI.

        ``infer(y=data)`` is shorthand for ``infer(observed={"y": data})``.
        Use the mapping form for variables named like inference options.
        ``guide="diagonal"`` selects native PyTorch mean-field VI, without Pyro.
        Omit it (or use ``guide="full"``) for the existing full-covariance backend.
        """
        from .inference import infer

        from collections.abc import Mapping
        supplied = {} if observed is None else observed
        if not isinstance(supplied, Mapping):
            raise ModelError("Observations must be a mapping from variable names to values.")
        overlap = supplied.keys() & observations.keys()
        if overlap:
            raise ModelError(f"Observations supplied twice: {sorted(overlap)}.")
        selected = self._backend(supplied, observations, backend=backend, device=device, dtype=dtype)
        values = self.graph.expand_values(to_numpy({**supplied, **observations}), allow_missing=True)
        values = _validate_values(self.graph, values)
        options = {key: value for key, value in {
            "steps": steps, "learning_rate": learning_rate, "particles": particles,
            "seed": seed, "guide": guide,
        }.items() if value is not None}
        result = infer(self.graph, values, method=method, backend=selected, **options)
        return Posterior(self.graph, result, selected)

    def log_prob(self, values, *, backend=None, device=None, dtype=None):
        """Joint log density/mass of a complete assignment, not marginal evidence."""
        selected = self._backend(values, backend=backend, device=device, dtype=dtype)
        flat = _validate_values(self.graph, self.graph.expand_values(to_numpy(values)))
        missing = {node.name for node in self.graph.nodes} - flat.keys()
        if missing:
            raise ModelError(f"log_prob needs every random variable; missing {sorted(missing)}.")
        if selected.name == "torch":
            from .arrays import torch_distribution
            native = {name: selected.array(value) for name, value in flat.items()}
            total = selected.array(0.0)
            for node in self.graph.nodes:
                value = selected.array(flat[node.name], integer=isinstance(node.distribution, Categorical))
                distribution = torch_distribution(node.distribution, native, selected)
                if type(node.distribution).__name__ == "HalfNormal" and flat[node.name] < 0:
                    total = total + selected.array(float("-inf"))
                else:
                    try:
                        total = total + distribution.log_prob(value)
                    except ValueError as exc:
                        raise ModelError(f"{node.name}: value is outside the distribution's support.") from exc
            return total
        return float(sum(node.distribution.log_prob(flat[node.name], flat) for node in self.graph.nodes))

    def math(self, *, labels=None):
        """Display the model's mathematical factorisation in a notebook."""
        from .display import ModelMath
        return ModelMath(self, labels=labels)

    def diagram(self, *, labels=None, observed=None):
        """Draw the model graph with distribution formulas and optional labels."""
        from .display import ModelDiagram
        return ModelDiagram(self, labels=labels, observed=observed)

    def describe(self):
        """List the exact dependency graph, with source-derived variable names."""
        lines = [f"{self.name}: {len(self.graph.nodes)} random variables"]
        for node in self.graph.nodes:
            parents = sorted(node.distribution.dependencies())
            family = type(node.distribution).__name__
            lines.append(f"  {node.name} ~ {family}" + (f"  given {', '.join(parents)}" if parents else ""))
        return "\n".join(lines)

    def __repr__(self):
        return f"BoundModel({self.name!r}, sites={len(self.graph.nodes)})"


class Posterior:
    """Posterior quantities with explicit exact/approximate method metadata."""

    def __init__(self, graph, result, backend=None):
        self.graph = graph
        self.backend = infer_backend(backend=backend)
        self._result = result
        self.method = result.method
        self.log_evidence = None if result.log_evidence is None else float(result.log_evidence)
        self.elbo = getattr(result, "elbo", None)
        self.approximate = self.method.startswith("vi-")
        self.diagnostics = dict(getattr(result, "diagnostics", {"approximate": False, "method": self.method}))
        self.diagnostics.update({"array_backend": self.backend.name, "output_device": self.backend.device,
                                 "output_dtype": self.backend.dtype})
        if not self.approximate:
            self.diagnostics.setdefault("device", self.backend.device)
            self.diagnostics.setdefault("dtype", self.backend.dtype)

    def _get(self, values, name):
        if name in values:
            return self.backend.array(values[name]) if self.backend.name == "torch" else values[name]
        if name not in self.graph.groups:
            raise ModelError(f"Unknown variable {name!r}. Available groups: {list(self.graph.groups)}")
        names = self.graph.groups[name]
        if not all(site in values for site in names):
            raise ModelError(f"{name!r} is not available for this posterior quantity.")
        if name in self.graph.series:
            return self.backend.stack([values[site] for site in names]) if names else self.backend.zeros((0,))
        return self.backend.array(values[names[0]]) if self.backend.name == "torch" else values[names[0]]

    def mean(self, name):
        return self._get(self._result.means, name)

    def sd(self, name):
        return self.backend.xp.sqrt(self._get(self._result.variances, name))

    def interval(self, name, probability=0.95):
        """Equal-tail marginal interval; VI uses its approximate guide quantiles."""
        if isinstance(probability, bool) or not isinstance(probability, Real) or not 0 < probability < 1:
            raise ModelError("probability must be between 0 and 1, exclusively.")
        if self.approximate:
            lower, upper = self._result.interval(float(probability))
            return self._get(lower, name), self._get(upper, name)
        if self.method != "exact-gaussian":
            raise InferenceError("Exact intervals are currently available only for Gaussian posteriors. For HMM states, use probabilities(name).")
        q = NormalDist().inv_cdf(0.5 + probability / 2)
        mean, sd = self.mean(name), self.sd(name)
        return mean - q * sd, mean + q * sd

    def probabilities(self, name):
        """Categorical state probabilities; series have shape (time, states)."""
        return self._get(self._result.marginals, name)

    def sample(self, draws=1000, *, seed=None):
        draws = _draw_count(draws)
        rng = np.random.default_rng(seed)
        if self.approximate and self.backend.name == "torch":
            # The fitted guide is detached for stable, independent posteriors.
            # Reconstitute its small factors, then draw and transform natively.
            result, backend = self._result, self.backend
            location, factor = backend.array(result._location), backend.array(result._factor)
            values = location + backend.normal((draws, len(result._latent_names)), rng) @ factor.T
            positive = backend.array(np.flatnonzero(result._positive), integer=True)
            values[:, positive] = backend.xp.exp(values[:, positive])
            if not bool(backend.xp.all(backend.xp.isfinite(values))) or bool(backend.xp.any(values[:, positive] <= 0)):
                raise InferenceError("Variational samples exceeded floating point range; rescale the model.")
            flat = {name: values[:, i] for i, name in enumerate(result._latent_names)}
            flat.update({name: backend.xp.full((draws,), value, dtype=backend.float_dtype, device=backend.device)
                         for name, value in result._observed.items()})
        else:
            flat = self._result.sample(draws, rng)
        return Samples(self.graph, flat, draws, self.backend)

    def __repr__(self):
        if self.approximate:
            return f"Posterior(method={self.method!r}, approximate=True, elbo={self.elbo:.6g})"
        return f"Posterior(method={self.method!r}, log_evidence={self.log_evidence:.6g})"
