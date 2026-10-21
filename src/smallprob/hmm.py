"""A fixed-parameter hidden Markov model without a Python time loop."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .api import BoundModel
from .arrays import infer_backend, to_numpy
from .core import Categorical, Expr, Graph, ModelError, Node, Normal, UnsupportedModelError, lookup


def _numeric_array(value, label, *, preserve_dtype=False):
    if isinstance(value, Expr):
        if value.dependencies():
            raise UnsupportedModelError(f"HMM {label} must be fixed parameters; fitting HMM parameters is not implemented.")
        value = value.eval({})
    try:
        result = np.asarray(to_numpy(value))
        if result.dtype.kind not in "iuf":
            raise ValueError("expected real numbers")
        result = result.copy() if preserve_dtype else result.astype(float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ModelError(f"HMM {label} must contain real numbers: {exc}") from exc
    if not np.all(np.isfinite(result)):
        raise ModelError(f"HMM {label} must contain only finite values.")
    return result


def _emission_parameter(expression, label, count):
    parameter = _numeric_array(expression, label)
    if parameter.ndim == 0:
        return float(parameter)
    if parameter.shape != (count,):
        raise ModelError(f"HMM {label} must be a scalar or a vector of length {count}, one value per state.")
    return parameter


def HMM(*, initial, transition, emissions, steps, states=None, backend=None,
        device=None, dtype=None):
    """Create a hidden Markov model from its probability tables and emissions.

    >>> weather = HMM(initial=[.5, .5],
    ...               transition=[[.9, .1], [.15, .85]],
    ...               emissions=Normal([-1.5, 1.5], .45), steps=30,
    ...               states=["low", "high"])
    >>> data = weather.simulate(seed=7)
    >>> posterior = weather.infer(y=data["y"][0])

    ``initial[k]`` is the probability of state ``k`` at the first time step.
    ``transition[j, k]`` (or ``transition[j][k]``) is the probability of moving
    from state ``j`` to state ``k``; every row must sum to one. ``emissions``
    is a Normal whose mean and SD are each a scalar or one value per state.
    All parameters are fixed: this constructor does not learn the tables.

    The returned ordinary bound model exposes hidden states as ``z`` and
    emissions as ``y``, each a series of length ``steps``. State values remain
    zero-based integers; optional ``states`` strings are display labels only.
    Parameter arrays, including Torch tensors, are detached host snapshots.
    Lists and NumPy tables default to NumPy; Torch initial/transition tables
    select Torch. ``backend``, ``device`` and ``dtype`` can override that choice.
    A Normal object has already snapshotted its parameters, so tensors supplied
    only inside ``emissions`` do not select the backend. Torch simulation is
    supported. Exact HMM inference is NumPy/CPU only: select
    ``infer(backend="numpy")`` explicitly when the model uses Torch.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps < 1:
        raise ModelError("HMM steps must be a positive integer.")
    steps = int(steps)
    selected = infer_backend(initial, transition, backend=backend, device=device, dtype=dtype)
    initial = _numeric_array(initial, "initial probabilities", preserve_dtype=True)
    # Categorical is the single source of truth for probability validation.
    initial = Categorical(initial)._probabilities({}).copy()
    count = len(initial)
    transition = _numeric_array(transition, "transition probabilities", preserve_dtype=True)
    if transition.shape != (count, count):
        raise ModelError(f"HMM transition must have shape ({count}, {count}), with one probability row per state.")
    rows = []
    for index, row in enumerate(transition):
        try:
            rows.append(Categorical(row)._probabilities({}))
        except ModelError as exc:
            raise ModelError(f"HMM transition row {index}: {exc}") from exc
    transition = np.stack(rows)

    if type(emissions) is not Normal:
        raise UnsupportedModelError("HMM emissions currently require Normal(mean, sd); other emission families are not implemented.")
    if emissions.dependencies():
        raise UnsupportedModelError("HMM emissions must have fixed parameters; fitting HMM parameters is not implemented.")
    means = _emission_parameter(emissions.mean, "emission mean", count)
    scales = _emission_parameter(emissions.sd, "emission SD", count)
    if np.any(np.asarray(scales) <= 0):
        raise ModelError("HMM emission SD must be strictly positive.")
    # Retain an explicit per-state lookup even when the means are shared.
    # This records the intended state/emission relationship in the graph.
    if np.ndim(means) == 0:
        means = np.full(count, means)

    labels = None
    if states is not None:
        if isinstance(states, (str, bytes)):
            raise ModelError("HMM states must be a sequence of unique nonempty names, one per state.")
        try:
            labels = tuple(states)
        except TypeError as exc:
            raise ModelError("HMM states must be a sequence of names, one per state.") from exc
        if len(labels) != count or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ModelError(f"HMM states must contain {count} nonempty string names, one per state.")
        if len(set(labels)) != count:
            raise ModelError("HMM state names must be unique.")

    state_names = [f"z[{time}]" for time in range(steps)]
    emission_names = [f"y[{time}]" for time in range(steps)]
    state_refs = [Expr("ref", (name,)) for name in state_names]
    nodes = [Node(state_names[0], Categorical(initial))]
    nodes.extend(Node(state_names[time], Categorical(lookup(transition, state_refs[time - 1])))
                 for time in range(1, steps))
    nodes.extend(Node(name, Normal(lookup(means, state),
                                   lookup(scales, state) if np.ndim(scales) else scales))
                 for name, state in zip(emission_names, state_refs))
    graph = Graph(nodes, {"z": state_names, "y": emission_names}, {"z", "y"})
    bound = BoundModel(graph, "Hidden Markov model", backend=selected)
    bound.hmm_state_labels = labels
    return bound
