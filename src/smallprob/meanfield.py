"""Small, PyTorch-only mean-field VI using reparameterized Normal samples.

The optimizer learns two vectors: locations and log standard deviations.
Positive sites use exp(u); their log-Jacobian is included in the objective.
No Pyro imports, covariance optimization, or exact-posterior initialization.
"""
from __future__ import annotations

import math
import numbers
import numpy as np

from .core import Normal, HalfNormal, StudentT, InferenceError, UnsupportedModelError
from .arrays import stable_student_t_class
from .vi import VIResult, _compiler, _positive_integer, _execution_options, _isolated_rng, _BACKEND_LOCK


def _parameters(distribution):
    if isinstance(distribution, StudentT):
        return distribution.df, distribution.loc, distribution.scale
    if isinstance(distribution, Normal):
        return distribution.mean, distribution.sd
    return (distribution.sd,)


def _log_joint(graph, observed, torch, device, dtype):
    """Compile the model density independently of the variational family."""
    compile_expr, compile_batch = _compiler(torch, device=device, dtype=dtype)
    classes = {Normal: torch.distributions.Normal, HalfNormal: torch.distributions.HalfNormal,
               StudentT: stable_student_t_class(torch.distributions.StudentT)}
    for node in graph.nodes:
        if any(compile_expr(expr).shape for expr in _parameters(node.distribution)):
            raise UnsupportedModelError(f"VI requires scalar distribution parameters at {node.name!r}.")
    groups = []
    for kind in classes:
        nodes = [node for node in graph.nodes if type(node.distribution) is kind]
        if nodes:
            batches = tuple(compile_batch(exprs) for exprs in zip(*[_parameters(n.distribution) for n in nodes]))
            groups.append((classes[kind], batches, [n.name for n in nodes]))
    constants = {name: torch.tensor([value], device=device, dtype=dtype) for name, value in observed.items()}
    latent = [n for n in graph.nodes if n.name not in observed]
    positive = torch.tensor([isinstance(n.distribution, HalfNormal) for n in latent], device=device)

    def transformed_log_joint(u):
        values = dict(constants)
        for i, node in enumerate(latent):
            values[node.name] = u[:, i].exp() if isinstance(node.distribution, HalfNormal) else u[:, i]
        total = u[:, positive].sum(-1)  # log |d exp(u) / du|, once per positive site
        for cls, batches, names in groups:
            distribution = cls(*(batch(values) for batch in batches), validate_args=True)
            data = torch.stack(torch.broadcast_tensors(*(values[name] for name in names)), dim=-1)
            total = total + distribution.log_prob(data).sum(-1)
        return total

    return transformed_log_joint


def _initial_location(graph, observed):
    """Prior conditional centers, without fitting or consulting exact inference."""
    values, location = dict(observed), []
    for node in graph.nodes:
        if node.name in observed:
            continue
        dist = node.distribution
        if isinstance(dist, Normal):
            value = float(dist.mean.eval(values))
        elif isinstance(dist, StudentT):
            value = float(dist.loc.eval(values))
        else:
            value = .6744897501960817 * float(dist.sd.eval(values))
            if not np.isfinite(value) or value <= 0:
                raise InferenceError(f"Invalid positive scale at initial site {node.name!r}.")
        values[node.name] = value
        location.append(math.log(value) if isinstance(dist, HalfNormal) else value)
    return location


def infer_meanfield(graph, observed_flat, *, steps=3000, learning_rate=.03,
                    particles=32, seed=0, device="cpu", dtype="float64"):
    """Fit independent Normals in unconstrained space with native Torch Adam.

    A Normal's scale in the model may depend on another latent. The guide is
    independent; the model need not be. HalfNormal guide marginals are lognormal.
    """
    steps, particles = _positive_integer(steps, "steps"), _positive_integer(particles, "particles")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(learning_rate, numbers.Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    names = [node.name for node in graph.nodes]
    if len(set(names)) != len(names):
        raise InferenceError("The model graph contains duplicate random-variable names.")
    if set(observed_flat) - set(names):
        raise InferenceError(f"Observations refer to unknown sites: {sorted(set(observed_flat) - set(names))}.")
    observed, seen = {}, set()
    for node in graph.nodes:
        if type(node.distribution) not in (Normal, HalfNormal, StudentT):
            raise UnsupportedModelError("Mean-field VI supports Normal, HalfNormal and StudentT; Categorical states require exact HMM inference.")
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
    latent = [n for n in graph.nodes if n.name not in observed]
    if not latent:
        raise UnsupportedModelError("VI requires at least one unobserved variable to optimise.")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Mean-field VI requires PyTorch: pip install 'smallprob[torch]'. Pyro is not required.") from exc
    device, dtype = _execution_options(torch, device, dtype)
    with _BACKEND_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            if device.type == "cpu":
                torch.set_num_threads(1)
            with _isolated_rng(torch, device, seed):
                log_joint = _log_joint(graph, observed, torch, device, dtype)
                loc = torch.nn.Parameter(torch.tensor(_initial_location(graph, observed), device=device, dtype=dtype))
                log_sd = torch.nn.Parameter(torch.full_like(loc, math.log(.25)))
                optimizer = torch.optim.Adam([loc, log_sd], lr=float(learning_rate))
                decay = .1 ** (1 / steps)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
                history = []
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    # Antithetic pairs remain unbiased and reduce mean-gradient noise.
                    eps = torch.randn(((particles + 1) // 2, len(latent)), device=device, dtype=dtype)
                    eps = torch.cat([eps, -eps], dim=0)[:particles]
                    u = loc + log_sd.exp() * eps
                    entropy = (log_sd + .5 * math.log(2 * math.pi * math.e)).sum()
                    try:
                        loss = -(log_joint(u).mean() + entropy)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("nonfinite ELBO")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([loc, log_sd], 10., error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()
                    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as exc:
                        raise InferenceError(f"Mean-field VI failed at step {step + 1}: rescale inputs or reduce learning_rate. {exc}") from exc
                    history.append(float(loss.detach()))
                with torch.no_grad():
                    if not bool(torch.isfinite(loc).all() & torch.isfinite(log_sd.exp()).all()) or bool((log_sd.exp() <= 0).any()):
                        raise InferenceError("Nonfinite or zero guide scales; rescale the model.")
                    evaluation_particles = 4096
                    q = torch.distributions.Normal(loc, log_sd.exp())
                    u = q.sample((evaluation_particles,))
                    estimates = log_joint(u) - q.log_prob(u).sum(-1)
                    if not bool(torch.isfinite(estimates).all()):
                        raise InferenceError("Held-out mean-field ELBO exceeded numeric range.")
                    elbo = float(estimates.mean())
                    elbo_se = float(estimates.std(unbiased=True) / math.sqrt(evaluation_particles))
                    location = loc.detach().cpu().numpy().copy()
                    scales = log_sd.exp().detach().cpu().numpy().copy()
        finally:
            torch.set_num_threads(previous_threads)
    order = tuple(node.name for node in latent)
    positive = np.array([isinstance(n.distribution, HalfNormal) for n in latent])
    diagonal = scales**2
    mean, variance = location.copy(), diagonal.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean[positive] = np.exp(location[positive] + diagonal[positive] / 2)
        variance[positive] = np.expm1(diagonal[positive]) * np.exp(2 * location[positive] + diagonal[positive])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)) or np.any(mean[positive] <= 0):
        raise InferenceError("Mean-field guide moments exceeded floating point range.")
    means = {**dict(zip(order, mean.tolist())), **observed}
    variances = {**dict(zip(order, variance.tolist())), **{name: 0. for name in observed}}
    window = min(100, max(1, steps // 10))
    warnings = ["Mean-field VI is approximate; a decreasing loss does not establish convergence.",
                "Independent guide variables cannot capture posterior correlations; marginal uncertainty may be underestimated."]
    if steps < 500:
        warnings.append("Fewer than 500 steps: smoke check only, not validated inference.")
    diagnostics = {
        "backend": "torch", "torch_version": torch.__version__, "family": "diagonal Gaussian in unconstrained space",
        "dtype": str(dtype).removeprefix("torch."), "device": str(device), "seed": int(seed),
        "steps": steps, "particles": particles, "learning_rate": float(learning_rate),
        "optimiser": "Adam", "gradient_clip_norm": 10., "learning_rate_schedule": {"type": "exponential", "final_factor": .1},
        "loss_history": history, "loss_summary_window": window, "initial_loss_mean": float(np.mean(history[:window])),
        "final_loss_mean": float(np.mean(history[-window:])), "latent_order": list(order),
        "unconstrained_mean": location.tolist(), "unconstrained_sd": scales.tolist(),
        "transforms": {name: "exp" if positive[i] else "identity" for i, name in enumerate(order)},
        "elbo_estimator": "independent held-out Monte Carlo", "elbo_particles": evaluation_particles,
        "elbo_standard_error": elbo_se, "interval_kind": "equal-tail marginal guide interval", "warnings": warnings,
    }
    # The common result adapter uses a factor for sampling; only its diagonal is
    # populated. No covariance entries are optimized.
    result = VIResult(means, variances, elbo, diagnostics, order, location, np.diag(scales), positive, observed)
    result.method = "vi-meanfield"
    return result
