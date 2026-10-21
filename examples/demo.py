"""Run: python -m examples.demo [--json outputs/demo-results.json]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import hmm, normal, regression


def run():
    population = normal(n=30, noise=1.0)
    population_data = population.simulate(seed=11, fixed={"mu": 2.0})
    population_posterior = population.infer(observed={"y": population_data["y"][0]})

    x = np.linspace(-2, 2, 40).tolist()
    line = regression(x, noise=0.5)
    line_data = line.simulate(seed=12, fixed={"intercept": 1.5, "slope": -0.8})
    line_posterior = line.infer(observed={"y": line_data["y"][0]})

    chain = hmm(
        n=35,
        initial=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.15, 0.85]],
        means=[-1.5, 1.5],
        noise=0.65,
    )
    chain_data = chain.simulate(seed=13)
    chain_posterior = chain.infer(observed={"y": chain_data["y"][0]})
    probabilities = chain_posterior.probabilities("z")
    decoded = probabilities.argmax(axis=1)

    def estimate(posterior, name, truth):
        return {
            "generating_value": truth,
            "posterior_mean": float(posterior.mean(name)),
            "posterior_sd": float(posterior.sd(name)),
            "credible_interval_95": [float(v) for v in posterior.interval(name)],
        }

    result = {
        "note": "Synthetic demonstrations with fixed random seeds. Results are computed, not benchmarks or performance guarantees.",
        "normal": {
            "method": population_posterior.method,
            "n": 30,
            "mu": estimate(population_posterior, "mu", 2.0),
            "y": population_data["y"][0].tolist(),
        },
        "regression": {
            "method": line_posterior.method,
            "n": len(x),
            "intercept": estimate(line_posterior, "intercept", 1.5),
            "slope": estimate(line_posterior, "slope", -0.8),
            "x": x,
            "y": line_data["y"][0].tolist(),
        },
        "hmm": {
            "method": chain_posterior.method,
            "n": 35,
            "log_evidence": chain_posterior.log_evidence,
            "generating_states": chain_data["z"][0].tolist(),
            "marginal_mode_states": decoded.tolist(),
            "state_probabilities": probabilities.tolist(),
            "state_agreement_on_this_draw": float(np.mean(decoded == chain_data["z"][0])),
            "posterior_joint_trajectory": chain_posterior.sample(draws=1, seed=14)["z"][0].tolist(),
            "y": chain_data["y"][0].tolist(),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optionally save the computed results")
    args = parser.parse_args()
    result = run()
    print("Smallprob: model -> generate -> infer\n")
    for family, variable in [("normal", "mu"), ("regression", "intercept"), ("regression", "slope")]:
        item = result[family][variable]
        lo, hi = item["credible_interval_95"]
        print(f"{family:10s} {variable:10s} truth={item['generating_value']: .3f}  posterior={item['posterior_mean']: .3f}  95% interval=[{lo:.3f}, {hi:.3f}]")
    print(f"\nHMM: {result['hmm']['method']}; log evidence={result['hmm']['log_evidence']:.3f}")
    print("Generated states: ", "".join(str(v) for v in result["hmm"]["generating_states"]))
    print("Marginal modes:   ", "".join(str(v) for v in result["hmm"]["marginal_mode_states"]))
    print("Posterior sample: ", "".join(str(v) for v in result["hmm"]["posterior_joint_trajectory"]))
    print("\nExact inference for these models. See examples.unknown_noise for approximate VI.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
