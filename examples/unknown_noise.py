"""Run after installing the vi extra: python -m examples.unknown_noise."""

import argparse
import json
from pathlib import Path

from .models import regression_unknown_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    x = [-2 + 4 * i / 39 for i in range(40)]
    m = regression_unknown_noise(x)
    truth = {"intercept": 1.5, "slope": -0.8, "sigma": 0.5}
    data = m.simulate(seed=12, fixed=truth)
    p = m.infer(y=data["y"][0], method="vi", seed=args.seed, steps=args.steps)
    print("Unknown-noise regression · approximate variational posterior\n")
    estimates = {}
    for name, generating_value in truth.items():
        lo, hi = p.interval(name)
        mean, sd = float(p.mean(name)), float(p.sd(name))
        estimates[name] = {"generating_value": generating_value, "mean": mean,
                           "sd": sd, "interval_95": [float(lo), float(hi)]}
        print(f"{name:10s} truth={generating_value: .3f}  VI mean={mean: .3f}  95% guide interval=[{lo:.3f}, {hi:.3f}]")
    print(f"\nMethod: {p.method}; Monte Carlo ELBO estimate: {p.elbo:.3f}")
    print("Intervals describe the learned variational approximation; exact evidence is unavailable.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        result = {"method": p.method, "approximate": p.approximate,
                  "elbo": p.elbo, "log_evidence": p.log_evidence,
                  "estimates": estimates, "x": x, "y": data["y"][0].tolist(),
                  "diagnostics": p.diagnostics}
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
