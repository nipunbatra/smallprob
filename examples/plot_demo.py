"""Optional figure: uv run --with matplotlib python -m examples.plot_demo."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .demo import run
from .models import regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/smallprob-demo.png"))
    args = parser.parse_args()
    result = run()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.5), layout="constrained")
    fig.set_facecolor("#faf9f6")
    for axis in axes:
        axis.set_facecolor("#faf9f6")
    fig.suptitle("Smallprob · write a model, generate data, infer", fontsize=19, weight="bold")

    item = result["normal"]["mu"]
    mean, sd = item["posterior_mean"], item["posterior_sd"]
    values = np.linspace(mean - 4 * sd, mean + 4 * sd, 400)
    density = np.exp(-0.5 * ((values - mean) / sd)**2) / (np.sqrt(2*np.pi) * sd)
    axes[0].plot(values, density, color="#176f76", label="Exact posterior density")
    lo, hi = item["credible_interval_95"]
    axes[0].fill_between(values, density, where=(values >= lo) & (values <= hi), color="#176f76", alpha=.16, label="95% marginal credible interval")
    axes[0].axvline(item["generating_value"], color="#bd5a32", linestyle="--", label="Generating mean = 2.0")
    axes[0].set(title="01   Normal mean · 30 observations", xlabel="Population mean", ylabel="Posterior density")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)

    line = result["regression"]
    x, y = np.asarray(line["x"]), np.asarray(line["y"])
    # Missing output nodes directly expose the exact predictive distribution.
    grid = np.linspace(-2.2, 2.2, 100)
    model = regression([*x.tolist(), *grid.tolist()], noise=.5)
    posterior = model.infer(observed={"y": [*y.tolist(), *([None] * len(grid))]})
    predicted_mean = posterior.mean("y")[len(x):]
    # Subtract known observation variance to show uncertainty in the mean line.
    mean_sd = np.sqrt(np.maximum(0, posterior.sd("y")[len(x):]**2 - .5**2))
    axes[1].scatter(x, y, color="#176f76", alpha=.75, s=23, label="Generated observations")
    axes[1].plot(grid, 1.5 - .8 * grid, color="#bd5a32", linestyle="--", label="Generating line")
    axes[1].plot(grid, predicted_mean, color="#176f76", label="Posterior mean line")
    axes[1].fill_between(grid, predicted_mean - 1.959963984540054 * mean_sd, predicted_mean + 1.959963984540054 * mean_sd, color="#176f76", alpha=.16, label="95% pointwise interval for mean")
    axes[1].set(title="02   Linear regression · exact joint Gaussian inference", xlabel="Input x", ylabel="Response y")
    axes[1].legend(frameon=False, loc="upper right", fontsize=10)

    chain = result["hmm"]
    time = np.arange(chain["n"])
    probabilities = np.asarray(chain["state_probabilities"])
    axes[2].step(time, chain["generating_states"], where="mid", color="#bd5a32", linestyle="--", alpha=.8, label="Generating hidden state")
    axes[2].plot(time, probabilities[:, 1], color="#176f76", marker="o", markersize=3, label="Smoothed P(state = 1 | all observations)")
    axes[2].set(title="03   HMM · exact forward–backward inference", xlabel="Time step", ylabel="State / probability", ylim=(-.08, 1.08), yticks=[0, .5, 1])
    axes[2].legend(frameon=False, loc="center right", fontsize=10)
    fig.supxlabel("Synthetic data · fixed seeds · exact inference with known noise and HMM parameters", fontsize=10, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
