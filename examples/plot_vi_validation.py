"""Plot the saved validation report, without running inference again.

uv run --extra vi --extra dev --with matplotlib python -m examples.plot_vi_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .vi_validation import quadrature_regression_reference


REFERENCE = "#183a44"
GUIDE_COLOURS = ("#d17b1f", "#2c7eb3", "#b94b69")
BACKGROUND = "#faf9f6"


def normal_density(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd)**2) / (np.sqrt(2 * np.pi) * sd)


def gaussian_joint_ellipse(mean, covariance, probability=0.95):
    """A bivariate Gaussian joint-probability contour, not marginal intervals."""
    angles = np.linspace(0, 2 * np.pi, 500)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    # Chi-square(df=2) CDF is 1-exp(-r^2/2).
    radius = np.sqrt(-2 * np.log1p(-probability))
    return np.asarray(mean)[:, None] + radius * np.linalg.cholesky(covariance) @ circle


def plot_normal(axis, case):
    reference = case["reference"]
    mean, sd = reference["mean"][0], reference["sd"][0]
    x = np.linspace(mean - 3.6 * sd, mean + 3.6 * sd, 600)
    density = normal_density(x, mean, sd)
    axis.fill_between(x, density, color=REFERENCE, alpha=0.08)
    axis.plot(x, density, color=REFERENCE, lw=2.7, label="Exact Gaussian")
    for colour, run in zip(GUIDE_COLOURS, case["runs"]):
        axis.plot(x, normal_density(x, run["mean"][0], run["sd"][0]),
                  color=colour, lw=1.7, linestyle="--", label=f"VI, seed {run['seed']}")
    axis.set(title="01   Normal mean", xlabel="Population mean", ylabel="Marginal posterior density")
    axis.legend(frameon=False, fontsize=10, loc="upper left")
    axis.set_ylim(bottom=0)


def plot_regression(axis, case):
    reference = case["reference"]
    ellipse = gaussian_joint_ellipse(reference["mean"], reference["covariance"])
    axis.fill(*ellipse, color=REFERENCE, alpha=0.08)
    axis.plot(*ellipse, color=REFERENCE, lw=2.7, label="Exact Gaussian")
    for colour, run in zip(GUIDE_COLOURS, case["runs"]):
        ellipse = gaussian_joint_ellipse(run["mean"], run["covariance"])
        axis.plot(*ellipse, color=colour, lw=1.7, linestyle="--", label=f"VI, seed {run['seed']}")
    axis.set(title="02   Correlated regression coefficients", xlabel="Intercept", ylabel="Slope")
    axis.legend(frameon=False, fontsize=10, loc="upper right")
    covariance = np.asarray(reference["covariance"])
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    axis.text(0.02, 0.05, f"95% joint posterior contours\nExact correlation: {correlation:.4f}",
              transform=axis.transAxes, fontsize=10, color=REFERENCE)
    axis.margins(0.1)


def plot_noise(axis, case):
    reference = case["reference"]
    x = np.asarray(case["x"])
    design = np.column_stack([np.ones_like(x), x])
    grid = reference["grid"]
    numerical = quadrature_regression_reference(design, case["y"],
                                                 bounds=grid["log_sigma_bounds"], points=grid["points"])
    # Verify that recomputing the density preserves the saved reference.
    np.testing.assert_allclose(numerical["mean"], reference["mean"], rtol=1e-10, atol=1e-10)
    log_sigma = numerical["_log_sigma"]
    sigma = np.exp(log_sigma)
    spacing = log_sigma[1] - log_sigma[0]
    density = numerical["_weights"] / (spacing * sigma)
    density[[0, -1]] *= 2  # Undo trapezoid endpoint weights.
    low = max(0.01, reference["mean"][-1] - 3.7 * reference["sd"][-1])
    high = reference["mean"][-1] + 4.7 * reference["sd"][-1]
    selection = (sigma >= low) & (sigma <= high)
    axis.fill_between(sigma[selection], density[selection], color=REFERENCE, alpha=0.08)
    axis.plot(sigma[selection], density[selection], color=REFERENCE, lw=2.7,
              label="Numerical quadrature")
    values = np.linspace(low, high, 700)
    for colour, run in zip(GUIDE_COLOURS, case["runs"]):
        diagnostics = run["diagnostics"]
        index = diagnostics["latent_order"].index("sigma")
        log_mean = diagnostics["unconstrained_mean"][index]
        log_sd = np.sqrt(diagnostics["unconstrained_covariance"][index][index])
        guide_density = normal_density(np.log(values), log_mean, log_sd) / values
        axis.plot(values, guide_density, color=colour, lw=1.7, linestyle="--",
                  label=f"VI, seed {run['seed']}")
    axis.set(title="03   Regression with unknown noise", xlabel="Noise standard deviation, σ",
             ylabel="Marginal posterior density", xlim=(low, high), ylim=(0, None))
    axis.legend(frameon=False, fontsize=10, loc="upper right")


def plot_summary(axis, cases, report):
    axis.axis("off")
    axis.set_title("04   Errors across all three seeds", pad=18)
    rows = []
    for label, case in zip(("Normal mean", "Known noise", "Unknown noise"), cases):
        errors = [run["errors"] for run in case["runs"]]
        rows.append([
            label,
            f"{max(e['max_mean_error_in_reference_sd'] for e in errors):.3f}",
            f"{100 * max(e['max_sd_relative_error'] for e in errors):.2f}%",
            f"{max(e['max_interval_error_in_reference_sd'] for e in errors):.3f}",
        ])
    thresholds = report["thresholds_predeclared"]
    rows.append(["Declared limit", f"{thresholds['max_mean_error_in_reference_sd']:.3f}",
                 f"{100 * thresholds['max_sd_relative_error']:.2f}%",
                 f"{thresholds['max_interval_error_in_reference_sd']:.3f}"])
    table = axis.table(cellText=rows, colLabels=["Case", "Mean error¹", "SD error²", "Interval error¹"],
                        cellLoc="left", colLoc="left", bbox=(0, 0.36, 1, 0.55),
                        colWidths=[0.32, 0.22, 0.20, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddeda")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#eef1ef" if row in (0, len(rows)) else BACKGROUND)
        if row == 0:
            cell.set_text_props(weight="bold", color=REFERENCE)
    axis.text(0, 0.29, "¹ Absolute error / reference marginal SD.\n"
              "   Interval error compares 95% marginal endpoints.\n"
              "² Relative error in the marginal posterior SD.", transform=axis.transAxes,
              fontsize=10, color="#4c5658", va="top", linespacing=1.4)
    count = sum(len(case["runs"]) for case in cases)
    passed = sum(run["passed"] for case in cases for run in case["runs"])
    axis.text(0, -0.015, f"{passed}/{count} fits passed all predeclared checks",
              transform=axis.transAxes, fontsize=12, weight="bold", color=REFERENCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/vi-validation.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vi-validation.png"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    cases_by_name = {case["name"]: case for case in report["cases"]}
    cases = [cases_by_name[name] for name in ("normal_mean", "known_noise_correlated_regression", "unknown_noise_regression")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.titlesize": 14, "axes.titlepad": 15,
                         "axes.labelcolor": "#34464a", "text.color": REFERENCE,
                         "xtick.color": "#4c5658", "ytick.color": "#4c5658"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.3))
    fig.set_facecolor(BACKGROUND)
    for axis in axes.flat:
        axis.set_facecolor(BACKGROUND)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.135, top=0.83, hspace=0.44, wspace=0.26)
    fig.text(0.075, 0.95, "Smallprob: VI against reference posteriors", fontsize=23, weight="bold")
    settings = report["settings"]
    seed_text = ", ".join(str(seed) for seed in settings["seeds"])
    fig.text(0.075, 0.914, f"Fixed synthetic datasets  ·  optimisation seeds {seed_text}  ·  "
              f"{settings['steps']:,} steps and {settings['particles']} particles per fit", fontsize=12, color="#4c5658")
    fig.text(0.075, 0.882, "Solid curves: exact Gaussian or numerical reference.  Dashed curves: approximate full-rank VI.",
              fontsize=11, color="#4c5658")
    plot_normal(axes[0, 0], cases[0])
    plot_regression(axes[0, 1], cases[1])
    plot_noise(axes[1, 0], cases[2])
    plot_summary(axes[1, 1], cases, report)
    fig.text(0.075, 0.065, "Unknown-noise reference: numerical quadrature over log σ, with regression coefficients integrated out analytically.",
              fontsize=10.5, color="#4c5658")
    fig.text(0.075, 0.038, "These are accuracy checks on three fixed datasets, not a study of interval coverage or calibration.  Source: outputs/vi-validation.json",
              fontsize=10, color="#4c5658")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved {args.output.resolve()} using {args.input.resolve()}; no VI fits rerun.")


if __name__ == "__main__":
    main()
