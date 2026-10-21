"""Build static site cases from real model code and executed notebooks.

Run: python -m examples.build_site_data
No VI fit is rerun here. Notebook outputs retain their execution provenance.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import textwrap

import nbformat

from examples.site_models import PYRO_MODELS, bound_cases, comparison_checks, smallprob_code

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"

RUN_CODE = {
    "normal-mean": '''
simple = normal_observation()
generated = simple.simulate(draws=5, seed=7)
posterior = simple.infer(y=2.0)
print(posterior.mean("mu"), posterior.sd("mu"))
simple.diagram(observed=["y"]).save("normal-mean-model.svg")
''',
    "linear-regression": '''
import numpy as np

x = np.linspace(-2, 2, 30)
line = regression(x)
data = line.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
posterior = line.infer(y=data["y"][0])
print(posterior.mean("slope"), posterior.interval("slope"))
line.diagram(labels={"intercept": "b", "slope": "m"}, observed=["y"]).save("linear-regression-model.svg")
''',
    "robust-regression": '''
import numpy as np

x = np.linspace(-2, 2, 30)
robust = robust_regression(x)
generated = robust.simulate(seed=7)
# Reproduce the notebook's Normal data, then add two outliers.
y = 1.5 - 0.8 * x + np.random.default_rng(7).normal(0, 0.35, len(x))
y[[5, 24]] += [3.0, -3.0]
posterior = robust.infer(y=y, method="vi", seed=7)
print(posterior.mean("m"), posterior.interval("m"))
robust.diagram(observed=["y"]).save("robust-regression-model.svg")
''',
    "hidden-markov-model": '''
data = chain.simulate(seed=7)
posterior = chain.infer(y=data["y"][0])
print(posterior.probabilities("z")[:, 1])
chain.diagram(observed=["y"]).save("hidden-markov-model.svg")
''',
    "weighing-scale": '''
scale = weighing_scale()
generated = scale.simulate(draws=5, seed=7)
posterior = scale.infer(measurement=9.5)
print(posterior.mean("weight"), posterior.sd("weight"))
scale.diagram(labels={"weight": "w", "measurement": "y"}, observed=["measurement"]).save("weighing-scale-model.svg")
''',
    "interaction-regression": '''
import numpy as np

x = np.tile(np.linspace(-1.5, 1.5, 20), 2)
group = np.repeat([0., 1.], 20)
regression = interaction_regression(group, x)
truth = {"intercept": 1.2, "group_effect": 0.7,
         "slope": -0.6, "interaction": 0.9, "sigma": 0.5}
data = regression.simulate(seed=103, fixed=truth)
posterior = regression.infer(y=data["y"][0], method="vi", steps=4000, seed=11)
print(posterior.mean("interaction"), posterior.interval("interaction"))
regression.diagram(labels={"intercept": "b", "slope": "m", "group_effect": "a", "interaction": "c"}, observed=["y"]).save("interaction-regression-model.svg")
''',
    "eight-schools": '''
se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
schools = eight_schools(se)
generated = schools.simulate(draws=2, seed=7)
posterior = schools.infer(y=y, method="vi", steps=4000, seed=11)
draws = posterior.sample(6000, seed=12)
effects = draws["mu"][:, None] + draws["tau"][:, None] * draws["eta"]
print(effects.mean(axis=0))
schools.diagram(observed=["y"]).save("eight-schools-model.svg")
''',
}

METADATA = [
    dict(id="normal-mean", title="One unknown mean", eyebrow="Start here", method="Exact Gaussian inference",
         description="An unknown mean and one noisy observation. Two lines specify the probability relationships.",
         notes=["Normal(mean, sd) uses standard deviation in Python; the mathematical diagram writes variance.",
                "The observation y = 2 is an illustrative input. The posterior calculation is analytic."], sources=[]),
    dict(id="linear-regression", title="A line through observations", eyebrow="Automatic observations", method="Exact Gaussian inference",
         description="Give x as an array. The priors are shared; the likelihood creates one observation per element.",
         notes=["Thirty synthetic observations, generated with intercept 1.5, slope −0.8 and noise SD 0.35.",
                "The displayed fit uses NumPy. Torch inputs select Torch execution; exact Gaussian inference also supports available Torch devices."], sources=[]),
    dict(id="robust-regression", title="A line with heavier tails", eyebrow="Unknown noise", method="Approximate variational inference",
         description="Replace Normal with StudentT and give the positive scale a prior. The observation syntax stays the same.",
         notes=["Synthetic Normal observations with two deliberately added outliers. This one dataset illustrates the model, not a general robustness guarantee.",
                "StudentT scale is not its SD. With df = 4, variance is twice scale squared.",
                "VI intervals describe the fitted guide; they are not exact posterior intervals. This VI implementation uses Pyro and PyTorch internally."], sources=[]),
    dict(id="hidden-markov-model", title="States, transitions, emissions", eyebrow="No time loop", method="Exact hidden-state inference",
         description="Supply the initial probabilities, transition table and Normal emissions. HMM constructs the time dependence.",
         notes=["Thirty synthetic time steps. State names label integer states 0 and 1; transition rows are current states and columns are next states.",
                "Transitions and emissions are fixed. Exact forward–backward inference recovers state probabilities on NumPy/CPU; learning parameters is not supported.",
                "The Pyro pane expresses this same joint law with named time sites. Pyro also has specialized HMM distributions and enumeration tools; this is not a port of its richer piano-sequence tutorial."],
         sources=[{"label": "Pyro HMM tutorial (broader scope)", "url": "https://pyro.ai/examples/hmm.html"}]),
    dict(id="weighing-scale", title="How much does it weigh?", eyebrow="From the Pyro tutorials", method="Exact Gaussian inference",
         description="A prior guess of 8.5 and a noisy reading of 9.5. Condition the same model in one call.",
         notes=["Same probability model as the original scale example: prior SD 1 and measurement SD 0.75.",
                "Pyro labels this older tutorial deprecated. The analytic posterior is Normal with mean 9.14 and SD 0.60."],
         sources=[{"label": "Pyro's scale example", "url": "https://pyro.ai/examples/intro_part_ii.html"}]),
    dict(id="interaction-regression", title="Two groups, two slopes", eyebrow="Adapted from Pyro", method="Approximate variational inference",
         description="An intercept, two predictors and their interaction. Group 1 has slope + interaction; group 0 has slope.",
         notes=["Adaptation: fixed synthetic data replace GDP data, and HalfNormal(1) replaces Uniform(0, 10) for noise SD. The prior change changes the model.",
                "Both code panes use the adapted prior. The plot shows 90% guide bands for mean responses, excluding new observation noise.",
                "Against independent numerical integration, the saved fit's largest mean error is 0.126 reference SD and largest SD error is 9.5%. These checks do not establish accurate tails."],
         sources=[{"label": "Pyro's regression tutorial", "url": "https://pyro.ai/examples/intro_long.html#example-from-maximum-likelihood-regression-to-bayesian-regression"}]),
    dict(id="eight-schools", title="Eight schools, shared information", eyebrow="Adapted from Pyro", method="Approximate variational inference",
         description="Eight effects share an unknown population mean and scale. An array of local effects handles the hierarchy.",
         notes=["The eight observations and standard errors are the published Eight Schools data, not synthetic data.",
                "Adaptation: HalfNormal(5) replaces Pyro's HalfCauchy(25) prior on tau, and VI replaces NUTS. Both code panes use the adapted law; its posterior differs from the original example.",
                "School effects come from joint draws of mu, tau and eta. Against numerical integration, hyperparameter mean errors are at most 0.083 reference SD and SD errors at most 7.4%; intervals remain approximate."],
         sources=[{"label": "Pyro's Eight Schools model", "url": "https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/mcmc.py"},
                  {"label": "Published observations", "url": "https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/data.py"}]),
]

# Select cells by source, not notebook position. A changed notebook that no
# longer matches must be reconciled explicitly rather than showing stale plots.
OUTPUTS = {
    "normal-mean": ("smallprob_examples.ipynb", "p = simple.infer", None),
    "linear-regression": ("smallprob_examples.ipynb", "p_numpy = line.infer", "plt.scatter(x, y, s=24"),
    "robust-regression": ("smallprob_examples.ipynb", "y_outliers = y.copy()", "plt.scatter(x, y_outliers"),
    "hidden-markov-model": ("smallprob_examples.ipynb", None, "sequence = chain.simulate"),
    "weighing-scale": ("smallprob_pyro_gallery.ipynb", "generated = scale.simulate", "grid = np.linspace(5, 12"),
    "interaction-regression": ("smallprob_pyro_gallery.ipynb", "p_interaction = regression.infer", "draws = p_interaction.sample"),
    "eight-schools": ("smallprob_pyro_gallery.ipynb", "generated_schools = schools.simulate", "positions = np.arange(8)"),
}


def _cell(notebook, prefix):
    matches = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.startswith(prefix)]
    if len(matches) != 1 or matches[0].execution_count is None:
        raise ValueError(f"Expected one executed notebook cell beginning {prefix!r}")
    if any(output.output_type == "error" for output in matches[0].outputs):
        raise ValueError(f"Notebook cell has an execution error: {prefix}")
    return matches[0]


def _result(notebook, case_id, prefix):
    if prefix is None:
        return "30 observations → 30 × 2 exact posterior state probabilities. The plot compares them with the generated states."
    cell = _cell(notebook, prefix)
    lines = "".join(output.get("text", "") for output in cell.outputs).strip().splitlines()
    # Select the measured summaries; avoid device-dependent diagnostics and
    # incidental prior-predictive samples in the compact result area.
    if case_id == "linear-regression":
        lines = [line for line in lines if line.startswith("NumPy slope:")]
    if case_id == "weighing-scale":
        lines = [line for line in lines if line.startswith("Exact posterior:")]
    if case_id == "interaction-regression":
        lines = [line for line in lines if not line.startswith("Known-noise")]
    if case_id == "eight-schools":
        lines = [line for line in lines if not line.startswith("Prior predictive")]
    # Keep the measured output, with long decimal strings rounded for display.
    return re.sub(r"(?<!\w)-?\d+\.\d{5,}(?:[eE][+-]?\d+)?",
                  lambda match: f"{float(match.group()):.4f}", "\n".join(lines))


def download_source(case):
    source_notes = "\n".join(f"- {note}" for note in case["notes"])
    sources = "\n".join(f"{source['label']}: {source['url']}" for source in case["sources"])
    return (f'"""{case["title"]} — runnable Smallprob example.\n\n'
            'After installing the cloned repository: uv sync --extra vi --extra notebook\n'
            f'Run: .venv/bin/python website/downloads/examples/{case["id"]}.py\n'
            'The script prints inference results and saves an SVG in the current directory.\n\n'
            f'{source_notes}\n\n{sources}\n"""\n\n'
            + case["smallprob"] + "\n\n" + case["run"] + "\n")


def build(destination=SITE):
    destination = Path(destination)
    for folder in ("assets/diagrams", "assets/plots", "downloads/examples"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    notebooks = {name: nbformat.read(ROOT / "notebooks" / name, as_version=4)
                 for name in {item[0] for item in OUTPUTS.values()}}
    provenance = {name: hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest()
                  for name in notebooks}
    models = bound_cases()
    cases = []
    # Matplotlib hashes SVG paths and embeds creation dates by default. Fix
    # both so rebuilding unchanged inputs produces the same assets.
    import matplotlib as mpl
    with mpl.rc_context({"svg.hashsalt": "smallprob-site"}):
        for metadata in METADATA:
            case = dict(metadata)
            case_id = case["id"]
            notebook_name, result_prefix, plot_prefix = OUTPUTS[case_id]
            notebook = notebooks[notebook_name]
            case.update(smallprob=smallprob_code(case_id), pyro=PYRO_MODELS[case_id].strip(),
                        run=textwrap.dedent(RUN_CODE[case_id]).strip(),
                        result=_result(notebook, case_id, result_prefix),
                        diagram=f"assets/diagrams/{case_id}.svg", plot=None,
                        download=f"downloads/examples/{case_id}.py")
            observed = ["measurement"] if case_id == "weighing-scale" else ["y"]
            labels = ({"weight": "w", "measurement": "y"} if case_id == "weighing-scale" else
                      {"intercept": "b", "slope": "m", "group_effect": "a", "interaction": "c"})
            labels = {name: label for name, label in labels.items() if name in models[case_id][0].graph.groups}
            diagram_path = destination / case["diagram"]
            models[case_id][0].diagram(labels=labels, observed=observed).save(diagram_path)
            # The renderer writes an SVG creation date; it is not evidence.
            diagram_path.write_text(re.sub(r"<dc:date>.*?</dc:date>", "", diagram_path.read_text()))
            if plot_prefix:
                plot_cell = _cell(notebook, plot_prefix)
                images = [output["data"]["image/png"] for output in plot_cell.outputs
                          if "image/png" in output.get("data", {})]
                if len(images) != 1:
                    raise ValueError(f"Expected one saved plot for {case_id}")
                case["plot"] = f"assets/plots/{case_id}.png"
                (destination / case["plot"]).write_bytes(base64.b64decode(images[0]))
            (destination / case["download"]).write_text(download_source(case))
            cases.append(case)
    checks = comparison_checks()
    if not all(check["passed"] for check in checks):
        raise AssertionError(f"Smallprob/Pyro joint-density mismatch: {checks}")
    evidence = {
        "scope": "Model probability laws at three generated complete assignments per case; not inference or speed equivalence.",
        "dtype": "float64", "tolerance": 1e-10, "checks": checks,
        "passed": True, "max_absolute_log_joint_error": max(check["absolute_error"] for check in checks),
        "notebook_sha256": provenance,
        "plot_provenance": "PNG outputs copied from executed committed notebooks. VI fits are not rerun by this build.",
        "adaptations": "Interaction noise prior and Eight Schools scale prior are adapted; both displayed code panes use the same adapted law.",
    }
    (destination / "assets/comparison-validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (destination / "assets/cases.js").write_text("window.SMALLPROB_CASES = " + json.dumps(cases, indent=2, ensure_ascii=False) + ";\n")
    print(f"Built {len(cases)} cases; {len(checks)} equivalent joint-density checks passed.")
    return cases


if __name__ == "__main__":
    build()
