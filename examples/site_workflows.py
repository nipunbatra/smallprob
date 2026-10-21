"""Complete, runnable generation and inference recipes for the comparison site.

The strings shown on the website are also the downloaded scripts. Pyro uses
its public inference APIs directly; there is no hidden Smallprob adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
import textwrap

from examples.site_models import MODEL_FUNCTIONS, PYRO_MODELS


def clean(source):
    return textwrap.dedent(source).strip()


SMALLPROB_GENERATE = {
    "normal-mean": '''
simple = normal_observation()
generated = simple.simulate(draws=5, seed=7)
y = 2.0  # Illustrative observed value; generated is prior predictive.
''',
    "linear-regression": '''
import numpy as np

x = np.linspace(-2, 2, 30)
line = regression(x)
data = line.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
y = data["y"][0]
''',
    "robust-regression": '''
import numpy as np

x = np.linspace(-2, 2, 30)
robust = robust_regression(x)
generated = robust.simulate(seed=7)
# Fit the same outlier-contaminated fixture in both implementations.
y = 1.5 - 0.8 * x + np.random.default_rng(7).normal(0, 0.35, len(x))
y[[5, 24]] += [3.0, -3.0]
''',
    "hidden-markov-model": '''
data = chain.simulate(seed=7)
y = data["y"][0]
''',
    "weighing-scale": '''
scale = weighing_scale()
generated = scale.simulate(draws=5, seed=7)
measurement = 9.5  # Illustrative reading; generated is prior predictive.
''',
    "interaction-regression": '''
import numpy as np

x = np.tile(np.linspace(-1.5, 1.5, 20), 2)
group = np.repeat([0., 1.], 20)
regression = interaction_regression(group, x)
truth = {"intercept": 1.2, "group_effect": 0.7,
         "slope": -0.6, "interaction": 0.9, "sigma": 0.5}
data = regression.simulate(seed=103, fixed=truth)
y = data["y"][0]
''',
    "eight-schools": '''
se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
schools = eight_schools(se)
generated = schools.simulate(draws=2, seed=7)
# Fit the published observations, not the prior-predictive draw.
''',
}

SMALLPROB_FIT = {
    "normal-mean": 'posterior = simple.infer(y=y)',
    "linear-regression": 'posterior = line.infer(y=y)',
    "robust-regression": 'posterior = robust.infer(y=y, method="vi", guide="full", steps=3000, seed=7)',
    "hidden-markov-model": 'posterior = chain.infer(y=y)',
    "weighing-scale": 'posterior = scale.infer(measurement=measurement)',
    "interaction-regression": 'posterior = regression.infer(y=y, method="vi", guide="full", steps=4000, seed=11)',
    "eight-schools": 'posterior = schools.infer(y=y, method="vi", guide="full", steps=4000, seed=11)',
}

SMALLPROB_SUMMARIZE = {
    "normal-mean": 'print(posterior.mean("mu"), posterior.sd("mu"))',
    "linear-regression": '''
print(posterior.mean("slope"), posterior.interval("slope"))
draws = posterior.sample(4000, seed=12)
x_new = np.linspace(-3, 3, 50)
mean_response = draws["intercept"][:, None] + draws["slope"][:, None] * x_new
print(np.quantile(mean_response, [0.05, 0.95], axis=0))
# A band for the mean response; add observation noise for predictive draws.
''',
    "robust-regression": 'print(posterior.mean("m"), posterior.interval("m"))',
    "hidden-markov-model": 'probabilities = posterior.probabilities("z")\nprint(probabilities[:, 1])',
    "weighing-scale": 'print(posterior.mean("weight"), posterior.sd("weight"))',
    "interaction-regression": '''
draws = posterior.sample(4000, seed=12)
group_0_slope = draws["slope"]
group_1_slope = draws["slope"] + draws["interaction"]
print(group_0_slope.mean(), group_1_slope.mean())
print(posterior.mean("interaction"), posterior.interval("interaction"))
''',
    "eight-schools": '''
draws = posterior.sample(6000, seed=12)
effects = draws["mu"][:, None] + draws["tau"][:, None] * draws["eta"]
print(effects.mean(axis=0))
''',
}

PYRO_SETUP = '''
import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)
'''

PYRO_GENERATE = {
    "normal-mean": '''
generated = Predictive(normal_observation, num_samples=5)()
y = torch.tensor(2.0)  # Illustrative observation.
''',
    "linear-regression": '''
x = torch.linspace(-2, 2, 30)
truth = {"intercept": torch.tensor(1.5), "slope": torch.tensor(-0.8)}
generator = pyro.poutine.condition(regression, data=truth)
data = Predictive(generator, num_samples=1)(x)
y = data["y"][0]
''',
    "robust-regression": '''
import numpy as np

x_np = np.linspace(-2, 2, 30)
x = torch.as_tensor(x_np)
generated = Predictive(robust_regression, num_samples=1)(x)
y_np = 1.5 - 0.8 * x_np + np.random.default_rng(7).normal(0, 0.35, len(x_np))
y_np[[5, 24]] += [3.0, -3.0]
y = torch.as_tensor(y_np)
''',
    "hidden-markov-model": '''
data = Predictive(hidden_markov_model, num_samples=1)(steps=30)
y = torch.stack([data[f"y[{t}]"][0] for t in range(30)])
''',
    "weighing-scale": '''
generated = Predictive(weighing_scale, num_samples=5)()
measurement = torch.tensor(9.5)  # Illustrative reading.
''',
    "interaction-regression": '''
x = torch.linspace(-1.5, 1.5, 20).repeat(2)
group = torch.tensor([0., 1.]).repeat_interleave(20)
truth = {name: torch.tensor(value) for name, value in {
    "intercept": 1.2, "group_effect": 0.7,
    "slope": -0.6, "interaction": 0.9, "sigma": 0.5}.items()}
generator = pyro.poutine.condition(interaction_regression, data=truth)
pyro.set_rng_seed(103)
data = Predictive(generator, num_samples=1)(group, x)
y = data["y"][0]
''',
    "eight-schools": '''
se = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])
y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
generated = Predictive(eight_schools, num_samples=2)(se)
# Fit the published observations, not the prior-predictive draw.
''',
}

PYRO_ARGUMENTS = {
    "normal-mean": "y=y",
    "linear-regression": "x, y=y",
    "robust-regression": "x, y=y",
    "weighing-scale": "measurement=measurement",
    "interaction-regression": "group, x, y=y",
    "eight-schools": "se, y=y",
}


def pyro_fit(case_id):
    if case_id == "hidden-markov-model":
        return clean('''
from pyro.infer import TraceEnum_ELBO, config_enumerate

enumerated = config_enumerate(hidden_markov_model, "parallel")
objective = TraceEnum_ELBO(max_plate_nesting=0)
marginals = objective.compute_marginals(enumerated, lambda **kwargs: None, y=y)
# There are no unknown continuous parameters, so no optimizer is needed.
''')
    steps = 4000 if case_id in {"eight-schools", "interaction-regression"} else 3000
    seed = 11 if steps == 4000 else 7
    return clean(f'''
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import ClippedAdam

pyro.clear_param_store()
pyro.set_rng_seed({seed})
guide = AutoMultivariateNormal({MODEL_FUNCTIONS[case_id].__name__})
steps = {steps}
optimizer = ClippedAdam({{"lr": 0.03, "lrd": 0.1 ** (1 / steps)}})
svi = SVI({MODEL_FUNCTIONS[case_id].__name__}, guide, optimizer,
          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    loss = svi.step({PYRO_ARGUMENTS[case_id]})
''')


def pyro_summarize(case_id):
    if case_id == "hidden-markov-model":
        return 'probabilities = torch.stack([marginals[f"z[{t}]"].probs.reshape(2) for t in range(30)])\nprint(probabilities[:, 1])'
    draws = 6000 if case_id == "eight-schools" else 4000
    source = (f'pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples={draws})({PYRO_ARGUMENTS[case_id]})\n')
    if case_id in {"normal-mean", "weighing-scale", "robust-regression"}:
        name = {"normal-mean": "mu", "weighing-scale": "weight", "robust-regression": "m"}[case_id]
        source += f'samples = draws["{name}"].reshape(-1)\n'
        source += 'print(samples.mean().item(), samples.std().item())\n'
        if case_id == "robust-regression":
            source += 'print(torch.quantile(samples, torch.tensor([0.025, 0.975])))\n'
    elif case_id == "linear-regression":
        source += clean('''
slopes = draws["slope"].reshape(-1)
intercepts = draws["intercept"].reshape(-1)
print(slopes.mean().item(), torch.quantile(slopes, torch.tensor([0.025, 0.975])))
x_new = torch.linspace(-3, 3, 50)
mean_response = intercepts[:, None] + slopes[:, None] * x_new
print(torch.quantile(mean_response, torch.tensor([0.05, 0.95]), dim=0))
# A band for the mean response; add observation noise for predictive draws.
''')
    elif case_id == "interaction-regression":
        source += clean('''
group_0_slope = draws["slope"].reshape(-1)
group_1_slope = group_0_slope + draws["interaction"].reshape(-1)
print(group_0_slope.mean().item(), group_1_slope.mean().item())
samples = draws["interaction"].reshape(-1)
print(samples.mean().item(), torch.quantile(samples, torch.tensor([0.025, 0.975])))
''')
    elif case_id == "eight-schools":
        source += clean('''
effects = draws["mu"].reshape(-1, 1) + draws["tau"].reshape(-1, 1) * draws["eta"]
print(effects.mean(dim=0))
''')
    return source.strip()


def workflows(case_id):
    """Display snippets, in their executable order after the model definition."""
    return [
        {"id": "generate", "label": "Generate data", "smallprob": clean(SMALLPROB_GENERATE[case_id]),
         "pyro": clean(PYRO_SETUP) + "\n\n" + clean(PYRO_GENERATE[case_id])},
        {"id": "fit", "label": "Fit the model", "smallprob": SMALLPROB_FIT[case_id],
         "pyro": pyro_fit(case_id)},
        {"id": "summarize", "label": "Use the posterior", "smallprob": clean(SMALLPROB_SUMMARIZE[case_id]),
         "pyro": pyro_summarize(case_id)},
    ]


WORKFLOW_NOTE = (
    "Run each column from Model through Use the posterior, or download its complete script. "
    "The generators use different random-number streams: equal seeds do not imply equal data. "
    "The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. "
    "Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide."
)


def pyro_download_source(case):
    source_notes = "\n".join(f"- {note}" for note in case["notes"])
    sources = "\n".join(f"{source['label']}: {source['url']}" for source in case["sources"])
    return (f'"""{case["title"]}: complete Pyro generation and inference example.\n\n'
            'Install: pip install pyro-ppl numpy\n'
            f'Run: python {case["id"]}-pyro.py\n\n'
            f'{case["pyro_method"]}.\n{WORKFLOW_NOTE}\n'
            f'\n{source_notes}\n\n{sources}\n\n'
            'torch.set_num_threads(1) below limits CPU overhead for these tiny examples.\n'
            'All generation, fitting and summary code is shown in the website panes.\n"""\n\n'
            + 'import torch\ntorch.set_num_threads(1)\n\n'
            + case["pyro"] + "\n\n"
            + "\n\n".join(step["pyro"] for step in case["workflows"]) + "\n")


@contextmanager
def isolated_pyro():
    """Keep running a downloaded recipe in a test from changing caller state."""
    import numpy as np
    import random
    import torch
    import pyro
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        torch.set_num_threads(1)
        with pyro.get_param_store().scope(), torch.random.fork_rng():
            yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def execute_pyro_workflow(case_id, *, observed=None, steps=None):
    """Run the displayed recipe; an override allows shared-data validation."""
    import torch
    namespace = {}
    with isolated_pyro():
        exec(PYRO_MODELS[case_id], namespace)
        recipe = workflows(case_id)
        exec(recipe[0]["pyro"], namespace)
        if observed is not None:
            key = "measurement" if case_id == "weighing-scale" else "y"
            namespace[key] = torch.as_tensor(observed, dtype=torch.float64)
        fit_code = recipe[1]["pyro"]
        if steps is not None and case_id != "hidden-markov-model":
            default_steps = 4000 if case_id in {"eight-schools", "interaction-regression"} else 3000
            fit_code = fit_code.replace(f"steps = {default_steps}", f"steps = {steps}")
        exec(fit_code, namespace)
        exec(recipe[2]["pyro"], namespace)
    return namespace
