"""The seven probability models used by the static comparison site.

Pyro strings are executable model definitions, not pseudocode. They keep the
same probability laws. Complete generation and inference recipes live in
``site_workflows.py`` so model structure remains separate from inference.
"""
from __future__ import annotations

import inspect
import textwrap

import numpy as np

from smallprob import HMM, HalfNormal, Normal, StudentT, model


@model
def normal_observation():
    mu = Normal(0, 3)
    y = Normal(mu, 1)


@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    y = Normal(intercept + slope * x, 0.35)


@model
def robust_regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(df=4, loc=b + m * x, scale=sigma)


def hidden_markov_model():
    return HMM(
        states=["quiet", "active"],
        initial=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.15, 0.85]],
        emissions=Normal([-1.5, 1.5], 0.45),
        steps=30,
    )


@model
def weighing_scale(guess=8.5):
    weight = Normal(guess, 1)
    measurement = Normal(weight, 0.75)


@model
def interaction_regression(group, x):
    intercept = Normal(0, 10)
    group_effect = Normal(0, 1)
    slope = Normal(0, 1)
    interaction = Normal(0, 1)
    sigma = HalfNormal(1)
    y = Normal(intercept + group_effect * group + slope * x + interaction * group * x, sigma)


@model
def eight_schools(se):
    mu = Normal(0, 10)
    tau = HalfNormal(5)
    eta = Normal(np.zeros(len(se)), 1)
    y = Normal(mu + tau * eta, se)


PYRO_IMPORTS = "import pyro\nimport pyro.distributions as dist\n"
PYRO_MODELS = {
    "normal-mean": PYRO_IMPORTS + '''
def normal_observation(y=None):
    mu = pyro.sample("mu", dist.Normal(0., 3.))
    pyro.sample("y", dist.Normal(mu, 1.), obs=y)
''',
    "linear-regression": PYRO_IMPORTS + '''
def regression(x, y=None):
    intercept = pyro.sample("intercept", dist.Normal(0., 5.))
    slope = pyro.sample("slope", dist.Normal(0., 5.))
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.Normal(intercept + slope * x, 0.35), obs=y)
''',
    "robust-regression": PYRO_IMPORTS + '''
def robust_regression(x, y=None):
    b = pyro.sample("b", dist.Normal(0., 5.))
    m = pyro.sample("m", dist.Normal(0., 5.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.StudentT(4., b + m * x, sigma), obs=y)
''',
    "hidden-markov-model": "import torch\n" + PYRO_IMPORTS + '''
def hidden_markov_model(y=None, steps=30):
    initial = torch.tensor([0.5, 0.5])
    transition = torch.tensor([[0.9, 0.1], [0.15, 0.85]])
    means = torch.tensor([-1.5, 1.5])
    for t in pyro.markov(range(steps)):
        probs = initial if t == 0 else transition[z]
        z = pyro.sample(f"z[{t}]", dist.Categorical(probs))
        observed = None if y is None else y[t]
        pyro.sample(f"y[{t}]", dist.Normal(means[z], 0.45), obs=observed)
''',
    "weighing-scale": PYRO_IMPORTS + '''
def weighing_scale(guess=8.5, measurement=None):
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    pyro.sample("measurement", dist.Normal(weight, 0.75), obs=measurement)
''',
    "interaction-regression": PYRO_IMPORTS + '''
def interaction_regression(group, x, y=None):
    intercept = pyro.sample("intercept", dist.Normal(0., 10.))
    group_effect = pyro.sample("group_effect", dist.Normal(0., 1.))
    slope = pyro.sample("slope", dist.Normal(0., 1.))
    interaction = pyro.sample("interaction", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    mean = intercept + group_effect * group + slope * x + interaction * group * x
    with pyro.plate("observations", len(x)):
        pyro.sample("y", dist.Normal(mean, sigma), obs=y)
''',
    "eight-schools": "import torch\n" + PYRO_IMPORTS + '''
def eight_schools(se, y=None):
    mu = pyro.sample("mu", dist.Normal(0., 10.))
    tau = pyro.sample("tau", dist.HalfNormal(5.))
    with pyro.plate("schools", len(se)):
        eta = pyro.sample("eta", dist.Normal(torch.zeros_like(se), 1.))
        pyro.sample("y", dist.Normal(mu + tau * eta, se), obs=y)
''',
}

MODEL_FUNCTIONS = {
    "normal-mean": normal_observation,
    "linear-regression": regression,
    "robust-regression": robust_regression,
    "hidden-markov-model": hidden_markov_model,
    "weighing-scale": weighing_scale,
    "interaction-regression": interaction_regression,
    "eight-schools": eight_schools,
}


def smallprob_code(case_id):
    """Extract displayed source from the functions used by validation."""
    if case_id == "hidden-markov-model":
        source = inspect.getsource(hidden_markov_model)
        body = textwrap.dedent(source.split("    return ", 1)[1])
        # The first line followed 'return'; the remaining lines were indented
        # for a function body. Remove exactly that outer indentation.
        lines = body.splitlines()
        body = "\n".join([lines[0]] + [line[4:] for line in lines[1:]])
        return "from smallprob import HMM, Normal\n\nchain = " + body
    function = MODEL_FUNCTIONS[case_id]
    source = inspect.getsource(inspect.unwrap(function))
    distributions = ["model", "Normal"]
    if case_id in {"robust-regression", "interaction-regression", "eight-schools"}:
        distributions.append("HalfNormal")
    if case_id == "robust-regression":
        distributions.append("StudentT")
    imports = "import numpy as np\n" if case_id == "eight-schools" else ""
    return imports + "from smallprob import " + ", ".join(distributions) + "\n\n" + source.strip()


def bound_cases():
    x = np.linspace(-2, 2, 30)
    group = np.repeat([0., 1.], 20)
    interaction_x = np.tile(np.linspace(-1.5, 1.5, 20), 2)
    se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
    return {
        "normal-mean": (normal_observation(), ()),
        "linear-regression": (regression(x), (x,)),
        "robust-regression": (robust_regression(x), (x,)),
        "hidden-markov-model": (hidden_markov_model(), ()),
        "weighing-scale": (weighing_scale(), ()),
        "interaction-regression": (interaction_regression(group, interaction_x), (group, interaction_x)),
        "eight-schools": (eight_schools(se), (se,)),
    }


def comparison_checks(seeds=(7, 19, 103)):
    """Compare complete log joints at generated assignments, including HMM paths.

    These are probability-law checks, not inference or performance comparisons.
    Both implementations use float64 for this numerical comparison.
    """
    import pyro
    import torch

    previous_dtype = torch.get_default_dtype()
    checks = []
    try:
        torch.set_default_dtype(torch.float64)
        for case_id, (bound, args) in bound_cases().items():
            namespace = {}
            exec(compile(PYRO_MODELS[case_id], f"<pyro-{case_id}>", "exec"), namespace)
            reference = namespace[MODEL_FUNCTIONS[case_id].__name__]
            torch_args = tuple(torch.as_tensor(value, dtype=torch.float64) for value in args)
            for seed in seeds:
                draws = bound.simulate(seed=seed)
                values = {name: draws[name][0] for name in bound.graph.groups}
                # Groups are vector sites in Pyro, except HMM's explicitly named
                # time sites. z must retain integer dtype for tensor indexing.
                if case_id == "hidden-markov-model":
                    values = bound.graph.expand_values(values)
                conditions = {name: torch.as_tensor(value,
                              dtype=torch.long if name.startswith("z[") else torch.float64)
                              for name, value in values.items()}
                conditioned = pyro.poutine.condition(reference, data=conditions)
                trace = pyro.poutine.trace(conditioned).get_trace(*torch_args)
                expected = float(trace.log_prob_sum())
                actual = float(bound.log_prob(values))
                error = abs(actual - expected)
                checks.append({"case": case_id, "seed": seed,
                               "smallprob_log_joint": actual, "pyro_log_joint": expected,
                               "absolute_error": error, "passed": error < 1e-10})
    finally:
        torch.set_default_dtype(previous_dtype)
    return checks
