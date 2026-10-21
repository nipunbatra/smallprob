"""Two groups, two slopes — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/interaction-regression.py
The script prints inference results and saves an SVG in the current directory.

- Adaptation: fixed synthetic data replace GDP data, and HalfNormal(1) replaces Uniform(0, 10) for noise SD. The prior change changes the model.
- Both code panes use the adapted prior. The plot shows 90% guide bands for mean responses, excluding new observation noise.
- Against independent numerical integration, the saved fit's largest mean error is 0.126 reference SD and largest SD error is 9.5%. These checks do not establish accurate tails.

Pyro's regression tutorial: https://pyro.ai/examples/intro_long.html#example-from-maximum-likelihood-regression-to-bayesian-regression
"""

from smallprob import model, Normal, HalfNormal

@model
def interaction_regression(group, x):
    intercept = Normal(0, 10)
    group_effect = Normal(0, 1)
    slope = Normal(0, 1)
    interaction = Normal(0, 1)
    sigma = HalfNormal(1)
    y = Normal(intercept + group_effect * group + slope * x + interaction * group * x, sigma)

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
