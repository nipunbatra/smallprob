"""A line through observations — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/linear-regression.py
The script prints inference results and saves an SVG in the current directory.

- Thirty synthetic observations, generated with intercept 1.5, slope −0.8 and noise SD 0.35.
- The displayed fit uses NumPy. Torch inputs select Torch execution; exact Gaussian inference also supports available Torch devices.

Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal
"""

from smallprob import model, Normal

@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    y = Normal(intercept + slope * x, 0.35)

import numpy as np

x = np.linspace(-2, 2, 30)
line = regression(x)
data = line.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
y = data["y"][0]

posterior = line.infer(y=y)

print(posterior.mean("slope"), posterior.interval("slope"))
draws = posterior.sample(4000, seed=12)
x_new = np.linspace(-3, 3, 50)
mean_response = draws["intercept"][:, None] + draws["slope"][:, None] * x_new
print(np.quantile(mean_response, [0.05, 0.95], axis=0))
# A band for the mean response; add observation noise for predictive draws.
line.diagram(labels={"intercept": "b", "slope": "m"}, observed=["y"]).save("linear-regression-model.svg")
