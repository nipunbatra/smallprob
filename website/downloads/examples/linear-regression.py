"""A line through observations — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/linear-regression.py
The script prints inference results and saves an SVG in the current directory.

- Thirty synthetic observations, generated with intercept 1.5, slope −0.8 and noise SD 0.35.
- The displayed fit uses NumPy. Torch inputs select Torch execution; exact Gaussian inference also supports available Torch devices.


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
posterior = line.infer(y=data["y"][0])
print(posterior.mean("slope"), posterior.interval("slope"))
line.diagram(labels={"intercept": "b", "slope": "m"}, observed=["y"]).save("linear-regression-model.svg")
