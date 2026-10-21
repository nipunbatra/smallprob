"""A line with heavier tails — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/robust-regression.py
The script prints inference results and saves an SVG in the current directory.

- Synthetic Normal observations with two deliberately added outliers. This one dataset illustrates the model, not a general robustness guarantee.
- StudentT scale is not its SD. With df = 4, variance is twice scale squared.
- VI intervals describe the fitted guide; they are not exact posterior intervals. This VI implementation uses Pyro and PyTorch internally.


"""

from smallprob import model, Normal, HalfNormal, StudentT

@model
def robust_regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(df=4, loc=b + m * x, scale=sigma)

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
