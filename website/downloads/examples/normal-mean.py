"""One unknown mean — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/normal-mean.py
The script prints inference results and saves an SVG in the current directory.

- Normal(mean, sd) uses standard deviation in Python; the mathematical diagram writes variance.
- The observation y = 2 is an illustrative input. The posterior calculation is analytic.

Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal
"""

from smallprob import model, Normal

@model
def normal_observation():
    mu = Normal(0, 3)
    y = Normal(mu, 1)

simple = normal_observation()
generated = simple.simulate(draws=5, seed=7)
y = 2.0  # Illustrative observed value; generated is prior predictive.

posterior = simple.infer(y=y)

print(posterior.mean("mu"), posterior.sd("mu"))
simple.diagram(observed=["y"]).save("normal-mean-model.svg")
