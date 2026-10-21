"""How much does it weigh? — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/weighing-scale.py
The script prints inference results and saves an SVG in the current directory.

- Same probability model as the original scale example: prior SD 1 and measurement SD 0.75.
- Pyro labels this older tutorial deprecated. The analytic posterior is Normal with mean 9.14 and SD 0.60.

Pyro's scale example: https://pyro.ai/examples/intro_part_ii.html
Pyro automatic Gaussian guides: https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal
"""

from smallprob import model, Normal

@model
def weighing_scale(guess=8.5):
    weight = Normal(guess, 1)
    measurement = Normal(weight, 0.75)

scale = weighing_scale()
generated = scale.simulate(draws=5, seed=7)
measurement = 9.5  # Illustrative reading; generated is prior predictive.

posterior = scale.infer(measurement=measurement)

print(posterior.mean("weight"), posterior.sd("weight"))
scale.diagram(labels={"weight": "w", "measurement": "y"}, observed=["measurement"]).save("weighing-scale-model.svg")
