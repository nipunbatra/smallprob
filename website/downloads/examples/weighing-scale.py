"""How much does it weigh? — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/weighing-scale.py
The script prints inference results and saves an SVG in the current directory.

- Same probability model as the original scale example: prior SD 1 and measurement SD 0.75.
- Pyro labels this older tutorial deprecated. The analytic posterior is Normal with mean 9.14 and SD 0.60.

Pyro's scale example: https://pyro.ai/examples/intro_part_ii.html
"""

from smallprob import model, Normal

@model
def weighing_scale(guess=8.5):
    weight = Normal(guess, 1)
    measurement = Normal(weight, 0.75)

scale = weighing_scale()
generated = scale.simulate(draws=5, seed=7)
posterior = scale.infer(measurement=9.5)
print(posterior.mean("weight"), posterior.sd("weight"))
scale.diagram(labels={"weight": "w", "measurement": "y"}, observed=["measurement"]).save("weighing-scale-model.svg")
