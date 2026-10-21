"""Eight schools, shared information — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/eight-schools.py
The script prints inference results and saves an SVG in the current directory.

- The eight observations and standard errors are the published Eight Schools data, not synthetic data.
- Adaptation: HalfNormal(5) replaces Pyro's HalfCauchy(25) prior on tau, and VI replaces NUTS. Both code panes use the adapted law; its posterior differs from the original example.
- School effects come from joint draws of mu, tau and eta. Against numerical integration, hyperparameter mean errors are at most 0.083 reference SD and SD errors at most 7.4%; intervals remain approximate.

Pyro's Eight Schools model: https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/mcmc.py
Published observations: https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/data.py
"""

import numpy as np
from smallprob import model, Normal, HalfNormal

@model
def eight_schools(se):
    mu = Normal(0, 10)
    tau = HalfNormal(5)
    eta = Normal(np.zeros(len(se)), 1)
    y = Normal(mu + tau * eta, se)

se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
schools = eight_schools(se)
generated = schools.simulate(draws=2, seed=7)
posterior = schools.infer(y=y, method="vi", steps=4000, seed=11)
draws = posterior.sample(6000, seed=12)
effects = draws["mu"][:, None] + draws["tau"][:, None] * draws["eta"]
print(effects.mean(axis=0))
schools.diagram(observed=["y"]).save("eight-schools-model.svg")
