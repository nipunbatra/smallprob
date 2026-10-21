"""States, transitions, emissions: complete Pyro generation and inference example.

Install: pip install pyro-ppl numpy
Run: python hidden-markov-model-pyro.py

Exact enumeration of hidden states.
Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.

- Thirty synthetic time steps. State names label integer states 0 and 1; transition rows are current states and columns are next states.
- Transitions and emissions are fixed. Exact forward–backward inference recovers state probabilities on NumPy/CPU; learning parameters is not supported.
- The Pyro pane expresses this same joint law with named time sites. Pyro also has specialized HMM distributions and enumeration tools; this is not a port of its richer piano-sequence tutorial.

Pyro HMM tutorial (broader scope): https://pyro.ai/examples/hmm.html
Pyro exact state marginals: https://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.traceenum_elbo.TraceEnum_ELBO.compute_marginals

torch.set_num_threads(1) below limits CPU overhead for these tiny examples.
All generation, fitting and summary code is shown in the website panes.
"""

import torch
torch.set_num_threads(1)

import torch
import pyro
import pyro.distributions as dist

def hidden_markov_model(y=None, steps=30):
    initial = torch.tensor([0.5, 0.5])
    transition = torch.tensor([[0.9, 0.1], [0.15, 0.85]])
    means = torch.tensor([-1.5, 1.5])
    for t in pyro.markov(range(steps)):
        probs = initial if t == 0 else transition[z]
        z = pyro.sample(f"z[{t}]", dist.Categorical(probs))
        observed = None if y is None else y[t]
        pyro.sample(f"y[{t}]", dist.Normal(means[z], 0.45), obs=observed)

import torch
from pyro.infer import Predictive

torch.set_default_dtype(torch.float64)
pyro.set_rng_seed(7)

data = Predictive(hidden_markov_model, num_samples=1)(steps=30)
y = torch.stack([data[f"y[{t}]"][0] for t in range(30)])

from pyro.infer import TraceEnum_ELBO, config_enumerate

enumerated = config_enumerate(hidden_markov_model, "parallel")
objective = TraceEnum_ELBO(max_plate_nesting=0)
marginals = objective.compute_marginals(enumerated, lambda **kwargs: None, y=y)
# There are no unknown continuous parameters, so no optimizer is needed.

probabilities = torch.stack([marginals[f"z[{t}]"].probs.reshape(2) for t in range(30)])
print(probabilities[:, 1])
