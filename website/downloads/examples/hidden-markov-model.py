"""States, transitions, emissions — runnable Smallprob example.

After installing the cloned repository: uv sync --extra vi --extra notebook
Run: .venv/bin/python website/downloads/examples/hidden-markov-model.py
The script prints inference results and saves an SVG in the current directory.

- Thirty synthetic time steps. State names label integer states 0 and 1; transition rows are current states and columns are next states.
- Transitions and emissions are fixed. Exact forward–backward inference recovers state probabilities on NumPy/CPU; learning parameters is not supported.
- The Pyro pane expresses this same joint law with named time sites. Pyro also has specialized HMM distributions and enumeration tools; this is not a port of its richer piano-sequence tutorial.

Pyro HMM tutorial (broader scope): https://pyro.ai/examples/hmm.html
"""

from smallprob import HMM, Normal

chain = HMM(
    states=["quiet", "active"],
    initial=[0.5, 0.5],
    transition=[[0.9, 0.1], [0.15, 0.85]],
    emissions=Normal([-1.5, 1.5], 0.45),
    steps=30,
)

data = chain.simulate(seed=7)
posterior = chain.infer(y=data["y"][0])
print(posterior.probabilities("z")[:, 1])
chain.diagram(observed=["y"]).save("hidden-markov-model.svg")
