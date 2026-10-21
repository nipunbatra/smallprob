"""Write the short user-facing notebook; run it with examples.run_notebook."""
from pathlib import Path
import nbformat as nb


def build():
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    cells = [
        md("""# Smallprob: four small generative models
Write a probability model once, then generate data, infer unknowns and draw its diagram.

Run from an installed project environment: `uv sync --extra vi --extra notebook`.
The examples use synthetic data. Exact results and approximate VI are labelled separately."""),
        code("""import warnings
warnings.filterwarnings("ignore", message="IProgress not found.*", module="tqdm.auto")

import numpy as np
import torch
import matplotlib.pyplot as plt
from smallprob import model, Normal, HalfNormal, StudentT, HMM

from smallprob.plotting import use_style
from matplotlib_inline.backend_inline import set_matplotlib_formats

use_style()  # readable labels, 150-dpi display, 300-dpi file export
set_matplotlib_formats("svg", "retina")  # sharp at any zoom + high-resolution PNG"""),
        md("""## 1. A Normal observation
An unknown mean, then one noisy observation. In Python, `Normal(mean, sd)` takes standard deviation. The diagram writes the mathematical variance explicitly."""),
        code("""@model
def normal_observation():
    mu = Normal(0, 3)
    y = Normal(mu, 1)

simple = normal_observation()
simple.diagram(observed=["y"])"""),
        code("""p = simple.infer(y=2.0)  # exact Gaussian inference
print(f"Posterior mean of mu: {p.mean('mu'):.3f}")
low, high = p.interval("mu")
print(f"95% marginal interval: [{low:.3f}, {high:.3f}]")
simple.math()"""),
        md(r"""The density is
$$p(y\mid\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].$$"""),
        md("""## 2. Linear regression, with either array library
A vector `x` creates a vector of observations. The two priors are shared by every observation. Display labels can use `m` and `b` while Python keeps readable variable names."""),
        code("""@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    y = Normal(intercept + slope * x, 0.35)

x = np.linspace(-2, 2, 30)
line = regression(x)
data = line.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8})
y = data["y"][0]
line.diagram(labels={"intercept": "b", "slope": "m"}, observed=["y"])"""),
        code("""p_numpy = line.infer(y=y)
p_torch = regression(torch.tensor(x)).infer(y=torch.tensor(y))
print("NumPy slope:", p_numpy.mean("slope"))
print("Torch slope:", p_torch.mean("slope"))
print("Sample arrays:", type(p_numpy.sample(2)["y"]).__name__,
      type(p_torch.sample(2)["y"]).__name__)"""),
        code("""plt.scatter(x, y, s=24, alpha=0.7, label="Synthetic observations")
plt.plot(x, p_numpy.mean("intercept") + p_numpy.mean("slope") * x,
         label="Exact posterior mean line")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()"""),
        md("""The next cell runs Gaussian conditioning on an available GPU, or CPU when neither accelerator is present. Computation and returned arrays use the selected device. Model tracing snapshots fixed inputs on the host. MPS uses float32."""),
        code("""device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
x_device = torch.tensor(x, device=device, dtype=torch.float32)
y_device = torch.tensor(y, device=device, dtype=torch.float32)
p_device = regression(x_device).infer(y=y_device)
print("Device:", p_device.diagnostics["device"], "| slope:", float(p_device.mean("slope")))"""),
        md("""## 3. Student's t for observations with outliers
Now the model uses mathematical names directly. A Student's t likelihood has heavier tails than a Normal. Here `df=4`; `scale` is **not** the standard deviation. We also learn the positive noise scale."""),
        code("""@model
def robust_regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(df=4, loc=b + m * x, scale=sigma)

robust = robust_regression(x)
robust.diagram(observed=["y"])"""),
        code("""y_outliers = y.copy()
y_outliers[[5, 24]] += [3.0, -3.0]  # two deliberately added outliers
p_robust = robust.infer(y=y_outliers, method="vi", seed=7)
p_normal = line.infer(y=y_outliers)
print("Approximate VI slope:", p_robust.mean("m"))
print("95% guide interval:", p_robust.interval("m"))
print("Learned Student t scale:", p_robust.mean("sigma"))"""),
        code("""plt.scatter(x, y_outliers, s=24, alpha=0.7, label="Synthetic data + added outliers")
plt.plot(x, p_robust.mean("b") + p_robust.mean("m") * x, label="Student t: VI mean line")
plt.plot(x, p_normal.mean("intercept") + p_normal.mean("slope") * x,
         linestyle="--", label="Normal: exact mean line")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()"""),
        md(r"""For location $\mu$, scale $s$ and degrees of freedom $\nu$:
$$p(y\mid\nu,\mu,s)=\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,s}\left(1+\frac{(y-\mu)^2}{\nu s^2}\right)^{-(\nu+1)/2}.$$
For $\nu>2$, the variance is $s^2\nu/(\nu-2)$. VI intervals describe the learned approximation, not exact posterior uncertainty. The fitted lines illustrate this fixed dataset, not a general robustness guarantee."""),
        md("""## 4. A hidden Markov model
Each hidden state depends on the previous state, and each observation depends on its current state. Specify the model directly with `HMM`: no loop, append or decorator. Rows of the transition matrix correspond to the current state; columns to the next state. Parameters are supplied here; state inference is exact and uses NumPy on CPU. Names label states 0 and 1."""),
        code("""chain = HMM(
    states=["quiet", "active"],
    initial=[0.5, 0.5],
    transition=[[0.9, 0.1], [0.15, 0.85]],
    emissions=Normal([-1.5, 1.5], 0.45),
    steps=30,
)
chain.diagram(observed=["y"])"""),
        code("""sequence = chain.simulate(seed=7)
states = chain.infer(y=sequence["y"][0])
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
axes[0].plot(sequence["y"][0], ".-", label="Generated observations")
axes[0].set_ylabel("y"); axes[0].legend()
axes[1].plot(states.probabilities("z")[:, 1], label="Exact P(state = 1 | observations)")
axes[1].step(range(30), sequence["z"][0], where="mid", linestyle="--", label="Generated hidden state")
axes[1].set(xlabel="Time step (Python index)", ylabel="State / probability", ylim=(-0.05, 1.05))
axes[1].legend(); plt.tight_layout(); plt.show()"""),
        md("""## Export a diagram
SVG keeps equations, arrows and distribution glyphs sharp. Display labels do not rename variables used by inference. Glyphs identify distribution families; they are not fitted density estimates."""),
        code("""robust.diagram(observed=["y"]).save("outputs/notebook-robust-model.svg")
print("Saved outputs/notebook-robust-model.svg")"""),
    ]
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11+"},
    })
    target=Path(__file__).resolve().parents[1]/'notebooks/smallprob_examples.ipynb'
    target.parent.mkdir(parents=True,exist_ok=True)
    nb.write(notebook,target)
    print(target)


if __name__ == '__main__':
    build()
