"""Build the native VI explanation and its two runnable downloads."""
from pathlib import Path
from html import escape
import json
import textwrap

SMALLPROB = '''import numpy as np
from smallprob import model, Normal, HalfNormal

@model
def regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(b + m * x, sigma)

x = np.linspace(-1.5, 1.5, 32)
line = regression(x)
data = line.simulate(seed=7, fixed={"b": 1.2, "m": -0.7, "sigma": 0.6})
posterior = line.infer(y=data["y"][0], method="vi", guide="diagonal", steps=4000, seed=7)
print(posterior.mean("m"), posterior.interval("m"))
print("Noise scale:", posterior.mean("sigma"))
draws = posterior.sample(2000, seed=8)
'''
PYRO = '''import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

# Standalone script settings: use double precision for this small example.
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
pyro.set_rng_seed(7)
pyro.clear_param_store()

def regression(x, y=None):
    b = pyro.sample("b", dist.Normal(0., 5.))
    m = pyro.sample("m", dist.Normal(0., 5.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    with pyro.plate("observations", len(x)):
        return pyro.sample("y", dist.Normal(b + m * x, sigma), obs=y)

x = torch.linspace(-1.5, 1.5, 32)
fixed = {"b": torch.tensor(1.2), "m": torch.tensor(-0.7), "sigma": torch.tensor(0.6)}
y = pyro.poutine.condition(regression, data=fixed)(x)
guide = AutoDiagonalNormal(regression)
steps = 4000
optimizer = ClippedAdam({"lr": .03, "lrd": .1**(1/steps)})
svi = SVI(regression, guide, optimizer, loss=Trace_ELBO(num_particles=32, vectorize_particles=True))
for _ in range(steps):
    svi.step(x, y)
draws = Predictive(guide, num_samples=2000)(x, y)
print(draws["m"].mean(), draws["m"].quantile(torch.tensor([.025, .975])))
print("Noise scale:", draws["sigma"].mean())
'''


def build():
    root = Path(__file__).resolve().parents[1]
    site = root/'website'
    report = json.loads((site/'assets/meanfield/validation.json').read_text())
    for name, source in [('smallprob', SMALLPROB), ('pyro', PYRO)]:
        (site/f'downloads/examples/meanfield-{name}.py').write_text('"""Unknown-noise regression with a diagonal Normal guide. Synthetic data."""\n'+source)
    errors = max(run['max_sd_error_diagonal_optimum'] for run in report['known_noise'])
    unknown = report['unknown_noise']
    figures = '<figure class="result-figure"><img src="assets/meanfield/validation.svg" alt="Exact correlated Gaussian ellipse compared with a narrower diagonal guide, and unknown-noise regression with guide bands"><figcaption>Left: 90% joint regions for correlated coefficients. Right: synthetic observations and a 95% guide band for the mean response, excluding future observation noise. <a href="assets/meanfield/validation.png">300-dpi PNG</a> · <a href="assets/meanfield/validation.svg">Vector SVG</a></figcaption></figure>'
    pairs = ''.join(f'<div class="code-panel"><div class="code-heading"><div><h3>{name}</h3><p class="code-method">{method}</p></div><a href="downloads/examples/meanfield-{key}.py" download>Download .py ↓</a></div><pre tabindex="0"><code>{escape(source)}</code></pre></div>' for name,key,method,source in [('Smallprob','smallprob','Native PyTorch · diagonal guide',SMALLPROB),('Pyro','pyro','SVI · AutoDiagonalNormal',PYRO)])
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mean-field VI · Smallprob</title><meta name="description" content="A PyTorch-only variational inference engine with independent Normal factors, checked against exact posteriors."><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&amp;display=swap" rel="stylesheet"><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="article.css"></head>
<body><a class="skip-link" href="#main">Skip to content</a><div class="site-shell"><header class="site-header"><a class="wordmark" href="./">smallprob<span>.</span></a><p class="site-byline">A project by<br><a href="https://nipunbatra.github.io/">Nipun Batra</a></p><nav class="top-nav" aria-label="Main navigation"><a href="./#examples">Examples</a><a href="neural.html">Neural models</a><a href="meanfield.html" aria-current="page">Mean-field VI</a><a href="./#get-started">Get started</a><a href="https://github.com/nipunbatra/smallprob">GitHub ↗</a></nav></header>
<main id="main" class="page-width article-page"><section class="intro"><div><p class="eyebrow">An inference engine you can inspect</p><h1>Two vectors.<br>A useful approximation.</h1><p class="intro-lead">Learn a mean and a standard deviation for every latent. The native engine uses PyTorch autograd and Adam; it does not import Pyro or optimize covariance parameters.</p></div></section>
<section class="article-section"><h2>The complete fit</h2><p>Use the same model and select <code>guide="diagonal"</code>. This example learns an intercept, a slope and the observation noise. Positive noise has a Normal guide on the log scale, so its guide on the original scale is lognormal.</p><div class="comparison-grid">{pairs}</div><p class="comparison-note">Both scripts generate synthetic data from the same probability model. NumPy and Torch random streams differ, so the same seed does not give identical observations or fitted summaries. Both guides are diagonal; the optimizers and ELBO estimators differ. The checks below use fixed shared observations and independent references.</p></section>
<section class="article-section"><h2>What the engine optimizes</h2><div class="formula">q(u) = ∏ᵢ Normal(uᵢ; μᵢ, sᵢ²)<br>u = μ + s ⊙ ε, &nbsp; ε ∼ Normal(0, I)<br>ELBO = E<sub>q</sub>[log p(y, T(u)) + log |J<sub>T</sub>(u)|] + H[q]</div><p>The trainable vectors are <code>μ</code> and <code>log s</code>. Reparameterization lets gradients pass through sampled values; Normal entropy is analytic. For a positive variable, <code>T(u) = exp(u)</code> and the log-Jacobian is <code>u</code>. The model itself can have dependencies and unknown scales; only the guide factors are independent.</p><p>Training uses antithetic Monte Carlo pairs, gradient clipping and a decaying learning rate. The returned ELBO is estimated from fresh samples, with its Monte Carlo standard error. It is not exact evidence or proof of convergence. <a href="https://github.com/nipunbatra/smallprob/blob/main/src/smallprob/meanfield.py">Read the engine</a>.</p></section>
<section class="article-section"><h2>Check it against answers we know</h2>{figures}<p>Six known-noise regression fits cover three seeds and both centered and correlated predictors. The means agree with the analytic Gaussian posterior; the largest scale error against the <em>optimal diagonal guide</em> is {errors:.1%}. The unknown-noise fit is compared with independently refined numerical quadrature: largest mean error {unknown['max_mean_error_reference_sd']:.3f} reference SD, largest SD difference {unknown['max_sd_relative_error']:.1%}.</p><p>For a Gaussian posterior with precision Λ, the best mean-field guide has variance <code>1 / Λᵢᵢ</code>. That generally differs from the marginal variance <code>(Λ⁻¹)ᵢᵢ</code>. The narrower red ellipse is the consequence of excluding correlation, not a claim of more accurate uncertainty.</p><p><a href="assets/meanfield/validation.json">Measured validation results</a> · <a href="https://github.com/nipunbatra/smallprob/blob/main/examples/meanfield.py">Reproduce the checks and figure</a></p></section>
<section class="article-section"><h2>Small scope, explicit choices</h2><p>The native engine supports continuous Normal, HalfNormal and StudentT graphs on available Torch CPU, CUDA or MPS devices. NumPy-facing results remain NumPy arrays. It does not add discrete VI, neural modules inside <code>@model</code>, general minibatching or multimodal guides. Use the separate <a href="neural.html">VAE helper</a> for amortized neural inference.</p><p>For compatibility, <code>method="vi"</code> without a guide still selects the existing full-covariance Pyro backend; <code>guide="full"</code> makes that choice explicit. Exact inference remains the default for supported Gaussian models and fixed-parameter HMMs.</p><pre><code>uv sync --extra torch --extra viz --locked
.venv/bin/python website/downloads/examples/meanfield-smallprob.py
.venv/bin/python -m examples.meanfield</code></pre></section>
<section class="article-section"><h2>Readable figures, sharp exports</h2><p>Enable shared Matplotlib settings explicitly. Notebooks show SVG plus high-resolution PNG; figure exports default to 300 dpi, with readable labels, a consistent color cycle, light grids and open top/right spines. Importing Smallprob does not change your plotting settings.</p><pre><code>import matplotlib.pyplot as plt
from smallprob.plotting import use_style, plot_style
use_style()  # apply once in a notebook

# Or temporarily, restoring your existing settings afterwards:
with plot_style():
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig("fit.svg")  # vector
    fig.savefig("fit.png")  # 300 dpi</code></pre></section>
<p class="comparison-note">Reference: <a href="https://docs.pytorch.org/docs/stable/distributions.html#pathwise-derivative">PyTorch's pathwise derivative estimator</a>. All figures above were produced by the committed Python example; they are not browser simulations.</p>
</main><footer class="site-footer"><p>Smallprob · a personal modelling toolkit</p><a href="./">Worked examples</a><a href="neural.html">Neural models →</a></footer></div></body></html>'''
    (site/'meanfield.html').write_text(body)
    print(site/'meanfield.html')


if __name__ == '__main__':
    build()
