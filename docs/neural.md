# A small VAE, with amortized inference

The optional `smallprob.neural.VAE` API supplies a complete, deliberately narrow
neural generative model. It uses PyTorch directly; Pyro is only needed to run the
side-by-side comparison. It does not change the finite-graph `@model` syntax.

```sh
python -m pip install 'smallprob[torch] @ git+https://github.com/nipunbatra/smallprob.git'
```

```python
from examples.vae import synthetic_bars
from smallprob.neural import VAE

train = synthetic_bars(1024, seed=23)
test = synthetic_bars(256, seed=24)

vae = VAE(input_dim=64, latent_dim=2, hidden_dim=128, seed=7)
history = vae.fit(train, validation_data=test, epochs=200, seed=11)
metrics = vae.evaluate(test, particles=128, seed=21)
reconstructed = vae.reconstruct(test[:12], particles=128, seed=29)
generated = vae.generate(12, seed=31)
```

`synthetic_bars` is an example data generator in this repository, not part of the
installed library. The website's downloadable scripts include its full source.
The images are synthetic 8×8 crossing bars with 1% random pixel flips; no dataset
or model is downloaded. Training and held-out sets use independent draws.

## What is specified by the helper?

For each binary observation vector \(x\) with \(D\) entries,

\[
z\sim\mathcal N(0,I),\qquad
x_j\mid z\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(g_\theta(z))_j).
\]

The encoder supplies

\[
q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),
\operatorname{diag}(\sigma_\phi(x)^2)\right),\quad
\sigma_\phi(x)=\operatorname{softplus}(r_\phi(x))+10^{-4}.
\]

Both networks have one fully connected SiLU hidden layer. The encoder emits
the location and raw scale; the decoder emits Bernoulli logits. `input_dim`,
`latent_dim` and `hidden_dim` select their sizes. This initial API fixes the
architecture and likelihood rather than accepting arbitrary neural modules.

The encoder is a shared function that predicts an approximate latent posterior
for each input, including a new input. This is **amortized VI**. It differs from
the global Gaussian guide used for regression coefficients in `m.infer(...)`.
Encoder and decoder weights are learned point estimates, not Bayesian random
variables with a weight posterior.

## The loss and its units

Training minimizes the negative ELBO per observation:

\[
\frac1N\sum_i\left[-\mathbb E_{q_\phi(z_i\mid x_i)}
  \log p_\theta(x_i\mid z_i)
 +\frac12\sum_k\{\mu_{ik}^2+\sigma_{ik}^2-1-\log\sigma_{ik}^2\}\right].
\]

The reconstruction term sums binary cross entropy over all \(D\) features;
the KL sums over latent dimensions. The outer mean averages images, so values
are **nats per image**, not per pixel. Reparameterized Normal draws estimate
the reconstruction expectation. The diagonal-Normal KL is analytic.

`fit` uses shuffled minibatches, Adam, and gradient-norm clipping at 10. There
is no annealed KL weight or beta-VAE modification hidden in the loss. Every
minibatch uses the mean per-image objective. The last partial batch is valid;
dataset evaluation weights each batch by its actual size.

The returned history contains epoch zero and every completed epoch. Its
training and validation entries evaluate the current networks using
`validation_particles` samples. Validation does not update the networks;
fixed evaluation seeds make curves less noisy. Changing seeds, batch sizes,
platforms or Torch versions can change finite Monte Carlo estimates and fits.
A finite estimated ELBO is not exact log evidence or a guaranteed numerical
bound. A falling loss does not prove the approximate posterior is accurate.

## Small API

| Call | Result |
| --- | --- |
| `vae.fit(x, epochs=50, batch_size=128, learning_rate=.003, seed=0)` | List of per-epoch dictionaries, including epoch zero. |
| `vae.evaluate(x, particles=32, seed=0)` | Mean `negative_elbo`, `reconstruction_nll`, and `kl`. |
| `vae.encode(x)` | Detached `(location, scale)` of the approximate posterior, each `(N, latent_dim)`. |
| `vae.reconstruct(x, particles=32, seed=0)` | Pixel probabilities averaged over guide draws, shape `(N, input_dim)`. |
| `vae.generate(n, seed=0)` | Binary observations after drawing latent codes from the standard Normal prior. |
| `vae.generate(n, probabilities=True, seed=0)` | Decoder probabilities at those prior latent codes, without sampling pixels. |
| `vae.loss(x, particles=1, seed=0, reduction="mean")` | Differentiable negative ELBO for a custom Torch optimizer. Also supports `"sum"` and `"none"`. |
| `vae.parameters()` | Torch parameters from both networks. |

Reconstruction probabilities are not samples from the prior or exact posterior
means. Prior probabilities and binary generations are different outputs; using
the same seed pairs their latent codes. A repeated `fit` continues the current
networks with a fresh optimizer. Constructor `seed` controls initialization;
`fit`'s seed controls minibatch order and latent noise.

`vae.encoder` and `vae.decoder` are inspectable Torch modules. Their standard
`state_dict()` / `load_state_dict()` methods can save and restore weights;
recreate a VAE with the same dimensions before loading them. General neural
graph tracing, alternative likelihoods, convolutional networks, flows, missing
pixels, and Bayesian weight inference are outside this helper's current scope.

## Arrays and devices

Inputs must be finite binary values with shape `(N, input_dim)` and `N > 0`.
NumPy arrays and numeric lists are converted to the constructor's dtype and
device. Fractional intensities are rejected: choose an explicit binarization
procedure if the data are grayscale, or use a different likelihood elsewhere.
Inputs are detached; fitting does not differentiate through the caller's data.

CPU float32 is the default. `dtype="float64"` is available on CPU/CUDA. Use
`device="cuda"` or `device="mps"` to select an available accelerator; MPS
requires float32. Unavailable devices fail rather than falling back. A Torch
input must already be on the selected device; move it explicitly with
`x.to(vae.device)` when needed. All array outputs are Torch tensors on that
device. Converting an output to NumPy is explicit: `tensor.cpu().numpy()`.

Importing `smallprob.neural` does not import Torch. Initialization, fitting and
sampling preserve caller CPU/selected-device RNG states. `fit(threads=1)`
temporarily limits CPU threads and restores the previous setting even on
failure. These scoped settings are serialized with other Smallprob Torch
operations; they do not make unrelated concurrent Torch operations globally
isolated or guarantee identical floating-point results across devices.

## Run and reproduce the comparison

From a repository checkout with the existing extras installed:

```sh
uv sync --extra dev --extra torch --extra vi --extra viz
.venv/bin/python -m examples.vae
.venv/bin/python -m examples.vae --pyro
.venv/bin/python -m examples.build_vae_example
.venv/bin/python -m pytest tests/test_neural.py -q
```

The website compares the entire workflow: synthetic data, networks and model,
fitting, evaluation, reconstruction, and prior generation. Both sides use the
same data, architecture, Gaussian guide and analytic-KL objective. Smallprob
uses its direct Torch optimizer; Pyro uses `SVI` with `TraceMeanField_ELBO`.
Their independent fits need not yield identical weights or samples. The
downloadable Pyro script imports no Smallprob code.

The tests independently check the Bernoulli log likelihood and Normal KL,
batch reductions, gradients, held-out improvement, seeds, input validation,
and available devices. A matched-weights, matched-noise check compares the
native objective and all parameter gradients with Pyro's implementation.
The website records actual 200-epoch CPU fits and exports SVG plus 300-dpi PNG.

This adapts [Pyro's VAE tutorial](https://pyro.ai/examples/vae.html): synthetic
bars replace MNIST; the networks use one SiLU hidden layer with 128 units and
two latent dimensions; softplus parameterizes the encoder scale; and analytic
KL (`TraceMeanField_ELBO`) replaces the tutorial's `Trace_ELBO`. These choices
keep the example small and offline. See also
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
