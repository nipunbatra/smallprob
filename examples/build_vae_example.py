"""Reproduce the VAE website's measured results, scripts and vector figures.

Run ``python -m examples.build_vae_example`` from the repository root. Training
is on CPU, needs the existing torch/vi/viz extras, and downloads no data.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import textwrap

import numpy as np

from examples.vae import (PYRO_FIT_CODE, PYRO_GENERATE_CODE, PYRO_MODEL_CODE,
                          PYRO_RECONSTRUCT_CODE, synthetic_bars, train_pyro,
                          train_smallprob)


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "website" / "assets" / "vae"

DATA_CODE = "import numpy as np\n\n" + inspect.getsource(synthetic_bars) + "\ntrain = synthetic_bars(1024, seed=23)\ntest = synthetic_bars(256, seed=24)\n"
SMALLPROB_MODEL_CODE = textwrap.dedent('''\
from smallprob.neural import VAE

vae = VAE(input_dim=64, latent_dim=2, hidden_dim=128, seed=7)
''')
SMALLPROB_FIT_CODE = textwrap.dedent('''\
history = vae.fit(
    train, validation_data=test,
    epochs=200, batch_size=128, learning_rate=0.003,
    validation_particles=8, seed=11,
)
metrics = vae.evaluate(test, particles=128, seed=21)
print(metrics)  # negative ELBO, reconstruction NLL, KL: nats per image
''')
SMALLPROB_RECONSTRUCT_CODE = "reconstructed = vae.reconstruct(test[:12], particles=128, seed=29)\n"
SMALLPROB_GENERATE_CODE = textwrap.dedent('''\
pixel_probabilities = vae.generate(12, probabilities=True, seed=31)
generated = vae.generate(12, seed=31)  # actual binary samples
''')


def _save(figure, name):
    import matplotlib.pyplot as plt
    figure.savefig(DESTINATION / f"{name}.svg", metadata={"Date": None})
    figure.savefig(DESTINATION / f"{name}.png", dpi=300)
    plt.close(figure)


def _image_grid(rows, labels, name):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(rows), 8, figsize=(10.5, 1.25 * len(rows)), squeeze=False)
    for row, values in enumerate(rows):
        for column in range(8):
            ax = axes[row, column]
            # Every pixel is a vector polygon in SVG; zooming never blurs the 8x8 grid.
            ax.pcolormesh(np.arange(9), np.arange(9), values[column].reshape(8, 8),
                          cmap="Greys", vmin=0, vmax=1, shading="flat", rasterized=False)
            ax.set(xlim=(0, 8), ylim=(8, 0), aspect="equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column == 0:
                ax.set_ylabel(labels[row], rotation=0, ha="right", va="center", labelpad=10, fontsize=9)
    figure.subplots_adjust(wspace=.12, hspace=.18, left=.15, right=.99, top=.95, bottom=.05)
    _save(figure, name)


def _diagram():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    figure, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")

    def box(x, y, title, subtitle, color):
        patch = FancyBboxPatch((x, y), 2.7, .7, boxstyle="round,pad=0.10,rounding_size=.04",
                               fc="white", ec=color, linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + 1.35, y + .46, title, ha="center", va="center", fontsize=12, color=color)
        ax.text(x + 1.35, y + .18, subtitle, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, color):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=13, linewidth=1.3, color=color))

    red, blue = "#b52d24", "#246892"
    ax.text(2.2, 3.85, "GENERATIVE MODEL", ha="center", fontsize=10, weight="bold", color=red)
    ax.text(7.6, 3.85, "AMORTIZED INFERENCE", ha="center", fontsize=10, weight="bold", color=blue)
    box(.85, 2.65, r"$z \sim \mathcal{N}(0, I_2)$", "one latent code per image", red)
    box(.85, 1.15, r"$x_j \sim \mathrm{Bernoulli}(\pi_j)$", r"$\pi = \mathrm{sigmoid}(g_\theta(z))$, 64 pixels", red)
    arrow(2.2, 2.50, 2.2, 1.95, red)
    ax.text(2.45, 2.23, "decoder", va="center", fontsize=9, color=red)
    box(6.25, 2.65, r"Observed image $x$", "64 binary pixels", blue)
    box(6.25, 1.15, r"$q_\phi(z\mid x)$", r"$\mathcal{N}(\mu_\phi(x),\;\mathrm{diag}(\sigma_\phi(x)^2))$", blue)
    arrow(7.6, 2.50, 7.6, 1.95, blue)
    ax.text(7.85, 2.23, "encoder", va="center", fontsize=9, color=blue)
    ax.text(5, .46, r"$\mathcal{L}(x) = \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]"
            r" - \mathrm{KL}[q_\phi(z\mid x)\,\Vert\,\mathcal{N}(0,I)]$", ha="center", fontsize=13)
    ax.text(5, .04, "Sample the reconstruction term; compute the Normal KL analytically. Learn θ and φ with minibatches.",
            ha="center", fontsize=9, color="#525650")
    _save(figure, "model-and-guide")


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import pyro
    from smallprob.plotting import plot_style

    DESTINATION.mkdir(parents=True, exist_ok=True)
    train, test = synthetic_bars(1024, 23), synthetic_bars(256, 24)
    vae, history, metrics = train_smallprob(train, test)
    _, pyro_history, pyro_metrics = train_pyro(train, test)

    with plot_style():
        _diagram()
        figure, ax = plt.subplots(figsize=(9.5, 3.8))
        ax.plot([row["epoch"] for row in history], [row["train_negative_elbo"] for row in history],
                label="Smallprob · training set", color="#b52d24")
        ax.plot([row["epoch"] for row in history], [row["validation_negative_elbo"] for row in history],
                label="Smallprob · held-out set", color="#246892")
        ax.plot([0] + [row["epoch"] for row in pyro_history],
                [pyro_metrics["before"]] + [row["validation_negative_elbo"] for row in pyro_history],
                label="Pyro · held-out set", color="#80704b", linestyle="--", alpha=.85)
        ax.set(xlabel="Training epoch", ylabel="Negative ELBO (nats / image)",
               xlim=(0, 200), title="Both implementations learn the synthetic distribution")
        ax.legend(loc="upper right")
        _save(figure, "learning-curve")
        _image_grid([test[:8], vae.reconstruct(test[:8], particles=128, seed=29).cpu().numpy()],
                    ["Held-out\nobservations", "Reconstructed\nprobabilities"], "reconstructions")
        _image_grid([vae.generate(8, probabilities=True, seed=31).cpu().numpy(),
                     vae.generate(8, seed=31).cpu().numpy()],
                    ["Prior-draw\nprobabilities", "Sampled\nbinary images"], "prior-samples")

        locations = vae.encode(test)[0].cpu().numpy()
        rows, columns = np.random.default_rng(24).integers(1, 7, size=(2, len(test)))
        figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True, sharey=True)
        for ax, factor, title in zip(axes, [rows, columns], ["Color: true row", "Color: true column"]):
            points = ax.scatter(locations[:, 0], locations[:, 1], c=factor, cmap="viridis", vmin=.5, vmax=6.5,
                                s=14, alpha=.8, edgecolors="none")
            ax.set(xlabel=r"Encoder mean $\mu_1(x)$", title=title)
        axes[0].set_ylabel(r"Encoder mean $\mu_2(x)$")
        figure.colorbar(points, ax=axes, ticks=range(1, 7), label="Synthetic bar position", fraction=.03, pad=.04)
        _save(figure, "latent-space")

    seeded_pyro_model = PYRO_MODEL_CODE.replace("encoder =", "torch.manual_seed(7)\n\nencoder =", 1)
    seeded_pyro_reconstruct = "torch.manual_seed(29)\n" + PYRO_RECONSTRUCT_CODE
    seeded_pyro_generate = "torch.manual_seed(31)\n" + PYRO_GENERATE_CODE
    steps = [
        {"id": "data", "title": "1. Generate the same synthetic training and test data",
         "smallprob": DATA_CODE, "pyro": DATA_CODE},
        {"id": "model", "title": "2. Specify the model and inference family",
         "smallprob": SMALLPROB_MODEL_CODE, "pyro": seeded_pyro_model},
        {"id": "fit", "title": "3. Fit with minibatches and evaluate held-out images",
         "smallprob": SMALLPROB_FIT_CODE, "pyro": PYRO_FIT_CODE},
        {"id": "reconstruct", "title": "4. Reconstruct images through the learned encoder",
         "smallprob": SMALLPROB_RECONSTRUCT_CODE, "pyro": seeded_pyro_reconstruct},
        {"id": "generate", "title": "5. Generate new images from the prior",
         "smallprob": SMALLPROB_GENERATE_CODE, "pyro": seeded_pyro_generate},
    ]
    header = "# Synthetic 8x8 crossing bars. Offline, CPU, independently trained VAE.\n"
    (DESTINATION / "smallprob_vae.py").write_text(header + "\n\n".join(step["smallprob"] for step in steps)
                                                + '\nprint("generated", tuple(generated.shape))\n', encoding="utf-8")
    (DESTINATION / "pyro_vae.py").write_text(header + "import torch\ntorch.set_num_threads(1)\n\n"
                                           + "\n\n".join(step["pyro"] for step in steps)
                                           + '\nprint({"test_negative_elbo": test_negative_elbo})\n'
                                           + 'print("generated", tuple(generated.shape))\n', encoding="utf-8")
    before = metrics["before"]["negative_elbo"]
    after = metrics["after"]["negative_elbo"]
    result = {
        "id": "variational-autoencoder", "title": "A VAE in a few calls",
        "description": "Learn a nonlinear latent representation, reconstruct held-out images, and generate new ones. "
                       "A focused Torch API handles the Bernoulli model and diagonal Normal encoder.",
        "steps": steps,
        "metrics": [
            {"label": "Held-out loss before training", "value": f"{before:.2f} nats", "detail": "Negative ELBO per image; lower is better."},
            {"label": "Smallprob after 200 epochs", "value": f"{after:.2f} nats", "detail": "128 posterior draws per image; exact Normal KL."},
            {"label": "Pyro after 200 epochs", "value": f"{pyro_metrics['after']:.2f} nats", "detail": "An independent fit with the same data, architecture and analytic-KL ELBO."},
        ],
        "figures": [
            {"path": "assets/vae/model-and-guide.svg", "png": "assets/vae/model-and-guide.png",
             "alt": "Generative model and amortized encoder with Bernoulli observations and diagonal Normal guide",
             "caption": "The decoder defines the likelihood. The encoder predicts a separate approximate latent posterior for every image."},
            {"path": "assets/vae/learning-curve.svg", "png": "assets/vae/learning-curve.png",
             "alt": "Training and held-out negative ELBO decrease over 200 epochs",
             "caption": "Measured CPU runs on 1,024 training and 256 independent held-out synthetic images. Curves use 8 Monte Carlo draws; headline metrics use 128. These are finite ELBO estimates, not exact log evidence."},
            {"path": "assets/vae/reconstructions.svg", "png": "assets/vae/reconstructions.png",
             "alt": "Held-out binary crossing bars and their reconstructed pixel probabilities",
             "caption": "First eight held-out images, without selection. Reconstructions average 128 decoder-probability draws from q(z | x); grey pixels show uncertainty, not higher-resolution data."},
            {"path": "assets/vae/prior-samples.svg", "png": "assets/vae/prior-samples.png",
             "alt": "Decoder pixel probabilities from prior latent draws and actual Bernoulli image samples",
             "caption": "First eight prior draws with seed 31. The upper row shows probabilities; the lower row samples binary pixels at the same latent codes. Imperfect bars remain visible."},
            {"path": "assets/vae/latent-space.svg", "png": "assets/vae/latent-space.png",
             "alt": "Two-dimensional encoder means colored by the synthetic true row and column of each crossing bar",
             "caption": "The same 256 held-out latent means, colored by two known simulation factors. Coordinates are learned and are not guaranteed to identify independent causal factors."},
        ],
        "equations": [
            "z ~ Normal(0, I₂)",
            "xⱼ | z ~ Bernoulli(sigmoid(gθ(z))ⱼ),  j = 1, …, 64",
            "qφ(z | x) = Normal(μφ(x), diag(σφ(x)²))",
            "ELBO(x) = E_q[log pθ(x | z)] − KL[qφ(z | x) || Normal(0, I)]",
        ],
        "scope": [
            "This is smallprob.neural.VAE, a separate focused API. The @model syntax still describes finite scalar graphs and does not trace arbitrary neural networks.",
            "Smallprob trains this VAE directly in PyTorch and does not require Pyro. The built-in encoder and decoder each have one SiLU hidden layer; the guide is a diagonal Normal per image.",
            "Data are synthetic 8×8 crossing bars with 1% independent pixel flips. The original Pyro tutorial uses MNIST; this example avoids network downloads and torchvision.",
            "The fit learns point estimates of encoder and decoder weights. It performs approximate amortized inference for image-specific z; it is not a Bayesian posterior over neural weights.",
            "Features must be binary and have shape (N, input_dim). Outputs are Torch tensors. CPU is the default; available CUDA or MPS must be requested explicitly. The figure run is CPU, not a GPU benchmark.",
            "Both implementations use analytic Normal KL and sampled Bernoulli reconstruction. Pyro uses TraceMeanField_ELBO; the published tutorial uses Trace_ELBO. Independent optimizer runs need not give identical parameters or images.",
            "The native loss and parameter gradients are checked against Pyro at matched weights and random draws. Held-out loss improvement is demonstrated; neither check establishes exact posterior accuracy.",
        ],
        "sources": [
            {"label": "Pyro: Variational Autoencoders", "url": "https://pyro.ai/examples/vae.html"},
            {"label": "Auto-Encoding Variational Bayes", "url": "https://arxiv.org/abs/1312.6114"},
        ],
        "downloads": [
            {"label": "Smallprob VAE script", "path": "assets/vae/smallprob_vae.py"},
            {"label": "Pyro VAE script", "path": "assets/vae/pyro_vae.py"},
        ],
        "validation": {"synthetic": True, "device": "cpu", "dtype": "float32", "epochs": 200,
                       "train_images": len(train), "test_images": len(test), "batch_size": 128,
                       "initialization_seed": 7, "training_seed": 11, "evaluation_seed": 21,
                       "evaluation_particles": 128, "smallprob": metrics, "pyro": pyro_metrics,
                       "torch_version": torch.__version__, "pyro_version": pyro.__version__,
                       "smallprob_history": history, "pyro_history": pyro_history},
    }
    (DESTINATION / "example.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"assets": str(DESTINATION), "smallprob": metrics, "pyro": pyro_metrics}, indent=2))
    return result


if __name__ == "__main__":
    build()
