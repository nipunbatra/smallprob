# Synthetic 8x8 crossing bars. Offline, CPU, independently trained VAE.
import numpy as np

def synthetic_bars(n, seed):
    """Independent synthetic images: crossing bars with 1% random pixel flips."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 8, 8), dtype=np.float32)
    rows, columns = rng.integers(1, 7, size=(2, n))
    images[np.arange(n), rows, :] = 1
    images[np.arange(n), :, columns] = 1
    flips = rng.random(images.shape) < .01
    return np.where(flips, 1 - images, images).reshape(n, 64)

train = synthetic_bars(1024, seed=23)
test = synthetic_bars(256, seed=24)


from smallprob.neural import VAE

vae = VAE(input_dim=64, latent_dim=2, hidden_dim=128, seed=7)


history = vae.fit(
    train, validation_data=test,
    epochs=200, batch_size=128, learning_rate=0.003,
    validation_particles=8, seed=11,
)
metrics = vae.evaluate(test, particles=128, seed=21)
print(metrics)  # negative ELBO, reconstruction NLL, KL: nats per image


reconstructed = vae.reconstruct(test[:12], particles=128, seed=29)


pixel_probabilities = vae.generate(12, probabilities=True, seed=31)
generated = vae.generate(12, seed=31)  # actual binary samples

print("generated", tuple(generated.shape))
