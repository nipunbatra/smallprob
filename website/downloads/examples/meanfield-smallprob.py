"""Unknown-noise regression with a diagonal Normal guide. Synthetic data."""
import numpy as np
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
