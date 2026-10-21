"""All model definitions use ordinary Python, without array libraries."""

from smallprob import Categorical, HalfNormal, Normal, model


@model
def normal(n, noise=1.0):
    """An unknown population mean with a Normal prior and known noise."""
    mu = Normal(mean=0, sd=5)
    y = [Normal(mean=mu, sd=noise) for _ in range(n)]


@model
def regression(x, noise=0.5):
    """A Bayesian line with known observation noise."""
    intercept = Normal(mean=0, sd=5)
    slope = Normal(mean=0, sd=5)
    y = Normal(mean=intercept + slope * x, sd=noise)


@model
def regression_unknown_noise(x, prior_noise=1.0):
    """A Bayesian line whose observation standard deviation is also unknown."""
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(prior_noise)
    y = Normal(intercept + slope * x, sigma)


@model
def hmm(n, initial, transition, means, noise=0.5):
    """Hidden states with supplied transition and emission parameters."""
    if n < 1:
        raise ValueError("An HMM sequence needs at least one time step.")
    z, y = [], []
    for t in range(n):
        probs = initial if t == 0 else transition[z[-1]]
        z.append(Categorical(probs))
        y.append(Normal(mean=means[z[-1]], sd=noise))
