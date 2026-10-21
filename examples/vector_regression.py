"""Regression with one Normal declaration for every observation.

Run: python -m examples.vector_regression
"""
from smallprob import HalfNormal, Normal, model


@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)


def main():
    x = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    m = regression(x)
    data = m.simulate(seed=7, fixed={"intercept": 1.5, "slope": -0.8, "sigma": 0.5})
    p = m.infer(y=data["y"][0], method="vi", seed=7)
    print(f"Generated y shape: {data['y'].shape} (draws, observations)")
    for name in ("intercept", "slope", "sigma"):
        low, high = p.interval(name)
        print(f"{name}: mean={p.mean(name):.3f}, 95% guide interval=[{low:.3f}, {high:.3f}]")
    print("VI intervals describe the learned approximation.")


if __name__ == "__main__":
    main()
