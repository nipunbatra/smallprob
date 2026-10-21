"""Generate presentation-only mathematical diagrams without Graphviz."""

from pathlib import Path

from smallprob import Categorical, HalfNormal, Normal, StudentT, model
from smallprob.hmm import HMM


@model
def normal_observations(n=8):
    mu = Normal(0, 3)
    y = [Normal(mu, 1) for _ in range(n)]


@model
def regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(intercept + slope * x, sigma)


@model
def robust_regression(x):
    intercept = Normal(0, 5)
    slope = Normal(0, 5)
    sigma = HalfNormal(1)
    y = StudentT(df=4, loc=intercept + slope * x, scale=sigma)


@model
def hidden_markov_model(n=4):
    initial = [.5, .5]
    transition = [[.9, .1], [.15, .85]]
    means = [-1.5, 1.5]
    z, y = [], []
    for t in range(n):
        z.append(Categorical(initial if t == 0 else transition[z[-1]]))
        y.append(Normal(means[z[-1]], .45))


def declarative_hmm():
    return HMM(
        states=["quiet", "active"],
        initial=[.6, .4],
        transition=[[.9, .1], [.2, .8]],
        emissions=Normal([-1., 2.], [.3, .8]),
        steps=30,
    )


def main():
    output = Path("outputs/diagrams")
    for bound in (normal_observations(), regression([-2., -1., 0., 1., 2.]), robust_regression([-2., -1., 0., 1., 2.]), hidden_markov_model()):
        labels = {key: value for key, value in {"intercept": "b", "slope": "m"}.items() if key in bound.graph.groups}
        diagram = bound.diagram(labels=labels, observed=["y"])
        for suffix in ("svg", "png"):
            diagram.save(output / f"{bound.name}.{suffix}")
        print(bound.math(labels=labels))
        print(output / f"{bound.name}.svg")
    diagram = declarative_hmm().diagram(observed="y")
    for suffix in ("svg", "png"):
        diagram.save(output / f"declarative_hmm.{suffix}")
    print(output / "declarative_hmm.svg")


if __name__ == "__main__":
    main()
