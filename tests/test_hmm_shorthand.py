"""The HMM constructor is graph syntax, not a second inference engine."""

from itertools import product

import numpy as np
import pytest

from smallprob import Categorical, HMM, ModelError, Normal, StudentT, UnsupportedModelError, draw, model
from smallprob.core import Expr


INITIAL = [.6, .4]
TRANSITION = [[.85, .15], [.25, .75]]
MEANS = [-1., 1.2]


@model
def explicit_hmm(steps, initial, transition, means, sd):
    z = [draw(Categorical(initial))]
    for time in range(1, steps):
        z.append(draw(Categorical(transition[z[time - 1]])))
    y = [draw(Normal(means[z[time]], sd)) for time in range(steps)]


@model
def explicit_state_sd_hmm(steps, initial, transition, means, sd):
    z = [draw(Categorical(initial))]
    for time in range(1, steps):
        z.append(draw(Categorical(transition[z[time - 1]])))
    y = [draw(Normal(means[z[time]], sd[z[time]])) for time in range(steps)]


def make_hmm(**changes):
    arguments = dict(initial=INITIAL, transition=TRANSITION, emissions=Normal(MEANS, .7), steps=3)
    return HMM(**(arguments | changes))


def expression_signature(value):
    if isinstance(value, Expr):
        return (value.op, tuple(expression_signature(argument) for argument in value.args))
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(expression_signature(item) for item in value)
    return value


def graph_signature(graph):
    return [(node.name, type(node.distribution).__name__,
             {key: expression_signature(value) for key, value in vars(node.distribution).items()})
            for node in graph.nodes]


@pytest.mark.parametrize("steps", [1, 3, 8])
@pytest.mark.parametrize("sd", [.7, [.8, .6]])
def test_constructor_graph_and_seeded_results_match_explicit_loop(steps, sd):
    concise = make_hmm(steps=steps, emissions=Normal(MEANS, sd))
    factory = explicit_hmm if np.ndim(sd) == 0 else explicit_state_sd_hmm
    explicit = factory(steps, INITIAL, TRANSITION, MEANS, sd)
    assert graph_signature(concise.graph) == graph_signature(explicit.graph)
    assert concise.graph.groups == explicit.graph.groups
    assert concise.graph.series == explicit.graph.series == {"z", "y"}
    simulated = concise.simulate(draws=7, seed=11)
    reference = explicit.simulate(draws=7, seed=11)
    for name in ("z", "y"):
        np.testing.assert_array_equal(simulated[name], reference[name])
    observed = simulated["y"][0].tolist()
    observed[-1] = None
    posterior, expected = concise.infer(y=observed), explicit.infer(y=observed)
    assert posterior.method == "exact-hmm"
    assert posterior.log_evidence == expected.log_evidence
    np.testing.assert_array_equal(posterior.probabilities("z"), expected.probabilities("z"))
    for name in ("z", "y"):
        np.testing.assert_array_equal(posterior.mean(name), expected.mean(name))
        np.testing.assert_array_equal(posterior.sd(name), expected.sd(name))
        np.testing.assert_array_equal(posterior.sample(100, seed=16)[name], expected.sample(100, seed=16)[name])


def test_constructor_smoothing_moments_and_joint_paths_match_exhaustive_reference():
    observed, scales = [-.2, None, .5], [.8, .6]
    paths = list(product(range(2), repeat=3))
    weights = []
    for path in paths:
        weight = INITIAL[path[0]]
        for time in range(1, 3):
            weight *= TRANSITION[path[time - 1]][path[time]]
        for time, value in enumerate(observed):
            if value is not None:
                state = path[time]
                weight *= np.exp(-.5 * ((value - MEANS[state]) / scales[state]) ** 2) / (scales[state] * np.sqrt(2 * np.pi))
        weights.append(weight)
    evidence = sum(weights)
    probabilities = np.array(weights) / evidence
    marginals = np.array([[sum(probability for path, probability in zip(paths, probabilities) if path[t] == state)
                           for state in range(2)] for t in range(3)])
    posterior = make_hmm(emissions=Normal(MEANS, scales)).infer(y=observed)
    assert posterior.log_evidence == pytest.approx(np.log(evidence), abs=1e-12)
    np.testing.assert_allclose(posterior.probabilities("z"), marginals, atol=1e-12)
    mean = marginals[1] @ MEANS
    variance = marginals[1] @ (np.square(MEANS) + np.square(scales)) - mean ** 2
    assert posterior.mean("y")[1] == pytest.approx(mean)
    assert posterior.sd("y")[1] == pytest.approx(np.sqrt(variance))
    samples = posterior.sample(10000, seed=81)
    frequencies = [np.mean(np.all(samples["z"] == path, axis=1)) for path in paths]
    np.testing.assert_allclose(frequencies, probabilities, atol=.018)


def test_named_states_are_metadata_and_do_not_change_numeric_categories():
    named = make_hmm(states=["low", "high"])
    unnamed = make_hmm()
    assert named.hmm_state_labels == ("low", "high")
    assert unnamed.hmm_state_labels is None
    assert graph_signature(named.graph) == graph_signature(unnamed.graph)
    np.testing.assert_array_equal(named.simulate(5, seed=7)["z"], unnamed.simulate(5, seed=7)["z"])
    assert named.simulate(seed=7)["z"].dtype.kind == "i"


def test_one_state_one_step_and_shared_mean():
    one = HMM(initial=[1], transition=[[1]], emissions=Normal(2., .4), steps=1, states=["only"])
    posterior = one.infer(y=[2.2])
    np.testing.assert_array_equal(posterior.probabilities("z"), [[1.]])
    assert posterior.sample(4, seed=9)["z"].shape == (4, 1)
    shared = make_hmm(emissions=Normal(2., [.4, .4])).infer()
    np.testing.assert_allclose(shared.mean("y"), 2.)
    np.testing.assert_allclose(shared.sd("y"), .4)


def test_input_mutation_cannot_change_bound_model():
    initial = np.array(INITIAL)
    transition = np.array(TRANSITION)
    means = np.array(MEANS)
    labels = ["low", "high"]
    concise = HMM(initial=initial, transition=transition, emissions=Normal(means, .7), steps=3, states=labels)
    before = concise.simulate(4, seed=7)
    initial[:] = [1, 0]
    transition[:] = [[0, 1], [1, 0]]
    means[:] = 100
    labels[:] = ["changed", "labels"]
    after = concise.simulate(4, seed=7)
    for name in ("z", "y"):
        np.testing.assert_array_equal(before[name], after[name])
    assert concise.hmm_state_labels == ("low", "high")


@pytest.mark.parametrize("steps", [True, np.bool_(True), 0, -1, 2., "3", None])
def test_invalid_steps(steps):
    with pytest.raises(ModelError, match="positive integer"):
        make_hmm(steps=steps)


@pytest.mark.parametrize("initial", [[], [.3, .4], [-.1, 1.1], [True, False], [np.nan, 0], [[.6, .4]], [".6", ".4"]])
def test_invalid_initial_probabilities(initial):
    with pytest.raises(ModelError):
        make_hmm(initial=initial)


@pytest.mark.parametrize("transition", [[], [.85, .15], [[1, 0]], [[.85], [.15, .85]],
                                          [[.85, .15, 0], [.25, .75, 0]],
                                          [[.8, .1], [.25, .75]], [[1.1, -.1], [.25, .75]],
                                          [[np.nan, 0], [.25, .75]]])
def test_invalid_transition_probabilities(transition):
    with pytest.raises(ModelError):
        make_hmm(transition=transition)


def test_invalid_unreachable_transition_row_is_rejected_even_at_one_step():
    with pytest.raises(ModelError, match="row 1"):
        make_hmm(initial=[1, 0], transition=[[1, 0], [.1, .1]], steps=1)


@pytest.mark.parametrize("emissions", [Normal([0], 1), Normal([0, 1, 2], 1), Normal([0, 1], [.5]),
                                         StudentT(4, [0, 1], .5), [Normal(0, 1), Normal(1, 1)]])
def test_unsupported_emissions_and_inconsistent_state_shapes(emissions):
    with pytest.raises(ModelError):
        make_hmm(emissions=emissions)


def test_random_emission_parameters_are_not_treated_as_fixed():
    with pytest.raises(UnsupportedModelError, match="fixed parameters"):
        make_hmm(emissions=Normal(Expr("ref", ("mu",)), .5))


@pytest.mark.parametrize("states", ["low", 2, ["low"], ["low", "low"], ["low", " "], [0, 1]])
def test_state_labels_must_match_support_and_be_unique(states):
    with pytest.raises(ModelError):
        make_hmm(states=states)


def test_torch_tables_preserve_backend_and_allow_explicit_cpu_inference():
    torch = pytest.importorskip("torch")
    initial = torch.tensor(INITIAL, dtype=torch.float64, requires_grad=True)
    transition = torch.tensor(TRANSITION, dtype=torch.float64, requires_grad=True)
    concise = make_hmm(initial=initial, transition=transition)
    assert concise.backend.name == "torch"
    assert concise.backend.dtype == "float64"
    samples = concise.simulate(3, seed=7)
    assert isinstance(samples["y"], torch.Tensor)
    assert samples["z"].dtype == torch.int64
    with pytest.raises(UnsupportedModelError, match="NumPy/CPU"):
        concise.infer(y=[-.8, None, .8])
    posterior = concise.infer(y=[-.8, None, .8], backend="numpy")
    assert posterior.method == "exact-hmm"
    assert isinstance(posterior.probabilities("z"), np.ndarray)
    assert make_hmm(initial=initial, transition=transition, backend="numpy").backend.name == "numpy"


def test_explicit_torch_backend_and_dtype_work_with_python_tables():
    torch = pytest.importorskip("torch")
    concise = make_hmm(backend="torch", dtype="float32", device="cpu")
    samples = concise.simulate(3, seed=7)
    assert samples["y"].dtype == torch.float32
    assert samples["y"].device.type == "cpu"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_native_float32_probability_tables_accept_only_roundoff(backend):
    initial = np.array(INITIAL, dtype=np.float32)
    transition = np.array(TRANSITION, dtype=np.float32)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        initial, transition = torch.from_numpy(initial), torch.from_numpy(transition)
    concise = make_hmm(initial=initial, transition=transition)
    assert concise.backend.name == backend
    generated = concise.simulate(5, seed=12)
    assert generated["z"].shape == (5, 3)
    posterior = concise.infer(y=[-.8, None, .8], backend="numpy")
    explicit = explicit_hmm(3, initial, transition, MEANS, .7).infer(y=[-.8, None, .8], backend="numpy")
    np.testing.assert_allclose(posterior.probabilities("z"), explicit.probabilities("z"), atol=1e-14)
    assert posterior.log_evidence == pytest.approx(explicit.log_evidence, abs=1e-14)
    # A visibly unnormalised table is still rejected, not repaired.
    with pytest.raises(ModelError, match="sum to 1"):
        make_hmm(initial=np.array([.6, .39], dtype=np.float32))
