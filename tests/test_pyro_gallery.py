"""Probability-law and reference checks for the translated tutorial models."""

import numpy as np
import pytest

from examples.pyro_gallery import (
    SCHOOL_EFFECTS, SCHOOL_SE, eight_schools, interaction_data,
    interaction_known_noise, interaction_reference, interaction_regression,
    pyro_log_joint_checks, quadrature_check, schools_reference, weighing_scale,
)


def test_scale_reproduces_tutorial_exact_posterior():
    posterior = weighing_scale().infer(measurement=9.5)
    assert posterior.mean("weight") == pytest.approx(9.14, abs=1e-12)
    assert posterior.sd("weight") == pytest.approx(0.6, abs=1e-12)


def test_gallery_log_joints_equal_independent_pyro_models():
    pytest.importorskip("pyro")
    checks = pyro_log_joint_checks()
    assert len(checks) == 3
    assert all(check["passed"] for check in checks), checks


def test_interaction_exact_matches_independent_conjugate_reference():
    group, x, y, truth = interaction_data()
    posterior = interaction_known_noise(group, x).infer(y=y)
    reference = interaction_reference(group, x, y, noise=truth["sigma"])
    names = ["intercept", "group_effect", "slope", "interaction"]
    np.testing.assert_allclose([posterior.mean(n) for n in names], reference["mean"], atol=1e-11)
    np.testing.assert_allclose([posterior.sd(n) for n in names], reference["sd"], atol=1e-11)


def test_quadrature_reference_moments_are_resolved():
    group, x, y, _ = interaction_data()
    for coarse, fine in [
        (interaction_reference(group, x, y), interaction_reference(group, x, y, points=16385)),
        (schools_reference(), schools_reference(points=16385)),
    ]:
        for name in ("mean", "sd"):
            np.testing.assert_allclose(coarse[name], fine[name], atol=1e-6)


def test_reference_range_expansion_resolves_moments_and_tail_mass():
    group, x, y, _ = interaction_data()
    cases = [
        (interaction_reference(group, x, y), interaction_reference(group, x, y, points=16385),
         interaction_reference(group, x, y, log_bounds=(-16, 4), points=10924)),
        (schools_reference(), schools_reference(points=16385),
         schools_reference(log_bounds=(-20, 5), points=10241)),
    ]
    for coarse, refined, expanded in cases:
        check = quadrature_check(coarse, refined, expanded)
        assert check["passed"], check
    assert "effect_mean_max_absolute_change" in check["range_expansion"]
    assert "effect_sd_max_absolute_change" in check["range_expansion"]


def test_quadrature_range_check_detects_truncation():
    narrow = schools_reference(log_bounds=(0, 1))
    refined = schools_reference(log_bounds=(0, 1), points=16385)
    expanded = schools_reference()
    check = quadrature_check(narrow, refined, expanded)
    assert not check["passed"]
    assert check["estimated_posterior_mass_outside_original_range"] > 0.1


def test_school_reference_without_observations_recovers_hyperpriors():
    reference = schools_reference(np.array([]), np.array([]))
    np.testing.assert_allclose(reference["mean"], [0, 5 * np.sqrt(2 / np.pi)], atol=2e-7)
    np.testing.assert_allclose(reference["sd"], [10, 5 * np.sqrt(1 - 2 / np.pi)], atol=2e-7)


def test_models_generate_reproducibly_with_complete_shapes():
    group, x, _, _ = interaction_data()
    for bound, shapes in [
        (weighing_scale(), {"weight": (4,), "measurement": (4,)}),
        (interaction_regression(group, x), {"y": (4, 40), "sigma": (4,)}),
        (eight_schools(), {"eta": (4, 8), "y": (4, 8), "mu": (4,), "tau": (4,)}),
    ]:
        left, right = bound.simulate(draws=4, seed=9), bound.simulate(draws=4, seed=9)
        for name, shape in shapes.items():
            assert left[name].shape == shape
            np.testing.assert_array_equal(left[name], right[name])
            assert np.isfinite(left[name]).all()


def test_school_data_and_conditional_law_match_published_values():
    assert SCHOOL_EFFECTS.tolist() == [28, 8, -3, 7, -1, 1, 18, 12]
    assert SCHOOL_SE.tolist() == [15, 10, 16, 11, 9, 11, 10, 18]
    bound = eight_schools()
    samples = bound.simulate(draws=3, seed=3, fixed={"mu": 4, "tau": 2, "eta": np.zeros(8)})
    np.testing.assert_array_equal(samples["mu"], np.full(3, 4))
    assert samples["y"].shape == (3, 8)


@pytest.mark.parametrize("case", ["interaction", "schools"])
def test_gallery_vi_executes_with_finite_joint_samples(case):
    pytest.importorskip("pyro")
    if case == "interaction":
        group, x, y, _ = interaction_data()
        bound = interaction_regression(group, x)
    else:
        bound, y = eight_schools(), SCHOOL_EFFECTS
    posterior = bound.infer(y=y, method="vi", steps=20, seed=11)
    assert posterior.approximate
    draws = posterior.sample(draws=10, seed=7)
    for values in draws.values():
        assert np.isfinite(values).all()
    if case == "schools":
        effects = draws["mu"][:, None] + draws["tau"][:, None] * draws["eta"]
        assert effects.shape == (10, 8)
        assert np.isfinite(effects).all()
