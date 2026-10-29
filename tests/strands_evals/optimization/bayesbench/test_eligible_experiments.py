"""Executor constraints and interaction estimands must survive planning."""

import warnings

import numpy as np
import pytest

from strands_evals.optimization.bayesbench import ExperimentSpace
from strands_evals.optimization.bayesbench.posterior import Posterior
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.seed import covers_all_levels


def grid():
    return ExperimentSpace({"feature": ["off", "on"], "model": ["small", "large"], "benchmark": ["short", "long"]})


def without_seed(space, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return InformationGainSampler(space, n_seed=0, **kwargs)


def test_mask_is_applied_to_seed_acquisition_and_observation():
    space = grid()
    eligible = [space.config(k) for k in space.cells() if not (k[1] == "small" and k[2] == "long")]
    sampler = InformationGainSampler(space, eligible_configs=eligible, seed=42)
    allowed = {space.key(c) for c in eligible}
    assert set(sampler._seed_plan) <= allowed
    assert covers_all_levels(space, sampler._seed_plan, candidates=sampler.eligible_keys)
    for c in sampler.suggest(n=20):
        sampler.observe(c, 0.5, m=100)
    again = sampler.suggest(n=20, allow_repeats=True)
    assert len(again) == len(allowed)
    assert {space.key(c) for c in again} == allowed
    with pytest.raises(ValueError, match="ineligible"):
        sampler.observe({"feature": "off", "model": "small", "benchmark": "long"}, 0.5, m=100)


def test_mask_order_and_duplicates_do_not_change_the_seed():
    space = grid()
    configs = [space.config(k) for k in space.cells()][1:]
    a = InformationGainSampler(space, eligible_configs=configs)
    b = InformationGainSampler(space, eligible_configs=list(reversed(configs)) + configs[:1])
    assert a.suggest(6) == b.suggest(6)
    with pytest.raises(ValueError, match="eligible"):
        InformationGainSampler(space, eligible_configs=[])


def test_candidate_noise_changes_the_pick_without_changing_the_posterior():
    space = grid()
    a = without_seed(space, candidate_variance=lambda c, p: 1e-4 if c["benchmark"] == "short" else 1)
    b = without_seed(space, candidate_variance=lambda c, p: 1e-4 if c["benchmark"] == "long" else 1)
    assert a.suggest(1)[0]["benchmark"] == "short"
    assert b.suggest(1)[0]["benchmark"] == "long"
    np.testing.assert_array_equal(a.posterior.cov, b.posterior.cov)
    bad = without_seed(space, candidate_variance=lambda c, p: float("nan"))
    with pytest.raises(ValueError, match="variance"):
        bad.suggest(1)
    grouped = without_seed(space, candidate_variance=lambda c, p: 2e-5)
    assert grouped.observation_variance(space.config(space.cells()[0]), 0.5) == 2e-5


def test_masked_contrast_matches_the_same_background_conditions():
    space = grid()
    # Only feature=on exists on the long benchmark for the small model.
    # Averaging each arm separately would leak the benchmark effect.
    eligible = [space.config(k) for k in space.cells() if k != ("off", "small", "long")]
    s = InformationGainSampler(space, eligible_configs=eligible)
    theta = np.zeros(s.encoder.p)
    theta[s.encoder.feature_names.index("benchmark=long")] = 0.3
    s.posterior = Posterior(theta, np.full(s.encoder.p, 1e-6))
    assert s.contrast("feature", "off", "on").mean == pytest.approx(0)


def test_big_feature_effect_is_not_evidence_of_an_interaction():
    s = InformationGainSampler(grid())
    theta = np.zeros(s.encoder.p)
    theta[s.encoder.feature_names.index("feature=on")] = 0.2
    s.posterior = Posterior(theta, np.full(s.encoder.p, 1e-6))
    assert s.contrast("feature", "off", "on").mean == pytest.approx(0.2)
    L = s.interaction_L("feature", "off", "on", "model", "small", "large")
    result = s.linear_contrast(L)
    assert result.mean == pytest.approx(0)
    assert result.ci90[0] < 0 < result.ci90[1]


def test_difference_of_effects_cancels_shared_uncertainty():
    s = InformationGainSampler(grid())
    theta = np.zeros(s.encoder.p)
    var = np.full(s.encoder.p, 0.2**2)
    interaction = s.encoder.feature_names.index("feature=on:model=large")
    theta[interaction], var[interaction] = 0.08, 0.03**2
    s.posterior = Posterior(theta, var)
    result = s.linear_contrast(s.interaction_L("feature", "off", "on", "model", "small", "large"))
    assert result.mean == pytest.approx(0.08)
    assert result.std == pytest.approx(0.03)  # the large shared main-effect uncertainty cancels


def test_interaction_requires_all_four_arms_on_common_support():
    space = grid()
    eligible = [space.config(k) for k in space.cells() if not (k[0] == "off" and k[1] == "large")]
    s = InformationGainSampler(space, eligible_configs=eligible)
    with pytest.raises(ValueError, match="four"):
        s.interaction_L("feature", "off", "on", "model", "small", "large")


def test_default_stopping_uses_only_eligible_cells():
    space = grid()
    configs = [space.config(k) for k in space.cells() if k[1] == "small"]
    s = InformationGainSampler(space, eligible_configs=configs)
    for _ in range(30):
        for c in configs:
            s.observe(c, 0.5, variance=1e-4)
    assert s.should_stop(delta=0.02)
    full_surface = s.encoder.design_matrix(space.cells())
    assert not s.should_stop(target=full_surface, delta=0.02)
