"""observe() is atomic: a rejected result must not corrupt the sampler.

Regression for the review finding that observe() appended the observation
before the refit validated it, so one NaN score made every later refit fail.
"""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def make_sampler():
    space = ExperimentSpace(
        factors={
            "model": ["model_a", "model_b"],
            "benchmark": ["bench_a", "bench_b"],
        }
    )
    return space, InformationGainSampler(space, tau_main=0.2, seed=0)


def snapshot(sampler):
    return (
        len(sampler.observations),
        sampler.posterior.mean.copy(),
        sampler._tracker.n_holdout,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": float("nan"), "m": 50},  # NaN score, variance derived from m
        {"value": float("nan"), "variance": 1e-3},  # NaN score, explicit variance
        {"value": float("inf"), "m": 50},
        {"value": 0.5, "variance": float("nan")},
        {"value": 0.5, "variance": float("inf")},
    ],
)
def test_observe_rejects_non_finite_input_and_preserves_state(kwargs):
    space, sampler = make_sampler()
    good = space.config(space.cells()[0])
    sampler.observe(good, 0.6, m=50)
    n_obs, mean, n_holdout = snapshot(sampler)

    with pytest.raises(ValueError, match="finite"):
        sampler.observe(space.config(space.cells()[1]), **kwargs)

    # Nothing was committed: same observations, posterior, and diagnostics.
    assert len(sampler.observations) == n_obs
    assert np.array_equal(sampler.posterior.mean, mean)
    assert sampler._tracker.n_holdout == n_holdout


def test_sampler_still_works_after_rejected_observation():
    space, sampler = make_sampler()
    configs = [space.config(k) for k in space.cells()]
    sampler.observe(configs[0], 0.6, m=50)
    with pytest.raises(ValueError):
        sampler.observe(configs[1], float("nan"), m=50)

    # The failed observe must not poison later updates or queries.
    sampler.observe(configs[1], 0.4, m=50)
    assert len(sampler.observations) == 2
    assert sampler.suggest(n=2)
    preds = sampler.predict(configs)
    assert all(np.isfinite(p.mean) and np.isfinite(p.std) for p in preds)


def test_zero_variance_is_still_floored_not_rejected():
    space, sampler = make_sampler()
    sampler.observe(space.config(space.cells()[0]), 0.5, variance=0.0)
    assert sampler.observations[0].variance > 0
