"""Paired bootstrap module + compare() paired-path wiring."""

import numpy as np
import pytest
from scipy.stats import norm

from strands_evals.optimization.bayesbench.paired import paired_bootstrap
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def make_space():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b"],
            "benchmark": ["bench_a", "bench_b"],
        }
    )


# -- pure bootstrap ---------------------------------------------------------


def test_bootstrap_recovers_mean_difference():
    rng = np.random.default_rng(0)
    a = (rng.random(200) < 0.7).astype(float)
    b = (rng.random(200) < 0.4).astype(float)
    r = paired_bootstrap(a, b, rng=np.random.default_rng(1))
    assert r["mean_difference"] == pytest.approx(a.mean() - b.mean(), abs=1e-12)
    lo, hi = r["credible_interval"]
    assert lo < r["mean_difference"] < hi
    assert 0.0 <= r["probability_a_better"] <= 1.0


def test_bootstrap_identical_vectors_have_zero_difference():
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    r = paired_bootstrap(x, x, rng=np.random.default_rng(2))
    assert r["mean_difference"] == 0.0
    assert r["std_difference"] == 0.0
    assert r["credible_interval"] == (0.0, 0.0)


def test_bootstrap_shared_tasks_narrow_vs_independent():
    """Perfectly correlated outcomes: paired difference has near-zero variance
    even when each cell's own pass rate is uncertain — the whole point."""
    rng = np.random.default_rng(3)
    d = rng.normal(0, 1, 300)
    a = (norm.ppf(0.6) - d > 0).astype(float)
    b = (norm.ppf(0.5) - d > 0).astype(float)  # same tasks, lower skill
    r = paired_bootstrap(a, b, rng=np.random.default_rng(4))
    lo, hi = r["credible_interval"]
    # Paired CI width is far below the independent binomial width
    # (~2*1.645*sqrt(2*0.25/300) ≈ 0.134).
    assert hi - lo < 0.07


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


# -- compare() wiring -------------------------------------------------------


def test_compare_uses_paired_path_when_outcomes_present():
    space = make_space()
    a = {"model": "model_a", "benchmark": "bench_a"}
    b = {"model": "model_b", "benchmark": "bench_a"}
    rng = np.random.default_rng(5)
    oa = (rng.random(100) < 0.65).astype(float)
    ob = (rng.random(100) < 0.5).astype(float)
    s = InformationGainSampler(space, tau_main=0.3)
    s.observe(a, float(oa.mean()), m=100, task_outcomes=oa)
    s.observe(b, float(ob.mean()), m=100, task_outcomes=ob)
    c = s.compare(a, b, rng=np.random.default_rng(6))
    assert c.method == "paired_bootstrap"
    assert c.shared_benchmark is True


def test_compare_falls_back_to_posterior_without_outcomes():
    space = make_space()
    a = {"model": "model_a", "benchmark": "bench_a"}
    b = {"model": "model_b", "benchmark": "bench_a"}
    s = InformationGainSampler(space, tau_main=0.3)
    s.observe(a, 0.65, m=100)
    s.observe(b, 0.50, m=100)
    c = s.compare(a, b, rng=np.random.default_rng(7))
    assert c.method == "posterior"
    assert c.shared_benchmark is True
    assert "conservative" in c.caveat


def test_compare_different_benchmarks_never_paired():
    space = make_space()
    a = {"model": "model_a", "benchmark": "bench_a"}
    b = {"model": "model_b", "benchmark": "bench_b"}
    rng = np.random.default_rng(8)
    oa = (rng.random(100) < 0.65).astype(float)
    ob = (rng.random(100) < 0.5).astype(float)
    s = InformationGainSampler(space, tau_main=0.3)
    s.observe(a, float(oa.mean()), m=100, task_outcomes=oa)
    s.observe(b, float(ob.mean()), m=100, task_outcomes=ob)
    c = s.compare(a, b, rng=np.random.default_rng(9))
    assert c.method == "posterior"  # not shared -> no pairing
    assert c.shared_benchmark is False
