"""M3 gate tests:

- closed-form EIG matches 10k-sample Monte Carlo within 2% on 20 random states
- greedy batch: no duplicates
- greedy batch joint information >= top-K-independent on >= 95% of random states
  (ties count as a pass: greedy may legitimately coincide with top-K)
"""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench.acquisition import batch_information, eig, greedy_batch, rank_one_update


def random_state(rng, p=12, n_obs=6):
    """A random posterior covariance + candidate set."""
    prior_var = rng.uniform(0.02, 0.3, p) ** 2
    cov = np.diag(prior_var)
    # Condition on a few observations to make cov non-diagonal.
    for _ in range(n_obs):
        a = (rng.random(p) < 0.4).astype(float)
        a[0] = 1.0
        cov = rank_one_update(cov, a, rng.uniform(1e-3, 1e-2))
    cands = (rng.random((30, p)) < 0.4).astype(float)
    cands[:, 0] = 1.0
    obs_var = rng.uniform(1e-3, 1e-2, 30)
    return cov, cands, obs_var


def mc_eig(cov, a, obs_var, n_samples, rng):
    """Monte Carlo EIG: E_y[H(theta) - H(theta|y)] via predictive entropy.

    For linear-Gaussian, EIG = 1/2 E_y[log(v + s2) - log(s2_post)] reduces to
    the marginal-entropy difference of y: H(y) - H(y|theta)
    = 1/2 log(2 pi e (v + s2)) - 1/2 log(2 pi e s2), estimated from samples.
    """
    v = a @ cov @ a
    # Simulate y draws and estimate the marginal entropy of y empirically.
    y = rng.normal(0.0, np.sqrt(v + obs_var), n_samples)
    # Gaussian entropy estimate from the sample variance.
    h_marginal = 0.5 * np.log(2 * np.pi * np.e * y.var(ddof=1))
    h_conditional = 0.5 * np.log(2 * np.pi * np.e * obs_var)
    return h_marginal - h_conditional


@pytest.mark.parametrize("seed", range(20))
def test_eig_matches_monte_carlo_within_2pct(seed):
    rng = np.random.default_rng(seed)
    cov, cands, obs_var = random_state(rng)
    i = rng.integers(len(cands))
    closed = eig(cov, cands[i], obs_var[i])[0]
    mc = mc_eig(cov, cands[i], obs_var[i], 10_000, rng)
    assert closed == pytest.approx(mc, rel=0.02)


def test_greedy_batch_no_duplicates():
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        cov, cands, obs_var = random_state(rng)
        picks = greedy_batch(cov, cands, obs_var, k=8)
        assert len(picks) == len(set(picks)) == 8


def test_greedy_beats_top_k_independent_on_95pct():
    wins = 0
    n_states = 100
    for seed in range(n_states):
        rng = np.random.default_rng(1000 + seed)
        cov, cands, obs_var = random_state(rng)
        k = 8
        greedy = greedy_batch(cov, cands, obs_var, k=k)
        top_k = list(np.argsort(-eig(cov, cands, obs_var))[:k])
        info_greedy = batch_information(cov, cands[greedy], obs_var[greedy])
        info_topk = batch_information(cov, cands[top_k], obs_var[top_k])
        if info_greedy >= info_topk - 1e-12:  # ties pass
            wins += 1
    assert wins / n_states >= 0.95, f"greedy won only {wins}/{n_states}"


def test_rank_one_update_matches_batch_information():
    # Information from greedy picks, summed stepwise, equals the joint logdet.
    rng = np.random.default_rng(42)
    cov, cands, obs_var = random_state(rng)
    picks = greedy_batch(cov, cands, obs_var, k=5)
    stepwise = 0.0
    c = cov.copy()
    for i in picks:
        stepwise += eig(c, cands[i], obs_var[i])[0]
        c = rank_one_update(c, cands[i], obs_var[i])
    joint = batch_information(cov, cands[picks], obs_var[picks])
    assert stepwise == pytest.approx(joint, abs=1e-10)


def test_cost_weighting_changes_ranking():
    rng = np.random.default_rng(7)
    cov, cands, obs_var = random_state(rng)
    free = greedy_batch(cov, cands, obs_var, k=3)
    # Make the free picks 100x more expensive than everything else.
    cost = np.ones(len(cands))
    cost[free] = 100.0
    weighted = greedy_batch(cov, cands, obs_var, k=3, cost=cost, alpha=1.0)
    assert set(weighted) != set(free)
    # alpha=0 ignores cost entirely.
    ignore = greedy_batch(cov, cands, obs_var, k=3, cost=cost, alpha=0.0)
    assert ignore == free


def test_greedy_rejects_oversized_k():
    rng = np.random.default_rng(0)
    cov, cands, obs_var = random_state(rng)
    with pytest.raises(ValueError):
        greedy_batch(cov, cands, obs_var, k=len(cands) + 1)
