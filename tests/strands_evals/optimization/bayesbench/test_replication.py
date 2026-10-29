"""Byte-identical replication across accounts (e.g. 2 accounts, 4 seeded
replicas per config).

Guarantees under test:
  1. Same seed + same observations in the same order -> identical suggest()
     sequences AND bit-for-bit identical compare()/sample_posterior() output.
  2. suggest() is RNG-free: identical without any seed at all.
  3. Different seeds -> different stochastic output (the seed actually bites).
  4. Observation ORDER: posterior is analytically order-invariant but floats
     are not associative; documented behavior is asserted honestly here.
"""

import numpy as np

from strands_evals.optimization.bayesbench import ledger
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace

A = {"model": "model_a", "harness": "harness_c", "benchmark": "bench_a"}
B = {"model": "model_b", "harness": "harness_c", "benchmark": "bench_a"}


def make_space():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
            "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
            "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
        }
    )


def _observations(space, n, seed=0):
    """A fixed list of (config, value) to feed identically to both samplers."""
    rng = np.random.default_rng(seed)
    cells = space.cells()
    idx = rng.choice(len(cells), size=n, replace=False)
    return [(space.config(cells[i]), float(rng.uniform(0.2, 0.8))) for i in idx]


def _run(space, obs, seed, order=None):
    s = InformationGainSampler(space, tau_main=0.1, seed=seed)
    stream = obs if order is None else [obs[i] for i in order]
    for config, value in stream:
        s.observe(config, value, m=50)
    return s


def _surface(space, seed=0):
    """A fixed truth surface: cell -> pass rate in [0.05, 0.95]."""
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(0.1, 0.9)) for k in space.cells()}


def _full_campaign(space, surface, run_seed, budget=40, k=8):
    """A whole-run lifecycle at one run-level seed (the replication scheme: an
    account runs the full table with seed 42/43/44/45).

    The sampler's own RNG drives the observation noise too, so the ENTIRE run —
    every suggest() (which depends on the posterior built by prior observes),
    every observed value, and the final queries — is governed by the single
    constructor seed. Returns (batch_trajectory, final_compare, final_samples).
    """
    s = InformationGainSampler(space, tau_main=0.1, seed=run_seed)
    trajectory = []
    while len(s.observations) < budget:
        n = min(k, budget - len(s.observations))
        batch = s.suggest(n=n)
        trajectory.append([space.key(c) for c in batch])
        for config in batch:
            p = surface[space.key(config)]
            # noise from the sampler's own seeded rng -> whole run is pinned
            value = float(s._rng.binomial(50, p) / 50)
            s.observe(config, value, m=50)
    return trajectory, s.compare(A, B), s.sample_posterior([A, B], n=2000)


# -- 1. same seed + same order -> byte-identical everything ------------------


def test_same_seed_same_order_is_bit_identical():
    space = make_space()
    obs = _observations(space, 25)
    s1 = _run(space, obs, seed=7)
    s2 = _run(space, obs, seed=7)

    # suggest() sequence identical (it is RNG-free, but confirm end to end).
    assert s1.suggest(n=8) == s2.suggest(n=8)

    # posterior mean/cov bit-for-bit (same data, same order).
    assert np.array_equal(s1.posterior.mean, s2.posterior.mean)
    assert np.array_equal(s1.posterior.cov, s2.posterior.cov)

    # sample_posterior bit-for-bit (seed pins the draw).
    assert np.array_equal(
        s1.sample_posterior([A, B], n=2000),
        s2.sample_posterior([A, B], n=2000),
    )

    # compare() bit-for-bit on every stochastic field.
    c1, c2 = s1.compare(A, B), s2.compare(A, B)
    assert c1.probability_a_better == c2.probability_a_better
    assert c1.mean_difference == c2.mean_difference
    assert c1.credible_interval == c2.credible_interval


def test_paired_compare_bit_identical_under_seed():
    """The paired-bootstrap path also inherits the constructor seed."""
    space = make_space()
    rng = np.random.default_rng(1)
    oa = (rng.random(200) < 0.6).astype(float)
    ob = (rng.random(200) < 0.5).astype(float)

    def build(seed):
        s = InformationGainSampler(space, tau_main=0.1, seed=seed)
        s.observe(A, float(oa.mean()), m=200, task_outcomes=oa)
        s.observe(B, float(ob.mean()), m=200, task_outcomes=ob)
        return s.compare(A, B)

    c1, c2 = build(3), build(3)
    assert c1.method == "paired_bootstrap"
    assert c1.probability_a_better == c2.probability_a_better
    assert c1.credible_interval == c2.credible_interval


# -- 1b. WHOLE-RUN lifecycle at a run-level seed ------------------------------


def test_full_campaign_same_run_seed_is_bit_identical():
    """Two accounts run the full table at the same run seed -> the entire
    suggest->observe->suggest trajectory and the final queries are identical.
    This is the real replication contract, not isolated calls."""
    space = make_space()
    surface = _surface(space, seed=0)

    traj1, cmp1, samp1 = _full_campaign(space, surface, run_seed=42)
    traj2, cmp2, samp2 = _full_campaign(space, surface, run_seed=42)

    assert traj1 == traj2  # every batch, in order
    assert len(traj1) >= 4 and sum(len(b) for b in traj1) == 40  # real loop ran
    assert cmp1.probability_a_better == cmp2.probability_a_better
    assert cmp1.credible_interval == cmp2.credible_interval
    assert np.array_equal(samp1, samp2)


def test_full_campaign_different_run_seeds_diverge():
    """Replica seeds 42 vs 43 must produce different runs (between-replica
    variance is the point of the 4-replica scheme)."""
    space = make_space()
    surface = _surface(space, seed=0)
    _, cmp42, samp42 = _full_campaign(space, surface, run_seed=42)
    _, cmp43, samp43 = _full_campaign(space, surface, run_seed=43)
    # Different observation noise -> different posterior draws and comparison.
    assert not np.array_equal(samp42, samp43)
    assert cmp42.probability_a_better != cmp43.probability_a_better


def test_full_campaign_replicates_through_the_ledger(tmp_path):
    """End-to-end path an executor will use: run a campaign, write its ledger
    with space_record (seed recorded once, line 1), reload on the other
    account, and confirm the reloaded sampler reproduces the posterior and
    queries bit-for-bit.

    Note the rng-lifecycle detail: the LIVE sampler consumed rng draws for its
    observation noise during the run, so its rng is mid-stream; a freshly
    reloaded sampler starts rng at seed=44. Posterior (a function of the data
    only) is therefore bit-identical, and two independent reloads — the actual
    cross-account situation — give identical draws."""
    space = make_space()
    surface = _surface(space, seed=0)
    path = tmp_path / "campaign.jsonl"

    live = InformationGainSampler(space, tau_main=0.1, seed=44)
    ledger.append(path, ledger.space_record(live, {"tau_main": 0.1}))
    while len(live.observations) < 24:
        for config in live.suggest(n=8):
            key = live.space.key(config)
            value = float(live._rng.binomial(50, surface[key]) / 50)
            live.observe(config, value, m=50)
            ledger.append(path, {"type": "observation", "config": config, "value": value, "m": 50})

    reloaded_a = ledger.load_sampler(path)  # account A replays the ledger
    reloaded_b = ledger.load_sampler(path)  # account B replays the ledger

    assert reloaded_a.seed == 44  # seed recorded once/run
    # Posterior depends only on the (data, order) -> bit-identical to the live
    # run and between accounts.
    assert np.array_equal(reloaded_a.posterior.mean, live.posterior.mean)
    assert np.array_equal(reloaded_a.posterior.cov, live.posterior.cov)
    # Two fresh reloads (the real cross-account case) -> identical queries.
    assert np.array_equal(
        reloaded_a.sample_posterior([A, B], n=2000),
        reloaded_b.sample_posterior([A, B], n=2000),
    )
    assert reloaded_a.compare(A, B).probability_a_better == reloaded_b.compare(A, B).probability_a_better


# -- 2. suggest() is RNG-free ------------------------------------------------


def test_suggest_is_deterministic_without_any_seed():
    space = make_space()
    obs = _observations(space, 20)
    s1 = InformationGainSampler(space, tau_main=0.1)  # no seed
    s2 = InformationGainSampler(space, tau_main=0.1)
    for c, v in obs:
        s1.observe(c, v, m=50)
        s2.observe(c, v, m=50)
    # Selection uses no RNG, so it matches with no seed set at all.
    for _ in range(3):
        b1, b2 = s1.suggest(n=8), s2.suggest(n=8)
        assert b1 == b2


# -- 3. the seed actually bites ----------------------------------------------


def test_different_seeds_differ():
    space = make_space()
    obs = _observations(space, 20)
    s1 = _run(space, obs, seed=1)
    s2 = _run(space, obs, seed=2)
    # Same data/order -> same posterior, but different draws.
    assert np.array_equal(s1.posterior.mean, s2.posterior.mean)
    assert not np.array_equal(
        s1.sample_posterior([A, B], n=1000),
        s2.sample_posterior([A, B], n=1000),
    )


# -- 4. observation ORDER: honest documented behavior ------------------------


def test_observe_order_posterior_is_analytically_equal_but_not_bit_identical():
    """Posterior is order-invariant in exact arithmetic; in float it differs
    at ~1e-15 (non-associative sums). Replication therefore requires feeding
    observations in the SAME order."""
    space = make_space()
    obs = _observations(space, 25)
    forward = _run(space, obs, seed=5)
    rng = np.random.default_rng(99)
    perm = list(rng.permutation(len(obs)))
    reversed_ = _run(space, obs, seed=5, order=perm)

    # Analytically equal: close to machine epsilon.
    assert np.allclose(forward.posterior.mean, reversed_.posterior.mean, atol=1e-10)
    # But NOT guaranteed bit-identical when the order differs.
    bit_identical = np.array_equal(forward.posterior.mean, reversed_.posterior.mean)
    # We do not assert which; we assert the contract we rely on: SAME order
    # reproduces exactly (already covered above). If a future change happens
    # to make permuted order bit-identical too, that is a strict improvement
    # and this test still passes.
    assert np.allclose(forward.posterior.mean, reversed_.posterior.mean, atol=1e-9) or bit_identical
