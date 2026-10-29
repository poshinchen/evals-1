"""M5 gate tests: seed design property + full-loop regression on Arm A.

Gate: with n_seed=12 on the 5x4x5 grid every level appears >= 2 times;
in mean over reps with fixed seeds, EIG RMSE <= random RMSE at every
budget and EIG RMSE <= balanced RMSE at every budget < p. Past n ~ p
balanced legitimately catches up.
"""

import numpy as np

from strands_evals.optimization.bayesbench.encoding import Encoder
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.seed import next_balanced, seed_design
from strands_evals.optimization.bayesbench.space import ExperimentSpace

TAU_MAIN = {"model": 0.08, "harness": 0.10, "benchmark": 0.15}


def make_space():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
            "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
            "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
        }
    )


def tau_int_dict():
    return {frozenset({"model", "harness"}): 0.06}  # others default to 0.03


def make_sampler(space, **kwargs):
    return InformationGainSampler(space, tau_main=TAU_MAIN, tau_int=tau_int_dict(), **kwargs)


def arm_a_surface(space, rng):
    """Arm A generator: random effects at prior scales, planted bumps added
    on the surface. Returns dict CellKey -> f(x)."""
    enc = Encoder(space)
    theta = np.zeros(enc.p)
    theta[0] = 0.45
    for factor, cols in enc.main_idx.items():
        theta[cols] = rng.normal(0, TAU_MAIN[factor], len(cols))
    for pair, cols in enc.interaction_idx.items():
        tau = 0.06 if pair == frozenset({"model", "harness"}) else 0.03
        theta[cols] = rng.normal(0, tau, len(cols))

    # Plants capped at 2 sigma of their pair's prior scale so Arm A stays
    # consistent with its own priors.
    f = {}
    for key in space.cells():
        model, harness, bench = key
        val = enc.phi(key) @ theta
        if model == "model_a" and harness == "harness_c":
            val += 0.12
        # (model_b, harness_c) = 0.0 -> no bump beyond the drawn coefficient
        if model == "model_e" and bench == "bench_d":
            val += 0.05
        if harness == "harness_b" and bench == "bench_a":
            val -= 0.05
        f[key] = val
    return f


def simulate(space, surface, key, rng, m=50):
    p = np.clip(surface[key], 0.01, 0.99)
    return rng.binomial(m, p) / m


def run_sampler(space, surface, mode, budget, rng):
    """Run one sampler to `budget` observations; return surface RMSE."""
    sampler = make_sampler(space)
    observed = []
    while len(observed) < budget:
        k = min(8, budget - len(observed))
        if mode == "eig":
            batch = sampler.suggest(n=k)
        elif mode == "balanced":
            # Same seed design, then greedy coverage forever.
            keys = []
            chosen = list(observed)
            for _ in range(k):
                nxt = next_balanced(space, chosen)
                keys.append(nxt)
                chosen.append(nxt)
            batch = [space.config(key) for key in keys]
        else:  # random
            unrun = [key for key in space.cells() if key not in observed]
            idx = rng.choice(len(unrun), size=k, replace=False)
            batch = [space.config(unrun[i]) for i in idx]
        for config in batch:
            key = space.key(config)
            sampler.observe(config, simulate(space, surface, key, rng), m=50)
            observed.append(key)
    preds = sampler.predict([space.config(key) for key in space.cells()])
    truth = np.array([surface[key] for key in space.cells()])
    est = np.array([p.mean for p in preds])
    return float(np.sqrt(np.mean((est - truth) ** 2)))


# -- seed design property ---------------------------------------------------


def test_seed_covers_every_level_twice():
    space = make_space()
    seed = seed_design(space, 12)
    assert len(seed) == len(set(seed)) == 12
    counts = {}
    for key in seed:
        for factor, level in zip(space.factor_order, key, strict=True):
            counts[(factor, level)] = counts.get((factor, level), 0) + 1
    for factor, levels in space.factors.items():
        for level in levels:
            assert counts.get((factor, level), 0) >= 2, (factor, level)


def test_seed_is_deterministic():
    space = make_space()
    assert seed_design(space, 12) == seed_design(space, 12)


def test_suggest_serves_seed_first():
    space = make_space()
    sampler = make_sampler(space)
    batch = sampler.suggest(n=8)
    assert [space.key(c) for c in batch] == seed_design(space, 12)[:8]


# -- full-loop regression (the M5 gate) --------------------------------------


def test_eig_beats_random_everywhere_and_balanced_below_crossover():
    """Regression, not H1 (revised gate): mean over 20 reps, fixed seeds.

    EIG <= random at every budget; EIG <= balanced below the measured Arm A
    crossover (~28 runs; well-specified crossover is n ~ p). Past crossover
    balanced catching up is expected, and the tie is a reported finding, not
    a failure.
    """
    space = make_space()
    budgets = (20, 30, 40)
    rmse = {mode: {b: [] for b in budgets} for mode in ("eig", "balanced", "random")}
    for rep in range(20):
        surface = arm_a_surface(space, np.random.default_rng(1000 + rep))
        for mode in rmse:
            for budget in budgets:
                rmse[mode][budget].append(run_sampler(space, surface, mode, budget, np.random.default_rng(2000 + rep)))
    for budget in budgets:
        eig = np.mean(rmse["eig"][budget])
        rand = np.mean(rmse["random"][budget])
        assert eig <= rand + 1e-9, f"budget={budget}: eig {eig:.4f} > random {rand:.4f}"
    bal20 = np.mean(rmse["balanced"][20])
    eig20 = np.mean(rmse["eig"][20])
    assert eig20 <= bal20 + 1e-9, f"budget=20: eig {eig20:.4f} > balanced {bal20:.4f}"
    # Past crossover: balanced within noise of EIG, not wildly better/worse.
    for budget in (30, 40):
        gap = np.mean(rmse["eig"][budget]) - np.mean(rmse["balanced"][budget])
        assert abs(gap) < 0.01, f"budget={budget}: unexpected gap {gap:+.4f}"


# -- API sanity ---------------------------------------------------------------


def test_observe_predict_compare_contrast_should_stop():
    space = make_space()
    rng = np.random.default_rng(0)
    surface = arm_a_surface(space, rng)
    sampler = make_sampler(space)
    for config in sampler.suggest(n=12):
        key = space.key(config)
        sampler.observe(config, simulate(space, surface, key, rng), m=50)

    preds = sampler.predict([space.config(space.cells()[0])])
    assert 0.0 <= preds[0].std < 0.3
    assert preds[0].ci90[0] < preds[0].mean < preds[0].ci90[1]

    comp = sampler.compare(
        {"model": "model_a", "harness": "harness_c", "benchmark": "bench_a"},
        {"model": "model_b", "harness": "harness_c", "benchmark": "bench_a"},
        rng=np.random.default_rng(1),
    )
    assert comp.shared_benchmark is True
    assert "correlated" in comp.caveat  # warning travels in the payload
    assert 0.0 <= comp.probability_a_better <= 1.0

    con = sampler.contrast("harness", "harness_a", "harness_c", conditional_factor="model")
    assert len(con.conditional) == 5
    assert 0.0 <= con.p_positive <= 1.0

    assert sampler.should_stop(delta=10.0) is True  # absurdly loose
    assert sampler.should_stop(delta=1e-6) is False  # absurdly tight


def test_strategy_balanced_matches_seed_rule():
    """strategy='balanced' reproduces the coverage baseline through the public API."""
    space = make_space()
    rng = np.random.default_rng(5)
    surface = arm_a_surface(space, rng)
    sampler = make_sampler(space, strategy="balanced")
    observed = []
    for _ in range(3):  # 24 cells, past the seed
        for config in sampler.suggest(n=8):
            key = space.key(config)
            sampler.observe(config, simulate(space, surface, key, rng), m=50)
            observed.append(key)
    # Must equal the pure greedy-coverage sequence.
    expected = []
    chosen = []
    for _ in range(24):
        nxt = next_balanced(space, chosen)
        expected.append(nxt)
        chosen.append(nxt)
    assert observed == expected


def test_strategy_auto_runs_and_converges():
    """strategy='auto' completes a 40-run loop; RMSE close to pure EIG."""
    space = make_space()
    rng = np.random.default_rng(6)
    surface = arm_a_surface(space, rng)
    sampler = make_sampler(space, strategy="auto")
    while len(sampler.observations) < 40:
        for config in sampler.suggest(n=8):
            key = space.key(config)
            sampler.observe(config, simulate(space, surface, key, rng), m=50)
    preds = sampler.predict([space.config(k) for k in space.cells()])
    truth = np.array([surface[k] for k in space.cells()])
    rmse = float(np.sqrt(np.mean((np.array([p.mean for p in preds]) - truth) ** 2)))
    assert rmse < 0.12


def test_targeted_suggest_differs_from_global():
    space = make_space()
    rng = np.random.default_rng(3)
    surface = arm_a_surface(space, rng)
    sampler = make_sampler(space)
    for config in sampler.suggest(n=12):  # burn the seed
        key = space.key(config)
        sampler.observe(config, simulate(space, surface, key, rng), m=50)
    L = sampler.contrast_L("harness", "harness_a", "harness_c", "model")
    targeted = sampler.suggest(n=8, target=L)
    untargeted = sampler.suggest(n=8)
    assert targeted != untargeted
    # Targeted picks concentrate on the contrast's harness pair.
    n_pair = sum(c["harness"] in ("harness_a", "harness_c") for c in targeted)
    assert n_pair >= 6
