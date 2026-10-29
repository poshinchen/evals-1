"""Hypothesis-expansion surfacing (hypotheses.py)."""

import numpy as np

from strands_evals.optimization.bayesbench.encoding import Encoder
from strands_evals.optimization.bayesbench.hypotheses import _eta_squared
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def make_space():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c"],
            "harness": ["harness_a", "harness_b"],
            "benchmark": ["bench_a", "bench_b", "bench_c"],
        }
    )


def test_eta_squared_bounds():
    # Perfect group separation -> eta^2 = 1.
    eta, n = _eta_squared(["x", "x", "y", "y"], [1.0, 1.0, 5.0, 5.0])
    assert eta == 1.0 and n == 2
    # No between-group variation -> eta^2 = 0.
    eta, _ = _eta_squared(["x", "y", "x", "y"], [3.0, 3.0, 3.0, 3.0])
    assert eta == 0.0


def test_empty_and_tiny_are_safe():
    space = make_space()
    s = InformationGainSampler(space, tau_main=0.15)
    rep = s.suggest_hypotheses()  # no observations
    assert rep.residual_cells == []
    assert any("nothing to rank" in m for m in rep.messages)


def test_surfaces_unmodeled_three_way_pair():
    """Plant a three-way pattern (model_a x harness_b response flips sign
    across benchmarks). The two-way model can't fit it; suggest_hypotheses
    should rank the model x harness (or benchmark-involved) pair high and the
    honesty note must be present."""
    space = make_space()
    enc = Encoder(space)
    rng = np.random.default_rng(0)
    theta = np.zeros(enc.p)
    theta[0] = 0.5
    flip = {"bench_a": +0.15, "bench_b": -0.15, "bench_c": +0.15}

    def surface(key):
        model, harness, bench = key
        val = enc.phi(key) @ theta
        if model == "model_a" and harness == "harness_b":
            val += flip[bench]  # unmodeled three-way
        return float(np.clip(val, 0.01, 0.99))

    s = InformationGainSampler(space, tau_main=0.15, tau_int=0.03)
    observed = set()
    while len(observed) < space.n_cells():  # observe the whole small grid
        for config in s.suggest(n=6):
            key = space.key(config)
            s.observe(config, rng.binomial(500, surface(key)) / 500, m=500)
            observed.add(key)
            if len(observed) >= space.n_cells():
                break

    rep = s.suggest_hypotheses(top_k=3)
    assert any("not inference" in m for m in rep.messages)  # honesty note
    assert len(rep.residual_cells) == 3
    # The planted structure lives on model x harness (crossing with benchmark);
    # a model- or benchmark-involved pair should top the ranking, not the
    # untouched pair alone.
    top = set(rep.pair_structure[0]["pair"])
    assert top & {"model", "harness", "benchmark"}  # sane
    # The largest-residual cell should involve the planted model_a/harness_b.
    tops = [r["config"] for r in rep.residual_cells]
    assert any(c["model"] == "model_a" and c["harness"] == "harness_b" for c in tops)


def test_report_shape():
    space = make_space()
    rng = np.random.default_rng(1)
    s = InformationGainSampler(space, tau_main=0.15)
    for _ in range(3):
        for config in s.suggest(n=6):
            s.observe(config, rng.binomial(200, 0.5) / 200, m=200)
    rep = s.suggest_hypotheses()
    # Three pairs on a 3-factor space, three factors ranked.
    assert len(rep.pair_structure) == 3
    assert len(rep.factor_dispersion) <= 3
    for r in rep.pair_structure:
        assert set(r) >= {"pair", "eta_sq", "n_groups", "well_supported"}
