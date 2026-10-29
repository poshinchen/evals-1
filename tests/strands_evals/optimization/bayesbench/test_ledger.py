"""Ledger round-trip: replaying the file rebuilds the same posterior."""

import time

import numpy as np
import pytest

from strands_evals.optimization.bayesbench import ledger
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace

FACTORS = {
    "model": ["model_a", "model_b", "model_c"],
    "harness": ["harness_a", "harness_b"],
    "benchmark": ["bench_a", "bench_b"],
}
TAU_MAIN = {"model": 0.08, "harness": 0.10, "benchmark": 0.15}
TAU_INT = {frozenset({"model", "harness"}): 0.06}


def test_round_trip(tmp_path):
    path = tmp_path / "exp.jsonl"
    space = ExperimentSpace(FACTORS)
    sampler = InformationGainSampler(space, tau_main=TAU_MAIN, tau_int=TAU_INT)
    ledger.append(
        path,
        {
            "type": "space",
            "factors": FACTORS,
            "sampler_kwargs": {"tau_main": TAU_MAIN, "tau_int": {"model|harness": 0.06}},
        },
    )

    rng = np.random.default_rng(0)
    for _ in range(15):
        config = sampler.suggest(n=1, allow_repeats=True)[0]
        value = float(rng.binomial(50, 0.55) / 50)
        sampler.observe(config, value, m=50)
        ledger.append(
            path,
            {
                "type": "observation",
                "config": config,
                "value": value,
                "variance": None,
                "m": 50,
                "cost": None,
                "ts": time.time(),
            },
        )

    rebuilt = ledger.load_sampler(path)
    np.testing.assert_allclose(rebuilt.posterior.mean, sampler.posterior.mean)
    np.testing.assert_allclose(rebuilt.posterior.cov, sampler.posterior.cov)
    assert len(rebuilt.observations) == 15


def test_corrupt_line_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    ledger.append(path, {"type": "space", "factors": FACTORS, "sampler_kwargs": {}})
    with open(path, "a") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        ledger.load_sampler(path)


def test_missing_space_header(tmp_path):
    path = tmp_path / "headless.jsonl"
    ledger.append(path, {"type": "observation", "config": {}, "value": 0.5})
    with pytest.raises(ValueError, match="space"):
        ledger.load_sampler(path)
