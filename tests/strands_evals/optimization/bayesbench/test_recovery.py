"""Interaction-recovery metric (recovery.py).

This module originally also tested a synthetic recovery-experiment script;
that script was not brought over, so only the library-level tests remain.
"""

import numpy as np

from strands_evals.optimization.bayesbench.encoding import Encoder
from strands_evals.optimization.bayesbench.recovery import recall_precision, significant_set
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def make_space():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c"],
            "harness": ["harness_a", "harness_b"],
            "benchmark": ["bench_a", "bench_b"],
        }
    )


def test_recall_precision_edge_cases():
    assert recall_precision(set(), set()) == (1.0, 1.0)  # nothing to find
    assert recall_precision(set(), {1, 2}) == (0.0, 1.0)  # missed all, no FP
    assert recall_precision({1, 2}, set()) == (1.0, 0.0)  # all FP
    assert recall_precision({1, 2}, {1, 3}) == (0.5, 0.5)


def test_significant_set_excludes_intercept_and_finds_large_effect():
    space = make_space()
    enc = Encoder(space)
    p = enc.p
    mean = np.zeros(p)
    mean[enc.intercept_idx] = 0.5  # big, but must be excluded
    col = enc.feature_names.index("model=model_b")
    mean[col] = 0.5  # clearly > tau=0.05
    cov = np.eye(p) * 1e-6  # tiny sd -> confident
    sig = significant_set(mean, cov, enc)
    assert col in sig
    assert enc.intercept_idx not in sig


def test_significant_set_ignores_uncertain_effect():
    space = make_space()
    enc = Encoder(space)
    mean = np.zeros(enc.p)
    col = enc.feature_names.index("model=model_b")
    mean[col] = 0.0  # centered near zero...
    cov = np.eye(enc.p) * 0.1**2  # ...sd=0.1 -> P(|theta|>0.05)≈0.62
    sig = significant_set(mean, cov, enc)
    assert col not in sig  # not confident it clears tau


def test_ground_truth_recovers_planted_effect_from_full_fit():
    """Full-data fit calls a planted large interaction significant."""
    space = make_space()
    enc = Encoder(space)
    theta = np.zeros(enc.p)
    theta[enc.intercept_idx] = 0.5
    planted = enc.feature_names.index("model=model_b:harness=harness_b")
    theta[planted] = 0.20
    s = InformationGainSampler(space, tau_main=0.15, tau_int=0.10)
    for key in space.cells():
        p = float(np.clip(enc.phi(key) @ theta, 0.01, 0.99))
        s.observe(space.config(key), p, m=500)  # large m -> confident fit
    sig = significant_set(s.posterior.mean, s.posterior.cov, s.encoder)
    assert planted in sig
