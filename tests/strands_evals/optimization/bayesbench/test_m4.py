"""M4 gate tests: rank-one targeted EIG matches direct logdet to 1e-8.
Plus contrast-builder sanity."""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench.acquisition import eig, rank_one_update
from strands_evals.optimization.bayesbench.contrasts import level_contrast_L, surface_L, targeted_eig
from strands_evals.optimization.bayesbench.encoding import Encoder
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def direct_ig_l(cov, L, a, obs_var):
    """Reference: build C and C' explicitly, take the logdet difference."""
    C = L @ cov @ L.T
    C_after = L @ rank_one_update(cov, a, obs_var) @ L.T
    return 0.5 * (np.linalg.slogdet(C)[1] - np.linalg.slogdet(C_after)[1])


@pytest.fixture
def space545():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
            "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
            "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
        }
    )


@pytest.mark.parametrize("seed", range(20))
def test_rank_one_identity_matches_direct_logdet(seed):
    rng = np.random.default_rng(seed)
    p, r, n_cand = 20, 5, 15
    # Random SPD covariance.
    m = rng.standard_normal((p, p))
    cov = m @ m.T / p + 0.05 * np.eye(p)
    L = rng.standard_normal((r, p))
    cands = (rng.random((n_cand, p)) < 0.4).astype(float)
    cands[:, 0] = 1.0
    obs_var = rng.uniform(1e-3, 1e-2, n_cand)

    fast = targeted_eig(cov, L, cands, obs_var)
    for i in range(n_cand):
        assert fast[i] == pytest.approx(direct_ig_l(cov, L, cands[i], obs_var[i]), abs=1e-8)


def test_global_mode_L_identity_equals_global_eig():
    # With L = I, targeted EIG must equal global EIG.
    rng = np.random.default_rng(3)
    p = 10
    m = rng.standard_normal((p, p))
    cov = m @ m.T / p + 0.1 * np.eye(p)
    cands = (rng.random((8, p)) < 0.5).astype(float)
    obs_var = rng.uniform(1e-3, 1e-2, 8)
    np.testing.assert_allclose(
        targeted_eig(cov, np.eye(p), cands, obs_var),
        eig(cov, cands, obs_var),
        atol=1e-10,
    )


def test_level_contrast_rows(space545):
    enc = Encoder(space545)
    L, names = level_contrast_L(
        space545,
        enc,
        factor="harness",
        baseline="harness_a",
        treatment="harness_c",
        conditional_factor="model",
    )
    assert L.shape == (6, 52)  # overall + 5 models
    assert names[0] == "harness: harness_c - harness_a"
    assert names[1].endswith("model=model_a")
    # A pure level contrast has no intercept component.
    np.testing.assert_allclose(L[:, 0], 0.0, atol=1e-12)


def test_level_contrast_value_on_known_theta(space545):
    # Plant a pure harness_c main effect and check the contrast reads it back.
    enc = Encoder(space545)
    theta = np.zeros(enc.p)
    col = enc.feature_names.index("harness=harness_c")
    theta[col] = 0.14
    L, _ = level_contrast_L(space545, enc, "harness", "harness_a", "harness_c", "model")
    z = L @ theta
    np.testing.assert_allclose(z, 0.14, atol=1e-12)  # same for overall and per model


def test_surface_L_is_design_matrix(space545):
    enc = Encoder(space545)
    keys = space545.cells()[:10]
    np.testing.assert_array_equal(surface_L(enc, keys), enc.design_matrix(keys))


def test_level_contrast_rejects_bad_input(space545):
    enc = Encoder(space545)
    with pytest.raises(ValueError):
        level_contrast_L(space545, enc, "nope", "a", "b")
    with pytest.raises(ValueError):
        level_contrast_L(space545, enc, "harness", "harness_a", "nope")
    with pytest.raises(ValueError):
        level_contrast_L(space545, enc, "harness", "harness_a", "harness_c", conditional_factor="harness")
