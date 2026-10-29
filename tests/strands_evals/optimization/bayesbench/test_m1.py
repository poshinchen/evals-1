"""M1 gate tests: posterior matches a naive-inverse reference to 1e-10,
handles n < p, never calls np.linalg.inv."""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench.posterior import Posterior


def naive_reference(prior_mean, prior_var, phi, y, obs_var):
    """Textbook update with explicit inverses. Test oracle only."""
    prior_prec = np.diag(1.0 / prior_var)
    w = np.diag(1.0 / obs_var)
    cov = np.linalg.inv(prior_prec + phi.T @ w @ phi)
    mean = cov @ (prior_prec @ prior_mean + phi.T @ w @ y)
    return mean, cov


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [3, 20, 80])  # includes n < p
def test_matches_naive_reference(seed, n):
    rng = np.random.default_rng(seed)
    p = 52
    prior_mean = np.zeros(p)
    prior_mean[0] = 0.45
    prior_var = rng.uniform(0.02, 0.3, p) ** 2
    phi = (rng.random((n, p)) < 0.3).astype(float)
    phi[:, 0] = 1.0
    y = rng.uniform(0, 1, n)
    obs_var = rng.uniform(1e-4, 1e-2, n)

    post = Posterior(prior_mean, prior_var)
    post.update(phi, y, obs_var)
    ref_mean, ref_cov = naive_reference(prior_mean, prior_var, phi, y, obs_var)

    np.testing.assert_allclose(post.mean, ref_mean, atol=1e-10)
    np.testing.assert_allclose(post.cov, ref_cov, atol=1e-10)


def test_no_data_returns_prior():
    prior_mean = np.array([0.45, 0.0, 0.0])
    prior_var = np.array([0.25, 0.15, 0.05]) ** 2
    post = Posterior(prior_mean, prior_var)
    np.testing.assert_allclose(post.mean, prior_mean, atol=1e-12)
    np.testing.assert_allclose(post.cov, np.diag(prior_var), atol=1e-12)


def test_predictive_matches_cov():
    rng = np.random.default_rng(7)
    p, n = 12, 5
    prior_mean = np.zeros(p)
    prior_var = np.full(p, 0.1)
    phi = (rng.random((n, p)) < 0.5).astype(float)
    post = Posterior(prior_mean, prior_var)
    post.update(phi, rng.uniform(0, 1, n), np.full(n, 1e-3))

    a = (rng.random((4, p)) < 0.5).astype(float)
    mean, var = post.predictive(a)
    np.testing.assert_allclose(mean, a @ post.mean, atol=1e-12)
    np.testing.assert_allclose(var, np.diag(a @ post.cov @ a.T), atol=1e-10)


def test_sample_moments():
    rng = np.random.default_rng(0)
    prior_mean = np.array([0.45, 0.0])
    prior_var = np.array([0.04, 0.01])
    post = Posterior(prior_mean, prior_var)
    samples = post.sample(200_000, rng)
    np.testing.assert_allclose(samples.mean(axis=0), prior_mean, atol=5e-3)
    np.testing.assert_allclose(samples.var(axis=0), prior_var, rtol=0.02)


def test_never_calls_np_linalg_inv(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("np.linalg.inv was called")

    monkeypatch.setattr(np.linalg, "inv", boom)
    rng = np.random.default_rng(1)
    p, n = 10, 4
    post = Posterior(np.zeros(p), np.full(p, 0.1))
    phi = (rng.random((n, p)) < 0.5).astype(float)
    post.update(phi, rng.uniform(0, 1, n), np.full(n, 1e-3))
    post.predictive(phi)
    _ = post.cov
    post.sample(10, rng)
