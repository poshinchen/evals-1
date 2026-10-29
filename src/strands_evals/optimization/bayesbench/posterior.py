"""Heteroscedastic conjugate Gaussian posterior.

Sigma_n = (Sigma_0^-1 + Phi' W Phi)^-1
mu_n    = Sigma_n (Sigma_0^-1 mu_0 + Phi' W y)

Implemented via Cholesky of the precision matrix. Never forms an explicit
inverse (hard rule: no np.linalg.inv).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _cho_factor_with_jitter(precision: np.ndarray):
    """Cholesky with escalating jitter: 1e-9, x10 up to 1e-3, then raise."""
    jitter = 0.0
    while True:
        try:
            return cho_factor(precision + jitter * np.eye(precision.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 1e-9 if jitter == 0.0 else jitter * 10
            if jitter > 1e-3:
                raise


class Posterior:
    """Gaussian posterior over theta, updated in one shot from all data.

    Prior is diagonal: prior_var is the length-p vector of prior variances,
    prior_mean the length-p mean (zeros except intercept 0.45 by convention).
    """

    def __init__(self, prior_mean: np.ndarray, prior_var: np.ndarray):
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_var = np.asarray(prior_var, dtype=float)
        if self.prior_mean.shape != self.prior_var.shape:
            raise ValueError("prior_mean and prior_var must have the same shape")
        if np.any(self.prior_var <= 0):
            raise ValueError("prior variances must be positive")
        self.p = self.prior_mean.shape[0]
        self._recompute(np.zeros((0, self.p)), np.zeros(0), np.zeros(0))

    def _recompute(self, phi: np.ndarray, y: np.ndarray, obs_var: np.ndarray) -> None:
        prior_prec = np.diag(1.0 / self.prior_var)
        if len(y):
            w_phi = phi / obs_var[:, None]  # W Phi, row-scaled
            precision = prior_prec + phi.T @ w_phi
            rhs = self.prior_mean / self.prior_var + phi.T @ (y / obs_var)
        else:
            precision = prior_prec
            rhs = self.prior_mean / self.prior_var
        self._cho = _cho_factor_with_jitter(precision)
        self.mean = cho_solve(self._cho, rhs)
        self.n = len(y)

    def update(self, phi: np.ndarray, y: np.ndarray, obs_var: np.ndarray) -> None:
        """Recompute the posterior from the full stacked data.

        Full O(np^2) refit on every update; at p=52 and n<=60 a refit costs
        microseconds, so incremental rank-one bookkeeping for the *posterior*
        is not worth its complexity. (Acquisition does use rank-one updates,
        on covariance copies.)
        """
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        obs_var = np.atleast_1d(np.asarray(obs_var, dtype=float))
        if np.any(obs_var <= 0):
            raise ValueError("observation variances must be positive")
        if not (len(phi) == len(y) == len(obs_var)):
            raise ValueError("phi, y, obs_var must have equal length")
        self._recompute(phi, y, obs_var)

    @property
    def cov(self) -> np.ndarray:
        """Posterior covariance Sigma_n (solved from the factored precision)."""
        return cho_solve(self._cho, np.eye(self.p))

    def predictive(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent mean and variance for feature rows a: (a mu, diag(a Sigma a'))."""
        a = np.atleast_2d(np.asarray(a, dtype=float))
        mean = a @ self.mean
        # a Sigma a' diag without forming Sigma: solve per row through the factor.
        var = np.einsum("ij,ji->i", a, cho_solve(self._cho, a.T))
        return mean, var

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples of theta, shape (n, p)."""
        # Sigma = L^-T L^-1 for the precision's Cholesky factor L, so
        # theta = mean + L^-T z has covariance Sigma.
        c, lower = self._cho
        z = rng.standard_normal((self.p, n))
        l_factor = np.tril(c) if lower else np.triu(c).T
        theta = self.mean[:, None] + np.linalg.solve(l_factor.T, z)
        return theta.T
