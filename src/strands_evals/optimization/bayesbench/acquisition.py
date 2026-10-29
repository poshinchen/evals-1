"""Acquisition: closed-form EIG, cost weighting, greedy batch selection.

Global EIG:      IG(x) = 1/2 log(1 + a' Sigma a / sigma_x^2)
Rank-one update: Sigma' = Sigma - Sigma a a' Sigma / (sigma_x^2 + v_x)
Cost weighting:  score = IG(x) / c(x)^alpha

The update does not depend on the observed value, so a batch can be planned
by pretending each pick was observed and re-scoring.
"""

import numpy as np


def eig(cov: np.ndarray, a: np.ndarray, obs_var: np.ndarray) -> np.ndarray:
    """Closed-form EIG for candidate feature rows `a` (k, p) under Sigma=cov."""
    a = np.atleast_2d(a)
    v = np.einsum("ij,jk,ik->i", a, cov, a)  # a' Sigma a per row
    return 0.5 * np.log1p(v / np.atleast_1d(obs_var))


def rank_one_update(cov: np.ndarray, a: np.ndarray, obs_var: float) -> np.ndarray:
    """Posterior covariance after hypothetically observing one cell."""
    sa = cov @ a
    return cov - np.outer(sa, sa) / (obs_var + a @ sa)


def greedy_batch(
    cov: np.ndarray,
    candidates: np.ndarray,
    obs_var: np.ndarray,
    k: int,
    cost: np.ndarray | None = None,
    alpha: float = 1.0,
) -> list[int]:
    """Select k candidate indices greedily, re-scoring after each pick.

    candidates: (n, p) feature rows of unrun cells.
    obs_var:    (n,) predicted observation noise per candidate.
    cost:       (n,) cost per candidate; None means uniform (no weighting).
    Returns indices into `candidates`, in pick order, no duplicates.
    """
    n = len(candidates)
    if k > n:
        raise ValueError(f"asked for k={k} from {n} candidates")
    obs_var = np.asarray(obs_var, dtype=float)
    weights = np.ones(n) if cost is None else np.asarray(cost, dtype=float) ** alpha
    if np.any(weights <= 0):
        raise ValueError("costs must be positive")

    cov = cov.copy()
    picked: list[int] = []
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        scores = eig(cov, candidates, obs_var) / weights
        scores[~available] = -np.inf
        best = int(np.argmax(scores))  # argmax takes the lowest index on ties
        picked.append(best)
        available[best] = False
        cov = rank_one_update(cov, candidates[best], obs_var[best])
    return picked


def batch_information(cov: np.ndarray, batch: np.ndarray, obs_var: np.ndarray) -> float:
    """Joint information of a batch: 1/2 logdet(I + W^1/2 A Sigma A' W^1/2)."""
    a = np.atleast_2d(batch)
    scale = 1.0 / np.sqrt(np.atleast_1d(obs_var))
    m = np.eye(len(a)) + (a * scale[:, None]) @ cov @ (a * scale[:, None]).T
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    return 0.5 * logdet
