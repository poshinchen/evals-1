"""Interaction-recovery metric.

A coefficient (main or two-way; intercept excluded) is "significant" when the
posterior puts > P_THRESH mass outside +/- TAU:

    P(|theta_j| > tau) > 0.9

evaluated from the Gaussian marginal (mean, sd = sqrt(cov_jj)). Recovery of a
sequential fit is recall/precision of its significant set against the set
computed the same way on the full exhaustive dataset (ground truth).
"""

import numpy as np
from scipy.stats import norm

TAU = 0.05
P_THRESH = 0.9


def significant_set(mean, cov, encoder, tau: float = TAU, p_thresh: float = P_THRESH) -> set[int]:
    """Column indices j (intercept excluded) with P(|theta_j| > tau) > p_thresh."""
    mean = np.asarray(mean, dtype=float)
    sd = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 1e-300, None))
    # P(|theta|>tau) = P(theta>tau) + P(theta<-tau), theta ~ N(mean, sd^2).
    p_out = norm.sf(tau, loc=mean, scale=sd) + norm.cdf(-tau, loc=mean, scale=sd)
    sig = {int(j) for j in np.flatnonzero(p_out > p_thresh)}
    sig.discard(encoder.intercept_idx)
    return sig


def recall_precision(recovered: set[int], truth: set[int]) -> tuple[float, float]:
    """(recall, precision) of a recovered set against ground truth.

    Conventions on empty sets: recall is 1.0 when truth is empty (nothing to
    miss); precision is 1.0 when nothing was recovered (no false positives).
    """
    if not truth:
        recall = 1.0
    else:
        recall = len(recovered & truth) / len(truth)
    if not recovered:
        precision = 1.0
    else:
        precision = len(recovered & truth) / len(recovered)
    return recall, precision
