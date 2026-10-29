"""Contrast matrices L and targeted EIG.

A contrast defines the scientific quantities z = L theta. Targeted EIG
scores a candidate by how much it shrinks uncertainty about z alone:

    IG_L(x) = 1/2 [ logdet(C) - logdet(C') ],   C = L Sigma L'

Because the covariance update is rank-one, so is the change in C. With
q = L Sigma a:

    IG_L(x) = -1/2 log(1 - q' C^-1 q / (sigma_x^2 + v_x))

Factorize C once per round; each candidate then costs O(pr + r^2).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .encoding import Encoder
from .space import ExperimentSpace


def targeted_eig(
    cov: np.ndarray,
    L: np.ndarray,
    candidates: np.ndarray,
    obs_var: np.ndarray,
) -> np.ndarray:
    """Targeted EIG for each candidate row, via the rank-one identity."""
    a = np.atleast_2d(candidates)
    obs_var = np.atleast_1d(obs_var)
    C = L @ cov @ L.T
    # C is singular when L has redundant rows (e.g. an overall contrast that
    # is the mean of its conditional rows). IG_L depends only on rowspace(L),
    # so a tiny ridge changes nothing that matters for ranking.
    jitter = 1e-10 * np.trace(C) / len(C)
    c_cho = cho_factor(C + jitter * np.eye(len(C)))
    sa = cov @ a.T  # (p, k)
    q = L @ sa  # (r, k)
    v = np.einsum("ij,ji->i", a, sa)  # a' Sigma a per candidate
    quad = np.einsum("ij,ij->j", q, cho_solve(c_cho, q))  # q' C^-1 q
    ratio = quad / (obs_var + v)
    # Guard: ratio is in [0, 1) mathematically; clip fp overshoot.
    return -0.5 * np.log1p(-np.clip(ratio, 0.0, 1.0 - 1e-15))


def surface_L(encoder: Encoder, keys: list) -> np.ndarray:
    """L = feature rows of the given cells (response-surface mode)."""
    return encoder.design_matrix(keys)


def level_contrast_L(
    space: ExperimentSpace,
    encoder: Encoder,
    factor: str,
    baseline: str,
    treatment: str,
    conditional_factor: str | None = None,
    eligible_keys: list | None = None,
) -> tuple[np.ndarray, list[str]]:
    """L rows for E[f | treatment] - E[f | baseline], averaged over other factors.

    Row 1 is the overall contrast. If conditional_factor is given, one more
    row per level of that factor (the contrast within that level).
    Returns (L, row_names).
    """
    if factor not in space.factors:
        raise ValueError(f"unknown factor {factor!r}")
    for level in (baseline, treatment):
        if level not in space.factors[factor]:
            raise ValueError(f"unknown level {level!r} for factor {factor!r}")

    def mean_feature(fixed: dict[str, str]) -> np.ndarray:
        keys = [k for k in space.cells() if all(k[space.factor_order.index(f)] == lv for f, lv in fixed.items())]
        return encoder.design_matrix(keys).mean(axis=0)

    def difference(fixed):
        if eligible_keys is None:
            return mean_feature({**fixed, factor: treatment}) - mean_feature({**fixed, factor: baseline})
        # Match every other factor, so an eligibility mask cannot turn a
        # feature contrast into a comparison of different populations.
        allowed = set(eligible_keys)
        fi = space.factor_order.index(factor)
        pairs = []
        for key in eligible_keys:
            if key[fi] != baseline or any(key[space.factor_order.index(f)] != v for f, v in fixed.items()):
                continue
            other = list(key)
            other[fi] = treatment
            other = tuple(other)
            if other in allowed:
                pairs.append(encoder.phi(other) - encoder.phi(key))
        if not pairs:
            raise ValueError(f"no common eligible support for {factor}: {treatment} - {baseline}, {fixed}")
        return np.mean(pairs, axis=0)

    rows = [difference({})]
    names = [f"{factor}: {treatment} - {baseline}"]
    if conditional_factor is not None:
        if conditional_factor == factor:
            raise ValueError("conditional_factor must differ from factor")
        for level in space.factors[conditional_factor]:
            rows.append(difference({conditional_factor: level}))
            names.append(f"{factor}: {treatment} - {baseline} | {conditional_factor}={level}")
    return np.vstack(rows), names


def interaction_contrast_L(
    space: ExperimentSpace,
    encoder: Encoder,
    factor: str,
    baseline: str,
    treatment: str,
    conditional_factor: str,
    conditional_baseline: str,
    conditional_treatment: str,
    eligible_keys: list | None = None,
) -> np.ndarray:
    """Difference of effects, conditional treatment minus conditional baseline.

    All four arms use exactly the same remaining-factor combinations.
    Summarize L @ theta with L @ Sigma @ L.T, retaining posterior covariance.
    """
    if factor == conditional_factor:
        raise ValueError("conditional_factor must differ from factor")
    for name, levels in (
        (factor, (baseline, treatment)),
        (conditional_factor, (conditional_baseline, conditional_treatment)),
    ):
        if name not in space.factors or any(v not in space.factors[name] for v in levels):
            raise ValueError(f"unknown factor or level in interaction contrast: {name}, {levels}")
    keys = space.cells() if eligible_keys is None else eligible_keys
    allowed = set(keys)
    fi, ci = space.factor_order.index(factor), space.factor_order.index(conditional_factor)
    rows = []
    for key in keys:
        if key[fi] != baseline or key[ci] != conditional_baseline:
            continue
        ft, ct, both = list(key), list(key), list(key)
        ft[fi] = both[fi] = treatment
        ct[ci] = both[ci] = conditional_treatment
        ft_key, ct_key, both_key = tuple(ft), tuple(ct), tuple(both)
        if all(k in allowed for k in (ft_key, ct_key, both_key)):
            rows.append(encoder.phi(both_key) - encoder.phi(ct_key) - encoder.phi(ft_key) + encoder.phi(key))
    if not rows:
        raise ValueError("no common eligible support for all four interaction arms")
    return np.mean(rows, axis=0, keepdims=True)
