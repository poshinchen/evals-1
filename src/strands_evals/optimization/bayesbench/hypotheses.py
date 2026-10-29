"""Hypothesis-expansion surfacing.

`diagnostics()` flags misspecification globally (one sticky bit). This makes it
ACTIONABLE: rank WHERE the two-way model fits worst, so a human (or later an
agent) can propose the next hypothesis. Three signals, computed from held-in
standardized residuals against the current posterior:

  1. residual_cells   — cells with the largest |standardized residual|.
  2. pair_structure    — factor pairs whose residuals still carry structure
                         (eta^2 of residuals grouped by the pair's level combo);
                         high => three-way structure or an over-shrunk two-way.
  3. factor_dispersion — factors whose per-level residual variance differs most;
                         high => a level the model explains worse than others.

This is exploratory candidate generation, NOT inference. No p-values, no
selection guarantee. The numbers rank candidates for review; they do not test
them.
"""

import itertools
from typing import Any

import numpy as np

from .types import HypothesisReport

MIN_GROUP = 2  # a residual group needs >= this many cells to be scored


def _residuals(sampler):
    """Held-in standardized residuals per observed cell, latest obs per cell.

    Returns (keys, z, y, pred_mean, obs_var) aligned arrays; one row per
    distinct observed cell (last observation wins if a cell was re-run).
    """
    latest = {}
    for o in sampler.observations:
        latest[sampler.space.key(o.config)] = o
    keys = list(latest)
    if not keys:
        return keys, *(np.zeros(0) for _ in range(4))
    phi = sampler.encoder.design_matrix(keys)
    pred_mean, pred_var = sampler.posterior.predictive(phi)
    y = np.array([latest[k].value for k in keys])
    obs_var = np.array([latest[k].variance for k in keys])
    z = (y - pred_mean) / np.sqrt(pred_var + obs_var)
    return keys, z, y, pred_mean, obs_var


def _eta_squared(labels, values) -> tuple[float, int]:
    """Fraction of variance in `values` explained by group `labels`.

    eta^2 = SS_between / SS_total. Returns (eta_sq, n_groups). Groups with a
    single member contribute 0 between-SS; total<=0 -> eta_sq 0.
    """
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    total = float(np.sum((values - grand) ** 2))
    if total <= 0:
        return 0.0, len(set(labels))
    between = 0.0
    groups: dict[Any, list[float]] = {}
    for lab, v in zip(labels, values, strict=True):
        groups.setdefault(lab, []).append(v)
    for members in groups.values():
        group = np.asarray(members)
        between += len(group) * (group.mean() - grand) ** 2
    return float(between / total), len(groups)


def suggest_hypotheses(sampler, top_k: int = 5) -> HypothesisReport:
    """Rank exploratory hypothesis candidates from residual structure."""
    space = sampler.space
    keys, z, y, pred_mean, obs_var = _residuals(sampler)
    messages = [
        "Exploratory candidate generation, not inference: these rank WHERE the "
        "two-way model fits worst; they do not test any hypothesis.",
    ]
    if len(keys) < MIN_GROUP:
        messages.append(f"only {len(keys)} observed cells; nothing to rank yet")
        return HypothesisReport([], [], [], messages)

    order = np.argsort(-np.abs(z))
    residual_cells = [
        {
            "config": space.config(keys[i]),
            "resid_z": float(z[i]),
            "y_observed": float(y[i]),
            "y_predicted": float(pred_mean[i]),
            "abs_resid": float(abs(z[i])),
        }
        for i in order[:top_k]
    ]

    # (2) pair structure: residual eta^2 grouped by each factor pair's combo.
    pair_rows: list[dict[str, Any]] = []
    for fa, fb in itertools.combinations(space.factor_order, 2):
        ia, ib = space.factor_order.index(fa), space.factor_order.index(fb)
        labels = [(k[ia], k[ib]) for k in keys]
        # need >= 2 populated groups with >= MIN_GROUP members to mean anything
        sizes: dict[tuple[str, str], int] = {}
        for lab in labels:
            sizes[lab] = sizes.get(lab, 0) + 1
        scored = sum(1 for c in sizes.values() if c >= MIN_GROUP)
        eta, n_groups = _eta_squared(labels, z)
        pair_rows.append(
            {
                "pair": (fa, fb),
                "eta_sq": eta,
                "n_groups": n_groups,
                "n_obs": len(keys),
                "well_supported": scored >= 2,
            }
        )
    pair_structure = sorted(pair_rows, key=lambda r: -r["eta_sq"])

    # (3) factor dispersion: spread of per-level residual variance.
    disp_rows: list[dict[str, Any]] = []
    for factor in space.factor_order:
        idx = space.factor_order.index(factor)
        per_level: dict[str, list[float]] = {}
        for i, k in enumerate(keys):
            per_level.setdefault(k[idx], []).append(z[i])
        var_by_level = {lv: float(np.var(v)) for lv, v in per_level.items() if len(v) >= MIN_GROUP}
        if len(var_by_level) < 2:
            continue
        vmax, vmin = max(var_by_level.values()), min(var_by_level.values())
        ratio = vmax / vmin if vmin > 0 else float("inf")
        disp_rows.append(
            {
                "factor": factor,
                "dispersion_ratio": ratio,
                "worst_level": max(var_by_level, key=lambda lv: var_by_level[lv]),
                "per_level_resid_var": var_by_level,
            }
        )
    factor_dispersion = sorted(
        disp_rows, key=lambda r: -r["dispersion_ratio"] if np.isfinite(r["dispersion_ratio"]) else -1e18
    )

    if not any(r["well_supported"] for r in pair_structure):
        messages.append(
            "pair-structure ranking is weakly supported at this budget (few cells per pair combo); treat as provisional"
        )
    return HypothesisReport(residual_cells, pair_structure, factor_dispersion, messages)
