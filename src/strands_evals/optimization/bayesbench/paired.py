"""Paired bootstrap for within-benchmark comparisons.

Cells that share a benchmark are evaluated on the same tasks, so their per-task
outcomes are positively correlated. The posterior `compare()` treats cells as
independent and is therefore conservative on shared benchmarks. When per-task
outcome vectors are available, a paired bootstrap over tasks keeps the
correlation and gives a calibrated comparison of the difference in means.

This is a check, not the full fix. The full fix is a task-level hierarchical
model — deferred by design.

Pairing contract: `a_outcomes[i]` and `b_outcomes[i]` are the SAME task i,
so the two vectors must be equal length and index-aligned.
"""

import numpy as np


def paired_bootstrap(
    a_outcomes,
    b_outcomes,
    n_boot: int = 10_000,
    ci: float = 0.90,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap the difference in mean outcome, resampling tasks jointly.

    Returns dict with mean_difference (observed a-b), std_difference
    (bootstrap sd), credible_interval (percentile CI at level `ci`),
    probability_a_better (fraction of bootstrap replicates with a > b).
    """
    a = np.asarray(a_outcomes, dtype=float)
    b = np.asarray(b_outcomes, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a_outcomes and b_outcomes must be 1-D and equal length")
    m = a.shape[0]
    if m == 0:
        raise ValueError("outcome vectors must be non-empty")
    rng = rng or np.random.default_rng()

    idx = rng.integers(0, m, size=(n_boot, m))  # joint resample of tasks
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)  # paired difference per rep
    lo_q, hi_q = (1 - ci) / 2, (1 + ci) / 2
    lo, hi = np.quantile(diffs, [lo_q, hi_q])
    return {
        "mean_difference": float(a.mean() - b.mean()),
        "std_difference": float(diffs.std(ddof=1)),
        "credible_interval": (float(lo), float(hi)),
        "probability_a_better": float((diffs > 0).mean()),
    }
