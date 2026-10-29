"""Held-out predictive checks.

On each observe(), BEFORE absorbing the observation, compute the held-out
z-score z = (y - a'mu) / sqrt(a'Sigma a + sigma_x^2). Test DISJOINT windows
of W z-scores. A window flags when either:

- sum(z_i^2) > chi2.ppf(q, df=W)          (overdispersion), or
- |mean(z)| > norm.ppf((1+q)/2) / sqrt(W)  (systematic bias).

Calibration record: a ROLLING window with W = 10, q = 0.95 ever-flags ~58% of
well-specified 40-observation runs on iid N(0,1) z alone — rolling windows
re-test the same draws ~W times and the 5% per-test rate compounds. Measured
on the real sampler over W in {10, 15, 20} x q in {0.95, 0.99, 0.995, 0.999},
rolling and disjoint: disjoint W = 15, q = 0.995 gives the best separation and
holds the false-alarm side at ~8% at the m = 50 operating point. See the
test_m6 gate tests for the full operating-characteristic table.
"""

from collections import deque

import numpy as np
from scipy.stats import chi2, norm

from .types import CellKey, Diagnostics

W = 15
Q = 0.995
P_BOUNDARY = (0.05, 0.95)


class DiagnosticsTracker:
    """Disjoint held-out z-score windows with the calibrated flag rule.

    `flagged` is sticky across windows: once any completed window flags, the
    tracker stays flagged (a real misspecification does not un-happen).
    """

    def __init__(self, window: int = W, q: float = Q):
        self.window = window
        self._z: deque[float] = deque(maxlen=window)  # current (incomplete) window
        self._all_z: list[float] = []  # full history, for snapshot stats
        self.n_holdout = 0
        self.boundary_cells: list[CellKey] = []
        self._chi2_bound = chi2.ppf(q, df=window)
        self._bias_bound = norm.ppf((1 + q) / 2) / np.sqrt(window)
        self._flagged = False

    def record(self, y: float, pred_mean: float, pred_var: float, obs_var: float, key: CellKey) -> None:
        """Score one observation against the pre-update posterior."""
        z = float((y - pred_mean) / np.sqrt(pred_var + obs_var))
        self._z.append(z)
        self._all_z.append(z)
        self.n_holdout += 1
        if not (P_BOUNDARY[0] <= y <= P_BOUNDARY[1]):
            self.boundary_cells.append(key)
        if len(self._z) == self.window:  # window complete: test and reset
            zz = np.array(self._z)
            if np.sum(zz**2) > self._chi2_bound or abs(np.mean(zz)) > self._bias_bound:
                self._flagged = True
            self._z.clear()

    @property
    def flagged(self) -> bool:
        """Whether any completed window has flagged (sticky)."""
        return self._flagged

    def snapshot(self) -> Diagnostics:
        """Current model-health summary over the full z history."""
        z = np.array(self._all_z) if self._all_z else np.zeros(0)
        messages = []
        if self.n_holdout < self.window:
            messages.append(f"first window not complete ({self.n_holdout}/{self.window}); no flagging yet")
        if self._flagged:
            messages.append(
                "a held-out z window exceeded its calibrated bound: the "
                "model's predictions are more wrong than its own uncertainty "
                "admits (possible unmodeled interaction structure)"
            )
        if self.boundary_cells:
            messages.append(
                f"{len(self.boundary_cells)} cell(s) with score outside "
                f"[{P_BOUNDARY[0]}, {P_BOUNDARY[1]}]: Gaussian approximation "
                "degrades near the boundary"
            )
        return Diagnostics(
            n_holdout=self.n_holdout,
            z_mean=float(z.mean()) if len(z) else 0.0,
            z_sq_sum=float(np.sum(z**2)) if len(z) else 0.0,
            flagged=self.flagged,
            boundary_cells=list(self.boundary_cells),
            messages=messages,
        )
