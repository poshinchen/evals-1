"""Dummy coding: turn a cell into the feature vector phi(x).

Feature order (fixed, tests index into theta by it):
    1. intercept
    2. main effects, factors in space.factor_order, levels in level order,
       reference level (the first) skipped
    3. two-way interactions, factor pairs in factor-order pairs
       (itertools.combinations), each pair's columns in (level_i, level_j)
       product order, reference levels skipped

For the 5x4x5 grid: p = 1 + (4+3+4) + (12+16+12) = 52.
"""

import itertools

import numpy as np

from .space import ExperimentSpace
from .types import CellKey


class Encoder:
    """Maps cells of an ExperimentSpace to dummy-coded feature vectors."""

    def __init__(self, space: ExperimentSpace, interactions: int = 2):
        if interactions not in (1, 2):
            raise ValueError("interactions must be 1 (mains only) or 2")
        self.space = space
        self.interactions = interactions

        # Column index of each non-reference (factor, level) main effect.
        self._main_col: dict[tuple[str, str], int] = {}
        # Column index of each interaction pair of main-effect levels.
        self._int_col: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}

        self.feature_names: list[str] = ["intercept"]
        for factor in space.factor_order:
            for level in space.factors[factor][1:]:
                self._main_col[(factor, level)] = len(self.feature_names)
                self.feature_names.append(f"{factor}={level}")
        if interactions == 2:
            for fa, fb in itertools.combinations(space.factor_order, 2):
                for la in space.factors[fa][1:]:
                    for lb in space.factors[fb][1:]:
                        key = ((fa, la), (fb, lb))
                        self._int_col[key] = len(self.feature_names)
                        self.feature_names.append(f"{fa}={la}:{fb}={lb}")

        self.p = len(self.feature_names)
        # Index ranges, handy for priors and metrics.
        self.intercept_idx = 0
        self.main_idx = {
            factor: [self._main_col[(factor, lv)] for lv in space.factors[factor][1:]] for factor in space.factor_order
        }
        self.interaction_idx = {
            frozenset((fa, fb)): [col for ((cfa, _), (cfb, _)), col in self._int_col.items() if {cfa, cfb} == {fa, fb}]
            for fa, fb in itertools.combinations(space.factor_order, 2)
        }

    def phi(self, key: CellKey) -> np.ndarray:
        """Feature vector for one cell."""
        level_of = dict(zip(self.space.factor_order, key, strict=True))
        x = np.zeros(self.p)
        x[0] = 1.0
        for factor, level in level_of.items():
            col = self._main_col.get((factor, level))
            if col is not None:
                x[col] = 1.0
        if self.interactions == 2:
            for fa, fb in itertools.combinations(self.space.factor_order, 2):
                col = self._int_col.get(((fa, level_of[fa]), (fb, level_of[fb])))
                if col is not None:
                    x[col] = 1.0
        return x

    def design_matrix(self, keys: list[CellKey]) -> np.ndarray:
        """Stack phi(x) for a list of cells into an (n, p) matrix."""
        return np.vstack([self.phi(k) for k in keys]) if keys else np.zeros((0, self.p))
