"""Experiment space: named categorical factors and their grid of cells."""

import itertools

from .types import CellKey, Config


class ExperimentSpace:
    """A finite grid of categorical factors.

    Example:
        space = ExperimentSpace(factors={
            "model": ["model_a", "model_b"],
            "harness": ["harness_a", "harness_b"],
        })
        space.cells()  # all 4 cells, as CellKey tuples

    The first level of each factor is the reference level for dummy coding.
    Factor order and level order are exactly as supplied.
    """

    def __init__(self, factors: dict[str, list[str]]):
        if not factors:
            raise ValueError("factors must not be empty")
        for name, levels in factors.items():
            if len(levels) < 2:
                raise ValueError(f"factor {name!r} needs >= 2 levels")
            if len(set(levels)) != len(levels):
                raise ValueError(f"factor {name!r} has duplicate levels")
        self.factors = {name: list(levels) for name, levels in factors.items()}
        self.factor_order = list(self.factors)

    def cells(self) -> list[CellKey]:
        """Every cell of the grid, in itertools.product order over factors."""
        return list(itertools.product(*self.factors.values()))

    def n_cells(self) -> int:
        """Total number of cells in the grid."""
        n = 1
        for levels in self.factors.values():
            n *= len(levels)
        return n

    def key(self, config: Config) -> CellKey:
        """Convert a Config dict to its canonical CellKey tuple."""
        missing = set(self.factor_order) - set(config)
        if missing:
            raise KeyError(f"config is missing factors: {sorted(missing)}")
        for factor in self.factor_order:
            if config[factor] not in self.factors[factor]:
                raise ValueError(f"unknown level {config[factor]!r} for factor {factor!r}")
        return tuple(config[factor] for factor in self.factor_order)

    def config(self, key: CellKey) -> Config:
        """Convert a CellKey tuple back to a Config dict."""
        if len(key) != len(self.factor_order):
            raise ValueError(f"key has {len(key)} levels, expected {len(self.factor_order)}")
        return dict(zip(self.factor_order, key, strict=True))
