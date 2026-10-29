"""Deterministic seed design and incremental balanced baseline.

Greedy max-coverage over factor levels. No RNG. The balanced baseline is the
same rule continued past n_seed, so the EIG-vs-balanced comparison is fair by
construction.

The design guarantee is "every level appears >= 2 times in the seed". That was
pinned to N_SEED = 12 for the 5x4x5 reference grid, where it holds. On other
grids a fixed 12 can silently break it (a factor with > 6 levels needs > 12
seed cells to cover each level twice). So the seed size now DERIVES from the
grid: `default_n_seed(space)` returns a size that provably keeps the >= 2
guarantee, and equals 12 on the 5x4x5 grid so existing results are unchanged.
An explicit n_seed is still honored, with a warning when it fails the
guarantee.
"""

import warnings

from .space import ExperimentSpace
from .types import CellKey

# Legacy constant: the 5x4x5 reference seed size. Kept for back-compat; new
# code should let the size derive from the grid (default_n_seed).
N_SEED = 12

# Guarantee: each factor level appears at least this many times in the seed.
MIN_APPEARANCES = 2


def next_balanced(
    space: ExperimentSpace,
    selected: list[CellKey],
    candidates: list[CellKey] | None = None,
) -> CellKey:
    """The next cell under the greedy coverage rule.

    score(x) = sum over factors f of 1 / (1 + count[level_f(x)]),
    tie-break: lowest index in space.cells() enumeration order.
    """
    counts: dict[tuple[str, str], int] = {}
    for key in selected:
        for factor, level in zip(space.factor_order, key, strict=True):
            counts[(factor, level)] = counts.get((factor, level), 0) + 1

    chosen = set(selected)
    best_key, best_score = None, -1.0
    for key in space.cells() if candidates is None else candidates:
        if key in chosen:
            continue
        score = sum(
            1.0 / (1 + counts.get((factor, level), 0)) for factor, level in zip(space.factor_order, key, strict=True)
        )
        if score > best_score:  # strict: first max wins ties
            best_key, best_score = key, score
    if best_key is None:
        raise ValueError("no unselected cells remain")
    return best_key


def _level_counts(space: ExperimentSpace, selected: list[CellKey]) -> dict:
    counts: dict[tuple[str, str], int] = {}
    for key in selected:
        for factor, level in zip(space.factor_order, key, strict=True):
            counts[(factor, level)] = counts.get((factor, level), 0) + 1
    return counts


def covers_all_levels(
    space: ExperimentSpace,
    selected: list[CellKey],
    min_appearances: int = MIN_APPEARANCES,
    candidates: list[CellKey] | None = None,
) -> bool:
    """True when every factor level appears >= min_appearances times."""
    counts = _level_counts(space, selected)
    available = (
        set((f, level) for key in candidates for f, level in zip(space.factor_order, key, strict=True))
        if candidates is not None
        else None
    )
    return all(
        counts.get((factor, level), 0) >= min_appearances
        for factor, levels in space.factors.items()
        for level in levels
        if available is None or (factor, level) in available
    )


def minimum_seed_for_coverage(
    space: ExperimentSpace,
    min_appearances: int = MIN_APPEARANCES,
    candidates: list[CellKey] | None = None,
) -> int:
    """Smallest greedy-seed size that gives every level >= min_appearances.

    Extends the greedy coverage sequence until the guarantee holds, capped at
    the grid size. Because the greedy rule spreads appearances as evenly as it
    can, the answer is close to `min_appearances * max_levels`; this computes
    the exact size rather than trusting the bound.
    """
    n_cells = space.n_cells() if candidates is None else len(candidates)
    selected: list[CellKey] = []
    for _ in range(n_cells):
        selected.append(next_balanced(space, selected, candidates))
        if covers_all_levels(space, selected, min_appearances, candidates):
            return len(selected)
    return n_cells  # grid too small to cover every level twice; use all of it


def default_n_seed(
    space: ExperimentSpace,
    min_appearances: int = MIN_APPEARANCES,
    candidates: list[CellKey] | None = None,
) -> int:
    """Grid-derived seed size that keeps the >= min_appearances guarantee.

    `2 * max_levels + 2` (the working rule already used by the crossover
    experiment) with a 2-cell cushion above the theoretical `min * max_levels`
    floor, but never below the exact `minimum_seed_for_coverage` and never
    above the grid size. Equals 12 on the 5x4x5 reference grid, so default
    behavior there is unchanged.
    """
    max_levels = (
        max(len({key[i] for key in candidates}) for i in range(len(space.factor_order)))
        if candidates is not None
        else max(len(levels) for levels in space.factors.values())
    )
    heuristic = min_appearances * max_levels + 2
    exact_floor = minimum_seed_for_coverage(space, min_appearances, candidates)
    size = space.n_cells() if candidates is None else len(candidates)
    return min(size, max(heuristic, exact_floor))


def seed_design(
    space: ExperimentSpace,
    n_seed: int | None = None,
    candidates: list[CellKey] | None = None,
) -> list[CellKey]:
    """The first n_seed cells of the greedy coverage sequence.

    n_seed=None (default) derives a grid-appropriate size via `default_n_seed`,
    which keeps the ">= 2 appearances per level" guarantee on any grid. An
    explicit n_seed is honored, but a warning fires when it is too small to
    cover every level (the guarantee the methodology relies on) or larger than
    the grid.
    """
    size = space.n_cells() if candidates is None else len(candidates)
    if size == 0:
        raise ValueError("seed design needs at least one eligible cell")
    if n_seed is None:
        n_seed = default_n_seed(space, candidates=candidates)
    elif n_seed < 0:
        raise ValueError("n_seed must be nonnegative")
    elif n_seed > size:
        warnings.warn(f"n_seed={n_seed} exceeds the {size}-cell grid; clamping to {size}.", stacklevel=2)
        n_seed = size

    selected: list[CellKey] = []
    for _ in range(n_seed):
        selected.append(next_balanced(space, selected, candidates))

    if not covers_all_levels(space, selected, candidates=candidates):
        need = minimum_seed_for_coverage(space, candidates=candidates)
        warnings.warn(
            f"seed of {n_seed} cells does not give every factor level the "
            f">= {MIN_APPEARANCES} appearances the seed design guarantees "
            f"(need >= {need} on this grid); posterior main effects for "
            f"under-covered levels will lean on the prior. Pass a larger "
            f"n_seed or leave it as None to auto-size.",
            stacklevel=2,
        )
    return selected
