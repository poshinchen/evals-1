"""Seed design scales with the grid and keeps the >= 2 coverage guarantee.

N_SEED=12 was originally hardcoded, but the ">= 2 appearances per level"
guarantee is only true for the 5x4x5 grid. On larger grids a fixed 12 silently
breaks it. These tests cover grids larger than 5x4x5, including a proposed
3-model-level 6-factor grid, and pin the default 5x4x5 behavior so the
earlier results still reproduce.
"""

import warnings

import pytest

from strands_evals.optimization.bayesbench.sampler import InformationGainSampler
from strands_evals.optimization.bayesbench.seed import (
    MIN_APPEARANCES,
    N_SEED,
    covers_all_levels,
    default_n_seed,
    minimum_seed_for_coverage,
    seed_design,
)
from strands_evals.optimization.bayesbench.space import ExperimentSpace


def grid(shape):
    """ExperimentSpace from a list of level counts."""
    return ExperimentSpace({f"f{i}": [f"f{i}_l{j}" for j in range(n)] for i, n in enumerate(shape)})


REFERENCE = ExperimentSpace(
    factors={
        "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
        "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
        "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
    }
)


# -- default behavior on 5x4x5 is unchanged (reproducibility) ----------------


def test_reference_grid_default_is_still_twelve():
    assert default_n_seed(REFERENCE) == N_SEED == 12


def test_reference_grid_seed_is_byte_identical_to_old_fixed_12():
    """The auto-derived seed must equal the old fixed-12 sequence exactly, so
    every earlier result on 5x4x5 reproduces."""
    assert seed_design(REFERENCE, None) == seed_design(REFERENCE, 12)


def test_reference_sampler_reports_twelve():
    s = InformationGainSampler(REFERENCE, tau_main=0.1)
    assert s.n_seed == 12
    assert s._seed_plan == seed_design(REFERENCE, 12)


# -- grids LARGER than 5x4x5 keep the guarantee ------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [8, 8],  # 64 cells, 8 levels: fixed-12 would FAIL (<16 needed)
        [10, 3, 3],  # 90 cells, 10 levels: fixed-12 FAILS
        [6, 5, 4, 3],  # 360 cells, 4 factors
        [3, 2, 2, 2, 2, 5],  # proposed widen: model=3 + 5 factors, 240 cells
        [3, 3, 3, 3, 3],  # 243 cells, symmetric
    ],
)
def test_auto_seed_covers_every_level_twice(shape):
    space = grid(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # auto-size must never warn
        seed = seed_design(space, None)
    assert covers_all_levels(space, seed, MIN_APPEARANCES)
    # Auto size is at least the exact floor and no larger than the grid.
    assert minimum_seed_for_coverage(space) <= len(seed) <= space.n_cells()


def test_proposed_grid_through_the_sampler():
    """A 3-model-level 6-factor grid builds and seeds cleanly end to end."""
    space = ExperimentSpace(
        factors={
            "model": ["m_a", "m_b", "m_c"],  # widened to 3 levels
            "thinking": ["off", "on"],
            "caching": ["off", "on"],
            "context_mgmt": ["off", "on"],
            "subagents": ["off", "on"],
            "benchmark": ["b_a", "b_b", "b_c", "b_d", "b_e"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = InformationGainSampler(space, tau_main=0.1)
    assert covers_all_levels(space, s._seed_plan)
    # suggest() serves the derived seed first, then EIG — no crash on 6 factors.
    batch = s.suggest(n=8)
    assert len(batch) == 8


# -- explicit n_seed too small warns but still returns -----------------------


def test_undersized_explicit_seed_warns():
    space = grid([10, 3, 3])  # needs >= 20 for 2x coverage
    with pytest.warns(UserWarning, match="appearances"):
        seed = seed_design(space, 12)
    assert len(seed) == 12  # honored, just flagged
    assert not covers_all_levels(space, seed)


def test_oversized_explicit_seed_clamps_and_warns():
    space = grid([2, 2, 2])  # 8 cells
    with pytest.warns(UserWarning, match="exceeds"):
        seed = seed_design(space, 20)
    assert len(seed) == 8  # clamped to grid size


def test_adequate_explicit_seed_is_silent():
    space = grid([10, 3, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seed = seed_design(space, 20)  # exactly enough
    assert covers_all_levels(space, seed)


# -- minimum_seed_for_coverage is a tight, correct floor ---------------------


@pytest.mark.filterwarnings("ignore:seed of")  # n-1 under-covers by design
def test_minimum_seed_matches_bruteforce_growth():
    for shape in ([5, 4, 5], [8, 8], [3, 2, 2, 2, 2, 5]):
        space = grid(shape)
        n = minimum_seed_for_coverage(space)
        # n covers; n-1 does not (tightness).
        assert covers_all_levels(space, seed_design(space, n))
        if n > 1:
            assert not covers_all_levels(space, seed_design(space, n - 1))


def test_minimum_seed_capped_when_grid_too_small():
    space = grid([2, 20])  # 40 cells but one factor has 20 levels
    # 2*20 = 40 needed for 2x; grid has exactly 40, so floor = n_cells.
    assert minimum_seed_for_coverage(space) <= space.n_cells()
