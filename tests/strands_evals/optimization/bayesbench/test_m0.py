"""M0 gate tests: space, encoding, types.

Gate: p = 52 on the 5x4x5 grid; feature order is fixed and documented;
Config <-> CellKey round-trips.
"""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench.encoding import Encoder
from strands_evals.optimization.bayesbench.space import ExperimentSpace


@pytest.fixture
def space545():
    return ExperimentSpace(
        factors={
            "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
            "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
            "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
        }
    )


def test_p_is_52_on_5x4x5(space545):
    enc = Encoder(space545, interactions=2)
    assert enc.p == 52
    assert space545.n_cells() == 100
    assert len(space545.cells()) == 100


def test_feature_order_is_fixed(space545):
    enc = Encoder(space545, interactions=2)
    # Intercept first, then mains in factor order, then pairs in factor-pair order.
    assert enc.feature_names[0] == "intercept"
    assert enc.feature_names[1] == "model=model_b"  # reference model_a skipped
    assert enc.feature_names[5] == "harness=harness_b"  # after 4 model mains
    assert enc.feature_names[8] == "benchmark=bench_b"
    assert enc.feature_names[12] == "model=model_b:harness=harness_b"
    # Mains: 4 + 3 + 4 = 11; interactions: 12 + 16 + 12 = 40.
    assert len(enc.main_idx["model"]) == 4
    assert len(enc.main_idx["harness"]) == 3
    assert len(enc.main_idx["benchmark"]) == 4
    assert len(enc.interaction_idx[frozenset({"model", "harness"})]) == 12
    assert len(enc.interaction_idx[frozenset({"model", "benchmark"})]) == 16
    assert len(enc.interaction_idx[frozenset({"harness", "benchmark"})]) == 12


def test_phi_reference_cell_is_intercept_only(space545):
    enc = Encoder(space545)
    x = enc.phi(("model_a", "harness_a", "bench_a"))
    assert x[0] == 1.0
    assert x[1:].sum() == 0.0


def test_phi_full_nonreference_cell(space545):
    enc = Encoder(space545)
    x = enc.phi(("model_b", "harness_b", "bench_b"))
    # intercept + 3 mains + 3 interactions
    assert x.sum() == 7.0
    names_on = [n for n, v in zip(enc.feature_names, x, strict=True) if v == 1.0]
    assert names_on == [
        "intercept",
        "model=model_b",
        "harness=harness_b",
        "benchmark=bench_b",
        "model=model_b:harness=harness_b",
        "model=model_b:benchmark=bench_b",
        "harness=harness_b:benchmark=bench_b",
    ]


def test_config_cellkey_round_trip(space545):
    for key in space545.cells():
        assert space545.key(space545.config(key)) == key


def test_key_rejects_bad_config(space545):
    with pytest.raises(KeyError):
        space545.key({"model": "model_a"})
    with pytest.raises(ValueError):
        space545.key({"model": "nope", "harness": "harness_a", "benchmark": "bench_a"})


def test_design_matrix_shape_and_rank(space545):
    enc = Encoder(space545)
    phi = enc.design_matrix(space545.cells())
    assert phi.shape == (100, 52)
    # Full column rank: the 52 parameters are identifiable from the full grid.
    assert np.linalg.matrix_rank(phi) == 52


def test_mains_only_encoder(space545):
    enc = Encoder(space545, interactions=1)
    assert enc.p == 12  # 1 + 4 + 3 + 4
