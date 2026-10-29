"""M2 gate tests: floor respected; arcsine round-trip."""

import numpy as np
import pytest

from strands_evals.optimization.bayesbench import noise


def test_plugin_variance_basic():
    # Agresti-Coull smoothing: p~ = (m p + 2)/(m + 4).
    p_smooth = (50 * 0.5 + 2) / 54
    assert noise.plugin_variance(0.5, 50) == pytest.approx(p_smooth * (1 - p_smooth) / 50)


def test_floor_and_smoothing_at_boundaries():
    # p_hat of 0 or 1 no longer gives zero raw variance (AC smoothing pulls
    # toward 1/2), and the floor still holds where smoothing dips below it.
    assert noise.plugin_variance(0.0, 50) == pytest.approx((2 / 54) * (52 / 54) / 50)
    assert noise.plugin_variance(0.0, 50) > noise.SIGMA_MIN_SQ
    # Very large m at extreme p: smoothed variance can dip under the floor.
    assert noise.plugin_variance(0.0, 5000) == noise.SIGMA_MIN_SQ


def test_candidate_variance_clips_prediction():
    # A predicted mean outside [0, 1] (linear model can produce it) is clipped.
    assert noise.candidate_variance(-0.3, m=50) == noise.plugin_variance(0.05, 50)
    assert noise.candidate_variance(1.7, m=50) == noise.plugin_variance(0.95, 50)
    assert noise.candidate_variance(0.5, m=50) == noise.plugin_variance(0.5, 50)


def test_arcsine_round_trip():
    p = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(noise.arcsine_inverse(noise.arcsine(p)), p, atol=1e-12)


def test_arcsine_variance():
    assert noise.arcsine_variance(50) == pytest.approx(1 / 200)
    with pytest.raises(ValueError):
        noise.arcsine_variance(0)


def test_plugin_rejects_bad_m():
    with pytest.raises(ValueError):
        noise.plugin_variance(0.5, 0)
