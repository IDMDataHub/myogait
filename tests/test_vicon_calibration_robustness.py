"""Input-validation regressions for static HJC calibration."""

import numpy as np
import pytest

from myogait.vicon_calibration import calibrate_hjc_from_static


def _static_pelvis():
    return {
        "RIGHT_ASIS": np.array([0.0, 1000.0, 100.0]),
        "LEFT_ASIS": np.array([0.0, 1000.0, -100.0]),
        "RIGHT_PSIS": np.array([-200.0, 1000.0, 100.0]),
        "LEFT_PSIS": np.array([-200.0, 1000.0, -100.0]),
    }


def test_static_hjc_calibration_accepts_pre_averaged_marker_positions():
    calibration = calibrate_hjc_from_static(_static_pelvis())

    assert np.isfinite(calibration.right_local).all()
    assert np.isfinite(calibration.left_local).all()


def test_static_hjc_calibration_rejects_an_entirely_occluded_marker():
    static_markers = {
        name: np.tile(position, (3, 1))
        for name, position in _static_pelvis().items()
    }
    static_markers["RIGHT_ASIS"][:] = np.nan

    with pytest.raises(ValueError, match="RIGHT_ASIS.*no finite frames"):
        calibrate_hjc_from_static(static_markers)


def test_static_hjc_calibration_rejects_malformed_marker_data():
    static_markers = _static_pelvis()
    static_markers["LEFT_PSIS"] = np.ones((2, 2))

    with pytest.raises(ValueError, match="LEFT_PSIS.*shape"):
        calibrate_hjc_from_static(static_markers)
