"""Tests for myogait.autoconfig: recipe detection and run_auto."""

import copy

import myogait as mg
from myogait.autoconfig import (
    DEFAULT_RECIPE,
    OVERGROUND_RECIPE,
    detect_recipe,
    has_direction_reversal,
    has_static_start,
)
from conftest import make_walking_data


def _set_hip_x(data, xs):
    for frame, x in zip(data["frames"], xs):
        for key in ("LEFT_HIP", "RIGHT_HIP"):
            frame["landmarks"][key]["x"] = float(x)
    return data


def test_static_start_detected():
    data = make_walking_data(n_frames=60)
    n = len(data["frames"])
    _set_hip_x(data, [0.3] * 20 + [0.3 + 0.01 * i for i in range(n - 20)])
    assert has_static_start(data["frames"])
    recipe, reasons = detect_recipe(data)
    assert recipe == DEFAULT_RECIPE
    assert reasons[-1].startswith("clean standing-start")


def test_moving_start_uses_overground_recipe():
    data = make_walking_data(n_frames=60)
    _set_hip_x(data, [0.1 + 0.01 * i for i in range(len(data["frames"]))])
    assert not has_static_start(data["frames"])
    recipe, _ = detect_recipe(data)
    assert recipe["name"] == "overground"
    assert recipe["calibrate"] is False
    assert recipe["direction_filter"] is False


def test_there_and_back_filters_direction():
    data = make_walking_data(n_frames=60)
    xs = [0.1 + 0.025 * i for i in range(30)] + [0.85 - 0.025 * i for i in range(30)]
    _set_hip_x(data, xs)
    assert has_direction_reversal(data["frames"])
    recipe, reasons = detect_recipe(data)
    assert recipe["name"] == "overground"
    assert recipe["direction_filter"] is True
    assert any("return-pass" in r for r in reasons)


def test_c3d_source_uses_overground_recipe():
    data = make_walking_data(n_frames=60)
    data.setdefault("meta", {})["source"] = "c3d"
    recipe, reasons = detect_recipe(data)
    assert recipe["name"] == "overground"
    assert reasons[0].startswith("marker (C3D)")


def test_detect_recipe_returns_copies():
    data = make_walking_data(n_frames=60)
    recipe, _ = detect_recipe(data)
    recipe["calibrate"] = "mutated"
    assert DEFAULT_RECIPE["calibrate"] is True
    assert OVERGROUND_RECIPE["calibrate"] is False


def test_detect_recipe_tolerates_missing_landmarks():
    data = {"frames": [{"landmarks": None}, {}, "bad"]}
    recipe, _ = detect_recipe(data)
    assert recipe["name"] == "overground"


def test_run_auto_on_dict_and_json(tmp_path):
    data = make_walking_data(n_frames=240)
    result = mg.run_auto(copy.deepcopy(data), show_progress=False)
    assert set(result) >= {"data", "cycles", "stats", "quality",
                           "source_type", "recipe", "reasons"}
    assert result["source_type"] == "json"
    assert result["recipe"]["name"] in {"default", "overground"}
    assert isinstance(result["cycles"].get("cycles", []), list)

    path = tmp_path / "walk.myogait.json"
    mg.save_json(data, str(path))
    from_file = mg.run_auto(str(path), show_progress=False)
    assert from_file["recipe"]["name"] == result["recipe"]["name"]


def test_filter_cycles_by_direction_is_public():
    from myogait.pipeline import _filter_cycles_by_direction
    assert mg.filter_cycles_by_direction is _filter_cycles_by_direction


def _cycle(side, hip_sign=1.0):
    import numpy as np
    pct = np.linspace(0, 1, 101)
    hip = hip_sign * 25.0 * np.cos(2 * np.pi * pct)           # flexed at contact
    knee = 30.0 + 25.0 * np.sin(2 * np.pi * (pct - 0.5)) ** 2
    return {"side": side, "angles_normalized": {"hip": list(hip), "knee": list(knee)}}


def test_enforce_flexion_positive_flips_inverted_hip_side():
    cycles = [_cycle("left", -1.0), _cycle("left", -1.0), _cycle("right", 1.0)]
    mg.enforce_flexion_positive(cycles)
    for c in cycles:
        hip = c["angles_normalized"]["hip"]
        assert hip[0] > 0 > hip[50]


def test_run_steps_enforces_sign_without_direction_filter():
    from myogait.pipeline import _run_steps
    data = make_walking_data(n_frames=240)
    res = _run_steps(copy.deepcopy(data), "json", direction_filter=False, analyze=False)
    for c in res["cycles"].get("cycles", []):
        hip = c["angles_normalized"].get("hip")
        if hip is not None:
            import numpy as np
            assert np.mean(hip[:15]) >= np.mean(hip[40:60])
