"""Coverage-focused tests for extract helper paths."""

import numpy as np
import pytest


def test_goliath_to_mediapipe_maps_and_builds_foot_index():
    from myogait.extract import _goliath_to_mediapipe
    from myogait.constants import MP_NAME_TO_INDEX

    g308 = np.full((308, 3), np.nan, dtype=float)
    # Directly mapped point (LEFT_HIP via GOLIATH_TO_MP)
    g308[9] = [0.4, 0.5, 0.9]
    # Left foot index midpoint from big/small toe (15, 16)
    g308[15] = [0.2, 0.8, 0.7]
    g308[16] = [0.4, 0.9, 0.6]

    mp = _goliath_to_mediapipe(g308)
    assert mp.shape == (33, 3)
    assert np.isclose(mp[MP_NAME_TO_INDEX["LEFT_HIP"], 0], 0.4)
    lf = mp[MP_NAME_TO_INDEX["LEFT_FOOT_INDEX"]]
    assert np.isclose(lf[0], 0.3)
    assert np.isclose(lf[1], 0.85)
    assert np.isclose(lf[2], 0.6)


def test_enrich_foot_landmarks_from_goliath308():
    from myogait.extract import _enrich_foot_landmarks

    frame = {
        "landmarks": {"LEFT_ANKLE": {"x": 0.2, "y": 0.8, "visibility": 1.0}},
        "goliath308": [[np.nan, np.nan, np.nan] for _ in range(308)],
    }
    # left big/small toe + heel
    frame["goliath308"][15] = [0.2, 0.8, 0.9]
    frame["goliath308"][16] = [0.4, 0.9, 0.8]
    frame["goliath308"][17] = [0.3, 0.85, 0.7]

    _enrich_foot_landmarks(frame)

    assert frame["foot_landmarks_source"] == "detected"
    assert "LEFT_BIG_TOE" in frame["landmarks"]
    assert "LEFT_HEEL" in frame["landmarks"]
    assert "LEFT_FOOT_INDEX" in frame["landmarks"]
    assert frame["landmarks"]["LEFT_FOOT_INDEX"]["x"] == pytest.approx(0.3)


def test_enrich_foot_landmarks_from_wholebody133():
    from myogait.extract import _enrich_foot_landmarks

    frame = {
        "landmarks": {"LEFT_ANKLE": {"x": 0.3, "y": 0.8, "visibility": 1.0}},
        "wholebody133": [[np.nan, np.nan, np.nan] for _ in range(133)],
    }
    # RTMW foot indices: left big toe, small toe, heel
    frame["wholebody133"][17] = [0.3, 0.8, 0.9]
    frame["wholebody133"][18] = [0.4, 0.82, 0.8]
    frame["wholebody133"][19] = [0.35, 0.84, 0.7]

    _enrich_foot_landmarks(frame)

    assert frame["foot_landmarks_source"] == "detected"
    assert "LEFT_BIG_TOE" in frame["landmarks"]
    assert "LEFT_SMALL_TOE" in frame["landmarks"]
    assert "LEFT_HEEL" in frame["landmarks"]


def test_flip_auxiliary_mirrors_and_swaps_pairs():
    from myogait.extract import _flip_auxiliary

    names = ["left_eye", "right_eye", "nose"]
    aux = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.9, 0.2, 1.0],
            [0.5, 0.3, 1.0],
        ],
        dtype=float,
    )

    flipped = _flip_auxiliary(aux, names)
    # left/right swapped after mirror
    assert flipped[0, 0] == pytest.approx(1.0 - aux[1, 0])
    assert flipped[1, 0] == pytest.approx(1.0 - aux[0, 0])
    # center point only mirrored
    assert flipped[2, 0] == pytest.approx(0.5)


def test_extract_missing_file_raises_file_not_found():
    from myogait.extract import extract

    with pytest.raises(FileNotFoundError):
        extract("/definitely/not/here.mp4")
