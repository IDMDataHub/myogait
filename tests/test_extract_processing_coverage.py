"""Additional useful coverage tests for extract processing helpers."""

import numpy as np


def test_detect_multi_person_confidence_drop_branch():
    from myogait.extract import detect_multi_person

    frames = []
    for i, conf in enumerate([0.9, 0.2, 0.9]):
        frames.append(
            {
                "frame_idx": i,
                "confidence": conf,
                "landmarks": {
                    "LEFT_HIP": {"x": 0.5, "y": 0.5},
                    "RIGHT_HIP": {"x": 0.55, "y": 0.5},
                    "LEFT_SHOULDER": {"x": 0.5, "y": 0.3},
                    "RIGHT_SHOULDER": {"x": 0.55, "y": 0.3},
                    "NOSE": {"x": 0.525, "y": 0.2},
                },
            }
        )
    data = {"frames": frames, "extraction": {}}
    out = detect_multi_person(data)
    assert out["extraction"]["multi_person_warning"] is True
    assert 1 in out["extraction"]["suspicious_frames"]


def test_swap_auxiliary_lr_l_r_prefix_convention():
    from myogait.extract import _swap_auxiliary_lr

    names = ["l_eye", "r_eye", "nose"]
    aux = np.array([[1, 0, 1], [2, 0, 1], [3, 0, 1]], dtype=float)
    swapped = _swap_auxiliary_lr(aux, names)
    assert np.array_equal(swapped[0], aux[1])
    assert np.array_equal(swapped[1], aux[0])
    assert np.array_equal(swapped[2], aux[2])


def test_correct_label_inversions_small_input_noop():
    from myogait.extract import _correct_label_inversions

    frames = [np.full((33, 3), np.nan, dtype=float)]
    out, mask = _correct_label_inversions(frames)
    assert out == frames
    assert mask == [False]
