"""Tests for letterbox resize and Sapiens padding-aware heatmap decoding."""

import numpy as np


def test_letterbox_preserves_aspect_ratio():
    """letterbox_resize should scale uniformly and center the content."""
    from myogait.models.base import letterbox_resize

    # Tall image (portrait) into a 3:4 canvas (768x1024)
    img = np.zeros((400, 200, 3), dtype=np.uint8)  # H=400, W=200, aspect 1:2
    canvas, pl, pt, cw, ch = letterbox_resize(img, 768, 1024)

    assert canvas.shape == (1024, 768, 3)
    # Scale limited by width: 768/200=3.84 vs 1024/400=2.56 → scale=2.56
    assert ch == 1024  # height fills target
    assert cw == 512   # 200 * 2.56 = 512
    assert pt == 0     # no vertical padding (height fills)
    assert pl == 128   # (768 - 512) // 2 = 128


def test_letterbox_wide_image():
    """A wide image should get top/bottom padding."""
    from myogait.models.base import letterbox_resize

    img = np.zeros((100, 768, 3), dtype=np.uint8)
    canvas, pl, pt, cw, ch = letterbox_resize(img, 768, 1024)

    assert canvas.shape == (1024, 768, 3)
    # Scale limited by width: 768/768=1.0 vs 1024/100=10.24 → scale=1.0
    assert cw == 768
    assert ch == 100
    assert pl == 0
    assert pt == 462  # (1024 - 100) // 2


def test_letterbox_exact_ratio():
    """Image already at target ratio should produce zero padding."""
    from myogait.models.base import letterbox_resize

    # 3:4 ratio like the Sapiens target
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    canvas, pl, pt, cw, ch = letterbox_resize(img, 768, 1024)

    assert canvas.shape == (1024, 768, 3)
    assert pl == 0
    assert pt == 0
    assert cw == 768
    assert ch == 1024


def test_letterbox_content_placed_correctly():
    """The content pixels should appear in the canvas at the right offset."""
    from myogait.models.base import letterbox_resize

    img = np.full((100, 200, 3), 42, dtype=np.uint8)
    canvas, pl, pt, cw, ch = letterbox_resize(img, 768, 1024)

    # Padding area should be black (0)
    assert canvas[0, 0, 0] == 0  # top-left corner is padding
    # Content area should be non-zero
    mid_y = pt + ch // 2
    mid_x = pl + cw // 2
    assert canvas[mid_y, mid_x, 0] == 42


def test_heatmap_to_coco_no_padding():
    """Without padding, coordinates should match the old behaviour."""
    from myogait.models.sapiens import _heatmaps_to_coco, _INPUT_W, _INPUT_H
    from myogait.constants import GOLIATH_TO_COCO

    hm_h, hm_w = 256, 192
    n_kp = max(GOLIATH_TO_COCO.keys()) + 1
    heatmaps = np.zeros((n_kp, hm_h, hm_w), dtype=np.float32)

    # Place a peak at the center of the heatmap for the first mapping
    goliath_idx = list(GOLIATH_TO_COCO.keys())[0]
    coco_idx = GOLIATH_TO_COCO[goliath_idx]
    heatmaps[goliath_idx, hm_h // 2, hm_w // 2] = 1.0

    # No padding: content fills the entire canvas
    pad_info = (0, 0, _INPUT_W, _INPUT_H)
    lm = _heatmaps_to_coco(heatmaps, pad_info)

    # Center should be ~0.5, 0.5
    assert abs(lm[coco_idx, 0] - 0.5) < 0.01
    assert abs(lm[coco_idx, 1] - 0.5) < 0.01
    assert lm[coco_idx, 2] == 1.0


def test_heatmap_to_coco_with_padding():
    """With left padding, x coordinates should shift appropriately."""
    from myogait.models.sapiens import _heatmaps_to_coco, _INPUT_H
    from myogait.constants import GOLIATH_TO_COCO

    hm_h, hm_w = 256, 192
    n_kp = max(GOLIATH_TO_COCO.keys()) + 1
    heatmaps = np.zeros((n_kp, hm_h, hm_w), dtype=np.float32)

    goliath_idx = list(GOLIATH_TO_COCO.keys())[0]
    coco_idx = GOLIATH_TO_COCO[goliath_idx]

    # Place peak at heatmap center (hm_w//2, hm_h//2)
    heatmaps[goliath_idx, hm_h // 2, hm_w // 2] = 1.0

    # Simulate padding: content is 512 wide, centered in 768 → pad_left=128
    content_w = 512
    pad_left = 128
    pad_info = (pad_left, 0, content_w, _INPUT_H)

    lm = _heatmaps_to_coco(heatmaps, pad_info)

    # hm center → input x = 96/192 * 768 = 384 → content x = (384-128)/512 = 0.5
    assert abs(lm[coco_idx, 0] - 0.5) < 0.01
    # y unchanged (no vertical padding)
    assert abs(lm[coco_idx, 1] - 0.5) < 0.01


def test_heatmap_to_all_with_padding():
    """_heatmaps_to_all should apply the same letterbox correction."""
    from myogait.models.sapiens import _heatmaps_to_all, _INPUT_H

    hm_h, hm_w = 256, 192
    heatmaps = np.zeros((2, hm_h, hm_w), dtype=np.float32)

    # Peak at top-left corner of the content region
    # Content starts at pad_left=128 in a 768-wide canvas
    # So input_x = 128 → hm_x = 128/768 * 192 = 32
    heatmaps[0, 0, 32] = 1.0

    pad_info = (128, 0, 512, _INPUT_H)
    lm = _heatmaps_to_all(heatmaps, pad_info)

    # Should map to x≈0.0 (left edge of content), y≈0.0 (top)
    assert lm[0, 0] < 0.02
    assert lm[0, 1] < 0.01
