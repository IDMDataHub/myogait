"""Experimental video degradation utilities for robustness benchmarks.

These helpers are intentionally opt-in and disabled by default.
They are designed for AIM benchmark scenarios where controlled
input degradation is required before pose extraction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


VIDEO_DEGRADATION_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "target_fps": None,
    "downscale": 1.0,
    "contrast": 1.0,
    "aspect_ratio": 1.0,
    "perspective_x": 0.0,
    "perspective_y": 0.0,
}


def build_video_degradation_config(config: Optional[dict] = None) -> dict:
    """Build and validate a video degradation config."""
    cfg = dict(VIDEO_DEGRADATION_DEFAULTS)
    if config:
        cfg.update(config)

    target_fps = cfg.get("target_fps")
    if target_fps is not None and float(target_fps) <= 0:
        raise ValueError("experimental.target_fps must be > 0 or None")

    downscale = float(cfg.get("downscale", 1.0))
    if not (0 < downscale <= 1.0):
        raise ValueError("experimental.downscale must be in (0, 1]")

    contrast = float(cfg.get("contrast", 1.0))
    if not (0 < contrast <= 1.0):
        raise ValueError("experimental.contrast must be in (0, 1]")

    aspect_ratio = float(cfg.get("aspect_ratio", 1.0))
    if not (0 < aspect_ratio <= 2.0):
        raise ValueError("experimental.aspect_ratio must be in (0, 2]")

    perspective_x = float(cfg.get("perspective_x", 0.0))
    perspective_y = float(cfg.get("perspective_y", 0.0))
    if abs(perspective_x) > 1.0:
        raise ValueError("experimental.perspective_x must be in [-1, 1]")
    if abs(perspective_y) > 1.0:
        raise ValueError("experimental.perspective_y must be in [-1, 1]")

    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["target_fps"] = None if target_fps is None else float(target_fps)
    cfg["downscale"] = downscale
    cfg["contrast"] = contrast
    cfg["aspect_ratio"] = aspect_ratio
    cfg["perspective_x"] = perspective_x
    cfg["perspective_y"] = perspective_y
    return cfg


def is_video_degradation_active(config: dict) -> bool:
    """Return True when any degradation parameter is active."""
    if config.get("enabled", False):
        return True
    return any(
        [
            config.get("target_fps") is not None,
            float(config.get("downscale", 1.0)) != 1.0,
            float(config.get("contrast", 1.0)) != 1.0,
            float(config.get("aspect_ratio", 1.0)) != 1.0,
            float(config.get("perspective_x", 0.0)) != 0.0,
            float(config.get("perspective_y", 0.0)) != 0.0,
        ]
    )


def compute_fps_sampling(original_fps: float, target_fps: Optional[float]) -> Tuple[int, float]:
    """Compute frame stride and effective FPS after sampling."""
    if original_fps <= 0 or target_fps is None or target_fps >= original_fps:
        return 1, float(original_fps)

    stride = max(1, int(round(original_fps / target_fps)))
    effective_fps = float(original_fps / stride) if stride > 0 else float(original_fps)
    return stride, effective_fps


def degraded_resolution(width: int, height: int, config: dict) -> Tuple[int, int]:
    """Return output resolution after downscale and aspect ratio transforms."""
    scale = float(config.get("downscale", 1.0))
    aspect_ratio = float(config.get("aspect_ratio", 1.0))
    out_w = max(32, int(round(width * scale * aspect_ratio)))
    out_h = max(32, int(round(height * scale)))
    return out_w, out_h


def apply_video_degradation(frame_bgr: np.ndarray, config: dict) -> np.ndarray:
    """Apply optional degradation transforms to a BGR frame."""
    if frame_bgr is None:
        return frame_bgr

    out = frame_bgr
    h, w = out.shape[:2]
    out_w, out_h = degraded_resolution(w, h, config)

    if out_w != w or out_h != h:
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_AREA)

    contrast = float(config.get("contrast", 1.0))
    if contrast < 1.0:
        arr = out.astype(np.float32)
        arr = (arr - 127.5) * contrast + 127.5
        out = np.clip(arr, 0, 255).astype(np.uint8)

    px = float(config.get("perspective_x", 0.0))
    py = float(config.get("perspective_y", 0.0))
    if px != 0.0 or py != 0.0:
        hh, ww = out.shape[:2]
        x_shift = px * 0.22 * ww
        y_shift = py * 0.22 * hh
        src = np.float32(
            [[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]]
        )
        dst = np.float32(
            [
                [0 + x_shift, 0 + y_shift],
                [ww - 1 - x_shift, 0 + y_shift],
                [ww - 1 + x_shift, hh - 1 - y_shift],
                [0 - x_shift, hh - 1 - y_shift],
            ]
        )
        m = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(
            out,
            m,
            (ww, hh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return out
