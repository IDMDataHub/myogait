"""Signal processing filters for pose landmark trajectories.

This module re-exports filter functions from normalize.py for backward
compatibility. New code should import directly from normalize.py or
use the normalize() orchestrator.
"""

from .normalize import (
    NORMALIZE_STEPS as FILTER_MAP,
)

FILTER_OPTIONS = list(FILTER_MAP.keys())


def apply_filters_pipeline(df, filter_configs, framerate=30.0):
    """Apply a sequence of filters (backward-compatible wrapper)."""
    result = df.copy()
    for config in filter_configs:
        ftype = config.get("type", "")
        params = config.get("params", {})
        if not ftype:
            continue
        func = FILTER_MAP.get(ftype)
        if func is None:
            continue
        if ftype == "butterworth":
            params.setdefault("fs", framerate)
        result = func(result, **params)
    return result
