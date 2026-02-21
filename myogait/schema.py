"""JSON pivot format for myogait.

The pivot JSON is the central data structure flowing through all
processing steps: extract -> normalize -> angles -> events -> cycles.

Functions
---------
create_empty
    Create an empty pivot JSON structure.
save_json
    Save pivot JSON to file with numpy type conversion.
load_json
    Load and validate a pivot JSON file.
set_subject
    Set subject metadata in the pivot JSON.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def create_empty(
    video_path: str = "",
    fps: float = 30.0,
    width: int = 0,
    height: int = 0,
    n_frames: int = 0,
) -> dict:
    """Create an empty pivot JSON structure.

    Parameters
    ----------
    video_path : str
        Source video path.
    fps : float
        Frame rate in Hz (default 30.0).
    width : int
        Video width in pixels.
    height : int
        Video height in pixels.
    n_frames : int
        Total number of frames.

    Returns
    -------
    dict
        Empty pivot dictionary ready to be populated.
    """
    from . import __version__
    duration = n_frames / fps if fps > 0 else 0.0
    return {
        "myogait_version": __version__,
        "meta": {
            "source": "video",
            "video_path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "duration_s": round(duration, 3),
        },
        "subject": None,
        "extraction": None,
        "frames": [],
        "normalization": None,
        "angles": None,
        "events": None,
    }


def set_subject(
    data: dict,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    height_m: Optional[float] = None,
    weight_kg: Optional[float] = None,
    pathology: Optional[str] = None,
    notes: Optional[str] = None,
    **extra,
) -> dict:
    """Set subject metadata in the pivot JSON.

    Parameters
    ----------
    data : dict
        Pivot JSON dict.
    age : int, optional
        Subject age in years.
    sex : {'M', 'F', 'X'}, optional
        Biological sex.
    height_m : float, optional
        Height in meters (e.g. 1.75).
    weight_kg : float, optional
        Weight in kilograms.
    pathology : str, optional
        Primary diagnosis or condition.
    notes : str, optional
        Additional clinical notes.
    **extra
        Any additional metadata key-value pairs.

    Returns
    -------
    dict
        Modified *data* dict with ``subject`` field populated.
    """
    subject = {}
    if age is not None:
        subject["age"] = age
    if sex is not None:
        subject["sex"] = sex
    if height_m is not None:
        subject["height_m"] = height_m
    if weight_kg is not None:
        subject["weight_kg"] = weight_kg
    if pathology is not None:
        subject["pathology"] = pathology
    if notes is not None:
        subject["notes"] = notes
    subject.update(extra)

    data["subject"] = subject
    return data


def save_json(data: dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save pivot JSON to file.

    Automatically converts numpy types to Python builtins before
    serialization.

    Parameters
    ----------
    data : dict
        Pivot dictionary.
    path : str or Path
        Output file path. Parent directories are created if needed.
    indent : int, optional
        JSON indentation level (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = _convert_numpy(data)
    with open(path, "w") as f:
        json.dump(converted, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> dict:
    """Load and validate a pivot JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    dict
        Pivot dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON content is not a valid pivot format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be a dict")

    # Minimal validation: must have meta and frames
    if "meta" not in data:
        raise ValueError("Missing 'meta' key in JSON")
    if "frames" not in data:
        raise ValueError("Missing 'frames' key in JSON")

    return data
