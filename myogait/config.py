"""Pipeline configuration management.

Supports JSON and YAML config files for reproducible analyses.
Configuration is merged against ``DEFAULT_CONFIG`` so partial
overrides work seamlessly.

Functions
---------
load_config
    Load pipeline config from a JSON or YAML file.
save_config
    Save pipeline config to a JSON or YAML file.

Attributes
----------
DEFAULT_CONFIG : dict
    Default configuration values for all pipeline stages.
"""

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "extract": {
        "model": "mediapipe",
        "max_frames": None,
        "flip_if_right": True,
        "correct_inversions": True,
        "experimental": {
            "enabled": False,
            "target_fps": None,
            "downscale": 1.0,
            "contrast": 1.0,
            "aspect_ratio": 1.0,
            "perspective_x": 0.0,
            "perspective_y": 0.0,
        },
    },
    "normalize": {
        "filters": ["butterworth"],
        "butterworth_cutoff": 4.0,
        "butterworth_order": 2,
        "center": False,
        "align": False,
        "correct_limbs": False,
    },
    "angles": {
        "method": "sagittal_vertical_axis",
        "correction_factor": 0.8,
        "calibrate": True,
        "calibration_frames": 30,
    },
    "events": {
        "method": "zeni",
        "min_cycle_duration": 0.4,
        "cutoff_freq": 6.0,
    },
    "cycles": {
        "n_points": 101,
        "min_duration": 0.4,
        "max_duration": 2.5,
    },
    "subject": {
        "age": None,
        "sex": None,
        "height_m": None,
        "weight_kg": None,
        "pathology": None,
    },
    "experimental_vicon": {
        "enabled": False,
        "scope": "AIM benchmark only",
        "trial_dir": None,
        "vicon_fps": 200.0,
        "max_lag_seconds": 10.0,
    },
}


def load_config(path: Union[str, Path]) -> dict:
    """Load pipeline config from a JSON or YAML file.

    The loaded configuration is merged against ``DEFAULT_CONFIG``
    so partial overrides work correctly.

    Parameters
    ----------
    path : str or Path
        Path to config file (``.json`` or ``.yaml``/``.yml``).

    Returns
    -------
    dict
        Merged configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ImportError
        If YAML is requested but ``pyyaml`` is not installed.
    ValueError
        If the file content is not a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    else:
        with open(path) as f:
            cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    # Merge with defaults
    merged = _deep_merge(DEFAULT_CONFIG.copy(), cfg)
    logger.info(f"Loaded config from {path}")
    return merged


def save_config(config: dict, path: Union[str, Path]) -> str:
    """Save pipeline config to a JSON or YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : str or Path
        Output file path (``.json`` or ``.yaml``/``.yml``).

    Returns
    -------
    str
        Path to the saved file.

    Raises
    ------
    ImportError
        If YAML is requested but ``pyyaml`` is not installed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    else:
        with open(path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved config to {path}")
    return str(path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
