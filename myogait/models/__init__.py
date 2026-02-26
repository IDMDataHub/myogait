"""Pose estimation model registry."""

from .base import BasePoseExtractor

EXTRACTORS = {}

def _register_lazy():
    """Register extractors with lazy imports to avoid heavy dependencies at import time."""
    global EXTRACTORS
    if EXTRACTORS:
        return EXTRACTORS

    EXTRACTORS["mediapipe"] = "myogait.models.mediapipe.MediaPipePoseExtractor"
    EXTRACTORS["yolo"] = "myogait.models.yolo.YOLOPoseExtractor"
    EXTRACTORS["sapiens-quick"] = "myogait.models.sapiens.SapiensQuickExtractor"
    EXTRACTORS["sapiens-mid"] = "myogait.models.sapiens.SapiensMidExtractor"
    EXTRACTORS["sapiens-top"] = "myogait.models.sapiens.SapiensTopExtractor"
    EXTRACTORS["hrnet"] = "myogait.models.hrnet.HRNETPoseExtractor"
    EXTRACTORS["mmpose"] = "myogait.models.mmpose.MMPosePoseExtractor"
    EXTRACTORS["vitpose"] = "myogait.models.vitpose.ViTPosePoseExtractor"
    EXTRACTORS["vitpose-large"] = "myogait.models.vitpose.ViTPosePoseExtractor"
    EXTRACTORS["vitpose-huge"] = "myogait.models.vitpose.ViTPosePoseExtractor"
    EXTRACTORS["rtmw"] = "myogait.models.rtmw.RTMWPoseExtractor"
    EXTRACTORS["openpose"] = "myogait.models.openpose.OpenPosePoseExtractor"
    EXTRACTORS["detectron2"] = "myogait.models.keypoint_rcnn.Detectron2PoseExtractor"
    EXTRACTORS["alphapose"] = "myogait.models.alphapose.AlphaPosePoseExtractor"
    return EXTRACTORS


def get_extractor(name: str, **kwargs) -> BasePoseExtractor:
    """Get a pose extractor by name.

    Args:
        name: Model name (mediapipe, yolo, sapiens-quick, sapiens-top, hrnet, mmpose)
        **kwargs: Passed to the extractor constructor

    Returns:
        Instantiated pose extractor

    Raises:
        ValueError: If model name is not recognized
        ImportError: If required dependencies are not installed
    """
    _register_lazy()

    if name not in EXTRACTORS:
        available = ", ".join(sorted(EXTRACTORS.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    class_path = EXTRACTORS[name]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Model '{name}' requires additional dependencies. "
            f"Install with: pip install myogait[{name.split('-')[0]}]\n"
            f"Original error: {e}"
        ) from e

    cls = getattr(module, class_name)

    # Pass model_size for ViTPose variants
    if name.startswith("vitpose") and name != "vitpose":
        size = name.replace("vitpose-", "")
        kwargs.setdefault("model_size", size)

    return cls(**kwargs)


def list_models():
    """List available model names."""
    _register_lazy()
    return sorted(EXTRACTORS.keys())


__all__ = ["get_extractor", "list_models", "BasePoseExtractor"]
