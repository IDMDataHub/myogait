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
    EXTRACTORS["sapiens2-quick"] = "myogait.models.sapiens2.Sapiens2QuickExtractor"
    EXTRACTORS["sapiens2-mid"] = "myogait.models.sapiens2.Sapiens2MidExtractor"
    EXTRACTORS["sapiens2-top"] = "myogait.models.sapiens2.Sapiens2TopExtractor"
    EXTRACTORS["sapiens2-ultra"] = "myogait.models.sapiens2.Sapiens2UltraExtractor"
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
        from ..exceptions import MissingDependencyError
        raise MissingDependencyError(
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


# Import-name requirements per backend (importlib module names, not pip
# names).  A model is "available" when every listed module is findable.
# Secondary, non-PyPI requirements (e.g. Meta's ``sapiens`` package for
# the Sapiens 2 SafeTensors loader) are included so that a "successful"
# ``pip install myogait[sapiens2]`` alone does not report as ready.
_MODEL_REQUIREMENTS = {
    "mediapipe": ["mediapipe"],
    "yolo": ["ultralytics"],
    "sapiens-quick": ["torch", "huggingface_hub"],
    "sapiens-mid": ["torch", "huggingface_hub"],
    "sapiens-top": ["torch", "huggingface_hub"],
    "sapiens2-quick": ["torch", "safetensors", "huggingface_hub"],
    "sapiens2-mid": ["torch", "safetensors", "huggingface_hub"],
    "sapiens2-top": ["torch", "safetensors", "huggingface_hub"],
    "sapiens2-ultra": ["torch", "safetensors", "huggingface_hub"],
    "hrnet": ["torch"],
    "mmpose": ["mmpose", "mmdet"],
    "vitpose": ["transformers", "torch"],
    "vitpose-large": ["transformers", "torch"],
    "vitpose-huge": ["transformers", "torch"],
    "rtmw": ["rtmlib", "onnxruntime"],
    "openpose": ["cv2"],
    "detectron2": ["detectron2", "torch"],
    "alphapose": ["torch", "torchvision", "ultralytics"],
}


def available_models() -> dict:
    """Report which pose backends are installed — without importing them.

    Returns ``{model_name: bool}`` for every registered model, using
    :func:`importlib.util.find_spec` on each backend's required import
    names.  Unlike calling :func:`get_extractor` speculatively, this is
    non-destructive: nothing heavy is imported, no GPU state is touched,
    and no ImportError is raised — suitable for a UI to grey out
    unavailable options.

    Note: ``find_spec`` proves the module is *findable*, not that it
    imports cleanly (a broken install can still fail at import time).
    """
    import importlib.util

    _register_lazy()
    out = {}
    for name in EXTRACTORS:
        reqs = _MODEL_REQUIREMENTS.get(name, [])
        ok = True
        for mod in reqs:
            try:
                if importlib.util.find_spec(mod) is None:
                    ok = False
                    break
            except (ImportError, ValueError):
                ok = False
                break
        out[name] = ok
    return out


__all__ = ["get_extractor", "list_models", "BasePoseExtractor"]
