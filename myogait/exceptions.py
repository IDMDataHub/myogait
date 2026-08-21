"""Dedicated exception hierarchy for myogait.

Every class also inherits the builtin exception that the code
historically raised (``ValueError``, ``ImportError``, ...), so existing
``except ValueError`` handlers keep working — adopting these types is
non-breaking.  Callers that want finer granularity (a UI distinguishing
"unreadable video" from "no person detected" without parsing English
messages) can catch the specific types; ``except MyogaitError`` catches
everything raised deliberately by the library.

Adoption is incremental: new code should raise these; existing raise
sites are migrated opportunistically.
"""

__all__ = [
    "MyogaitError",
    "ExtractionError",
    "NoPersonDetectedError",
    "UnreadableVideoError",
    "InvalidPivotError",
    "MissingDependencyError",
    "InvalidC3DError",
]


class MyogaitError(Exception):
    """Root of all deliberate myogait errors."""


class ExtractionError(MyogaitError, RuntimeError):
    """Pose extraction failed (backend error, decoding failure, ...)."""


class UnreadableVideoError(ExtractionError, ValueError):
    """The video file cannot be opened or decoded."""


class NoPersonDetectedError(ExtractionError, ValueError):
    """The pose backend ran but found no usable person in the video."""


class InvalidPivotError(MyogaitError, ValueError):
    """A pivot dict is missing required structure (frames, angles, ...)."""


class MissingDependencyError(MyogaitError, ImportError):
    """An optional backend/feature dependency is not installed."""


class InvalidC3DError(MyogaitError, ValueError):
    """A C3D file cannot be used (unreadable, no matching markers, ...)."""
