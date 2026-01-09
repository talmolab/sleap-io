"""This module exposes all high level APIs for sleap-io."""

import os
import sys

import lazy_loader as lazy

# Version is lightweight, keep it eager
from sleap_io.version import __version__

# Lazy load everything else using lazy_loader
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["io", "model", "codecs"],
    submod_attrs={
        # I/O functions from sleap_io.io.main
        "io.main": [
            "load_alphatracker",
            "load_coco",
            "load_dlc",
            "load_file",
            "load_jabs",
            "load_labels_set",
            "load_labelstudio",
            "load_leap",
            "load_nwb",
            "load_skeleton",
            "load_slp",
            "load_ultralytics",
            "load_video",
            "save_coco",
            "save_file",
            "save_jabs",
            "save_labelstudio",
            "save_nwb",
            "save_skeleton",
            "save_slp",
            "save_ultralytics",
            "save_video",
        ],
        # Video reading functions from sleap_io.io.video_reading
        "io.video_reading": [
            "VideoBackend",
            "get_default_image_plugin",
            "get_default_video_plugin",
            "set_default_image_plugin",
            "set_default_video_plugin",
            "get_available_video_backends",
            "get_available_image_backends",
            "get_installation_instructions",
        ],
        # Video writing from sleap_io.io.video_writing
        "io.video_writing": ["VideoWriter"],
        # Rendering from sleap_io.rendering
        "rendering.core": ["render_video", "render_image"],
        "rendering.colors": ["get_palette"],
        "rendering.callbacks": ["RenderContext", "InstanceContext"],
        # Model classes from sleap_io.model.*
        "model.camera": [
            "Camera",
            "CameraGroup",
            "FrameGroup",
            "InstanceGroup",
            "RecordingSession",
        ],
        "model.instance": [
            "Instance",
            "PredictedInstance",
            "Track",
        ],
        "model.labeled_frame": ["LabeledFrame"],
        "model.labels": ["Labels"],
        "model.labels_set": ["LabelsSet"],
        "model.skeleton": ["Edge", "Node", "Skeleton", "Symmetry"],
        "model.suggestions": ["SuggestionFrame"],
        "model.video": ["Video"],
    },
)

# Add __version__ to __all__ (it's not in lazy_loader's __all__)
__all__ = ["__version__"] + __all__

# Force eager imports when EAGER_IMPORT=1 (used for testing and docs)
# The EAGER_IMPORT environment variable is set by:
# - tests/conftest.py (for pytest)
# - .github/workflows/docs.yml (for docs build)
# This ensures griffe/mkdocstrings can find all documented symbols
#
# Also auto-detect mkdocs runs (mkdocs imports griffe which imports us)
_is_mkdocs = "mkdocs" in sys.modules or "griffe" in sys.modules
if os.getenv("EAGER_IMPORT") or _is_mkdocs:
    # Trigger all lazy imports and add them to __dict__ for griffe/mkdocstrings
    _current_module = sys.modules[__name__]
    for _attr in list(__all__):
        if _attr != "__version__":
            try:
                # Get the attribute (triggers lazy import)
                _obj = __getattr__(_attr)
                # Add to module __dict__ so griffe can find it
                setattr(_current_module, _attr, _obj)
            except Exception:
                pass  # Ignore any import errors
    del _attr, _obj, _current_module
