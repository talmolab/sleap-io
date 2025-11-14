"""This module exposes all high level APIs for sleap-io."""

import lazy_loader as lazy

# Version is lightweight, keep it eager
from sleap_io.version import __version__

# Lazy load everything else using lazy_loader
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["io", "model"],
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
        ],
        # Video writing from sleap_io.io.video_writing
        "io.video_writing": ["VideoWriter"],
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
