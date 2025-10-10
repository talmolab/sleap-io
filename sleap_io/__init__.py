"""This module exposes all high level APIs for sleap-io."""

from sleap_io.io.main import (
    load_alphatracker,
    load_coco,
    load_dlc,
    load_file,
    load_jabs,
    load_labels_set,
    load_labelstudio,
    load_leap,
    load_nwb,
    load_skeleton,
    load_slp,
    load_ultralytics,
    load_video,
    save_file,
    save_jabs,
    save_labelstudio,
    save_nwb,
    save_skeleton,
    save_slp,
    save_ultralytics,
    save_video,
)
from sleap_io.io.video_reading import (
    VideoBackend,
    get_default_image_plugin,
    get_default_video_plugin,
    set_default_image_plugin,
    set_default_video_plugin,
)
from sleap_io.io.video_writing import VideoWriter
from sleap_io.model.camera import (
    Camera,
    CameraGroup,
    FrameGroup,
    InstanceGroup,
    RecordingSession,
)
from sleap_io.model.instance import (
    Instance,
    PredictedInstance,
    Track,
)
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.labels_set import LabelsSet
from sleap_io.model.skeleton import Edge, Node, Skeleton, Symmetry
from sleap_io.model.suggestions import SuggestionFrame
from sleap_io.model.video import Video
from sleap_io.version import __version__

__all__ = [
    "__version__",
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
    "save_file",
    "save_jabs",
    "save_labelstudio",
    "save_nwb",
    "save_skeleton",
    "save_slp",
    "save_ultralytics",
    "save_video",
    "get_default_image_plugin",
    "get_default_video_plugin",
    "set_default_image_plugin",
    "set_default_video_plugin",
    "VideoBackend",
    "VideoWriter",
    "Camera",
    "CameraGroup",
    "FrameGroup",
    "InstanceGroup",
    "RecordingSession",
    "Instance",
    "PredictedInstance",
    "Track",
    "LabeledFrame",
    "Labels",
    "LabelsSet",
    "Edge",
    "Node",
    "Skeleton",
    "Symmetry",
    "SuggestionFrame",
    "Video",
]
