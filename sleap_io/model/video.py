"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations
from attrs import define
from typing import Any, Tuple, Optional


@define(auto_attribs=True)
class Video:
    """`Video` class used by sleap to represent videos and data associated with them.

    This class is used to store information regarding a video and its components.
    It is used to store the video's `filename`, `shape`, and the video's `backend`.

    Args:
        filename: The filename of the video.
        shape: The shape of the video. In the format: (frames, height, width, channels).
        backend: An object that implements the basic methods for reading and
            manipulating frames of a specific video type.

    """

    filename: str
    shape: Optional[Tuple[int, int, int, int]] = None
    backend: Any = None
