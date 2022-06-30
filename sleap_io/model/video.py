"""Data model for Videos.

Videos are SLEAP data structures that store information regarding a video and its
components used in SLEAP.
"""

from __future__ import annotations
from attr import attrs
from attrs import define, field, Factory
from typing import Any, List, Optional, Tuple, Union, Dict
import h5py as h5
import numpy as np
import os


@define
class Video:
    """Video class used by sleap to represent videos and data associated with them.

    This class is used to store information regarding a video and its components.
    It is used to store the video's filename, shape, and the video's skeleton.

    Args:
        filename: The filename of the video.
        shape: The shape of the video.
        backend: An object that implements the basic methods for reading and
            manipulating frames of a specific video type.

    """

    filename: str
    shape: Tuple[int, int, int, int] = field(default=None)
    backend: Any = None
