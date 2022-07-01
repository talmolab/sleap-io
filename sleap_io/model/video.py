<<<<<<< HEAD
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
        backend: A backend is an object that implements the following basic
            required methods and properties
=======
"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations
from attrs import define
from typing import Any, Tuple


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
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472

    """

    filename: str
<<<<<<< HEAD
    shape: Tuple[int, int, int, int] = field(default=None)
=======
    shape: Tuple[int, int, int, int]
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    backend: Any = None
