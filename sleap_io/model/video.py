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
    """
    The top-level interface to any Video data used by SLEAP.

    This class provides a common interface for various supported video data
    backends. It provides the bare minimum of properties and methods that
    any video data needs to support in order to function with other SLEAP
    components. This interface currently only supports reading of video
    data, there is no write support. Unless one is creating a new video
    backend, this class should be instantiated from its various class methods
    for different formats. For example: ::

       >>> video = Video.from_hdf5(filename="test.h5", dataset="box")
       >>> video = Video.from_media(filename="test.mp4")

    Or we can use auto-detection based on filename: ::

       >>> video = Video.from_filename(filename="test.mp4")

    Args:
        backend: A backend is an object that implements the following basic
            required methods and properties

        * Properties

            * :code:`frames`: The number of frames in the video
            * :code:`channels`: The number of channels in the video
              (e.g. 1 for grayscale, 3 for RGB)
            * :code:`width`: The width of each frame in pixels
            * :code:`height`: The height of each frame in pixels

        * Methods

            * :code:`get_frame(frame_index: int) -> np.ndarray`:
              Get a single frame from the underlying video data with
              output shape=(height, width, channels).

    """

    filename: str
    shape: Tuple[int, int, int, int] = field(default=None)
    backend: Any = None

    # @classmethod
    # def from_filename(cls, filename: str, *args, **kwargs) -> Video:
    #     """Create an instance of a video object, auto-detecting the backend.

    #     Args:
    #         filename: The path to the video filename.
    #             Currently supported types are:

    #             * Media Videos - AVI, MP4, etc. handled by OpenCV directly
    #             * HDF5 Datasets - .h5 files
    #             * Numpy Arrays - npy files
    #             * imgstore datasets - produced by loopbio's Motif recording
    #                 system. See: https://github.com/loopbio/imgstore.

    #         args: Arguments to pass to :class:`NumpyVideo`
    #         kwargs: Arguments to pass to :class:`NumpyVideo`

    #     Returns:
    #         A Video object with the detected backend.
    #     """
    #     filename = Video.fixup_path(filename)
    #     backend_class: Any
    #     if filename.lower().endswith(("h5", "hdf5", "slp")):
    #         backend_class = HDF5Video
    #     elif filename.endswith(("npy")):
    #         backend_class = NumpyVideo
    #     elif filename.lower().endswith(("mp4", "avi", "mov")):
    #         backend_class = MediaVideo
    #         kwargs["dataset"] = ""  # prevent serialization from breaking
    #     elif os.path.isdir(filename) or "metadata.yaml" in filename:
    #         backend_class = ImgStoreVideo
    #     else:
    #         raise ValueError("Could not detect backend for specified filename.")

    #     kwargs["filename"] = filename

    #     return cls(backend=cls.make_specific_backend(backend_class, kwargs))

    # @classmethod
    # def from_hdf5(
    #     cls,
    #     filename: str,
    #     shape: Tuple[int, int, int, int],
    # ) -> Video:
    #     """
    #     Create an instance of a video object from an HDF5 file and dataset.

    #     This is a helper method that invokes the HDF5Video backend.

    #     Args:
    #         dataset: The name of the dataset or and h5.Dataset object. If
    #             filename is h5.File, dataset must be a str of the dataset name.
    #         filename: The name of the HDF5 file or and open h5.File object.
    #         input_format: Whether the data is oriented with "channels_first"
    #             or "channels_last"
    #         convert_range: Whether we should convert data to [0, 255]-range

    #     Returns:
    #         A Video object with HDF5Video backend.
    #     """
    #     filename = Video.fixup_path(filename)
    #     backend = HDF5Video(
    #         filename=filename,
    #         shape=shape,
    #         backend=backend,
    #     )
    #     return cls(backend=backend)
