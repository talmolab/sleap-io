"""Data model for Videos.

Videos are SLEAP data structures that store information regarding a video and its
components used in SLEAP.
"""

from __future__ import annotations
from attrs import define, field, Factory
from typing import List, Optional, Tuple, Union
import h5py as h5
import numpy as np
import os
import cv2


@define
class DummyVideo:
    """
    Fake video backend,returns frames with all zeros.

    This can be useful when you want to look at labels for a dataset but don't
    have access to the real video.
    """

    filename: str = field(default="")
    height: int = field(default=2000)
    width: int = field(default=2000)
    frames: int = field(default=10000)
    channels: int = field(default=1)
    dummy: bool = field(default=True)

    @property
    def test_frame(self):
        return self.get_frame(0)

    def get_frame(self, idx) -> np.ndarray:
        return np.zeros((self.height, self.width, self.channels))


@define(auto_attribs=True, eq=False)
class HDF5Video:
    """
    Video data stored as 4D datasets in HDF5 files.

    Args:
        filename: The name of the HDF5 file where the dataset with video data
            is stored.
        dataset: The name of the HDF5 dataset where the video data is stored.
        file_h5: The h5.File object that the underlying dataset is stored.
        dataset_h5: The h5.Dataset object that the underlying data is stored.
        input_format: A string value equal to either "channels_last" or
            "channels_first".
            This specifies whether the underlying video data is stored as:

                * "channels_first": shape = (frames, channels, height, width)
                * "channels_last": shape = (frames, height, width, channels)
        convert_range: Whether we should convert data to [0, 255]-range
    """

    filename: str = field(default=None)
    dataset: str = field(default=None)
    input_format: str = "channels_last"
    convert_range: bool = field(default=True)

    def __attrs_post_init__(self):
        """Called by attrs after __init__()."""

        self.enable_source_video = True
        self._test_frame_ = None
        self.__original_to_current_frame_idx = dict()
        self.__dataset_h5 = None
        self.__tried_to_load = False


    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Lets grab a test frame to help us figure things out about the video
            self._test_frame_ = self.get_frame(self.last_frame_idx)

        # Return stored test frame
        return self._test_frame_

    @property
    def enable_source_video(self) -> bool:

        """If set to `True`, will attempt to read from original video for frames not
        saved in the file."""
        return self._enable_source_video

    @enable_source_video.setter
    def enable_source_video(self, val: bool):
        self._enable_source_video = val

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__dataset_h5.shape[0]

    @property
    def channels(self):
        """See :class:`Video`."""
        if "channels" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["channels"])
        return self.__dataset_h5.shape[self.__channel_idx]

    @property
    def width(self):
        """See :class:`Video`."""
        if "width" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["width"])
        return self.__dataset_h5.shape[self.__width_idx]

    @property
    def height(self):
        """See :class:`Video`."""
        if "height" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["height"])
        return self.__dataset_h5.shape[self.__height_idx]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.test_frame.dtype



@define
class MediaVideo:
    """
    Video data stored in traditional media formats readable by FFMPEG

    This class provides bare minimum read only interface on top of
    OpenCV's VideoCapture class.

    Args:
        filename: The name of the file (.mp4, .avi, etc)
        grayscale: Whether the video is grayscale or not. "auto" means detect
            based on first frame.
        bgr: Whether color channels ordered as (blue, green, red).
    """

    filename: str
    grayscale: bool
    bgr: bool = True

    # Unused attributes still here so we don't break deserialization
    dataset: str = ""
    input_format: str = ""

    _detect_grayscale = False
    _reader_ = field(default=None)
    _test_frame_ = field(default=None)

    @property
    def __lock(self):
        if not hasattr(self, "_lock"):
            self._lock = multiprocessing.RLock()
        return self._lock

    @property
    def __reader(self):
        # Load if not already loaded
        if self._reader_ is None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(
                    f"Could not find filename video filename named {self.filename}"
                )

            # Try and open the file either locally in current directory or with full
            # path
            self._reader_ = cv2.VideoCapture(self.filename)

            # If the user specified None for grayscale bool, figure it out based on the
            # the first frame of data.
            if self._detect_grayscale is True:
                self.grayscale = bool(
                    np.alltrue(self.test_frame[..., 0] == self.test_frame[..., -1])
                )

        # Return cached reader
        return self._reader_

    @property
    def __frames_float(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Lets grab a test frame to help us figure things out about the video
            self._test_frame_ = self.get_frame(0, grayscale=False)

        # Return stored test frame
        return self._test_frame_

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        return self.__reader.get(cv2.CAP_PROP_FPS)

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        """See :class:`Video`."""
        return int(self.__frames_float)

    @property
    def channels(self):
        """See :class:`Video`."""
        if self.grayscale:
            return 1
        else:
            return self.test_frame.shape[2]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.test_frame.shape[1]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.test_frame.shape[0]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.test_frame.dtype


@define
class NumpyVideo:
    """
    Video data stored as Numpy array.

    Args:
        filename: Either a file to load or a numpy array of the data.

        * numpy data shape: (frames, height, width, channels)
    """

    filename: Union[str, np.ndarray]

    @property
    def test_frame(self):
        return self.get_frame(0)

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__frame_idx]

    @property
    def channels(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__channel_idx]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__width_idx]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__height_idx]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__data.dtype



@define
class ImgStoreVideo:
    """
    Video data stored as an ImgStore dataset.

    See: https://github.com/loopbio/imgstore
    This class is just a lightweight wrapper for reading such datasets as
    video sources for SLEAP.

    Args:
        filename: The name of the file or directory to the imgstore.
        index_by_original: ImgStores are great for storing a collection of
            selected frames from an larger video. If the index_by_original is
            set to True then the get_frame function will accept the original
            frame numbers of from original video. If False, then it will
            accept the frame index from the store directly.
            Default to True so that we can use an ImgStoreVideo in a dataset
            to replace another video without having to update all the frame
            indices on :class:`LabeledFrame` objects in the dataset.
    """

    filename: str = field(default=None)
    index_by_original: bool = True
    _store_ = field(default=None)
    _img_ = field(default=None)

    @property
    def __store(self):
        if self._store_ is None:
            self.open()
        return self._store_

    @__store.setter
    def __store(self, val):
        self._store_ = val

    @property
    def __img(self):
        if self._img_ is None:
            self.open()
        return self._img_

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__store.frame_count

    @property
    def channels(self):
        """See :class:`Video`."""
        if len(self.__img.shape) < 3:
            return 1
        else:
            return self.__img.shape[2]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.__img.shape[1]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.__img.shape[0]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__img.dtype

    @property
    def last_frame_idx(self) -> int:
        """
        The idx number of the last frame.

        Overrides method of base :class:`Video` class for videos with
        select frames indexed by number from original video, since the last
        frame index here will not match the number of frames in video.
        """
        if self.index_by_original:
            return self.__store.frame_max
        return self.frames - 1

    @property
    def imgstore(self):
        """
        Get the underlying ImgStore object for this Video.

        Returns:
            The imgstore that is backing this video object.
        """
        return self.__store


@define
class SingleImageVideo:
    """
    Video wrapper for individual image files.

    Args:
        filenames: Files to load as video.
    """

    filename: Optional[str] = field(default=None)
    filenames: Optional[list] = field(default=Factory(list))
    height_: Optional[int] = field(default=None)
    width_: Optional[int] = field(default=None)
    channels_: Optional[int] = field(default=None)

    @property
    def frames(self):
        """See :class:`Video`."""
        return len(self.filenames)

    @property
    def channels(self):
        """See :class:`Video`."""
        if self.channels_ is None:
            self._load_test_frame()

        return self.channels_

    @property
    def width(self):
        """See :class:`Video`."""
        if self.width_ is None:
            self._load_test_frame()

        return self.width_

    @width.setter
    def width(self, val):
        self.width_ = val

    @property
    def height(self):
        """See :class:`Video`."""
        if self.height_ is None:
            self._load_test_frame()

        return self.height_

    @height.setter
    def height(self, val):
        self.height_ = val

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__data.dtype


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

    backend: Union[
        HDF5Video, NumpyVideo, MediaVideo, ImgStoreVideo, SingleImageVideo, DummyVideo
    ]

    def __getattr__(self, item):
        return getattr(self.backend, item)

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the video."""
        return self.frames


    @property
    def shape(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Return tuple of (frame count, height, width, channels)."""
        try:
            return (self.frames, self.height, self.width, self.channels)
        except:
            return (None, None, None, None)

    @staticmethod
    def make_specific_backend(backend_class, kwargs):
        # Only pass through the kwargs that match attributes for the backend
        attribute_kwargs = {
            key: val
            for (key, val) in kwargs.items()
            if key in attr.fields_dict(backend_class).keys()
        }

        return backend_class(**attribute_kwargs)
