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


@define
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
    convert_range: bool = True

    @property
    def __dataset_h5(self) -> h5.Dataset:
        if self.__loaded_dataset is None and not self.__tried_to_load:
            self._load()
        return self.__loaded_dataset

    @__dataset_h5.setter
    def __dataset_h5(self, val):
        self.__loaded_dataset = val

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

    @property
    def is_missing(self) -> bool:
        """Return True if the video comes from a file and is missing."""
        if self.filename == "Raw Video Data":
            return False
        return not os.path.exists(self.filename)


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
    filenames: Optional[List[str]] = Factory(list)
    height_: Optional[int] = field(default=None)
    width_: Optional[int] = field(default=None)
    channels_: Optional[int] = field(default=None)

    @property
    def test_frame(self) -> np.ndarray:
        self._load_test_frame()
        return self.test_frame_

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

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the video."""
        return self.frames

    @property
    def last_frame_idx(self) -> int:
        """Return the index number of the last frame. Usually `num_frames - 1`."""
        if hasattr(self.backend, "last_frame_idx"):
            return self.backend.last_frame_idx
        return self.frames - 1

    @property
    def shape(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Return tuple of (frame count, height, width, channels)."""
        try:
            return (self.frames, self.height, self.width, self.channels)
        except:
            return (None, None, None, None)

    @property
    def is_missing(self) -> bool:
        """Return True if the video is a file and is not present."""
        if not hasattr(self.backend, "filename"):
            return True
        elif hasattr(self.backend, "is_missing"):
            return self.backend.is_missing
        else:
            return not os.path.exists(self.backend.filename)

    @classmethod
    def from_hdf5(
        cls,
        dataset: Union[str, h5.Dataset],
        filename: Union[str, h5.File] = None,
        input_format: str = "channels_last",
        convert_range: bool = True,
    ) -> "Video":
        """
        Create an instance of a video object from an HDF5 file and dataset.

        This is a helper method that invokes the HDF5Video backend.

        Args:
            dataset: The name of the dataset or and h5.Dataset object. If
                filename is h5.File, dataset must be a str of the dataset name.
            filename: The name of the HDF5 file or and open h5.File object.
            input_format: Whether the data is oriented with "channels_first"
                or "channels_last"
            convert_range: Whether we should convert data to [0, 255]-range

        Returns:
            A Video object with HDF5Video backend.
        """
        filename = Video.fixup_path(filename)
        backend = HDF5Video(
            filename=filename,
            dataset=dataset,
            input_format=input_format,
            convert_range=convert_range,
        )
        return cls(backend=backend)

    @classmethod
    def from_numpy(cls, filename: Union[str, np.ndarray], *args, **kwargs) -> "Video":
        """
        Create an instance of a video object from a numpy array.

        Args:
            filename: The numpy array or the name of the file
            args: Arguments to pass to :class:`NumpyVideo`
            kwargs: Arguments to pass to :class:`NumpyVideo`

        Returns:
            A Video object with a NumpyVideo backend
        """
        filename = Video.fixup_path(filename)
        backend = NumpyVideo(filename=filename, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_media(cls, filename: str, *args, **kwargs) -> "Video":
        """Create an instance of a video object from a typical media file.

        For example, mp4, avi, or other types readable by FFMPEG.

        Args:
            filename: The name of the file
            args: Arguments to pass to :class:`MediaVideo`
            kwargs: Arguments to pass to :class:`MediaVideo`

        Returns:
            A Video object with a MediaVideo backend
        """
        filename = Video.fixup_path(filename)
        backend = MediaVideo(filename=filename, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_image_filenames(
        cls,
        filenames: List[str],
        height: Optional[int] = field(default=None),
        width: Optional[int] = field(default=None),
        *args,
        **kwargs,
    ) -> "Video":
        """Create an instance of a SingleImageVideo from individual image file(s)."""
        backend = SingleImageVideo(filenames=filenames)
        if height:
            backend.height = height
        if width:
            backend.width = width
        return cls(backend=backend)

    @classmethod
    def from_filename(cls, filename: str, *args, **kwargs) -> "Video":
        """Create an instance of a video object, auto-detecting the backend.

        Args:
            filename: The path to the video filename.
                Currently supported types are:

                * Media Videos - AVI, MP4, etc. handled by OpenCV directly
                * HDF5 Datasets - .h5 files
                * Numpy Arrays - npy files
                * imgstore datasets - produced by loopbio's Motif recording
                    system. See: https://github.com/loopbio/imgstore.

            args: Arguments to pass to :class:`NumpyVideo`
            kwargs: Arguments to pass to :class:`NumpyVideo`

        Returns:
            A Video object with the detected backend.
        """
        filename = Video.fixup_path(filename)

        if filename.lower().endswith(("h5", "hdf5", "slp")):
            backend_class = HDF5Video
        elif filename.endswith(("npy")):
            backend_class = NumpyVideo
        elif filename.lower().endswith(("mp4", "avi", "mov")):
            backend_class = MediaVideo
            kwargs["dataset"] = ""  # prevent serialization from breaking
        elif os.path.isdir(filename) or "metadata.yaml" in filename:
            backend_class = ImgStoreVideo
        else:
            raise ValueError("Could not detect backend for specified filename.")

        kwargs["filename"] = filename

        return cls(backend=cls.make_specific_backend(backend_class, kwargs))

    @classmethod
    def imgstore_from_filenames(
        cls, filenames: list, output_filename: str, *args, **kwargs
    ) -> "Video":
        """Create an imgstore from a list of image files.

        Args:
            filenames: List of filenames for the image files.
            output_filename: Filename for the imgstore to create.

        Returns:
            A `Video` object for the new imgstore.
        """
        # get the image size from the first file
        first_img = cv2.imread(filenames[0], flags=cv2.IMREAD_COLOR)
        img_shape = first_img.shape

        # create the imgstore
        store = imgstore.new_for_format(
            "png", mode="w", basedir=output_filename, imgshape=img_shape
        )

        # read each frame and write it to the imgstore
        # unfortunately imgstore doesn't let us just add the file
        for i, img_filename in enumerate(filenames):
            img = cv2.imread(img_filename, flags=cv2.IMREAD_COLOR)
            store.add_image(img, i, i)

        store.close()

        # Return an ImgStoreVideo object referencing this new imgstore.
        return cls(backend=ImgStoreVideo(filename=output_filename))

    @staticmethod
    def make_specific_backend(backend_class, kwargs):
        # Only pass through the kwargs that match attributes for the backend
        attribute_kwargs = {
            key: val
            for (key, val) in kwargs.items()
            if key in attr.fields_dict(backend_class).keys()
        }

        return backend_class(**attribute_kwargs)

    @staticmethod
    def cattr():
        """Return a cattr converter for serialiazing/deserializing Video objects.

        Returns:
            A cattr converter.
        """

        # When we are structuring video backends, try to fixup the video file paths
        # in case they are coming from a different computer or the file has been moved.
        def fixup_video(x, cl):
            if "filename" in x:
                x["filename"] = Video.fixup_path(x["filename"])
            if "file" in x:
                x["file"] = Video.fixup_path(x["file"])

            return Video.make_specific_backend(cl, x)

        vid_cattr = cattr.Converter()

        # Check the type hint for backend and register the video path
        # fixup hook for each type in the Union.
        for t in attr.fields(Video).backend.type.__args__:
            vid_cattr.register_structure_hook(t, fixup_video)

        return vid_cattr

    @staticmethod
    def fixup_path(
        path: str, raise_error: bool = False, raise_warning: bool = False
    ) -> str:
        """Try to locate video if the given path doesn't work.

        Given a path to a video try to find it. This is attempt to make the
        paths serialized for different video objects portable across multiple
        computers. The default behavior is to store whatever path is stored
        on the backend object. If this is an absolute path it is almost
        certainly wrong when transferred when the object is created on
        another computer. We try to find the video by looking in the current
        working directory as well.

        Note that when loading videos during the process of deserializing a
        saved :class:`Labels` dataset, it's usually preferable to fix video
        paths using a `video_search` callback or path list.

        Args:
            path: The path the video asset.
            raise_error: Whether to raise error if we cannot find video.
            raise_warning: Whether to raise warning if we cannot find video.

        Raises:
            FileNotFoundError: If file still cannot be found and raise_error
                is True.

        Returns:
            The fixed up path
        """
        # If path is not a string then just return it and assume the backend
        # knows what to do with it.
        if type(path) is not str:
            return path

        if os.path.exists(path):
            return path

        # Strip the directory and lets see if the file is in the current working
        # directory.
        elif os.path.exists(os.path.basename(path)):
            return os.path.basename(path)

        # Special case: this is an ImgStore path! We cant use
        # basename because it will strip the directory name off
        elif path.endswith("metadata.yaml"):

            # Get the parent dir of the YAML file.
            img_store_dir = os.path.basename(os.path.split(path)[0])

            if os.path.exists(img_store_dir):
                return img_store_dir

        if raise_error:
            raise FileNotFoundError(f"Cannot find a video file: {path}")
        else:
            if raise_warning:
                logger.warning(f"Cannot find a video file: {path}")
            return path


def load_video(
    filename: str,
    grayscale: Optional[bool] = field(default=None),
    dataset=Optional[None],
    channels_first: bool = field(default=False),
) -> Video:
    """Open a video from disk.

    Args:
        filename: Path to a video file. The video reader backend will be determined by
            the file extension. Support extensions include: `.mp4`, `.avi`, `.h5`,
            `.hdf5` and `.slp` (for embedded images in a labels file). If the path to a
            folder is provided, images within that folder will be treated as video
            frames.
        grayscale: Read frames as a single channel grayscale images. If `None` (the
            default), this will be auto-detected.
        dataset: Name of the dataset that contains the video if loading a video stored
            in an HDF5 file. This has no effect for non-HDF5 inputs.
        channels_first: If `False` (the default), assume the data in the HDF5 dataset
            are formatted in `(frames, height, width, channels)` order. If `False`,
            assume the data are in `(frames, channels, width, height)` format. This has
            no effect for non-HDF5 inputs.

    Returns:
        A `sleap.Video` instance with the appropriate backend for its format.

        This enables numpy-like access to video data.

    Example: ::

        >>> video = sleap.load_video("centered_pair_small.mp4")
        >>> video.shape
        (1100, 384, 384, 1)
        >>> imgs = video[0:3]
        >>> imgs.shape
        (3, 384, 384, 1)

    See also:
        sleap.io.video.Video
    """
    kwargs = {}
    if grayscale is not None:
        kwargs["grayscale"] = grayscale
    if dataset is not None:
        kwargs["dataset"] = dataset
    kwargs["input_format"] = "channels_first" if channels_first else "channels_last"
    return Video.from_filename(filename, **kwargs)
