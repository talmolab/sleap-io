"""Backends for reading videos."""

from __future__ import annotations
from pathlib import Path

import simplejson as json
import sys
from io import BytesIO
from typing import Optional, Tuple

import attrs
import h5py
import imageio.v3 as iio
import numpy as np

try:
    import cv2
except ImportError:
    pass

try:
    import imageio_ffmpeg
except ImportError:
    pass

try:
    import av
except ImportError:
    pass


def _get_valid_kwargs(cls, kwargs: dict) -> dict:
    """Filter a list of kwargs to the ones that are valid for a class."""
    valid_fields = [a.name for a in attrs.fields(cls)]
    return {k: v for k, v in kwargs.items() if k in valid_fields}


@attrs.define
class VideoBackend:
    """Base class for video backends.

    This class is not meant to be used directly. Instead, use the `from_filename`
    constructor to create a backend instance.

    Attributes:
        filename: Path to video file(s).
        grayscale: Whether to force grayscale. If None, autodetect on first frame load.
        keep_open: Whether to keep the video reader open between calls to read frames.
            If False, will close the reader after each call. If True (the default), it
            will keep the reader open and cache it for subsequent calls which may
            enhance the performance of reading multiple frames.
    """

    filename: str | Path | list[str] | list[Path]
    grayscale: Optional[bool] = None
    keep_open: bool = True
    _cached_shape: Optional[Tuple[int, int, int, int]] = None
    _open_reader: Optional[object] = None

    @classmethod
    def from_filename(
        cls,
        filename: str | list[str],
        dataset: Optional[str] = None,
        grayscale: Optional[bool] = None,
        keep_open: bool = True,
        **kwargs,
    ) -> VideoBackend:
        """Create a VideoBackend from a filename.

        Args:
            filename: Path to video file(s).
            dataset: Name of dataset in HDF5 file.
            grayscale: Whether to force grayscale. If None, autodetect on first frame
                load.
            keep_open: Whether to keep the video reader open between calls to read
                frames. If False, will close the reader after each call. If True (the
                default), it will keep the reader open and cache it for subsequent calls
                which may enhance the performance of reading multiple frames.

        Returns:
            VideoBackend subclass instance.
        """
        if isinstance(filename, Path):
            filename = filename.as_posix()

        if type(filename) == str and Path(filename).is_dir():
            filename = ImageVideo.find_images(filename)

        if type(filename) == list:
            filename = [Path(f).as_posix() for f in filename]
            return ImageVideo(
                filename, grayscale=grayscale, **_get_valid_kwargs(ImageVideo, kwargs)
            )
        elif filename.endswith(ImageVideo.EXTS):
            return ImageVideo(
                [filename], grayscale=grayscale, **_get_valid_kwargs(ImageVideo, kwargs)
            )
        elif filename.endswith(MediaVideo.EXTS):
            return MediaVideo(
                filename,
                grayscale=grayscale,
                keep_open=keep_open,
                **_get_valid_kwargs(MediaVideo, kwargs),
            )
        elif filename.endswith(HDF5Video.EXTS):
            return HDF5Video(
                filename,
                dataset=dataset,
                grayscale=grayscale,
                keep_open=keep_open,
                **_get_valid_kwargs(HDF5Video, kwargs),
            )
        else:
            raise ValueError(f"Unknown video file type: {filename}")

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the video. Must be implemented in subclasses."""
        raise NotImplementedError

    def _read_frames(self, frame_inds: list) -> np.ndarray:
        """Read a list of frames from the video."""
        return np.stack([self.get_frame(i) for i in frame_inds], axis=0)

    def read_test_frame(self) -> np.ndarray:
        """Read a single frame from the video to test for grayscale.

        Note:
            This reads the frame at index 0. This may not be appropriate if the first
            frame is not available in a given backend.
        """
        return self._read_frame(0)

    def detect_grayscale(self, test_img: np.ndarray | None = None) -> bool:
        """Detect whether the video is grayscale.

        This works by reading in a test frame and comparing the first and last channel
        for equality. It may fail in cases where, due to compression, the first and
        last channels are not exactly the same.

        Args:
            test_img: Optional test image to use. If not provided, a test image will be
                loaded via the `read_test_frame` method.

        Returns:
            Whether the video is grayscale. This value is also cached in the `grayscale`
            attribute of the class.
        """
        if test_img is None:
            test_img = self.read_test_frame()
        is_grayscale = np.array_equal(test_img[..., 0], test_img[..., -1])
        self.grayscale = is_grayscale
        return is_grayscale

    @property
    def num_frames(self) -> int:
        """Number of frames in the video. Must be implemented in subclasses."""
        raise NotImplementedError

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        """Shape of a single frame in the video."""
        height, width, channels = self.read_test_frame().shape
        if self.grayscale is None:
            self.detect_grayscale()
        if self.grayscale is False:
            channels = 3
        elif self.grayscale is True:
            channels = 1
        return int(height), int(width), int(channels)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of the video as a tuple of `(frames, height, width, channels)`.

        On first call, this will defer to `num_frames` and `img_shape` to determine the
        full shape. This call may be expensive for some subclasses, so the result is
        cached and returned on subsequent calls.
        """
        if self._cached_shape is not None:
            return self._cached_shape
        else:
            shape = (self.num_frames,) + self.img_shape
            self._cached_shape = shape
            return shape

    @property
    def frames(self) -> int:
        """Number of frames in the video."""
        return self.shape[0]

    def __len__(self) -> int:
        """Return number of frames in the video."""
        return self.shape[0]

    def has_frame(self, frame_idx: int) -> bool:
        """Check if a frame index is contained in the video.

        Args:
            frame_idx: Index of frame to check.

        Returns:
            `True` if the index is contained in the video, otherwise `False`.
        """
        return frame_idx < len(self)

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the video.

        Args:
            frame_idx: Index of frame to read.

        Returns:
            Frame as a numpy array of shape `(height, width, channels)` where the
            `channels` dimension is 1 for grayscale videos and 3 for color videos.

        Notes:
            If the `grayscale` attribute is set to `True`, the `channels` dimension will
            be reduced to 1 if an RGB frame is loaded from the backend.

            If the `grayscale` attribute is set to `None`, the `grayscale` attribute
            will be automatically set based on the first frame read.

        See also: `get_frames`
        """
        if not self.has_frame(frame_idx):
            raise IndexError(f"Frame index {frame_idx} out of range.")

        img = self._read_frame(frame_idx)

        if self.grayscale is None:
            self.detect_grayscale(img)

        if self.grayscale:
            img = img[..., [0]]

        return img

    def get_frames(self, frame_inds: list[int]) -> np.ndarray:
        """Read a list of frames from the video.

        Depending on the backend implementation, this may be faster than reading frames
        individually using `get_frame`.

        Args:
            frame_inds: List of frame indices to read.

        Returns:
            Frames as a numpy array of shape `(frames, height, width, channels)` where
            `channels` dimension is 1 for grayscale videos and 3 for color videos.

        Notes:
            If the `grayscale` attribute is set to `True`, the `channels` dimension will
            be reduced to 1 if an RGB frame is loaded from the backend.

            If the `grayscale` attribute is set to `None`, the `grayscale` attribute
            will be automatically set based on the first frame read.

        See also: `get_frame`
        """
        imgs = self._read_frames(frame_inds)

        if self.grayscale is None:
            self.detect_grayscale(imgs[0])

        if self.grayscale:
            imgs = imgs[..., [0]]

        return imgs

    def __getitem__(self, ind: int | list[int] | slice) -> np.ndarray:
        """Return a single frame or a list of frames from the video.

        Args:
            ind: Index or list of indices of frames to read.

        Returns:
            Frame or frames as a numpy array of shape `(height, width, channels)` if a
            scalar index is provided, or `(frames, height, width, channels)` if a list
            of indices is provided.

        See also: get_frame, get_frames
        """
        if np.isscalar(ind):
            return self.get_frame(ind)
        else:
            if type(ind) is slice:
                start = (ind.start or 0) % len(self)
                stop = ind.stop or len(self)
                if stop < 0:
                    stop = len(self) + stop
                step = ind.step or 1
                ind = range(start, stop, step)
            return self.get_frames(ind)


@attrs.define
class MediaVideo(VideoBackend):
    """Video backend for reading videos stored as common media files.

    This backend supports reading through FFMPEG (the default), pyav, or OpenCV. Here
    are their trade-offs:

        - "opencv": Fastest video reader, but only supports a limited number of codecs
            and may not be able to read some videos. It requires `opencv-python` to be
            installed. It is the fastest because it uses the OpenCV C++ library to read
            videos, but is limited by the version of FFMPEG that was linked into it at
            build time as well as the OpenCV version used.
        - "FFMPEG": Slowest, but most reliable. This is the default backend. It requires
            `imageio-ffmpeg` and a `ffmpeg` executable on the system path (which can be
            installed via conda). The `imageio` plugin for FFMPEG reads frames into raw
            bytes which are communicated to Python through STDOUT on a subprocess pipe,
            which can be slow. However, it is the most reliable and feature-complete. If
            you install the conda-forge version of ffmpeg, it will be compiled with
            support for many codecs, including GPU-accelerated codecs like NVDEC for
            H264 and others.
        - "pyav": Supports most codecs that FFMPEG does, but not as complete or reliable
            of an implementation in `imageio` as FFMPEG for some video types. It is
            faster than FFMPEG because it uses the `av` package to read frames directly
            into numpy arrays in memory without the need for a subprocess pipe. These
            are Python bindings for the C library libav, which is the same library that
            FFMPEG uses under the hood.

    Attributes:
        filename: Path to video file.
        grayscale: Whether to force grayscale. If None, autodetect on first frame load.
        keep_open: Whether to keep the video reader open between calls to read frames.
            If False, will close the reader after each call. If True (the default), it
            will keep the reader open and cache it for subsequent calls which may
            enhance the performance of reading multiple frames.
        plugin: Video plugin to use. One of "opencv", "FFMPEG", or "pyav". If `None`,
            will use the first available plugin in the order listed above.
    """

    plugin: str = attrs.field(
        validator=attrs.validators.in_(["opencv", "FFMPEG", "pyav"])
    )

    EXTS = ("mp4", "avi", "mov", "mj2", "mkv")

    @plugin.default
    def _default_plugin(self) -> str:
        if "cv2" in sys.modules:
            return "opencv"
        elif "imageio_ffmpeg" in sys.modules:
            return "FFMPEG"
        elif "av" in sys.modules:
            return "pyav"
        else:
            raise ImportError(
                "No video plugins found. Install opencv-python, imageio-ffmpeg, or av."
            )

    @property
    def reader(self) -> object:
        """Return the reader object for the video, caching if necessary."""
        if self.keep_open:
            if self._open_reader is None:
                if self.plugin == "opencv":
                    self._open_reader = cv2.VideoCapture(self.filename)
                elif self.plugin == "pyav" or self.plugin == "FFMPEG":
                    self._open_reader = iio.imopen(
                        self.filename, "r", plugin=self.plugin
                    )
            return self._open_reader
        else:
            if self.plugin == "opencv":
                return cv2.VideoCapture(self.filename)
            elif self.plugin == "pyav" or self.plugin == "FFMPEG":
                return iio.imopen(self.filename, "r", plugin=self.plugin)

    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        if self.plugin == "opencv":
            return int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            props = iio.improps(self.filename, plugin=self.plugin)
            n_frames = props.n_images
            if np.isinf(n_frames):
                legacy_reader = self.reader.legacy_get_reader()
                # Note: This might be super slow for some videos, so maybe we should
                # defer evaluation of this or give the user control over it.
                n_frames = legacy_reader.count_frames()
            return n_frames

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the video.

        Args:
            frame_idx: Index of frame to read.

        Returns:
            The frame as a numpy array of shape `(height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frame` method of the `VideoBackend` class instead.
        """
        failed = False
        if self.plugin == "opencv":
            if self.reader.get(cv2.CAP_PROP_POS_FRAMES) != frame_idx:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, img = self.reader.read()
        elif self.plugin == "pyav" or self.plugin == "FFMPEG":
            if self.keep_open:
                img = self.reader.read(index=frame_idx)
            else:
                with iio.imopen(self.filename, "r", plugin=self.plugin) as reader:
                    img = reader.read(index=frame_idx)

        success = (not failed) and (img is not None)
        if not success:
            raise IndexError(f"Failed to read frame index {frame_idx}.")
        return img

    def _read_frames(self, frame_inds: list) -> np.ndarray:
        """Read a list of frames from the video.

        Args:
            frame_inds: List of indices of frames to read.

        Returns:
            The frame as a numpy array of shape `(frames, height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frames` method of the `VideoBackend` class instead.
        """
        if self.plugin == "opencv":
            if self.keep_open:
                if self._open_reader is None:
                    self._open_reader = cv2.VideoCapture(self.filename)
                reader = self._open_reader
            else:
                reader = cv2.VideoCapture(self.filename)

            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_inds[0])
            imgs = []
            for idx in frame_inds:
                if reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, img = reader.read()
                img = img[..., ::-1]  # BGR -> RGB
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)

        elif self.plugin == "pyav" or self.plugin == "FFMPEG":
            if self.keep_open:
                if self._open_reader is None:
                    self._open_reader = iio.imopen(
                        self.filename, "r", plugin=self.plugin
                    )
                reader = self._open_reader
                imgs = np.stack([reader.read(index=idx) for idx in frame_inds], axis=0)
            else:
                with iio.imopen(self.filename, "r", plugin=self.plugin) as reader:
                    imgs = np.stack(
                        [reader.read(index=idx) for idx in frame_inds], axis=0
                    )
        return imgs


@attrs.define
class HDF5Video(VideoBackend):
    """Video backend for reading videos stored in HDF5 files.

    This backend supports reading videos stored in HDF5 files, both in rank-4 datasets
    as well as in datasets with lists of binary-encoded images.

    Embedded image datasets are used in SLEAP when exporting package files (`.pkg.slp`)
    with videos embedded in them. This is useful for bundling training or inference data
    without having to worry about the videos (or frame images) being moved or deleted.
    It is expected that these types of datasets will be in a `Group` with a `int8`
    variable length dataset called `"video"`. This dataset must also contain an
    attribute called "format" with a string describing the image format (e.g., "png" or
    "jpg") which will be used to decode it appropriately.

    If a `frame_numbers` dataset is present in the group, it will be used to map from
    source video frames to the frames in the dataset. This is useful to preserve frame
    indexing when exporting a subset of frames in the video. It will also be used to
    populate `frame_map` and `source_inds` attributes.

    Attributes:
        filename: Path to HDF5 file (.h5, .hdf5 or .slp).
        grayscale: Whether to force grayscale. If None, autodetect on first frame load.
        keep_open: Whether to keep the video reader open between calls to read frames.
            If False, will close the reader after each call. If True (the default), it
            will keep the reader open and cache it for subsequent calls which may
            enhance the performance of reading multiple frames.
        dataset: Name of dataset to read from. If `None`, will try to find a rank-4
            dataset by iterating through datasets in the file. If specifying an embedded
            dataset, this can be the group containing a "video" dataset or the dataset
            itself (e.g., "video0" or "video0/video").
        input_format: Format of the data in the dataset. One of "channels_last" (the
            default) in `(frames, height, width, channels)` order or "channels_first" in
            `(frames, channels, width, height)` order. Embedded datasets should use the
            "channels_last" format.
        frame_map: Mapping from frame indices to indices in the dataset. This is used to
            translate between the frame indices of the images within their source video
            and the indices of the images in the dataset. This is only used when reading
            embedded image datasets.
        source_filename: Path to the source video file. This is metadata and only used
            when reading embedded image datasets.
        source_inds: Indices of the frames in the source video file. This is metadata
            and only used when reading embedded image datasets.
        image_format: Format of the images in the embedded dataset. This is metadata and
            only used when reading embedded image datasets.
    """

    dataset: Optional[str] = None
    input_format: str = attrs.field(
        default="channels_last",
        validator=attrs.validators.in_(["channels_last", "channels_first"]),
    )
    frame_map: dict[int, int] = attrs.field(init=False, default=attrs.Factory(dict))
    source_filename: Optional[str] = None
    source_inds: Optional[np.ndarray] = None
    image_format: str = "hdf5"

    EXTS = ("h5", "hdf5", "slp")

    def __attrs_post_init__(self):
        """Auto-detect dataset and frame map heuristically."""
        # Check if the file accessible before applying heuristics.
        try:
            f = h5py.File(self.filename, "r")
        except OSError:
            return

        if self.dataset is None:
            # Iterate through datasets to find a rank 4 array.
            def find_movies(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                    self.dataset = name
                    return True

            f.visititems(find_movies)

        if self.dataset is None:
            # Iterate through datasets to find an embedded video dataset.
            def find_embedded(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("/video"):
                    self.dataset = name
                    return True

            f.visititems(find_embedded)

        if self.dataset is None:
            # Couldn't find video datasets.
            return

        if isinstance(f[self.dataset], h5py.Group):
            # If this is a group, assume it's an embedded video dataset.
            if "video" in f[self.dataset]:
                self.dataset = f"{self.dataset}/video"

        if self.dataset.split("/")[-1] == "video":
            # This may be an embedded video dataset. Check for frame map.
            ds = f[self.dataset]

            if "format" in ds.attrs:
                self.image_format = ds.attrs["format"]

            if "frame_numbers" in ds.parent:
                frame_numbers = ds.parent["frame_numbers"][:].astype(int)
                self.frame_map = {frame: idx for idx, frame in enumerate(frame_numbers)}
                self.source_inds = frame_numbers

            if "source_video" in ds.parent:
                self.source_filename = json.loads(
                    ds.parent["source_video"].attrs["json"]
                )["backend"]["filename"]

        f.close()

    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        with h5py.File(self.filename, "r") as f:
            return f[self.dataset].shape[0]

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        """Shape of a single frame in the video as `(height, width, channels)`."""
        with h5py.File(self.filename, "r") as f:
            ds = f[self.dataset]

            img_shape = None
            if "height" in ds.attrs:
                # Try to get shape from the attributes.
                img_shape = (
                    ds.attrs["height"],
                    ds.attrs["width"],
                    ds.attrs["channels"],
                )

                if img_shape[0] == 0 or img_shape[1] == 0:
                    # Invalidate the shape if the attributes are zero.
                    img_shape = None

            if img_shape is None and self.image_format == "hdf5" and ds.ndim == 4:
                # Use the dataset shape if just stored as a rank-4 array.
                img_shape = ds.shape[1:]

                if self.input_format == "channels_first":
                    img_shape = img_shape[::-1]

        if img_shape is None:
            # Fall back to reading a test frame.
            return super().img_shape

        return int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

    def read_test_frame(self) -> np.ndarray:
        """Read a single frame from the video to test for grayscale."""
        if self.frame_map:
            frame_idx = list(self.frame_map.keys())[0]
        else:
            frame_idx = 0
        return self._read_frame(frame_idx)

    @property
    def has_embedded_images(self) -> bool:
        """Return True if the dataset contains embedded images."""
        return self.image_format is not None and self.image_format != "hdf5"

    @property
    def embedded_frame_inds(self) -> list[int]:
        """Return the frame indices of the embedded images."""
        return list(self.frame_map.keys())

    def decode_embedded(self, img_string: np.ndarray) -> np.ndarray:
        """Decode an embedded image string into a numpy array.

        Args:
            img_string: Binary string of the image as a `int8` numpy vector with the
                bytes as values corresponding to the format-encoded image.

        Returns:
            The decoded image as a numpy array of shape `(height, width, channels)`. If
            a rank-2 image is decoded, it will be expanded such that channels will be 1.

            This method does not apply grayscale conversion as per the `grayscale`
            attribute. Use the `get_frame` or `get_frames` methods of the `VideoBackend`
            to apply grayscale conversion rather than calling this function directly.
        """
        if "cv2" in sys.modules:
            img = cv2.imdecode(img_string, cv2.IMREAD_UNCHANGED)
        else:
            img = iio.imread(BytesIO(img_string), extension=f".{self.image_format}")

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img

    def has_frame(self, frame_idx: int) -> bool:
        """Check if a frame index is contained in the video.

        Args:
            frame_idx: Index of frame to check.

        Returns:
            `True` if the index is contained in the video, otherwise `False`.
        """
        if self.frame_map:
            return frame_idx in self.frame_map
        else:
            return frame_idx < len(self)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the video.

        Args:
            frame_idx: Index of frame to read.

        Returns:
            The frame as a numpy array of shape `(height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frame` method of the `VideoBackend` class instead.
        """
        if self.keep_open:
            if self._open_reader is None:
                self._open_reader = h5py.File(self.filename, "r")
            f = self._open_reader
        else:
            f = h5py.File(self.filename, "r")

        ds = f[self.dataset]

        if self.frame_map:
            frame_idx = self.frame_map[frame_idx]

        img = ds[frame_idx]

        if self.has_embedded_images:
            img = self.decode_embedded(img)

        if self.input_format == "channels_first":
            img = np.transpose(img, (2, 1, 0))

        if not self.keep_open:
            f.close()
        return img

    def _read_frames(self, frame_inds: list) -> np.ndarray:
        """Read a list of frames from the video.

        Args:
            frame_inds: List of indices of frames to read.

        Returns:
            The frame as a numpy array of shape `(frames, height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frames` method of the `VideoBackend` class instead.
        """
        if self.keep_open:
            if self._open_reader is None:
                self._open_reader = h5py.File(self.filename, "r")
            f = self._open_reader
        else:
            f = h5py.File(self.filename, "r")

        if self.frame_map:
            frame_inds = [self.frame_map[idx] for idx in frame_inds]

        ds = f[self.dataset]
        imgs = ds[frame_inds]

        if "format" in ds.attrs:
            imgs = np.stack(
                [self.decode_embedded(img) for img in imgs],
                axis=0,
            )

        if self.input_format == "channels_first":
            imgs = np.transpose(imgs, (0, 3, 2, 1))

        if not self.keep_open:
            f.close()

        return imgs


@attrs.define
class ImageVideo(VideoBackend):
    """Video backend for reading videos stored as image files.

    This backend supports reading videos stored as a list of images.

    Attributes:
        filename: Path to image files.
        grayscale: Whether to force grayscale. If None, autodetect on first frame load.
    """

    EXTS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")

    @staticmethod
    def find_images(folder: str) -> list[str]:
        """Find images in a folder and return a list of filenames."""
        folder = Path(folder)
        return sorted(
            [f.as_posix() for f in folder.glob("*") if f.suffix[1:] in ImageVideo.EXTS]
        )

    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        return len(self.filename)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the video.

        Args:
            frame_idx: Index of frame to read.

        Returns:
            The frame as a numpy array of shape `(height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frame` method of the `VideoBackend` class instead.
        """
        img = iio.imread(self.filename[frame_idx])
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img
