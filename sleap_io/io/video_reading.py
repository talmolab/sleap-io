"""Backends for reading videos."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import attrs
import h5py
import imageio.v3 as iio
import numpy as np
import simplejson as json

try:
    import cv2
except ImportError:
    pass

try:
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    pass

try:
    import av  # noqa: F401
except ImportError:
    pass


# Track available backends (populated on module import)
_AVAILABLE_VIDEO_BACKENDS = {
    "opencv": "cv2" in sys.modules,
    "FFMPEG": "imageio_ffmpeg" in sys.modules,
    "pyav": "av" in sys.modules,
}

_AVAILABLE_IMAGE_BACKENDS = {
    "opencv": "cv2" in sys.modules,
    "imageio": True,  # Always available (core dependency)
}


# Global default video plugin
_default_video_plugin: Optional[str] = None


def normalize_plugin_name(plugin: str) -> str:
    """Normalize plugin names to standard format.

    Args:
        plugin: Plugin name or alias (case-insensitive).

    Returns:
        Normalized plugin name ("opencv", "FFMPEG", or "pyav").

    Raises:
        ValueError: If plugin name is not recognized.
    """
    plugin_lower = plugin.lower()

    # Map aliases to standard names
    aliases = {
        "cv": "opencv",
        "cv2": "opencv",
        "opencv": "opencv",
        "ocv": "opencv",
        "ffmpeg": "FFMPEG",
        "imageio-ffmpeg": "FFMPEG",
        "imageio_ffmpeg": "FFMPEG",
        "pyav": "pyav",
        "av": "pyav",
    }

    if plugin_lower not in aliases:
        raise ValueError(
            f"Unknown plugin: {plugin}. Valid options: opencv, FFMPEG, pyav"
        )

    return aliases[plugin_lower]


def set_default_video_plugin(plugin: Optional[str]) -> None:
    """Set the default video plugin for all subsequently loaded videos.

    Args:
        plugin: Video plugin name. One of "opencv", "FFMPEG", or "pyav".
            Also accepts aliases: "cv", "cv2", "ocv" for opencv;
            "imageio-ffmpeg", "imageio_ffmpeg" for FFMPEG; "av" for pyav.
            Case-insensitive. If None, clears the default preference.

    Examples:
        >>> import sleap_io as sio
        >>> sio.set_default_video_plugin("opencv")
        >>> sio.set_default_video_plugin("cv2")  # Same as "opencv"
        >>> sio.set_default_video_plugin(None)  # Clear preference
    """
    global _default_video_plugin
    if plugin is not None:
        plugin = normalize_plugin_name(plugin)
    _default_video_plugin = plugin


def get_default_video_plugin() -> Optional[str]:
    """Get the current default video plugin.

    Returns:
        The current default video plugin name, or None if not set.

    Examples:
        >>> import sleap_io as sio
        >>> sio.get_default_video_plugin()
        None
        >>> sio.set_default_video_plugin("opencv")
        >>> sio.get_default_video_plugin()
        'opencv'
    """
    return _default_video_plugin


# Global default image plugin for encoding/decoding embedded images
_default_image_plugin: Optional[str] = None


def normalize_image_plugin_name(plugin: str) -> str:
    """Normalize image plugin names to standard format.

    Args:
        plugin: Plugin name or alias (case-insensitive).

    Returns:
        Normalized plugin name ("opencv" or "imageio").

    Raises:
        ValueError: If plugin name is not recognized.
    """
    plugin_lower = plugin.lower()

    # Map aliases to standard names (only opencv and imageio for images)
    aliases = {
        "cv": "opencv",
        "cv2": "opencv",
        "opencv": "opencv",
        "ocv": "opencv",
        "imageio": "imageio",
        "iio": "imageio",
    }

    if plugin_lower not in aliases:
        raise ValueError(
            f"Unknown image plugin: {plugin}. Valid options: opencv, imageio"
        )

    return aliases[plugin_lower]


def set_default_image_plugin(plugin: Optional[str]) -> None:
    """Set the default image plugin for encoding/decoding embedded images.

    Args:
        plugin: Image plugin name. One of "opencv" or "imageio".
            Also accepts aliases: "cv", "cv2", "ocv" for opencv;
            "iio" for imageio. Case-insensitive.
            If None, clears the default preference.

    Examples:
        >>> import sleap_io as sio
        >>> sio.set_default_image_plugin("opencv")
        >>> sio.set_default_image_plugin("imageio")
        >>> sio.set_default_image_plugin(None)  # Clear preference
    """
    global _default_image_plugin
    if plugin is not None:
        plugin = normalize_image_plugin_name(plugin)
    _default_image_plugin = plugin


def get_default_image_plugin() -> Optional[str]:
    """Get the current default image plugin.

    Returns:
        The current default image plugin name ("opencv" or "imageio"), or None.

    Examples:
        >>> import sleap_io as sio
        >>> sio.get_default_image_plugin()
        None
        >>> sio.set_default_image_plugin("opencv")
        >>> sio.get_default_image_plugin()
        'opencv'
    """
    return _default_image_plugin


def get_available_video_backends() -> list[str]:
    """Get list of available video backend plugins.

    Returns:
        List of plugin names that are currently available. Possible values include
        "opencv", "FFMPEG", and "pyav".

    Examples:
        >>> import sleap_io as sio
        >>> sio.get_available_video_backends()
        ['FFMPEG', 'pyav']
        >>> 'opencv' in sio.get_available_video_backends()
        False
    """
    return [k for k, v in _AVAILABLE_VIDEO_BACKENDS.items() if v]


def get_available_image_backends() -> list[str]:
    """Get list of available image backend plugins.

    Returns:
        List of plugin names that are currently available. Will always include
        "imageio" (core dependency), and may include "opencv" if installed.

    Examples:
        >>> import sleap_io as sio
        >>> sio.get_available_image_backends()
        ['imageio']
        >>> 'opencv' in sio.get_available_image_backends()
        False
    """
    return [k for k, v in _AVAILABLE_IMAGE_BACKENDS.items() if v]


def get_installation_instructions(
    plugin: Optional[str] = None, backend_type: str = "video"
) -> str:
    """Get installation instructions for backend plugins.

    Args:
        plugin: Specific plugin name (e.g., "opencv", "FFMPEG", "pyav"), or None to
            get instructions for all plugins. Case-insensitive, accepts aliases.
        backend_type: Either "video" or "image". Determines which backend type to
            provide instructions for.

    Returns:
        Installation instructions as a formatted string.

    Examples:
        >>> import sleap_io as sio
        >>> print(sio.get_installation_instructions("opencv"))
        pip install sleap-io[opencv]

        >>> print(sio.get_installation_instructions())
        Video backend installation options:
          FFMPEG (bundled):        Included by default
          opencv (fastest):        pip install sleap-io[opencv]
          pyav (balanced):         pip install sleap-io[pyav]
    """
    if backend_type == "video":
        instructions = {
            "opencv": "pip install sleap-io[opencv]",
            "FFMPEG": "Included by default (imageio-ffmpeg)",
            "pyav": "pip install sleap-io[pyav]",
        }

        if plugin is not None:
            plugin = normalize_plugin_name(plugin)
            return instructions.get(plugin, "pip install sleap-io[all]")
        else:
            return (
                "Video backend installation options:\n"
                "  FFMPEG (bundled):        Included by default\n"
                "  opencv (fastest):        pip install sleap-io[opencv]\n"
                "  pyav (balanced):         pip install sleap-io[pyav]"
            )
    else:
        instructions = {
            "opencv": "pip install sleap-io[opencv]",
            "imageio": "Already installed (core dependency)",
        }

        if plugin is not None:
            plugin = normalize_image_plugin_name(plugin)
            return instructions.get(plugin, "pip install sleap-io[all]")
        else:
            return (
                "Image backend installation options:\n"
                "  opencv: pip install sleap-io[opencv]\n"
                "  imageio: Already installed (core dependency)"
            )


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
            **kwargs: Additional backend-specific arguments. These are filtered to only
                include parameters that are valid for the specific backend being
                created:
                - For ImageVideo: plugin (str): Image plugin to use. One of "opencv"
                  or "imageio". Also accepts aliases (case-insensitive).
                  If None, uses global default if set, otherwise auto-detects.
                - For MediaVideo: plugin (str): Video plugin to use. One of "opencv",
                  "FFMPEG", or "pyav". Also accepts aliases (case-insensitive).
                  If None, uses global default if set, otherwise auto-detects.
                - For HDF5Video: input_format (str), frame_map (dict),
                  source_filename (str),
                  source_inds (np.ndarray), image_format (str). See HDF5Video for
                  details.

        Returns:
            VideoBackend subclass instance.
        """
        if isinstance(filename, Path):
            filename = filename.as_posix()

        if type(filename) is str and Path(filename).is_dir():
            filename = ImageVideo.find_images(filename)

        if type(filename) is list:
            filename = [Path(f).as_posix() for f in filename]
            return ImageVideo(
                filename, grayscale=grayscale, **_get_valid_kwargs(ImageVideo, kwargs)
            )
        elif filename.lower().endswith(("tif", "tiff")):
            # Detect TIFF format
            format_type, metadata = TiffVideo.detect_format(filename)

            if format_type in ("multi_page", "rank3_video", "rank4_video"):
                # Use TiffVideo for multi-page or multi-dimensional TIFFs
                tiff_kwargs = _get_valid_kwargs(TiffVideo, kwargs)
                # Add format if detected
                if format_type in ("rank3_video", "rank4_video"):
                    tiff_kwargs["format"] = metadata.get("format")
                return TiffVideo(
                    filename,
                    grayscale=grayscale,
                    keep_open=keep_open,
                    **tiff_kwargs,
                )
            else:
                # Single-page TIFF, treat as regular image
                return ImageVideo(
                    [filename],
                    grayscale=grayscale,
                    **_get_valid_kwargs(ImageVideo, kwargs),
                )
        elif filename.lower().endswith(tuple(ext.lower() for ext in ImageVideo.EXTS)):
            return ImageVideo(
                [filename], grayscale=grayscale, **_get_valid_kwargs(ImageVideo, kwargs)
            )
        elif filename.lower().endswith(tuple(ext.lower() for ext in MediaVideo.EXTS)):
            return MediaVideo(
                filename,
                grayscale=grayscale,
                keep_open=keep_open,
                **_get_valid_kwargs(MediaVideo, kwargs),
            )
        elif filename.lower().endswith(tuple(ext.lower() for ext in HDF5Video.EXTS)):
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

    plugin: str = attrs.field()

    @plugin.validator
    def _validate_plugin(self, attribute, value):
        # Normalize the plugin name
        normalized = normalize_plugin_name(value)
        # Update the actual value to the normalized version
        object.__setattr__(self, attribute.name, normalized)

    EXTS = ("mp4", "avi", "mov", "mj2", "mkv")

    @plugin.default
    def _default_plugin(self) -> str:
        # Check global default first
        if _default_video_plugin is not None:
            # Warn if preferred plugin not available
            if not _AVAILABLE_VIDEO_BACKENDS.get(_default_video_plugin, False):
                import warnings

                available = get_available_video_backends()
                install_cmd = get_installation_instructions(_default_video_plugin)
                warnings.warn(
                    f"Preferred video plugin '{_default_video_plugin}' is not "
                    f"available. Available plugins: {available}\n"
                    f"Install with: {install_cmd}"
                )
                # Fall through to auto-detection
            else:
                return _default_video_plugin

        # Auto-detect based on what's available
        if "cv2" in sys.modules:
            return "opencv"
        elif "imageio_ffmpeg" in sys.modules:
            return "FFMPEG"
        elif "av" in sys.modules:
            return "pyav"
        else:
            # Enhanced error message with installation instructions
            raise ImportError(
                "No video backend plugins are available.\n\n"
                "The bundled imageio-ffmpeg should be available by default.\n"
                "If you see this error, try reinstalling sleap-io:\n"
                "  pip install --force-reinstall sleap-io\n\n"
                "Alternative backends:\n"
                "  opencv (fastest):  pip install sleap-io[opencv]\n"
                "  pyav (balanced):   pip install sleap-io[pyav]\n\n"
                "For more information, see: https://io.sleap.ai"
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
        if self.plugin == "opencv":
            if self.keep_open:
                if self._open_reader is None:
                    self._open_reader = cv2.VideoCapture(self.filename)
                reader = self._open_reader
            else:
                reader = cv2.VideoCapture(self.filename)

            if reader.get(cv2.CAP_PROP_POS_FRAMES) != frame_idx:
                reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, img = reader.read()

            if success:
                img = img[..., ::-1]  # BGR -> RGB

        elif self.plugin == "pyav" or self.plugin == "FFMPEG":
            if self.keep_open:
                img = self.reader.read(index=frame_idx)
            else:
                with iio.imopen(self.filename, "r", plugin=self.plugin) as reader:
                    img = reader.read(index=frame_idx)
            success = img is not None

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
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)

            imgs = imgs[..., ::-1]  # BGR -> RGB

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
        channel_order: Channel order of embedded images, either "RGB" or "BGR". This is
            used to ensure consistent color channel ordering when decoding embedded
            images. If the encoding and decoding plugins have different channel orders,
            the channels will be automatically flipped during decoding.
        plugin: Plugin to use for decoding embedded images. One of "opencv" or
            "FFMPEG". If None, uses the global default or auto-detects based on
            available packages. Note that "pyav" is automatically mapped to "FFMPEG"
            since PyAV doesn't support image decoding.
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
    channel_order: str = "RGB"
    plugin: Optional[str] = None

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

            # Read channel_order, with backwards compatibility
            if "channel_order" in ds.attrs:
                self.channel_order = ds.attrs["channel_order"]
            else:
                # Backwards compatibility: Check format_id for older files
                # Prior to format 1.4, embedded images were primarily encoded with
                # OpenCV which uses BGR, so default to BGR for older formats
                if "metadata" in f and "format_id" in f["metadata"].attrs:
                    format_id = f["metadata"].attrs["format_id"]
                    if format_id < 1.4:
                        self.channel_order = "BGR"  # Legacy default
                # If no format_id found, assume BGR (safest legacy default)
                # since most embedded images before this change used OpenCV

            if "frame_numbers" in ds.parent:
                frame_numbers = ds.parent["frame_numbers"][:].astype(int)
                self.frame_map = {frame: idx for idx, frame in enumerate(frame_numbers)}
                self.source_inds = frame_numbers

            if "source_video" in ds.parent:
                self.source_filename = json.loads(
                    ds.parent["source_video"].attrs["json"]
                )["backend"]["filename"]

        f.close()

        # Set default plugin if not specified (use image plugin, not video plugin)
        if self.plugin is None:
            # Check image plugin default first (for embedded images)
            if _default_image_plugin is not None:
                self.plugin = _default_image_plugin
            # Otherwise auto-detect (for embedded image decoding)
            elif "cv2" in sys.modules:
                self.plugin = "opencv"
            else:
                self.plugin = "imageio"  # imageio fallback

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
        # Decode based on plugin
        if self.plugin == "opencv":
            img = cv2.imdecode(img_string, cv2.IMREAD_UNCHANGED)
            decoder_order = "BGR"  # OpenCV decodes to BGR
        else:
            # Use imageio for FFMPEG or any other plugin
            img = iio.imread(BytesIO(img_string), extension=f".{self.image_format}")
            decoder_order = "RGB"  # imageio decodes to RGB

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Convert channel order if needed
        # If the stored order doesn't match the decoder order, flip channels
        if img.shape[-1] == 3 and self.channel_order != decoder_order:
            img = img[..., ::-1]  # Flip RGB <-> BGR

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
        plugin: Image plugin to use for reading. One of "opencv" or "imageio".
            If None, uses global default from get_default_image_plugin(), or
            auto-detects.
    """

    EXTS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")

    plugin: str = attrs.field()

    @plugin.validator
    def _validate_plugin(self, attribute, value):
        """Validate and normalize plugin name."""
        normalized = normalize_image_plugin_name(value)
        object.__setattr__(self, attribute.name, normalized)

    @plugin.default
    def _default_plugin(self) -> str:
        """Get default plugin, checking global default first."""
        # Check global default first
        if _default_image_plugin is not None:
            # Warn if preferred plugin not available
            if not _AVAILABLE_IMAGE_BACKENDS.get(_default_image_plugin, False):
                import warnings

                available = get_available_image_backends()
                install_cmd = get_installation_instructions(
                    _default_image_plugin, "image"
                )
                warnings.warn(
                    f"Preferred image plugin '{_default_image_plugin}' is not "
                    f"available. Available plugins: {available}\n"
                    f"Install with: {install_cmd}"
                )
                # Fall through to auto-detection
            else:
                return _default_image_plugin

        # Otherwise auto-detect
        if "cv2" in sys.modules:
            return "opencv"
        else:
            return "imageio"

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
            The frame as a numpy array of shape `(height, width, channels)` in RGB
            order.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frame` method of the `VideoBackend` class instead.

            Images are always returned in RGB order regardless of plugin:
            - imageio: Returns RGB natively
            - opencv: Returns BGR, automatically flipped to RGB
        """
        if self.plugin == "opencv":
            # OpenCV reads as BGR, flip to RGB
            img = cv2.imread(self.filename[frame_idx], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {self.filename[frame_idx]}")
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[..., ::-1]  # BGR -> RGB
        else:  # imageio
            # imageio reads as RGB natively
            img = iio.imread(self.filename[frame_idx])

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        return img


@attrs.define
class TiffVideo(VideoBackend):
    """Video backend for reading multi-page TIFF stacks.

    This backend supports reading multi-page TIFF files as video sequences.
    Each page in the TIFF is treated as a frame.

    Attributes:
        filename: Path to the multi-page TIFF file.
        grayscale: Whether to force grayscale. If None, autodetect on first frame load.
        keep_open: Whether to keep the reader open between calls to read frames.
        format: Format of the TIFF file ("multi_page", "THW", "HWT", "THWC", "CHWT").
    """

    EXTS = ("tif", "tiff")
    format: Optional[str] = None

    @staticmethod
    def is_multipage(filename: str) -> bool:
        """Check if a TIFF file contains multiple pages.

        Args:
            filename: Path to the TIFF file.

        Returns:
            True if the TIFF contains multiple pages, False otherwise.
        """
        try:
            # Try to read the second frame
            iio.imread(filename, index=1)
            return True
        except (IndexError, ValueError):
            return False
        except Exception:
            # For any other error, assume it's not multi-page
            return False

    @staticmethod
    def detect_format(filename: str) -> tuple[str, dict]:
        """Detect TIFF format and shape for single files.

        Args:
            filename: Path to the TIFF file.

        Returns:
            Tuple of (format_type, metadata) where:
            - format_type: "single_frame", "multi_page", "rank3_video", or "rank4_video"
            - metadata: dict with shape info and inferred format
        """
        try:
            # Read first frame to check shape
            img = iio.imread(filename, index=0)
            shape = img.shape

            # Check if multi-page first
            is_multi = TiffVideo.is_multipage(filename)

            if is_multi:
                return "multi_page", {"shape": shape}

            # Single page cases
            if img.ndim == 2:
                # Rank-2: single channel image
                return "single_frame", {"shape": shape}
            elif img.ndim == 3:
                # Rank-3: could be HWC (single frame) or THW/HWT (video)
                return TiffVideo._detect_rank3_format(shape)
            elif img.ndim == 4:
                # Rank-4: video with channels
                return TiffVideo._detect_rank4_format(shape)
            else:
                return "single_frame", {"shape": shape}

        except Exception:
            return "single_frame", {"shape": None}

    @staticmethod
    def _detect_rank3_format(shape: tuple) -> tuple[str, dict]:
        """Detect format for rank-3 TIFF files.

        Args:
            shape: Shape tuple (dim1, dim2, dim3)

        Returns:
            Tuple of (format_type, metadata)
        """
        dim1, dim2, dim3 = shape

        # If last dimension is 1 or 3, likely HWC (single frame)
        if dim3 in (1, 3):
            return "single_frame", {"shape": shape, "format": "HWC"}

        # If first two dims are equal, it's likely HWT format
        # (most common case for square frames stored as H x W x T)
        if dim1 == dim2:
            # Default to HWT format for square frames
            return "rank3_video", {
                "shape": shape,
                "format": "HWT",
                "height": dim1,
                "width": dim2,
                "n_frames": dim3,
            }
        else:
            # For non-square frames, check if it could be THW
            # This is less common but possible
            if dim2 == dim3:
                # Could be THW format
                return "rank3_video", {
                    "shape": shape,
                    "format": "THW",
                    "n_frames": dim1,
                    "height": dim2,
                    "width": dim3,
                }
            else:
                # Default to HWT format
                return "rank3_video", {
                    "shape": shape,
                    "format": "HWT",
                    "height": dim1,
                    "width": dim2,
                    "n_frames": dim3,
                }

    @staticmethod
    def _detect_rank4_format(shape: tuple) -> tuple[str, dict]:
        """Detect format for rank-4 TIFF files.

        Args:
            shape: Shape tuple (dim1, dim2, dim3, dim4)

        Returns:
            Tuple of (format_type, metadata)
        """
        dim1, dim2, dim3, dim4 = shape

        # Check if first or last dimension is 1 or 3 (channels)
        if dim1 in (1, 3):
            # CHWT format
            return "rank4_video", {
                "shape": shape,
                "format": "CHWT",
                "channels": dim1,
                "height": dim2,
                "width": dim3,
                "n_frames": dim4,
            }
        elif dim4 in (1, 3):
            # THWC format
            return "rank4_video", {
                "shape": shape,
                "format": "THWC",
                "n_frames": dim1,
                "height": dim2,
                "width": dim3,
                "channels": dim4,
            }
        else:
            # Default to THWC
            return "rank4_video", {
                "shape": shape,
                "format": "THWC",
                "n_frames": dim1,
                "height": dim2,
                "width": dim3,
                "channels": dim4,
            }

    def __attrs_post_init__(self):
        """Initialize format if not provided."""
        if self.format is None:
            # Auto-detect format
            format_type, metadata = TiffVideo.detect_format(self.filename)
            if format_type == "multi_page":
                self.format = "multi_page"
            elif format_type in ("rank3_video", "rank4_video"):
                self.format = metadata.get("format", "multi_page")
            else:
                self.format = "multi_page"

    @property
    def num_frames(self) -> int:
        """Number of frames in the TIFF stack."""
        if self.format == "multi_page":
            # Count frames by trying to read each one until we get an error
            frame_count = 0
            while True:
                try:
                    iio.imread(self.filename, index=frame_count)
                    frame_count += 1
                except (IndexError, ValueError):
                    break
            return frame_count
        else:
            # For rank3/rank4 formats, detect from shape
            format_type, metadata = TiffVideo.detect_format(self.filename)
            return metadata.get("n_frames", 1)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the TIFF stack.

        Args:
            frame_idx: Index of frame to read.

        Returns:
            The frame as a numpy array of shape `(height, width, channels)`.

        Notes:
            This does not apply grayscale conversion. It is recommended to use the
            `get_frame` method of the `VideoBackend` class instead.
        """
        if self.format == "multi_page":
            img = iio.imread(self.filename, index=frame_idx)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            return img
        else:
            # Read entire array for rank3/rank4 formats
            img = iio.imread(self.filename)

            if self.format == "THW":
                # Extract frame from THW format
                frame = img[frame_idx, :, :]
                return np.expand_dims(frame, axis=-1)
            elif self.format == "HWT":
                # Extract frame from HWT format
                frame = img[:, :, frame_idx]
                return np.expand_dims(frame, axis=-1)
            elif self.format == "THWC":
                # Extract frame from THWC format
                return img[frame_idx, :, :, :]
            elif self.format == "CHWT":
                # Extract frame from CHWT format
                frame = img[:, :, :, frame_idx]
                return np.moveaxis(frame, 0, -1)  # CHW -> HWC
            else:
                raise ValueError(f"Unknown format: {self.format}")

    def _read_frames(self, frame_inds: list) -> np.ndarray:
        """Read multiple frames from the TIFF stack.

        Args:
            frame_inds: List of frame indices to read.

        Returns:
            Frames as a numpy array of shape `(frames, height, width, channels)`.
        """
        if self.format == "multi_page":
            imgs = []
            for idx in frame_inds:
                imgs.append(self._read_frame(idx))
            return np.stack(imgs, axis=0)
        else:
            # For rank3/rank4, read all at once and extract
            img = iio.imread(self.filename)

            if self.format == "THW":
                frames = img[frame_inds, :, :]
                return np.expand_dims(frames, axis=-1)
            elif self.format == "HWT":
                frames = img[:, :, frame_inds]
                frames = np.moveaxis(frames, -1, 0)  # HWT -> THW
                return np.expand_dims(frames, axis=-1)
            elif self.format == "THWC":
                return img[frame_inds, :, :, :]
            elif self.format == "CHWT":
                frames = img[:, :, :, frame_inds]
                frames = np.moveaxis(frames, -1, 0)  # CHWT -> TCHW
                frames = np.moveaxis(frames, 1, -1)  # TCHW -> THWC
                return frames
            else:
                raise ValueError(f"Unknown format: {self.format}")
