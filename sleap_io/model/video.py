"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import attrs
import h5py
import numpy as np

from sleap_io.io.utils import is_file_accessible
from sleap_io.io.video_reading import HDF5Video, ImageVideo, MediaVideo, VideoBackend
from sleap_io.io.video_writing import VideoWriter


@attrs.define(eq=False)
class Video:
    """`Video` class used by sleap to represent videos and data associated with them.

    This class is used to store information regarding a video and its components.
    It is used to store the video's `filename`, `shape`, and the video's `backend`.

    To create a `Video` object, use the `from_filename` method which will select the
    backend appropriately.

    Attributes:
        filename: The filename(s) of the video. Supported extensions: "mp4", "avi",
            "mov", "mj2", "mkv", "h5", "hdf5", "slp", "png", "jpg", "jpeg", "tif",
            "tiff", "bmp". If the filename is a list, a list of image filenames are
            expected. If filename is a folder, it will be searched for images.
        backend: An object that implements the basic methods for reading and
            manipulating frames of a specific video type.
        backend_metadata: A dictionary of metadata specific to the backend. This is
            useful for storing metadata that requires an open backend (e.g., shape
            information) without having access to the video file itself.
        source_video: The source video object if this is a proxy video. This is present
            when the video contains an embedded subset of frames from another video.
        open_backend: Whether to open the backend when the video is available. If `True`
            (the default), the backend will be automatically opened if the video exists.
            Set this to `False` when you want to manually open the backend, or when the
            you know the video file does not exist and you want to avoid trying to open
            the file.

    Notes:
        Instances of this class are hashed by identity, not by value. This means that
        two `Video` instances with the same attributes will NOT be considered equal in a
        set or dict.

    Media Video Plugin Support:
        For media files (mp4, avi, etc.), the following plugins are supported:
        - "opencv": Uses OpenCV (cv2) for video reading
        - "FFMPEG": Uses imageio-ffmpeg for video reading
        - "pyav": Uses PyAV for video reading

        Plugin aliases (case-insensitive):
        - opencv: "opencv", "cv", "cv2", "ocv"
        - FFMPEG: "FFMPEG", "ffmpeg", "imageio-ffmpeg", "imageio_ffmpeg"
        - pyav: "pyav", "av"

        Plugin selection priority:
        1. Explicitly specified plugin parameter
        2. Backend metadata plugin value
        3. Global default (set via sio.set_default_video_plugin)
        4. Auto-detection based on available packages

    See Also:
        VideoBackend: The backend interface for reading video data.
        sleap_io.set_default_video_plugin: Set global default plugin.
        sleap_io.get_default_video_plugin: Get current default plugin.
    """

    filename: str | list[str]
    backend: Optional[VideoBackend] = None
    backend_metadata: dict[str, any] = attrs.field(factory=dict)
    source_video: Optional[Video] = None
    original_video: Optional[Video] = None
    open_backend: bool = True

    EXTS = MediaVideo.EXTS + HDF5Video.EXTS + ImageVideo.EXTS

    def __attrs_post_init__(self):
        """Post init syntactic sugar."""
        if self.open_backend and self.backend is None and self.exists():
            try:
                self.open()
            except Exception:
                # If we can't open the backend, just ignore it for now so we don't
                # prevent the user from building the Video object entirely.
                pass

    def __deepcopy__(self, memo):
        """Deep copy the video object."""
        if id(self) in memo:
            return memo[id(self)]

        reopen = False
        if self.is_open:
            reopen = True
            self.close()

        new_video = Video(
            filename=self.filename,
            backend=None,
            backend_metadata=self.backend_metadata.copy(),
            source_video=self.source_video,
            original_video=self.original_video,
            open_backend=self.open_backend,
        )

        memo[id(self)] = new_video

        if reopen:
            self.open()

        return new_video

    @classmethod
    def from_filename(
        cls,
        filename: str | list[str],
        dataset: Optional[str] = None,
        grayscale: Optional[bool] = None,
        keep_open: bool = True,
        source_video: Optional[Video] = None,
        **kwargs,
    ) -> VideoBackend:
        """Create a Video from a filename.

        Args:
            filename: The filename(s) of the video. Supported extensions: "mp4", "avi",
                "mov", "mj2", "mkv", "h5", "hdf5", "slp", "png", "jpg", "jpeg", "tif",
                "tiff", "bmp". If the filename is a list, a list of image filenames are
                expected. If filename is a folder, it will be searched for images.
            dataset: Name of dataset in HDF5 file.
            grayscale: Whether to force grayscale. If None, autodetect on first frame
                load.
            keep_open: Whether to keep the video reader open between calls to read
                frames. If False, will close the reader after each call. If True (the
                default), it will keep the reader open and cache it for subsequent calls
                which may enhance the performance of reading multiple frames.
            source_video: The source video object if this is a proxy video. This is
                present when the video contains an embedded subset of frames from
                another video.
            **kwargs: Additional backend-specific arguments passed to
                VideoBackend.from_filename. See VideoBackend.from_filename for supported
                arguments.

        Returns:
            Video instance with the appropriate backend instantiated.
        """
        backend = VideoBackend.from_filename(
            filename,
            dataset=dataset,
            grayscale=grayscale,
            keep_open=keep_open,
            **kwargs,
        )
        # If filename is a directory, VideoBackend.from_filename will expand it
        # to a list of paths to images contained within the directory. In this
        # case we want to use the expanded list as filename
        return cls(
            filename=backend.filename,
            backend=backend,
            source_video=source_video,
        )

    @property
    def shape(self) -> Tuple[int, int, int, int] | None:
        """Return the shape of the video as (num_frames, height, width, channels).

        If the video backend is not set or it cannot determine the shape of the video,
        this will return None.
        """
        return self._get_shape()

    def _get_shape(self) -> Tuple[int, int, int, int] | None:
        """Return the shape of the video as (num_frames, height, width, channels).

        This suppresses errors related to querying the backend for the video shape, such
        as when it has not been set or when the video file is not found.
        """
        try:
            return self.backend.shape
        except Exception:
            if "shape" in self.backend_metadata:
                return self.backend_metadata["shape"]
            return None

    @property
    def grayscale(self) -> bool | None:
        """Return whether the video is grayscale.

        If the video backend is not set or it cannot determine whether the video is
        grayscale, this will return None.
        """
        shape = self.shape
        if shape is not None:
            return shape[-1] == 1
        else:
            grayscale = None
            if "grayscale" in self.backend_metadata:
                grayscale = self.backend_metadata["grayscale"]
            return grayscale

    @grayscale.setter
    def grayscale(self, value: bool):
        """Set the grayscale value and adjust the backend."""
        if self.backend is not None:
            self.backend.grayscale = value
            self.backend._cached_shape = None

        self.backend_metadata["grayscale"] = value

    def __len__(self) -> int:
        """Return the length of the video as the number of frames."""
        shape = self.shape
        return 0 if shape is None else shape[0]

    def __repr__(self) -> str:
        """Informal string representation (for print or format)."""
        dataset = (
            f"dataset={self.backend.dataset}, "
            if getattr(self.backend, "dataset", "")
            else ""
        )
        return (
            "Video("
            f'filename="{self.filename}", '
            f"shape={self.shape}, "
            f"{dataset}"
            f"backend={type(self.backend).__name__}"
            ")"
        )

    def __str__(self) -> str:
        """Informal string representation (for print or format)."""
        return self.__repr__()

    def __getitem__(self, inds: int | list[int] | slice) -> np.ndarray:
        """Return the frames of the video at the given indices.

        Args:
            inds: Index or list of indices of frames to read.

        Returns:
            Frame or frames as a numpy array of shape `(height, width, channels)` if a
            scalar index is provided, or `(frames, height, width, channels)` if a list
            of indices is provided.

        See also: VideoBackend.get_frame, VideoBackend.get_frames
        """
        if not self.is_open:
            if self.open_backend:
                self.open()
            else:
                raise ValueError(
                    "Video backend is not open. Call video.open() or set "
                    "video.open_backend to True to do automatically on frame read."
                )
        return self.backend[inds]

    def exists(self, check_all: bool = False, dataset: str | None = None) -> bool:
        """Check if the video file exists and is accessible.

        Args:
            check_all: If `True`, check that all filenames in a list exist. If `False`
                (the default), check that the first filename exists.
            dataset: Name of dataset in HDF5 file. If specified, this will function will
                return `False` if the dataset does not exist.

        Returns:
            `True` if the file exists and is accessible, `False` otherwise.
        """
        if isinstance(self.filename, list):
            if check_all:
                for f in self.filename:
                    if not is_file_accessible(f):
                        return False
                return True
            else:
                return is_file_accessible(self.filename[0])

        file_is_accessible = is_file_accessible(self.filename)
        if not file_is_accessible:
            return False

        if dataset is None or dataset == "":
            dataset = self.backend_metadata.get("dataset", None)

        if dataset is not None and dataset != "":
            has_dataset = False
            if (
                self.backend is not None
                and type(self.backend) is HDF5Video
                and self.backend._open_reader is not None
            ):
                has_dataset = dataset in self.backend._open_reader
            else:
                with h5py.File(self.filename, "r") as f:
                    has_dataset = dataset in f
            return has_dataset

        return True

    @property
    def is_open(self) -> bool:
        """Check if the video backend is open."""
        return self.exists() and self.backend is not None

    def open(
        self,
        filename: Optional[str] = None,
        dataset: Optional[str] = None,
        grayscale: Optional[str] = None,
        keep_open: bool = True,
        plugin: Optional[str] = None,
    ):
        """Open the video backend for reading.

        Args:
            filename: Filename to open. If not specified, will use the filename set on
                the video object.
            dataset: Name of dataset in HDF5 file.
            grayscale: Whether to force grayscale. If None, autodetect on first frame
                load.
            keep_open: Whether to keep the video reader open between calls to read
                frames. If False, will close the reader after each call. If True (the
                default), it will keep the reader open and cache it for subsequent calls
                which may enhance the performance of reading multiple frames.
            plugin: Video plugin to use for MediaVideo files. One of "opencv",
                "FFMPEG", or "pyav". Also accepts aliases (case-insensitive).
                If not specified, uses the backend metadata, global default,
                or auto-detection in that order.

        Notes:
            This is useful for opening the video backend to read frames and then closing
            it after reading all the necessary frames.

            If the backend was already open, it will be closed before opening a new one.
            Values for the HDF5 dataset and grayscale will be remembered if not
            specified.
        """
        if filename is not None:
            self.replace_filename(filename, open=False)

        # Try to remember values from previous backend if available and not specified.
        if self.backend is not None:
            if dataset is None:
                dataset = getattr(self.backend, "dataset", None)
            if grayscale is None:
                grayscale = getattr(self.backend, "grayscale", None)

        else:
            if dataset is None and "dataset" in self.backend_metadata:
                dataset = self.backend_metadata["dataset"]
            if grayscale is None:
                if "grayscale" in self.backend_metadata:
                    grayscale = self.backend_metadata["grayscale"]
                elif "shape" in self.backend_metadata:
                    grayscale = self.backend_metadata["shape"][-1] == 1

        if not self.exists(dataset=dataset):
            msg = (
                f"Video does not exist or cannot be opened for reading: {self.filename}"
            )
            if dataset is not None:
                msg += f" (dataset: {dataset})"
            raise FileNotFoundError(msg)

        # Close previous backend if open.
        self.close()

        # Handle plugin parameter
        backend_kwargs = {}
        if plugin is not None:
            from sleap_io.io.video_reading import normalize_plugin_name

            plugin = normalize_plugin_name(plugin)
            self.backend_metadata["plugin"] = plugin

        if "plugin" in self.backend_metadata:
            backend_kwargs["plugin"] = self.backend_metadata["plugin"]

        # Create new backend.
        self.backend = VideoBackend.from_filename(
            self.filename,
            dataset=dataset,
            grayscale=grayscale,
            keep_open=keep_open,
            **backend_kwargs,
        )

    def close(self):
        """Close the video backend."""
        if self.backend is not None:
            # Try to remember values from previous backend if available and not
            # specified.
            try:
                self.backend_metadata["dataset"] = getattr(
                    self.backend, "dataset", None
                )
                self.backend_metadata["grayscale"] = getattr(
                    self.backend, "grayscale", None
                )
                self.backend_metadata["shape"] = getattr(self.backend, "shape", None)
            except Exception:
                pass

            del self.backend
            self.backend = None

    def replace_filename(
        self, new_filename: str | Path | list[str] | list[Path], open: bool = True
    ):
        """Update the filename of the video, optionally opening the backend.

        Args:
            new_filename: New filename to set for the video.
            open: If `True` (the default), open the backend with the new filename. If
                the new filename does not exist, no error is raised.
        """
        if isinstance(new_filename, Path):
            new_filename = new_filename.as_posix()

        if isinstance(new_filename, list):
            new_filename = [
                p.as_posix() if isinstance(p, Path) else p for p in new_filename
            ]

        self.filename = new_filename
        self.backend_metadata["filename"] = new_filename

        if open:
            if self.exists():
                self.open()
            else:
                self.close()

    def matches_path(self, other: "Video", strict: bool = False) -> bool:
        """Check if this video has the same path as another video.

        Args:
            other: Another video to compare with.
            strict: If True, require exact path match. If False, consider videos
                with the same filename (basename) as matching.

        Returns:
            True if the videos have matching paths, False otherwise.

        Notes:
            For HDF5 video backends (e.g., embedded videos in .pkg.slp files),
            matching prioritizes the source_filename attribute since multiple
            videos can share the same HDF5 file path but reference different
            source videos. Falls back to dataset name matching if source_filename
            is not available.
        """
        # Handle HDF5 backends specially - prioritize source_filename matching
        self_is_hdf5 = isinstance(self.backend, HDF5Video)
        other_is_hdf5 = isinstance(other.backend, HDF5Video)

        if self_is_hdf5 and other_is_hdf5:
            # Both are HDF5 videos - match by source_filename first
            self_source = self.backend.source_filename
            other_source = other.backend.source_filename

            if self_source is not None and other_source is not None:
                if strict:
                    return Path(self_source).resolve() == Path(other_source).resolve()
                else:
                    return Path(self_source).name == Path(other_source).name

            # Fall back to dataset name matching if source_filename is not available
            self_dataset = self.backend.dataset
            other_dataset = other.backend.dataset

            if self_dataset is not None and other_dataset is not None:
                return self_dataset == other_dataset

            # If neither source_filename nor dataset available, cannot match
            return False

        if isinstance(self.filename, list) and isinstance(other.filename, list):
            # Both are image sequences
            if strict:
                return self.filename == other.filename
            else:
                # Compare basenames
                self_basenames = [Path(f).name for f in self.filename]
                other_basenames = [Path(f).name for f in other.filename]
                return self_basenames == other_basenames
        elif isinstance(self.filename, list) or isinstance(other.filename, list):
            # One is image sequence, other is single file
            return False
        else:
            # Both are single files
            if strict:
                return Path(self.filename).resolve() == Path(other.filename).resolve()
            else:
                return Path(self.filename).name == Path(other.filename).name

    def matches_content(self, other: "Video") -> bool:
        """Check if this video has the same content as another video.

        Args:
            other: Another video to compare with.

        Returns:
            True if the videos have the same shape and backend type.

        Notes:
            This compares metadata like shape and backend type, not actual frame data.
        """
        # Compare shapes
        self_shape = self.shape
        other_shape = other.shape

        if self_shape != other_shape:
            return False

        # Compare backend types
        if self.backend is None and other.backend is None:
            return True
        elif self.backend is None or other.backend is None:
            return False

        return type(self.backend).__name__ == type(other.backend).__name__

    def matches_shape(self, other: "Video") -> bool:
        """Check if this video has the same shape as another video.

        Args:
            other: Another video to compare with.

        Returns:
            True if the videos have the same height, width, and channels.

        Notes:
            This only compares spatial dimensions, not the number of frames.
        """
        # Try to get shape from backend metadata first if shape is not available
        if self.backend is None and "shape" in self.backend_metadata:
            self_shape = self.backend_metadata["shape"]
        else:
            self_shape = self.shape

        if other.backend is None and "shape" in other.backend_metadata:
            other_shape = other.backend_metadata["shape"]
        else:
            other_shape = other.shape

        # Handle None shapes
        if self_shape is None or other_shape is None:
            return False

        # Compare only height, width, channels (not frames)
        return self_shape[1:] == other_shape[1:]

    def has_overlapping_images(self, other: "Video") -> bool:
        """Check if this video has overlapping images with another video.

        This method is specifically for ImageVideo backends (image sequences).

        Args:
            other: Another video to compare with.

        Returns:
            True if both are ImageVideo instances with overlapping image files.
            False if either video is not an ImageVideo or no overlap exists.

        Notes:
            Only works with ImageVideo backends where filename is a list.
            Compares individual image filenames (basenames only).
        """
        # Both must be image sequences
        if not (isinstance(self.filename, list) and isinstance(other.filename, list)):
            return False

        # Get basenames for comparison
        self_basenames = set(Path(f).name for f in self.filename)
        other_basenames = set(Path(f).name for f in other.filename)

        # Check if there's any overlap
        return len(self_basenames & other_basenames) > 0

    def deduplicate_with(self, other: "Video") -> "Video":
        """Create a new video with duplicate images removed.

        This method is specifically for ImageVideo backends (image sequences).

        Args:
            other: Another video to deduplicate against. Must also be ImageVideo.

        Returns:
            A new Video object with duplicate images removed from this video,
            or None if all images were duplicates.

        Raises:
            ValueError: If either video is not an ImageVideo backend.

        Notes:
            Only works with ImageVideo backends where filename is a list.
            Images are considered duplicates if they have the same basename.
            The returned video contains only images from this video that are
            not present in the other video.
        """
        if not isinstance(self.filename, list):
            raise ValueError("deduplicate_with only works with ImageVideo backends")
        if not isinstance(other.filename, list):
            raise ValueError("Other video must also be ImageVideo backend")

        # Get basenames from other video
        other_basenames = set(Path(f).name for f in other.filename)

        # Keep only non-duplicate images
        deduplicated_paths = [
            f for f in self.filename if Path(f).name not in other_basenames
        ]

        if not deduplicated_paths:
            # All images were duplicates
            return None

        # Create new video with deduplicated images
        return Video.from_filename(deduplicated_paths, grayscale=self.grayscale)

    def merge_with(self, other: "Video") -> "Video":
        """Merge another video's images into this one.

        This method is specifically for ImageVideo backends (image sequences).

        Args:
            other: Another video to merge with. Must also be ImageVideo.

        Returns:
            A new Video object with unique images from both videos.

        Raises:
            ValueError: If either video is not an ImageVideo backend.

        Notes:
            Only works with ImageVideo backends where filename is a list.
            The merged video contains all unique images from both videos,
            with automatic deduplication based on image basename.
        """
        if not isinstance(self.filename, list):
            raise ValueError("merge_with only works with ImageVideo backends")
        if not isinstance(other.filename, list):
            raise ValueError("Other video must also be ImageVideo backend")

        # Get all unique images (by basename) preserving order
        seen_basenames = set()
        merged_paths = []

        for path in self.filename:
            basename = Path(path).name
            if basename not in seen_basenames:
                merged_paths.append(path)
                seen_basenames.add(basename)

        for path in other.filename:
            basename = Path(path).name
            if basename not in seen_basenames:
                merged_paths.append(path)
                seen_basenames.add(basename)

        # Create new video with merged images
        return Video.from_filename(merged_paths, grayscale=self.grayscale)

    def save(
        self,
        save_path: str | Path,
        frame_inds: list[int] | np.ndarray | None = None,
        video_kwargs: dict[str, Any] | None = None,
    ) -> Video:
        """Save video frames to a new video file.

        Args:
            save_path: Path to the new video file. Should end in MP4.
            frame_inds: Frame indices to save. Can be specified as a list or array of
                frame integers. If not specified, saves all video frames.
            video_kwargs: A dictionary of keyword arguments to provide to
                `sio.save_video` for video compression.

        Returns:
            A new `Video` object pointing to the new video file.
        """
        video_kwargs = {} if video_kwargs is None else video_kwargs
        frame_inds = np.arange(len(self)) if frame_inds is None else frame_inds

        with VideoWriter(save_path, **video_kwargs) as vw:
            for frame_ind in frame_inds:
                vw(self[frame_ind])

        new_video = Video.from_filename(save_path, grayscale=self.grayscale)
        return new_video

    def set_video_plugin(self, plugin: str) -> None:
        """Set the video plugin and reopen the video.

        Args:
            plugin: Video plugin to use. One of "opencv", "FFMPEG", or "pyav".
                Also accepts aliases (case-insensitive).

        Raises:
            ValueError: If the video is not a MediaVideo type.

        Examples:
            >>> video.set_video_plugin("opencv")
            >>> video.set_video_plugin("CV2")  # Same as "opencv"
        """
        from sleap_io.io.video_reading import MediaVideo, normalize_plugin_name

        if not self.filename.endswith(MediaVideo.EXTS):
            raise ValueError(f"Cannot set plugin for non-media video: {self.filename}")

        plugin = normalize_plugin_name(plugin)

        # Close current backend if open
        was_open = self.is_open
        if was_open:
            self.close()

        # Update backend metadata
        self.backend_metadata["plugin"] = plugin

        # Reopen with new plugin if it was open
        if was_open:
            self.open()
