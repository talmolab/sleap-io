"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations
import attrs
from typing import Tuple, Optional, Optional, Any
import numpy as np
from sleap_io.io.video_reading import VideoBackend, MediaVideo, HDF5Video, ImageVideo
from sleap_io.io.video_writing import VideoWriter
from sleap_io.io.utils import is_file_accessible
from pathlib import Path
import h5py


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

    See also: VideoBackend
    """

    filename: str | list[str]
    backend: Optional[VideoBackend] = None
    backend_metadata: dict[str, any] = attrs.field(factory=dict)
    source_video: Optional[Video] = None
    open_backend: bool = True

    EXTS = MediaVideo.EXTS + HDF5Video.EXTS + ImageVideo.EXTS

    def __attrs_post_init__(self):
        """Post init syntactic sugar."""
        if self.open_backend and self.backend is None and self.exists():
            try:
                self.open()
            except Exception as e:
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
            backend_metadata=self.backend_metadata,
            source_video=self.source_video,
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

        Returns:
            Video instance with the appropriate backend instantiated.
        """
        return cls(
            filename=filename,
            backend=VideoBackend.from_filename(
                filename,
                dataset=dataset,
                grayscale=grayscale,
                keep_open=keep_open,
                **kwargs,
            ),
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
        except:
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
                and type(self.backend) == HDF5Video
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
            msg = f"Video does not exist or is inaccessible: {self.filename}"
            if dataset is not None:
                msg += f" (dataset: {dataset})"
            raise FileNotFoundError(msg)

        # Close previous backend if open.
        self.close()

        # Create new backend.
        self.backend = VideoBackend.from_filename(
            self.filename,
            dataset=dataset,
            grayscale=grayscale,
            keep_open=keep_open,
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
            except:
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
