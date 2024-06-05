"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations
import attrs
from typing import Tuple, Optional, Optional
import numpy as np
from sleap_io.io.video import VideoBackend, MediaVideo, HDF5Video, ImageVideo
from pathlib import Path


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

    EXTS = MediaVideo.EXTS + HDF5Video.EXTS + ImageVideo.EXTS

    def __attrs_post_init__(self):
        """Post init syntactic sugar."""
        if self.backend is None and self.exists():
            self.open()

    def __attrs_post_init__(self):
        """Post init syntactic sugar."""
        if self.backend is None and self.exists():
            self.open()

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
            if "grayscale" in self.backend_metadata:
                return self.backend_metadata["grayscale"]
            return None

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
            self.open()
        return self.backend[inds]

    def exists(self, check_all: bool = False) -> bool:
        """Check if the video file exists.

        Args:
            check_all: If `True`, check that all filenames in a list exist. If `False`
                (the default), check that the first filename exists.
        """
        if isinstance(self.filename, list):
            if check_all:
                for f in self.filename:
                    if not Path(f).exists():
                        return False
                return True
            else:
                return Path(self.filename[0]).exists()
        return Path(self.filename).exists()

    @property
    def is_open(self) -> bool:
        """Check if the video backend is open."""
        return self.exists() and self.backend is not None

    def open(
        self,
        dataset: Optional[str] = None,
        grayscale: Optional[str] = None,
        keep_open: bool = True,
    ):
        """Open the video backend for reading.

        Args:
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
        if not self.exists():
            raise FileNotFoundError(f"Video file not found: {self.filename}")

        # Try to remember values from previous backend if available and not specified.
        if self.backend is not None:
            if dataset is None:
                dataset = getattr(self.backend, "dataset", None)
            if grayscale is None:
                grayscale = getattr(self.backend, "grayscale", None)

        else:
            if dataset is None and "dataset" in self.backend_metadata:
                dataset = self.backend_metadata["dataset"]
            if grayscale is None and "grayscale" in self.backend_metadata:
                grayscale = self.backend_metadata["grayscale"]

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
            new_filename = str(new_filename)

        if isinstance(new_filename, list):
            new_filename = [
                p.as_posix() if isinstance(p, Path) else p for p in new_filename
            ]

        self.filename = new_filename

        if open:
            if self.exists():
                self.open()
            else:
                self.close()
