"""Data model for videos.

The `Video` class is a SLEAP data structure that stores information regarding
a video and its components used in SLEAP.
"""

from __future__ import annotations
from attrs import define
from typing import Tuple, Optional, Optional
import numpy as np
from sleap_io.io.video import VideoBackend


@define
class Video:
    """`Video` class used by sleap to represent videos and data associated with them.

    This class is used to store information regarding a video and its components.
    It is used to store the video's `filename`, `shape`, and the video's `backend`.

    To create a `Video` object, use the `from_filename` method which will select the
    backend appropriately.

    Args:
        filename: The filename of the video.
        backend: An object that implements the basic methods for reading and
            manipulating frames of a specific video type.

    See also: VideoBackend
    """

    filename: str
    backend: Optional[VideoBackend] = None

    @classmethod
    def from_filename(
        cls,
        filename: str,
        dataset: Optional[str] = None,
        grayscale: Optional[str] = None,
        **kwargs,
    ) -> VideoBackend:
        """Create a Video from a filename.

        Args:
            filename: Path to video file.
            dataset: Name of dataset in HDF5 file.
            grayscale: Whether to force grayscale. If None, autodetect on first frame
                load.

        Returns:
            Video instance with the appropriate backend instantiated.
        """
        return cls(
            filename=filename,
            backend=VideoBackend.from_filename(
                filename, dataset=dataset, grayscale=grayscale, **kwargs
            ),
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
            ind: Index or list of indices of frames to read.

        Returns:
            Frame or frames as a numpy array of shape `(height, width, channels)` if a
            scalar index is provided, or `(frames, height, width, channels)` if a list
            of indices is provided.

        See also: VideoBackend.get_frame, VideoBackend.get_frames
        """
        if self.backend is None:
            raise ValueError(
                "Video backend is not set. "
                "This may be because the video reader could not be determined "
                "automatically from the filename."
            )
        return self.backend[inds]
