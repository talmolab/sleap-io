"""Media video backend reader for standard video formats (MP4, AVI, etc.)."""
from __future__ import annotations
from pims import PyAVReaderIndexed
from sleap_io.io.utils import resolve_path
import numpy as np


class MediaVideoReader(PyAVReaderIndexed):
    """Class for reading and manipulating frames of standard video formats (MP4, AVI, etc.).

    Attributes:
        file: The path of the video file as a string.
        channels: The number of unique color channels.
        video_shape: The shape of the video as a tuple (height, width, unique channels, frames)
        frame_shape: The shape of a single frame in the video (height, width, pixel channels)

    Examples:
        >>> video = Video('video.avi')  # or .mov, etc.
        >>> imshow(video[0]) # Show the first frame.
        >>> imshow(video[-1]) # Show the last frame.
        >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

        >>> for frame in video[:]:
        ...    # Do something with every frame.

        >>> for frame in video[10:20]:
        ...    # Do something with frames 10-20.

        >>> for frame in video[[5, 7, 13]]:
        ...    # Do something with frames 5, 7, and 13.

        >>> video_shape = video.video_shape  # Dimensions of video (height, width, channels, frames)
        >>> frame_shape = video.frame_shape  # Pixel dimensions of video (height, width, channels)
    """

    def __init__(self, filename):
        """Initialize attributes of MediaVideoReader by reading a frame from the video.

        Args:
            filename: path to the video to read
        """
        super().__init__(file=filename)
        self.test_frame = self[0]
        self.grayscale = bool(
            np.alltrue(self.test_frame[..., 0] == self.test_frame[..., -1])
        )
        self.channels = 1 if self.grayscale else self.frame_shape[2]
        self.video_shape = self.frame_shape[:2] + (
            self.channels,
            len(self),
        )

    @classmethod
    def read_media_video(cls, filename: str, video_dirs: list[str] = []):
        """Read the video at `filename` by creating an instance of `MediaVideoReader`.

        Args:
            filename: path to the video to read
            video_dirs: list of paths pointing to folders which might contain the video

        Returns:
            An instance of `MediaVideoReader`.
        """
        filename = resolve_path(filename, video_dirs)
        return cls(filename)
