"""Utilities for writing videos."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import List, Optional, Type

import attrs
import imageio
import imageio.v2 as iio_v2
import numpy as np


@attrs.define
class VideoWriter:
    """Simple video writer using imageio and FFMPEG.

    Attributes:
        filename: Path to output video file.
        fps: Frames per second. Defaults to 30.
        pixelformat: Pixel format for video. Defaults to "yuv420p".
        codec: Codec to use for encoding. Defaults to "libx264".
        crf: Constant rate factor to control lossiness of video. Values go from 2 to 32,
            with numbers in the 18 to 30 range being most common. Lower values mean less
            compressed/higher quality. Defaults to 25. No effect if codec is not
            "libx264".
        preset: H264 encoding preset. Defaults to "superfast". No effect if codec is not
            "libx264".
        output_params: Additional output parameters for FFMPEG. This should be a list of
            strings corresponding to command line arguments for FFMPEG and libx264. Use
            `ffmpeg -h encoder=libx264` to see all options for libx264 output_params.

    Notes:
        This class can be used as a context manager to ensure the video is properly
        closed after writing. For example:

        ```python
        with VideoWriter("output.mp4") as writer:
            for frame in frames:
                writer(frame)
        ```
    """

    filename: Path = attrs.field(converter=Path)
    fps: float = 30
    pixelformat: str = "yuv420p"
    codec: str = "libx264"
    crf: int = 25
    preset: str = "superfast"
    output_params: list[str] = attrs.field(factory=list)
    _writer: "imageio.plugins.ffmpeg.FfmpegFormat.Writer" | None = None

    def build_output_params(self) -> list[str]:
        """Build the output parameters for FFMPEG."""
        output_params = []
        if self.codec == "libx264":
            output_params.extend(
                [
                    "-crf",
                    str(self.crf),
                    "-preset",
                    self.preset,
                ]
            )
        return output_params + self.output_params

    def open(self):
        """Open the video writer."""
        self.close()

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._writer = iio_v2.get_writer(
            self.filename.as_posix(),
            format="FFMPEG",
            fps=self.fps,
            codec=self.codec,
            pixelformat=self.pixelformat,
            output_params=self.build_output_params(),
        )

    def close(self):
        """Close the video writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame: Frame to write to video. Should be a 2D or 3D numpy array with
                dimensions (height, width) or (height, width, channels).
        """
        if self._writer is None:
            self.open()

        self._writer.append_data(frame)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Context manager exit."""
        self.close()
        return False

    def __call__(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame: Frame to write to video. Should be a 2D or 3D numpy array with
                dimensions (height, width) or (height, width, channels).
        """
        self.write_frame(frame)


@attrs.define
class MJPEGFrameWriter:
    """Video writer for MJPEG format optimized for seekable frame containers.

    This writer is designed for scientific/archival use where frames from arbitrary
    indices need to be independently seekable. Each frame is intra-coded (I-frame only)
    to ensure reliable random access.

    Attributes:
        filename: Path to output MJPEG video file.
        fps: Nominal frames per second. Defaults to 30.
        quality: MJPEG quality level (2-31, lower is better). Defaults to 2.

        output_params: Additional output parameters for FFMPEG.

    Notes:
        This class can be used as a context manager:

        ```python
        with MJPEGFrameWriter("output.avi") as writer:
            for frame in frames:
                writer.write_frame(frame)
        ```
    """

    filename: Path = attrs.field(converter=Path)
    fps: float = 30
    quality: int = 2

    output_params: list[str] = attrs.field(factory=list)
    _writer: "imageio.plugins.ffmpeg.FfmpegFormat.Writer" | None = None
    _frame_index: int = 0

    def build_output_params(self) -> list[str]:
        """Build the output parameters for FFMPEG MJPEG encoding."""
        params = [
            # MJPEG quality (2-32)
            "-q:v",
            str(self.quality),
            # All frames are keyframes (I-frames)
            "-g",
            "1",
            # Use full range (JPEG) color
            "-vf",
            "scale=in_range=pc:out_range=pc,format=yuv420p",
            "-color_range",
            "pc",
        ]

        return params + self.output_params

    def open(self):
        """Open the MJPEG writer."""
        self.close()
        self._frame_index = 0

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._writer = iio_v2.get_writer(
            self.filename.as_posix(),
            format="FFMPEG",
            fps=self.fps,
            codec="mjpeg",
            pixelformat="yuv420p",  # Use full range YUV for MJPEG
            output_params=self.build_output_params(),
        )

    def close(self):
        """Close the MJPEG writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._frame_index = 0

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the MJPEG video.

        Args:
            frame: Frame to write. Should be a 2D or 3D numpy array with
                dimensions (height, width) or (height, width, channels).
        """
        if self._writer is None:
            self.open()

        self._writer.append_data(frame)
        self._frame_index += 1

    def write_frames(self, frames: List[np.ndarray]):
        """Write multiple frames to the MJPEG video.

        Args:
            frames: List of frames to write.
        """
        for frame in frames:
            self.write_frame(frame)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Context manager exit."""
        self.close()
        return False
