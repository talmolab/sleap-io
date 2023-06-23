"""Tests for methods in the sleap_io.model.video file."""
from sleap_io import Video

import imageio
import imageio_ffmpeg
import av


def test_video_class():
    """Test initialization of `Video` object."""
    test_video = Video(filename="123.mp4")
    assert test_video.filename == "123.mp4"
