"""Tests for methods in the sleap_io.model.video file."""
from sleap_io import Video
import numpy as np


def test_video_class():
    """Test initialization of `Video` object."""
    test_video = Video(filename="123.mp4")
    assert test_video.filename == "123.mp4"
    assert test_video.backend is None
    assert test_video.shape is None


def test_video_from_filename(centered_pair_low_quality_path):
    """Test initialization of `Video` object from filename."""
    test_video = Video.from_filename(centered_pair_low_quality_path)
    assert test_video.filename == centered_pair_low_quality_path
    assert test_video.shape == (1100, 384, 384, 1)


def test_video_getitem(centered_pair_low_quality_video):
    img = centered_pair_low_quality_video[0]
    assert img.shape == (384, 384, 1)
    assert img.dtype == np.uint8


def test_video_repr(centered_pair_low_quality_video):
    assert str(centered_pair_low_quality_video) == (
        'Video(filename="tests/data/videos/centered_pair_low_quality.mp4", '
        "shape=(1100, 384, 384, 1), backend=MediaVideo)"
    )
