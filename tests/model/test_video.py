"""Tests for methods in the sleap_io.model.video file."""

from sleap_io import Video
from sleap_io.io.video import MediaVideo
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
    assert type(test_video.backend) == MediaVideo


def test_video_getitem(centered_pair_low_quality_video):
    img = centered_pair_low_quality_video[0]
    assert img.shape == (384, 384, 1)
    assert img.dtype == np.uint8


def test_video_repr(centered_pair_low_quality_video):
    assert str(centered_pair_low_quality_video) == (
        'Video(filename="tests/data/videos/centered_pair_low_quality.mp4", '
        "shape=(1100, 384, 384, 1), backend=MediaVideo)"
    )


def test_video_exists(centered_pair_low_quality_video):
    video = Video("test.mp4")
    assert video.exists() is False

    assert centered_pair_low_quality_video.exists() is True


def test_video_open_close(centered_pair_low_quality_path):
    video = Video(centered_pair_low_quality_path)
    assert video.is_open is False
    assert video.backend is None

    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.is_open is True
    assert type(video.backend) == MediaVideo

    video.close()
    assert video.is_open is False
    assert video.backend is None
    assert video.shape is None

    video.open()
    assert video.is_open is True
    assert type(video.backend) == MediaVideo
    assert video[0].shape == (384, 384, 1)

    video = Video.from_filename(centered_pair_low_quality_path, grayscale=False)
    assert video.shape == (1100, 384, 384, 3)
    video.open()
    assert video.shape == (1100, 384, 384, 3)
    video.open(grayscale=True)
    assert video.shape == (1100, 384, 384, 1)


def test_video_replace_filename(centered_pair_low_quality_path):
    video = Video.from_filename("test.mp4")
    assert video.exists() is False

    video.replace_filename(centered_pair_low_quality_path)
    assert video.exists() is True
    assert video.is_open is True
    assert type(video.backend) == MediaVideo

    video.replace_filename("test.mp4")
    assert video.exists() is False
    assert video.is_open is False

    video.replace_filename(centered_pair_low_quality_path, open=False)
    assert video.exists() is True
    assert video.is_open is False
    assert video.backend is None
