"""Fixtures for video and media files."""

import pytest
import sleap_io


@pytest.fixture
def centered_pair_low_quality_path():
    """Path to a video with two flies in the center."""
    return "tests/data/videos/centered_pair_low_quality.mp4"


@pytest.fixture
def centered_pair_low_quality_video(centered_pair_low_quality_path):
    """Video with two flies in the center."""
    return sleap_io.Video.from_filename(centered_pair_low_quality_path)


@pytest.fixture
def centered_pair_frame_paths():
    """Paths to three frames of a video with two flies in the center."""
    return [
        "tests/data/videos/imgs/img.00.jpg",
        "tests/data/videos/imgs/img.01.jpg",
        "tests/data/videos/imgs/img.02.jpg",
    ]
