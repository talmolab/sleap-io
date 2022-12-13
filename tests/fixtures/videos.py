"""Fixtures that return paths to video files."""
import pytest


@pytest.fixture
def video_slp_predictions():
    """Path to 1100 X 384 X 384 grayscale .mp4 video. Used in `slp_predictions` fixture."""
    return "tests/data/videos/centered_pair_low_quality.mp4"
