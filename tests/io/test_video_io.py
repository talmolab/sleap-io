"""Tests for functions in the sleap_io.io.video folder."""
import pytest
from pathlib import Path
from sleap_io.io.video.media import MediaVideoReader


def test_media_video_backend(video_slp_predictions):
    """Test media videos can be read."""
    filename = video_slp_predictions
    video = MediaVideoReader.read_media_video(filename)

    # Ensure metadata read correctly
    assert Path(video.file).resolve() == Path(filename).resolve()
    assert video.frame_shape == (384, 384, 3)
    assert video.grayscale == True
    assert video.channels == 1
    assert video.video_shape == (384, 384, 1, 1100)
