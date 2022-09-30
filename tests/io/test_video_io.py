"""Tests for functions in the sleap_io.io.video folder."""
import pytest
from pathlib import Path
from sleap_io.io.video.media import MediaVideoReader


def test_media_video_backend(video_slp_predictions):
    filename = video_slp_predictions
    video = MediaVideoReader.read_media_video(filename)

    # Load frames without error - manually tested that correct frames loaded
    video[0]
    video[-1]

    # Ensure metadata read correctly
    assert Path(video.file).resolve() == Path(filename).resolve()
    assert video.video_shape == (384, 384, 3, 1100)
