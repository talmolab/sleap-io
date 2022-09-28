"""Tests for functions in the sleap_io.io.video folder."""
import pytest
from sleap_io.io.video.media import MediaVideoReader

def test_media_video_backend(video_slp_predictions):
    filename = video_slp_predictions
    video = MediaVideoReader(filename)

    assert video.file == filename
    assert video.height == 384
    assert video.width == 384
    assert video.channels == 3
    assert video.n_frames == 1100
