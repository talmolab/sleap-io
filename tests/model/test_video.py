"""Tests for methods in the sleap_io.model.video file."""

from pathlib import Path

import numpy as np
import pytest

from sleap_io import Video
from sleap_io.io.video_reading import ImageVideo, MediaVideo


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
    assert len(test_video) == 1100
    assert type(test_video.backend) is MediaVideo


def test_video_getitem(centered_pair_low_quality_video):
    img = centered_pair_low_quality_video[0]
    assert img.shape == (384, 384, 1)
    assert img.dtype == np.uint8


def test_video_repr(centered_pair_low_quality_video):
    assert str(centered_pair_low_quality_video) == (
        'Video(filename="tests/data/videos/centered_pair_low_quality.mp4", '
        "shape=(1100, 384, 384, 1), backend=MediaVideo)"
    )


def test_video_exists(centered_pair_low_quality_video, centered_pair_frame_paths):
    video = Video("test.mp4")
    assert video.exists() is False

    assert centered_pair_low_quality_video.exists() is True

    video = Video(centered_pair_frame_paths)
    assert video.exists() is True
    assert video.exists(check_all=True) is True

    video = Video([centered_pair_frame_paths[0], "fake.jpg"])
    assert video.exists() is True
    assert video.exists(check_all=True) is False

    video = Video(["fake.jpg", centered_pair_frame_paths[0]])
    assert video.exists() is False
    assert video.exists(check_all=True) is False


def test_video_open_close(centered_pair_low_quality_path, centered_pair_frame_paths):
    video = Video(centered_pair_low_quality_path)
    assert video.is_open
    assert type(video.backend) is MediaVideo

    img = video[0]
    assert img.shape == (384, 384, 1)
    assert video.is_open is True

    video = Video("test.mp4")
    assert video.is_open is False
    with pytest.raises(FileNotFoundError):
        video[0]

    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.is_open is True
    assert type(video.backend) is MediaVideo

    video.close()
    assert video.is_open is False
    assert video.backend is None
    assert video.shape == (1100, 384, 384, 1)

    video.open()
    assert video.is_open is True
    assert type(video.backend) is MediaVideo
    assert video[0].shape == (384, 384, 1)

    video = Video.from_filename(centered_pair_low_quality_path, grayscale=False)
    assert video.shape == (1100, 384, 384, 3)
    video.open()
    assert video.shape == (1100, 384, 384, 3)
    video.open(grayscale=True)
    assert video.shape == (1100, 384, 384, 1)

    video.open(centered_pair_frame_paths)
    assert video.shape == (3, 384, 384, 1)
    assert type(video.backend) is ImageVideo


def test_video_replace_filename(
    centered_pair_low_quality_path, centered_pair_frame_paths
):
    video = Video.from_filename("test.mp4")
    assert video.exists() is False

    video.replace_filename(centered_pair_low_quality_path)
    assert video.exists() is True
    assert video.is_open is True
    assert type(video.backend) is MediaVideo

    video.replace_filename(Path(centered_pair_low_quality_path))
    assert video.exists() is True
    assert video.is_open is True
    assert type(video.backend) is MediaVideo

    video.replace_filename("test.mp4")
    assert video.exists() is False
    assert video.is_open is False
    assert video.backend_metadata["filename"] == "test.mp4"

    video.replace_filename(centered_pair_low_quality_path, open=False)
    assert video.exists() is True
    assert video.is_open is False
    assert video.backend is None

    video = Video.from_filename(["fake.jpg", "fake2.jpg", "fake3.jpg"])
    assert type(video.backend) is ImageVideo
    video.replace_filename(centered_pair_frame_paths)
    assert type(video.backend) is ImageVideo
    assert video.exists(check_all=True) is True


def test_grayscale(centered_pair_low_quality_path):
    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.grayscale is True
    assert video.shape[-1] == 1

    video.grayscale = False
    assert video.shape[-1] == 3

    video.close()
    video.open()
    assert video.grayscale is False
    assert video.shape[-1] == 3

    video.grayscale = True
    video.close()
    video.open()
    assert video.grayscale is True
    assert video.shape[-1] == 1

    video.close()
    assert "grayscale" in video.backend_metadata
    assert video.grayscale is True

    video.backend_metadata = {"shape": (1100, 384, 384, 3)}
    assert video.grayscale is False

    video.open()
    assert video.grayscale is False


def test_open_backend_preference(centered_pair_low_quality_path):
    video = Video(centered_pair_low_quality_path)
    assert video.is_open
    assert type(video.backend) is MediaVideo

    video = Video(centered_pair_low_quality_path, open_backend=False)
    assert video.is_open is False
    assert video.backend is None
    with pytest.raises(ValueError):
        video[0]

    video.open_backend = True
    _ = video[0]
    assert video.is_open
    assert type(video.backend) is MediaVideo


def test_safe_video_open(slp_minimal_pkg):
    video = Video(slp_minimal_pkg, backend_metadata={"dataset": "video0/video"})
    assert video.is_open

    video = Video(slp_minimal_pkg, backend_metadata={"dataset": "video999/video"})
    assert not video.is_open


def test_video_set_plugin(centered_pair_low_quality_path):
    """Test Video.set_video_plugin() method."""
    import sleap_io as sio

    # Clear any global default
    sio.set_default_video_plugin(None)

    # Create video
    video = Video.from_filename(centered_pair_low_quality_path)
    video.open()
    assert video.is_open

    # Get initial plugin (auto-detected)
    initial_plugin = video.backend.plugin
    assert initial_plugin in ["opencv", "FFMPEG", "pyav"]

    # Change plugin
    video.set_video_plugin("FFMPEG")
    assert video.backend.plugin == "FFMPEG"
    assert video.backend_metadata["plugin"] == "FFMPEG"

    # Test with alias
    video.set_video_plugin("cv2")
    assert video.backend.plugin == "opencv"

    # Test error for non-media video
    hdf5_video = Video.from_filename("test.h5")
    with pytest.raises(ValueError, match="Cannot set plugin for non-media video"):
        hdf5_video.set_video_plugin("opencv")


def test_video_open_with_plugin(centered_pair_low_quality_path):
    """Test Video.open() with plugin parameter."""
    import sleap_io as sio

    # Clear any global default
    sio.set_default_video_plugin(None)

    # Create video without opening
    video = Video(centered_pair_low_quality_path, open_backend=False)
    assert not video.is_open

    # Open with specific plugin
    video.open(plugin="FFMPEG")
    assert video.is_open
    assert video.backend.plugin == "FFMPEG"
    assert video.backend_metadata["plugin"] == "FFMPEG"

    # Close and reopen with different plugin
    video.close()
    video.open(plugin="opencv")
    assert video.backend.plugin == "opencv"

    # Test plugin alias
    video.close()
    video.open(plugin="cv2")
    assert video.backend.plugin == "opencv"


def test_labels_set_video_plugin(centered_pair_low_quality_path):
    """Test Labels.set_video_plugin() method."""
    import sleap_io as sio
    from sleap_io import LabeledFrame, Labels

    # Clear any global default
    sio.set_default_video_plugin(None)

    # Create labels with multiple videos
    video1 = Video.from_filename(centered_pair_low_quality_path)
    video2 = Video.from_filename(
        centered_pair_low_quality_path
    )  # Same file, different instance

    labels = Labels()
    labels.videos.extend([video1, video2])
    labels.labeled_frames.append(LabeledFrame(video=video1, frame_idx=0))
    labels.labeled_frames.append(LabeledFrame(video=video2, frame_idx=1))

    # Open videos
    video1.open()
    video2.open()

    # Change plugin for all videos
    labels.set_video_plugin("FFMPEG")

    # Check both videos were updated
    assert video1.backend.plugin == "FFMPEG"
    assert video2.backend.plugin == "FFMPEG"

    # Test with alias
    labels.set_video_plugin("cv2")
    assert video1.backend.plugin == "opencv"
    assert video2.backend.plugin == "opencv"
