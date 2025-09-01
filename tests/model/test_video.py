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


def test_video_matches_path():
    """Test Video.matches_path() method."""
    # Create test videos with different paths
    video1 = Video(filename="/path/to/video.mp4", open_backend=False)
    video2 = Video(filename="/path/to/video.mp4", open_backend=False)
    video3 = Video(filename="/different/path/video.mp4", open_backend=False)
    video4 = Video(filename="/path/to/other.mp4", open_backend=False)

    # Test strict path matching
    assert video1.matches_path(video2, strict=True)
    assert not video1.matches_path(video3, strict=True)
    assert not video1.matches_path(video4, strict=True)

    # Test basename matching
    assert video1.matches_path(video2, strict=False)
    assert video1.matches_path(video3, strict=False)  # Same basename
    assert not video1.matches_path(video4, strict=False)


def test_video_matches_shape():
    """Test Video.matches_shape() method."""
    # Test with backend metadata (when backend is None)
    video1 = Video(filename="video1.mp4", open_backend=False)
    video1.backend_metadata["shape"] = (100, 480, 640, 3)

    video2 = Video(filename="video2.mp4", open_backend=False)
    video2.backend_metadata["shape"] = (50, 480, 640, 3)  # Different frames, same shape

    video3 = Video(filename="video3.mp4", open_backend=False)
    video3.backend_metadata["shape"] = (100, 720, 1280, 3)  # Different shape

    # Should match (same height, width, channels)
    assert video1.matches_shape(video2)
    assert not video1.matches_shape(video3)

    # Test with None shape (should return False)
    video4 = Video(filename="video4.mp4", open_backend=False)
    assert not video1.matches_shape(video4)  # video4 has None shape
    assert not video4.matches_shape(video1)  # video4 has None shape

    # Test with ImageVideo list
    video5 = Video(filename=["img1.jpg", "img2.jpg"], open_backend=False)
    video5.backend_metadata["shape"] = (2, 480, 640, 3)
    assert video1.matches_shape(video5)  # Same spatial dimensions


def test_video_has_overlapping_images():
    """Test Video.has_overlapping_images() method."""
    # Test with ImageVideo (lists)
    video1 = Video(
        filename=["/path/to/img1.jpg", "/path/to/img2.jpg"], open_backend=False
    )
    video2 = Video(filename=["/other/img2.jpg", "/other/img3.jpg"], open_backend=False)
    video3 = Video(filename=["/path/img4.jpg", "/path/img5.jpg"], open_backend=False)

    # Should detect overlap (img2.jpg is in both)
    assert video1.has_overlapping_images(video2)

    # No overlap
    assert not video1.has_overlapping_images(video3)

    # Test with non-ImageVideo (should return False)
    regular_video = Video(filename="video.mp4", open_backend=False)
    assert not video1.has_overlapping_images(regular_video)
    assert not regular_video.has_overlapping_images(video1)
    assert not regular_video.has_overlapping_images(regular_video)


def test_video_deduplicate_with():
    """Test Video.deduplicate_with() method."""
    # Test with ImageVideo
    video1 = Video(
        filename=["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg"],
        open_backend=False,
    )
    video2 = Video(filename=["/other/img2.jpg", "/other/img4.jpg"], open_backend=False)

    # Deduplicate - should remove img2.jpg from video1
    deduped = video1.deduplicate_with(video2)
    assert len(deduped.filename) == 2
    assert "img1.jpg" in Path(deduped.filename[0]).name
    assert "img3.jpg" in Path(deduped.filename[1]).name

    # Test when all images are duplicates (returns None)
    video3 = Video(filename=["/path/img2.jpg"], open_backend=False)
    assert video3.deduplicate_with(video2) is None

    # Test with non-ImageVideo (should raise ValueError)
    regular_video = Video(filename="video.mp4", open_backend=False)
    with pytest.raises(ValueError, match="deduplicate_with only works with ImageVideo"):
        regular_video.deduplicate_with(video2)

    with pytest.raises(ValueError, match="Other video must also be ImageVideo"):
        video1.deduplicate_with(regular_video)


def test_video_merge_with():
    """Test Video.merge_with() method."""
    # Test with ImageVideo
    video1 = Video(filename=["/path/img1.jpg", "/path/img2.jpg"], open_backend=False)
    video1.grayscale = False
    video2 = Video(filename=["/other/img2.jpg", "/other/img3.jpg"], open_backend=False)

    # Merge - should combine unique images
    merged = video1.merge_with(video2)
    assert len(merged.filename) == 3  # img1, img2, img3 (deduplicated)
    basenames = [Path(f).name for f in merged.filename]
    assert "img1.jpg" in basenames
    assert "img2.jpg" in basenames
    assert "img3.jpg" in basenames
    assert basenames.count("img2.jpg") == 1  # No duplicates

    # Test with non-ImageVideo (should raise ValueError)
    regular_video = Video(filename="video.mp4", open_backend=False)
    with pytest.raises(ValueError, match="merge_with only works with ImageVideo"):
        regular_video.merge_with(video2)

    with pytest.raises(ValueError, match="Other video must also be ImageVideo"):
        video1.merge_with(regular_video)  # Different basename

    # Test with image sequences
    video_seq1 = Video(filename=["img1.png", "img2.png"], open_backend=False)
    video_seq2 = Video(filename=["img1.png", "img2.png"], open_backend=False)
    video_seq3 = Video(filename=["img3.png", "img4.png"], open_backend=False)

    assert video_seq1.matches_path(video_seq2, strict=True)
    assert not video_seq1.matches_path(video_seq3, strict=True)

    # Test mixed types
    assert not video1.matches_path(video_seq1, strict=False)


def test_video_matches_content():
    """Test Video.matches_content() method."""
    # Create videos with mock shapes
    video1 = Video(filename="test1.mp4", open_backend=False)
    video1.backend_metadata["shape"] = (100, 480, 640, 3)

    video2 = Video(filename="test2.mp4", open_backend=False)
    video2.backend_metadata["shape"] = (100, 480, 640, 3)

    video3 = Video(filename="test3.mp4", open_backend=False)
    video3.backend_metadata["shape"] = (50, 480, 640, 3)  # Different frames

    video4 = Video(filename="test4.mp4", open_backend=False)
    video4.backend_metadata["shape"] = (100, 240, 320, 3)  # Different resolution

    # Test matching content
    assert video1.matches_content(video2)  # Same shape
    assert not video1.matches_content(video3)  # Different frame count
    assert not video1.matches_content(video4)  # Different resolution

    # Test with no shape metadata
    video5 = Video(filename="test5.mp4", open_backend=False)
    video6 = Video(filename="test6.mp4", open_backend=False)
    assert video5.matches_content(video6)  # Both have no shape


def test_video_matches_path_image_sequences_strict_false():
    """Test Video.matches_path() with image sequences when strict=False."""
    # Test image sequences with strict=False (lines 454-458 in video.py)
    video_seq1 = Video(
        filename=["frame001.jpg", "frame002.jpg", "frame003.jpg"], open_backend=False
    )
    video_seq3 = Video(
        filename=[
            "/different/path/frame001.jpg",
            "/different/path/frame002.jpg",
            "/different/path/frame003.jpg",
        ],
        open_backend=False,
    )

    # Same basenames, strict=False should match (lines 455-457)
    assert video_seq1.matches_path(video_seq3, strict=False)

    # Different basenames
    video_seq4 = Video(
        filename=["img001.jpg", "img002.jpg", "img003.jpg"], open_backend=False
    )
    assert not video_seq1.matches_path(video_seq4, strict=False)


def test_video_matches_content_backend_edge_cases():
    """Test Video.matches_content() backend comparison edge cases."""
    # Test backend comparison edge cases (lines 491-494 in video.py)
    from sleap_io.io.video_reading import HDF5Video, ImageVideo, MediaVideo

    video1 = Video(filename="test1.mp4", open_backend=False)
    video2 = Video(filename="test2.mp4", open_backend=False)

    # Both have None backend
    video1.backend = None
    video2.backend = None
    assert video1.matches_content(video2)  # Line 492: both None returns True

    # One has backend, other doesn't (lines 493-494)
    video3 = Video(filename="test3.mp4", open_backend=False)
    video3.backend = None

    video4 = Video(filename="test4.mp4", open_backend=False)
    # Mock a backend
    video4.backend = type("MockBackend", (), {})()

    assert not video3.matches_content(video4)  # One is None, other isn't
    assert not video4.matches_content(video3)  # Reverse check

    # Test different backend types (line 494)
    video5 = Video(filename="test5.mp4", open_backend=False)
    video6 = Video(filename="test6.h5", open_backend=False)

    # Set same shape for both
    video5.backend_metadata["shape"] = (100, 480, 640, 3)
    video6.backend_metadata["shape"] = (100, 480, 640, 3)

    # Create backends of different types
    video5.backend = MediaVideo(filename="test5.mp4", grayscale=False)
    video6.backend = HDF5Video(filename="test6.h5", grayscale=False)

    # Should not match even with same shape because backend types differ
    assert not video5.matches_content(video6)

    # Also test with ImageVideo
    video7 = Video(filename=["img1.jpg", "img2.jpg"], open_backend=False)
    video7.backend_metadata["shape"] = (100, 480, 640, 3)
    video7.backend = ImageVideo(filename=["img1.jpg", "img2.jpg"], grayscale=False)

    assert not video5.matches_content(video7)  # MediaVideo vs ImageVideo
    assert not video6.matches_content(video7)  # HDF5Video vs ImageVideo
