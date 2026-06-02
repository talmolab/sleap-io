"""Tests for methods in the sleap_io.model.video file."""

import copy
from pathlib import Path

import numpy as np
import pytest

from sleap_io import Video
from sleap_io.io.video_reading import (
    CropVideoBackend,
    HDF5Video,
    ImageVideo,
    MediaVideo,
)
from sleap_io.model.video import _resolve_crop_rect
from sleap_io.transform.frame import crop_frame


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


def _count_requests(httpserver, path: str) -> int:
    """Count the requests in the server log matching the given path."""
    return sum(1 for req, _ in httpserver.log if req.path == path)


def test_video_exists_url_cached_within_ttl(httpserver):
    """A URL `exists()` probes the server once and reuses the cache within TTL."""
    httpserver.expect_request("/labels.slp").respond_with_data(
        b"hello world data", content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    video = Video(filename=url, open_backend=False)

    assert video.exists() is True
    first = _count_requests(httpserver, "/labels.slp")
    assert first == 1

    # Second call within the TTL must NOT hit the server again.
    assert video.exists() is True
    second = _count_requests(httpserver, "/labels.slp")
    assert second == first


def test_video_exists_url_404(httpserver):
    """A URL that 404s returns False from `exists()`."""
    httpserver.expect_request("/missing.slp").respond_with_data("nope", status=404)
    url = httpserver.url_for("/missing.slp")

    video = Video(filename=url, open_backend=False)
    assert video.exists() is False


def test_video_exists_replace_filename_clears_cache(httpserver, monkeypatch):
    """`replace_filename` clears the cached URL existence result."""
    # Force a short TTL so we control caching behavior deterministically.
    monkeypatch.setenv("SLEAP_IO_EXISTS_TTL", "1000")
    httpserver.expect_request("/labels.slp").respond_with_data(
        b"hello world data", content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    video = Video(filename=url, open_backend=False)
    assert video.exists() is True
    assert video._exists_cache  # populated

    # Replacing the filename must invalidate the cache.
    video.replace_filename(url, open=False)
    assert video._exists_cache == {}

    # The next call re-probes the server (a second request appears).
    assert video.exists() is True
    assert _count_requests(httpserver, "/labels.slp") == 2


def test_video_exists_url_malformed_ttl_env(httpserver, monkeypatch):
    """A malformed `SLEAP_IO_EXISTS_TTL` must not break the never-raise contract.

    `exists()` (and `is_open`, which a GUI render loop polls) is contractually a
    bool predicate; a bad env value should fall back to the default TTL rather
    than raising `ValueError`.
    """
    monkeypatch.setenv("SLEAP_IO_EXISTS_TTL", "notanumber")
    httpserver.expect_request("/labels.slp").respond_with_data(
        b"hello world data", content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    video = Video(filename=url, open_backend=False)
    # Must return a bool, not raise.
    assert video.exists() is True


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


def test_video_matches_path_hdf5_source_filename():
    """Test Video.matches_path() for HDF5 backends requires matching datasets.

    For HDF5 videos, different datasets always mean different videos, even if
    they share the same source_filename. This is important for pkg.slp files
    where all embedded videos may share the same intermediate source_filename.
    """
    # Create two HDF5 videos with same source_filename but different datasets
    video1 = Video(filename="/path/to/labels.slp", open_backend=False)
    video1.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1.backend.source_filename = "/original/path/video_a.mp4"
    video1.backend.dataset = "video0/video"

    video2 = Video(filename="/path/to/labels.slp", open_backend=False)
    video2.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video2.backend.source_filename = "/original/path/video_a.mp4"
    video2.backend.dataset = "video1/video"

    # Same source_filename AND same dataset = match
    video1_copy = Video(filename="/path/to/labels.slp", open_backend=False)
    video1_copy.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1_copy.backend.source_filename = "/original/path/video_a.mp4"
    video1_copy.backend.dataset = "video0/video"

    assert video1.matches_path(video1_copy, strict=False)
    assert video1.matches_path(video1_copy, strict=True)

    # Same source_filename but different datasets = NO match (different videos)
    assert not video1.matches_path(video2, strict=False)
    assert not video1.matches_path(video2, strict=True)

    # Different source_filenames should NOT match
    video3 = Video(filename="/path/to/labels.slp", open_backend=False)
    video3.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video3.backend.source_filename = "/original/path/video_b.mp4"
    video3.backend.dataset = "video2/video"

    assert not video1.matches_path(video3, strict=False)
    assert not video1.matches_path(video3, strict=True)


def test_video_matches_path_hdf5_source_filename_basename():
    """Test basename matching for HDF5 source_filename when strict=False.

    For HDF5 videos, datasets must match first. Then source_filename is compared
    (basename only for strict=False, full path for strict=True).
    """
    # Videos with same dataset and same source_filename basename
    video1 = Video(filename="/path/to/labels.slp", open_backend=False)
    video1.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1.backend.source_filename = "/original/path/video.mp4"
    video1.backend.dataset = "video0/video"

    video2 = Video(filename="/path/to/labels.slp", open_backend=False)
    video2.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video2.backend.source_filename = "/different/path/video.mp4"
    video2.backend.dataset = "video0/video"  # Same dataset

    # Same dataset + same basename = match when strict=False
    assert video1.matches_path(video2, strict=False)

    # Same dataset + different full path = no match when strict=True
    assert not video1.matches_path(video2, strict=True)

    # Different datasets = no match, regardless of source_filename
    video3 = Video(filename="/path/to/labels.slp", open_backend=False)
    video3.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video3.backend.source_filename = "/original/path/video.mp4"
    video3.backend.dataset = "video1/video"  # Different dataset

    assert not video1.matches_path(video3, strict=False)
    assert not video1.matches_path(video3, strict=True)


def test_video_matches_path_hdf5_fallback_to_dataset():
    """Test fallback to dataset matching when source_filename is None."""
    video1 = Video(filename="/path/to/labels.slp", open_backend=False)
    video1.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1.backend.source_filename = None
    video1.backend.dataset = "video0/video"

    video2 = Video(filename="/path/to/labels.slp", open_backend=False)
    video2.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video2.backend.source_filename = None
    video2.backend.dataset = "video0/video"

    video3 = Video(filename="/path/to/labels.slp", open_backend=False)
    video3.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video3.backend.source_filename = None
    video3.backend.dataset = "video1/video"

    # Same dataset should match when source_filename is None
    assert video1.matches_path(video2, strict=False)
    assert video1.matches_path(video2, strict=True)

    # Different datasets should NOT match
    assert not video1.matches_path(video3, strict=False)
    assert not video1.matches_path(video3, strict=True)


def test_video_matches_path_hdf5_no_source_or_dataset():
    """Test returns False when neither source_filename nor dataset is available."""
    video1 = Video(filename="/path/to/labels.slp", open_backend=False)
    video1.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1.backend.source_filename = None
    video1.backend.dataset = None

    video2 = Video(filename="/path/to/labels.slp", open_backend=False)
    video2.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video2.backend.source_filename = None
    video2.backend.dataset = None

    # Should return False when neither source_filename nor dataset is available
    assert not video1.matches_path(video2, strict=False)
    assert not video1.matches_path(video2, strict=True)


def test_video_matches_path_hdf5_mixed_with_non_hdf5():
    """Test Video.matches_path() with one HDF5 and one non-HDF5 video."""
    video1 = Video(filename="/path/to/labels.slp", open_backend=False)
    video1.backend = HDF5Video(filename="/path/to/labels.slp", grayscale=False)
    video1.backend.source_filename = "/original/path/video.mp4"
    video1.backend.dataset = "video0/video"

    video2 = Video(filename="/path/to/video.mp4", open_backend=False)
    # No backend (or non-HDF5 backend), so it uses default filename matching

    # Should use default filename matching (basenames don't match)
    assert not video1.matches_path(video2, strict=False)


def test_video_matches_content_backend_edge_cases():
    """Test Video.matches_content() backend comparison edge cases."""
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


def test_video_deepcopy_preserves_source_video_chain():
    """Test that Video.__deepcopy__() preserves source_video chain.

    The source_video attribute (and the computed original_video property) is used
    for provenance tracking during merge operations. If source_video is not
    preserved during deepcopy, the provenance chain breaks and _get_root_video()
    in matching.py cannot correctly identify related videos.

    Note: original_video is now a computed property that traverses source_video chain.
    """
    # Create a chain: original -> source -> video
    original = Video(filename="/data/original.mp4", open_backend=False)
    source = Video(
        filename="/embedded/source.slp", source_video=original, open_backend=False
    )
    video = Video(
        filename="/current/video.mp4",
        source_video=source,
        open_backend=False,
    )

    # Verify original_video is computed correctly from chain
    assert video.original_video is original

    # Deepcopy should preserve source_video chain
    video_copy = copy.deepcopy(video)

    assert video_copy.filename == video.filename
    assert video_copy.source_video is video.source_video
    # original_video (computed) should give same result through the chain
    assert video_copy.original_video is video.original_video
    assert video_copy.original_video is original


# =============================================================================
# Tests for is_same_file() - Function in matching module
# =============================================================================
# These tests document expected behavior for the is_same_file() function
# that definitively checks if two Video objects reference the same underlying file.


class TestVideoIsSameFile:
    """Tests for is_same_file() function from sleap_io.model.matching.

    This function is needed to prevent duplicate video creation and enable
    correct video matching during merge operations. It differs from
    Video.matches_path() in that it:

    1. Uses os.path.samefile() when files are accessible (handles symlinks)
    2. Resolves paths before comparison (handles relative vs absolute)
    3. Checks source_video for embedded/PKG.SLP videos
    4. Handles ImageVideo list comparison correctly

    Background: The Video class uses eq=False for identity-based comparison,
    but this causes duplicate video creation when the same file is added
    multiple times as different Video objects. The is_same_file() function
    provides definitive file identity checking.
    """

    def test_is_same_file_same_path_different_objects(self):
        """Two Video objects with identical paths should be same file.

        Use case: User accidentally adds the same video twice via GUI.
        The GUI uses `video not in labels.videos` which fails because
        Video uses identity comparison (eq=False). is_same_file() should
        correctly identify these as the same file.
        """
        from sleap_io.model.matching import is_same_file

        video1 = Video(filename="/data/video.mp4", open_backend=False)
        video2 = Video(filename="/data/video.mp4", open_backend=False)

        # Identity comparison fails (this is expected)
        assert video1 is not video2

        # is_same_file should recognize they're the same file
        assert is_same_file(video1, video2)
        assert is_same_file(video2, video1)  # Symmetric

    def test_is_same_file_different_paths(self):
        """Videos with different paths should not be same file."""
        from sleap_io.model.matching import is_same_file

        video1 = Video(filename="/data/video_a.mp4", open_backend=False)
        video2 = Video(filename="/data/video_b.mp4", open_backend=False)

        assert not is_same_file(video1, video2)
        assert not is_same_file(video2, video1)

    def test_is_same_file_source_video_match(self):
        """Embedded video with source_video pointing to external should match.

        Use case (UC3): Merging predictions from PKG.SLP back into parent SLP.
        The PKG.SLP has embedded frames with source_video pointing to the
        original external video. The AUTO matcher should recognize these
        as the same video by checking source_video.

        This is a CRITICAL gap identified in the merge red-team investigation:
        The current AUTO matcher does NOT check source_video, causing
        predictions to be attributed to wrong videos or creating duplicates.
        """
        from sleap_io.model.matching import is_same_file

        # External video in base labels
        external = Video(filename="/data/recordings/video.mp4", open_backend=False)

        # Embedded video from PKG.SLP with source_video
        source = Video(filename="/data/recordings/video.mp4", open_backend=False)
        embedded = Video(
            filename="predictions.pkg.slp", source_video=source, open_backend=False
        )

        # is_same_file(embedded, external) should return True
        # because embedded.source_video points to external
        assert is_same_file(embedded, external)
        assert is_same_file(external, embedded)

    def test_is_same_file_nested_source_video(self):
        """Provenance chain: embedded → intermediate → original.

        Real-world example from Tiernon dataset: A PKG.SLP may have been
        created from another PKG.SLP, creating a 3-level provenance chain.
        is_same_file() should traverse the chain to find the original.
        """
        from sleap_io.model.matching import is_same_file

        # Original external video
        original = Video(filename="/data/video.mp4", open_backend=False)

        # First embedding (intermediate)
        intermediate = Video(
            filename="intermediate.pkg.slp", source_video=original, open_backend=False
        )

        # Second embedding (final)
        final = Video(
            filename="final.pkg.slp", source_video=intermediate, open_backend=False
        )

        # final should be recognized as same file as original
        assert is_same_file(final, original)
        assert is_same_file(original, final)

    def test_is_same_file_imagevideo_same_images(self):
        """ImageVideo with same image list should be same file.

        ImageVideo uses a list of paths. Two ImageVideos with the same
        image list should be recognized as the same file.
        """
        from sleap_io.model.matching import is_same_file

        paths = ["/data/img_000.jpg", "/data/img_001.jpg", "/data/img_002.jpg"]
        video1 = Video(filename=paths.copy(), open_backend=False)
        video2 = Video(filename=paths.copy(), open_backend=False)

        assert is_same_file(video1, video2)

    def test_is_same_file_imagevideo_different_order(self):
        """ImageVideo with same images in different order - NOT same file.

        Frame indices are tied to image order, so different ordering
        means different frame indexing = different video.
        """
        from sleap_io.model.matching import is_same_file

        video1 = Video(
            filename=["/data/img_000.jpg", "/data/img_001.jpg"], open_backend=False
        )
        video2 = Video(
            filename=["/data/img_001.jpg", "/data/img_000.jpg"], open_backend=False
        )

        assert not is_same_file(video1, video2)

    def test_is_same_file_cross_platform_paths(self):
        """Cross-platform paths with same basename should NOT auto-match.

        is_same_file() should be conservative - different absolute paths
        on different operating systems should NOT be considered the same
        file unless we can verify they point to the same file (which
        requires file access or explicit path mapping).

        This differs from matches_path(strict=False) which only checks
        basenames and can incorrectly match different files.
        """
        from sleap_io.model.matching import is_same_file

        windows_path = Video(filename=r"C:\data\video.mp4", open_backend=False)
        linux_path = Video(filename="/home/user/data/video.mp4", open_backend=False)

        # is_same_file should be conservative - can't verify across systems
        assert not is_same_file(windows_path, linux_path)

    def test_is_same_file_same_basename_different_dir(self):
        """Same basename in different directories should NOT match.

        This is a critical safety check: two files with the same name
        but in different directories are different files.

        Real-world example from Sky Shi dataset: Multiple experiments
        may have `fly.mp4` in different directories. These should never
        be confused.
        """
        from sleap_io.model.matching import is_same_file

        video1 = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video2 = Video(filename="/data/exp2/fly.mp4", open_backend=False)

        assert not is_same_file(video1, video2)


def test_video_fps_from_mediavideo(centered_pair_low_quality_path):
    """Test FPS property on Video with MediaVideo backend."""
    video = Video.from_filename(centered_pair_low_quality_path)
    assert isinstance(video.backend, MediaVideo)

    # Should read FPS from video container
    assert video.fps == 15.0


def test_video_fps_explicit_set(centered_pair_low_quality_path):
    """Test setting FPS explicitly on Video."""
    video = Video.from_filename(centered_pair_low_quality_path)

    # Override FPS
    video.fps = 30.0
    assert video.fps == 30.0

    # Reset to container value
    video.fps = None
    assert video.fps == 15.0


def test_video_fps_backend_metadata(centered_pair_low_quality_path):
    """Test FPS is cached in backend_metadata on close."""
    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.fps == 15.0

    # Close the backend
    video.close()
    assert video.backend is None

    # FPS should still be available from backend_metadata
    assert video.fps == 15.0
    assert video.backend_metadata.get("fps") == 15.0


def test_video_fps_imagevideo(centered_pair_frame_paths):
    """Test FPS property on Video with ImageVideo backend."""
    video = Video.from_filename(centered_pair_frame_paths)
    assert isinstance(video.backend, ImageVideo)

    # ImageVideo has no inherent FPS
    assert video.fps is None

    # Can set FPS explicitly
    video.fps = 24.0
    assert video.fps == 24.0
    assert video.backend_metadata.get("fps") == 24.0


def test_video_fps_validation():
    """Test FPS setter validates input."""
    video = Video(filename="test.mp4", open_backend=False)

    with pytest.raises(ValueError, match="FPS must be positive"):
        video.fps = 0

    with pytest.raises(ValueError, match="FPS must be positive"):
        video.fps = -5.0


def test_video_frame_to_seconds(centered_pair_low_quality_path):
    """Test frame_to_seconds conversion."""
    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.fps == 15.0

    # Test conversion
    assert video.frame_to_seconds(0) == 0.0
    assert video.frame_to_seconds(15) == 1.0
    assert video.frame_to_seconds(30) == 2.0
    assert video.frame_to_seconds(150) == 10.0

    # Test with fractional results
    assert abs(video.frame_to_seconds(1) - 1 / 15) < 1e-10


def test_video_seconds_to_frame(centered_pair_low_quality_path):
    """Test seconds_to_frame conversion."""
    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.fps == 15.0

    # Test conversion
    assert video.seconds_to_frame(0.0) == 0
    assert video.seconds_to_frame(1.0) == 15
    assert video.seconds_to_frame(2.0) == 30
    assert video.seconds_to_frame(10.0) == 150

    # Test rounding down
    assert video.seconds_to_frame(0.5) == 7  # 0.5 * 15 = 7.5 -> 7


def test_video_timestamp_with_no_fps():
    """Test timestamp conversion returns None when FPS is unknown."""
    video = Video(filename="test.mp4", open_backend=False)
    assert video.fps is None

    assert video.frame_to_seconds(10) is None
    assert video.seconds_to_frame(1.0) is None


def test_video_save_preserves_fps(centered_pair_low_quality_path, tmp_path):
    """Test that Video.save() preserves FPS from source video."""
    video = Video.from_filename(centered_pair_low_quality_path)
    assert video.fps == 15.0

    # Save a subset of frames
    output_path = tmp_path / "output.mp4"
    new_video = video.save(output_path, frame_inds=[0, 1, 2])

    # New video should have same FPS
    assert new_video.fps == 15.0


def test_video_save_custom_fps(centered_pair_low_quality_path, tmp_path):
    """Test that Video.save() can use custom FPS."""
    video = Video.from_filename(centered_pair_low_quality_path)

    # Save with custom FPS
    output_path = tmp_path / "output.mp4"
    new_video = video.save(output_path, frame_inds=[0, 1, 2], fps=30.0)

    # New video should have custom FPS
    assert new_video.fps == 30.0


# ---------------------------------------------------------------------------
# Video crop facade (UNIT U3)
# ---------------------------------------------------------------------------


def test_crop_facade_shape_len_grayscale(small_robot_path):
    """Cropped video reports cropped shape/len and inherits grayscale intent."""
    v = Video.from_filename(small_robot_path)
    assert v.shape == (3, 320, 560, 3)

    rect = (10, 20, 100, 120)
    c = v.crop(rect)

    assert isinstance(c.backend, CropVideoBackend)
    assert c.backend.crop == rect
    # Width = 100 - 10 = 90, height = 120 - 20 = 100, frames/channels preserved.
    assert c.shape == (3, 100, 90, 3)
    assert len(c) == 3
    assert c.grayscale is False
    assert c.source_video is v


def test_crop_facade_grayscale_video(centered_pair_low_quality_path):
    """A crop of a grayscale video stays grayscale."""
    v = Video.from_filename(centered_pair_low_quality_path)
    assert v.grayscale is True

    c = v.crop((0, 0, 100, 100))
    assert c.shape == (1100, 100, 100, 1)
    assert c.grayscale is True


def test_crop_byte_parity(small_robot_path):
    """Cropped frames are byte-identical to crop_frame(full_frame, rect)."""
    v = Video.from_filename(small_robot_path)
    rect = (10, 20, 100, 120)
    c = v.crop(rect)

    for i in range(len(v)):
        got = c[i]
        expected = crop_frame(v[i], rect)
        assert np.array_equal(got, expected)
        assert got.dtype == expected.dtype


def test_crop_byte_parity_out_of_bounds(small_robot_path):
    """An out-of-bounds crop pads with fill, byte-identical to crop_frame."""
    v = Video.from_filename(small_robot_path)
    rect = (-10, -5, 30, 40)
    c = v.crop(rect, fill=7)

    got = c[0]
    expected = crop_frame(v[0], rect, fill=7)
    assert got.shape == (45, 40, 3)
    assert np.array_equal(got, expected)


def test_from_crop_path(small_robot_path):
    """from_crop opens a path and returns a virtual crop."""
    rect = (10, 20, 100, 120)
    c = Video.from_crop(small_robot_path, rect)

    assert isinstance(c.backend, CropVideoBackend)
    assert c.backend.crop == rect
    assert c.shape == (3, 100, 90, 3)
    src = Video.from_filename(small_robot_path)
    assert np.array_equal(c[0], crop_frame(src[0], rect))


def test_from_crop_video(small_robot_path):
    """from_crop accepts an existing Video and sets it as the source."""
    v = Video.from_filename(small_robot_path)
    rect = (10, 20, 100, 120)
    c = Video.from_crop(v, rect)

    assert c.source_video is v
    assert c.backend.crop == rect


def test_from_crop_fill_forwarded(small_robot_path):
    """from_crop forwards the fill value to the crop."""
    c = Video.from_crop(small_robot_path, (-5, -5, 10, 10), fill=3)
    assert c.backend.fill == 3
    src = Video.from_filename(small_robot_path)
    assert np.array_equal(c[0], crop_frame(src[0], (-5, -5, 10, 10), fill=3))


def test_resolve_crop_rect_bbox():
    """The bbox spec floors mins / ceils maxs to an integer rect."""
    assert _resolve_crop_rect(bbox=(1.2, 2.8, 10.4, 20.1)) == (1, 2, 11, 21)


def test_resolve_crop_rect_roi_margin():
    """The roi spec uses axis-aligned .bounds (floor/ceil) plus a margin."""
    from shapely.geometry import box

    roi = box(1.5, 2.5, 10.5, 20.5)
    assert _resolve_crop_rect(roi=roi, margin=2) == (-1, 0, 13, 23)


def test_resolve_crop_rect_center_size():
    """center+size yields a fixed output shape rect (round, not truncate)."""
    # cx - w/2 = 50 - 10 = 40, cy - h/2 = 60 - 15 = 45.
    rect = _resolve_crop_rect(center=(50, 60), size=(20, 30))
    assert rect == (40, 45, 60, 75)
    assert rect[2] - rect[0] == 20
    assert rect[3] - rect[1] == 30


def test_resolve_crop_rect_explicit():
    """An explicit crop rect is returned as integers without truncation."""
    assert _resolve_crop_rect(crop=(1, 2, 3, 4)) == (1, 2, 3, 4)


def test_resolve_crop_rect_exactly_one_required():
    """Not providing exactly one region spec raises ValueError."""
    with pytest.raises(ValueError):
        _resolve_crop_rect()
    with pytest.raises(ValueError):
        _resolve_crop_rect(crop=(0, 0, 5, 5), bbox=(0, 0, 5, 5))
    # center without size is not a complete spec.
    with pytest.raises(ValueError):
        _resolve_crop_rect(center=(5, 5))


def test_resolve_crop_rect_stray_center_or_size():
    """A lone center or size alongside another spec is rejected, not dropped."""
    with pytest.raises(ValueError, match="center and size"):
        _resolve_crop_rect(crop=(0, 0, 5, 5), center=(99, 99))
    with pytest.raises(ValueError, match="center and size"):
        _resolve_crop_rect(bbox=(0, 0, 5, 5), size=(99, 99))


def test_resolve_crop_rect_inverted_raises():
    """Inverted crop rects (x2<x1 / y2<y1) raise rather than yield a negative shape."""
    with pytest.raises(ValueError, match="Inverted crop"):
        _resolve_crop_rect(crop=(100, 100, 50, 200))
    with pytest.raises(ValueError, match="Inverted crop"):
        _resolve_crop_rect(bbox=(100.0, 100.0, 50.0, 200.0))


def test_resolve_crop_rect_size_rounds_not_truncates():
    """Float size is rounded (not int()-truncated) so output shape matches size."""
    assert _resolve_crop_rect(center=(5, 5), size=(3.9, 3.9)) == (3, 3, 7, 7)


def test_crop_closed_video_no_backend_raises():
    """Cropping a video with no openable backend raises a clear ValueError."""
    v = Video(filename="/nonexistent_video.mp4", open_backend=False)
    assert v.backend is None
    with pytest.raises(ValueError, match="no open backend"):
        v.crop((0, 0, 10, 10))


def test_crop_persists_source_shape_and_fill(small_robot_path):
    """Video.crop records the uncropped source_shape and exposes _crop_fill."""
    v = Video.from_filename(small_robot_path)
    c = v.crop((2, 3, 12, 13), fill=7)
    assert tuple(c.backend_metadata["source_shape"]) == tuple(v.shape)
    assert c._crop_fill() == 7
    assert v._crop_fill() == 0  # uncropped


def test_crop_center_size_fixed_shape(small_robot_path):
    """A center+size crop produces a fixed output shape even partly OOB."""
    v = Video.from_filename(small_robot_path)
    # Center near the top-left corner so the window runs off the edge.
    c = v.crop(center=(5, 5), size=(40, 40))
    assert c.shape[1:3] == (40, 40)


def test_crop_validation_error_on_video(small_robot_path):
    """Video.crop surfaces the exactly-one-spec validation error."""
    v = Video.from_filename(small_robot_path)
    with pytest.raises(ValueError):
        v.crop()
    with pytest.raises(ValueError):
        v.crop((0, 0, 5, 5), bbox=(0, 0, 5, 5))


def test_crop_close_open_preserves_crop_and_shape(small_robot_path):
    """close()/open() on a cropped media video preserves crop + cropped shape."""
    v = Video.from_filename(small_robot_path)
    rect = (10, 20, 100, 120)
    c = v.crop(rect)
    expected = crop_frame(v[0], rect)

    c.close()
    # Closed video reports the cropped shape from metadata (no double-crop).
    assert c.shape == (3, 100, 90, 3)
    assert c.backend is None
    assert c.backend_metadata["crop"] == list(rect)

    # Reading reopens and re-wraps exactly once.
    got = c[0]
    assert isinstance(c.backend, CropVideoBackend)
    assert c.backend.crop == rect
    assert np.array_equal(got, expected)


def test_crop_close_open_preserves_dataset_hdf5(slp_minimal_pkg):
    """A cropped HDF5 video keeps its dataset across close()/open() (D-116)."""
    import sleap_io as sio

    labels = sio.load_slp(slp_minimal_pkg)
    v = labels.videos[0]
    rect = (5, 10, 50, 60)
    c = v.crop(rect)
    expected = crop_frame(v[0], rect)
    assert c.backend.dataset == v.backend.dataset

    c.close()
    assert c.backend_metadata["dataset"] == v.backend.dataset

    got = c[0]
    assert isinstance(c.backend, CropVideoBackend)
    # Dataset delegates to the inner backend after reopen.
    assert c.backend.dataset == v.backend.dataset
    assert np.array_equal(got, expected)


def test_crop_deepcopy_preserves_crop_and_reads(small_robot_path):
    """A deepcopy of a cropped video preserves the crop and reads correctly."""
    v = Video.from_filename(small_robot_path)
    rect = (10, 20, 100, 120)
    c = v.crop(rect)
    expected = crop_frame(v[0], rect)

    d = copy.deepcopy(c)
    assert isinstance(d.backend, CropVideoBackend)
    assert d.backend.crop == rect
    assert d.backend_metadata["crop"] == list(rect)
    assert np.array_equal(d[0], expected)


def test_crop_mosaic_shared_inner_close_one_keeps_sibling(small_robot_path):
    """Mosaic tiles share the source's inner; closing one keeps the others."""
    src = Video.from_filename(small_robot_path)
    W, H = src.shape[2], src.shape[1]
    W2 = W // 2

    t1 = src.crop((0, 0, W2, H))
    t2 = src.crop((W2, 0, W, H))

    # Both tiles reuse the source's backend instance as the shared inner.
    assert t1.backend.inner is src.backend
    assert t2.backend.inner is src.backend
    # Shared-decode tiles do not own the shared decoder.
    assert t1.backend.owns_inner is False
    assert t2.backend.owns_inner is False

    # Prime the shared reader, then close one tile.
    _ = t1[0]
    t1.close()

    # The shared decoder is untouched: the source and sibling still read.
    assert src.backend._open_reader is not None
    assert src[0].shape == (H, W, 3)
    assert t2[0].shape == (H, W2, 3)


def test_crop_no_share_decode_owns_inner(small_robot_path):
    """share_decode=False reuses the source backend as inner but TAKES ownership."""
    src = Video.from_filename(small_robot_path)
    t = src.crop((0, 0, 100, 100), share_decode=False)
    assert t.backend.owns_inner is True
    # share_decode=False still reuses self.backend as the inner, but the tile owns
    # it, so closing the tile cascades close() into the shared source backend.
    assert t.backend.inner is src.backend
    _ = src[0]
    t.close()
    # owns_inner=True released the shared inner's reader...
    assert src.backend._open_reader is None
    # ...but the source transparently re-opens on the next read (lazy reopen).
    assert src[0].shape[0] > 0


def test_crop_closed_video_reports_cropped_shape(small_robot_path):
    """A never-opened cropped video reports the cropped shape from metadata."""
    v = Video.from_filename(small_robot_path)
    c = v.crop((10, 20, 100, 120))
    # Drop the backend so shape must come from backend_metadata.
    c.close()
    assert c.backend is None
    assert c.shape == (3, 100, 90, 3)
    assert c.grayscale is False


def test_crop_of_crop_composition(small_robot_path):
    """crop-of-crop flattens to a composed source rect (backend + metadata)."""
    v = Video.from_filename(small_robot_path)
    c1 = v.crop((10, 20, 110, 120))
    c2 = c1.crop((5, 5, 50, 50))

    # Composed source rect: (10+5, 20+5, 10+50, 20+50).
    assert c2.backend.crop == (15, 25, 60, 70)
    assert c2.backend_metadata["crop"] == [15, 25, 60, 70]
    # Flattened: the inner is the plain source backend, never another crop.
    assert not isinstance(c2.backend.inner, CropVideoBackend)

    # Byte-identical to a direct composed crop on the source.
    direct = v.crop((15, 25, 60, 70))
    assert np.array_equal(c2[0], direct[0])


def test_crop_of_crop_byte_parity_via_nested_frames(small_robot_path):
    """crop-of-crop equals cropping the already-cropped frame."""
    v = Video.from_filename(small_robot_path)
    c1 = v.crop((10, 20, 110, 120))
    c2 = c1.crop((5, 5, 50, 50))
    expected = crop_frame(c1[0], (5, 5, 50, 50))
    assert np.array_equal(c2[0], expected)


def test_video_to_crop_and_source_coords(small_robot_path):
    """to_crop_coords/to_source_coords translate by the crop origin."""
    v = Video.from_filename(small_robot_path)
    rect = (10, 20, 100, 120)
    c = v.crop(rect)

    pts = np.array([[15.0, 25.0], [np.nan, 5.0]])
    in_crop = c.to_crop_coords(pts)
    assert np.array_equal(in_crop[0], np.array([5.0, 5.0]))
    # NaN x is preserved; y shifts by -y1.
    assert np.isnan(in_crop[1, 0])
    assert in_crop[1, 1] == -15.0

    # Round-trip back to source coordinates.
    back = c.to_source_coords(in_crop)
    valid = ~np.isnan(pts)
    assert np.allclose(back[valid], pts[valid])


def test_video_coords_uncropped_passthrough(small_robot_path):
    """Coordinate methods are identity copies on an uncropped video."""
    v = Video.from_filename(small_robot_path)
    pts = np.array([[15.0, 25.0], [3.0, 4.0]])

    out_c = v.to_crop_coords(pts)
    out_s = v.to_source_coords(pts)
    assert np.array_equal(out_c, pts)
    assert np.array_equal(out_s, pts)
    # Copies, not the same object (no accidental in-place aliasing).
    assert out_c is not pts
    assert out_s is not pts


def test_video_crop_tuple(small_robot_path):
    """_crop_tuple returns the rect for a crop and None when uncropped."""
    v = Video.from_filename(small_robot_path)
    assert v._crop_tuple() is None

    rect = (10, 20, 100, 120)
    c = v.crop(rect)
    assert c._crop_tuple() == rect

    # Closed path reads the crop from backend_metadata.
    c.close()
    assert c._crop_tuple() == rect
