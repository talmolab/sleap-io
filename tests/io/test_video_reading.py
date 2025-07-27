"""Tests for methods in the sleap_io.io.video_reading file."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_equal

from sleap_io.io.video_reading import (
    HDF5Video,
    ImageVideo,
    MediaVideo,
    TiffVideo,
    VideoBackend,
)


def test_video_backend_from_filename(centered_pair_low_quality_path, slp_minimal_pkg):
    """Test initialization of `VideoBackend` object from filename."""
    backend = VideoBackend.from_filename(centered_pair_low_quality_path)
    assert type(backend) is MediaVideo
    assert backend.filename == centered_pair_low_quality_path
    assert backend.shape == (1100, 384, 384, 1)

    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) is HDF5Video
    assert backend.filename == slp_minimal_pkg
    assert backend.shape == (1, 384, 384, 1)


def test_shape_caching(centered_pair_low_quality_path):
    backend = VideoBackend.from_filename(centered_pair_low_quality_path)
    assert backend._cached_shape is None
    assert backend.shape == (1100, 384, 384, 1)
    assert backend._cached_shape == (1100, 384, 384, 1)

    assert len(backend) == 1100
    assert backend.frames == 1100


def test_get_frame(centered_pair_low_quality_path):
    backend = VideoBackend.from_filename(centered_pair_low_quality_path)

    # First frame
    img = backend.get_frame(0)
    assert img.shape == (384, 384, 1)
    assert img.dtype == "uint8"

    # Last frame
    img = backend.get_frame(len(backend) - 1)
    assert img.shape == (384, 384, 1)
    assert img.dtype == "uint8"

    # Multi-frame
    imgs = backend.get_frames(np.arange(3))
    assert imgs.shape == (3, 384, 384, 1)
    assert imgs.dtype == "uint8"

    # __getitem__
    assert backend[0].shape == (384, 384, 1)
    assert backend[:3].shape == (3, 384, 384, 1)
    assert_equal(backend[:3], backend.get_frames(np.arange(3)))
    assert_equal(backend[-3:], backend.get_frames(range(1097, 1100)))
    assert_equal(backend[-3:-1], backend.get_frames(range(1097, 1099)))

    with pytest.raises(IndexError):
        backend.get_frame(1100)


@pytest.mark.parametrize("keep_open", [False, True])
def test_mediavideo(centered_pair_low_quality_path, keep_open):
    # Test with FFMPEG backend
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="FFMPEG", keep_open=keep_open
    )
    assert type(backend) is MediaVideo
    assert backend.filename == centered_pair_low_quality_path
    assert backend.shape == (1100, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:3].shape == (3, 384, 384, 1)
    if keep_open:
        assert backend._open_reader is not None
        assert backend[0].shape == (384, 384, 1)
        assert type(backend._open_reader).__name__ == "LegacyPlugin"
    else:
        assert backend._open_reader is None

    # Test with pyav backend (if installed)
    try:
        import av  # noqa: F401

        backend = VideoBackend.from_filename(
            centered_pair_low_quality_path, plugin="pyav", keep_open=keep_open
        )
        assert type(backend) is MediaVideo
        assert backend.filename == centered_pair_low_quality_path
        assert backend.shape == (1100, 384, 384, 1)
        assert backend[0].shape == (384, 384, 1)
        assert backend[:3].shape == (3, 384, 384, 1)
        if keep_open:
            assert backend._open_reader is not None
            assert backend[0].shape == (384, 384, 1)
            assert type(backend._open_reader).__name__ == "PyAVPlugin"
        else:
            assert backend._open_reader is None
    except ImportError:
        pass

    # Test with opencv backend
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="opencv", keep_open=keep_open
    )
    assert type(backend) is MediaVideo
    assert backend.filename == centered_pair_low_quality_path
    assert backend.shape == (1100, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:3].shape == (3, 384, 384, 1)
    if keep_open:
        assert backend._open_reader is not None
        assert backend[0].shape == (384, 384, 1)
        assert type(backend._open_reader).__name__ == "VideoCapture"
    else:
        assert backend._open_reader is None


@pytest.mark.parametrize("keep_open", [False, True])
def test_hdf5video_rank4(centered_pair_low_quality_path, tmp_path, keep_open):
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, keep_open=keep_open
    )
    imgs = backend[:3]
    assert imgs.shape == (3, 384, 384, 1)

    with h5py.File(tmp_path / "test.h5", "w") as f:
        f.create_dataset("images", data=imgs)

    backend = VideoBackend.from_filename(tmp_path / "test.h5")
    assert type(backend) is HDF5Video

    assert backend.shape == (3, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:].shape == (3, 384, 384, 1)
    assert not backend.has_embedded_images
    if keep_open:
        assert backend._open_reader is not None
        assert backend[0].shape == (384, 384, 1)


def test_hdf5video_embedded(slp_minimal_pkg):
    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) is HDF5Video

    assert backend.shape == (1, 384, 384, 1)
    assert backend.dataset == "video0/video"
    assert backend[0].shape == (384, 384, 1)
    assert backend[[0]].shape == (1, 384, 384, 1)
    assert (
        backend.source_filename
        == "tests/data/json_format_v1/centered_pair_low_quality.mp4"
    )
    assert backend.has_embedded_images


def test_imagevideo(centered_pair_frame_paths):
    backend = VideoBackend.from_filename(centered_pair_frame_paths)
    assert type(backend) is ImageVideo
    assert backend.shape == (3, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:3].shape == (3, 384, 384, 1)

    img_folder = Path(centered_pair_frame_paths[0]).parent
    imgs = ImageVideo.find_images(img_folder)
    assert imgs == centered_pair_frame_paths

    backend = VideoBackend.from_filename(img_folder)
    assert type(backend) is ImageVideo
    assert backend.shape == (3, 384, 384, 1)

    backend = VideoBackend.from_filename(centered_pair_frame_paths[0])
    assert type(backend) is ImageVideo
    assert backend.shape == (1, 384, 384, 1)


def test_tiff_single_page(single_page_tiff_path):
    """Test that single-page TIFF files are handled by ImageVideo backend."""
    backend = VideoBackend.from_filename(single_page_tiff_path)
    assert type(backend) is ImageVideo
    assert backend.num_frames == 1

    # Test frame reading
    frame = backend.get_frame(0)
    assert frame.shape == (128, 128, 1)  # Grayscale
    assert frame.dtype == np.uint8

    # Test that we can't read beyond the single frame
    with pytest.raises(IndexError):
        backend.get_frame(1)


def test_tiff_multipage(multipage_tiff_path):
    """Test that multi-page TIFF files are handled by TiffVideo backend."""
    backend = VideoBackend.from_filename(multipage_tiff_path)
    assert type(backend) is TiffVideo
    assert backend.num_frames == 8

    # Test reading individual frames
    frame0 = backend.get_frame(0)
    assert frame0.shape == (128, 128, 1)
    assert frame0.dtype == np.uint8

    frame7 = backend.get_frame(7)
    assert frame7.shape == (128, 128, 1)

    # Test reading multiple frames
    frames = backend.get_frames([0, 3, 7])
    assert frames.shape == (3, 128, 128, 1)

    # Test frame index bounds
    with pytest.raises(IndexError):
        backend.get_frame(8)

    # Test slicing
    frames_slice = backend[2:5]
    assert frames_slice.shape == (3, 128, 128, 1)


def test_tiff_stacked_channels(stacked_tiff_path):
    """Test that stacked TIFF (H,W,C) is handled by ImageVideo backend."""
    backend = VideoBackend.from_filename(stacked_tiff_path)
    assert type(backend) is ImageVideo
    assert backend.num_frames == 1

    # This should read as a single multi-channel image
    frame = backend.get_frame(0)
    assert frame.shape == (128, 128, 8)  # 8 channels


def test_tiff_image_sequence(tiff_image_sequence_path):
    """Test that a directory of TIFF files is handled by ImageVideo backend."""
    backend = VideoBackend.from_filename(tiff_image_sequence_path)
    assert type(backend) is ImageVideo
    assert backend.num_frames == 8

    # Test reading frames
    frame0 = backend.get_frame(0)
    assert frame0.shape == (128, 128, 1)

    frames = backend.get_frames([0, 4, 7])
    assert frames.shape == (3, 128, 128, 1)


def test_is_multipage_tiff(
    single_page_tiff_path, multipage_tiff_path, stacked_tiff_path
):
    """Test the is_multipage_tiff utility function."""
    from sleap_io.io.video_reading import is_multipage_tiff

    assert is_multipage_tiff(single_page_tiff_path) is False
    assert is_multipage_tiff(multipage_tiff_path) is True
    assert is_multipage_tiff(stacked_tiff_path) is False

    # Test with non-existent file
    assert is_multipage_tiff("non_existent.tif") is False
