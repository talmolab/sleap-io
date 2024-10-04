"""Tests for methods in the sleap_io.io.video_reading file."""

from sleap_io.io.video_reading import VideoBackend, MediaVideo, HDF5Video, ImageVideo
import numpy as np
from numpy.testing import assert_equal
import h5py
import pytest
from pathlib import Path


def test_video_backend_from_filename(centered_pair_low_quality_path, slp_minimal_pkg):
    """Test initialization of `VideoBackend` object from filename."""
    backend = VideoBackend.from_filename(centered_pair_low_quality_path)
    assert type(backend) == MediaVideo
    assert backend.filename == centered_pair_low_quality_path
    assert backend.shape == (1100, 384, 384, 1)

    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) == HDF5Video
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
    assert type(backend) == MediaVideo
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
        import av

        backend = VideoBackend.from_filename(
            centered_pair_low_quality_path, plugin="pyav", keep_open=keep_open
        )
        assert type(backend) == MediaVideo
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
    assert type(backend) == MediaVideo
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
    assert type(backend) == HDF5Video

    assert backend.shape == (3, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:].shape == (3, 384, 384, 1)
    assert not backend.has_embedded_images
    if keep_open:
        assert backend._open_reader is not None
        assert backend[0].shape == (384, 384, 1)


def test_hdf5video_embedded(slp_minimal_pkg):
    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) == HDF5Video

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
    assert type(backend) == ImageVideo
    assert backend.shape == (3, 384, 384, 1)
    assert backend[0].shape == (384, 384, 1)
    assert backend[:3].shape == (3, 384, 384, 1)

    img_folder = Path(centered_pair_frame_paths[0]).parent
    imgs = ImageVideo.find_images(img_folder)
    assert imgs == centered_pair_frame_paths

    backend = VideoBackend.from_filename(img_folder)
    assert type(backend) == ImageVideo
    assert backend.shape == (3, 384, 384, 1)

    backend = VideoBackend.from_filename(centered_pair_frame_paths[0])
    assert type(backend) == ImageVideo
    assert backend.shape == (1, 384, 384, 1)
