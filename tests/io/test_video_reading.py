"""Tests for methods in the sleap_io.io.video_reading file."""

from pathlib import Path

import h5py
import imageio.v3 as iio
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

try:
    import cv2
except ImportError:
    cv2 = None


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


def test_opencv_bgr_to_rgb_conversion(tmp_path):
    """Test that OpenCV backend correctly converts BGR to RGB."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("OpenCV not installed")

    # Create a test video with known color pattern
    # We'll create a simple video with distinct R, G, B channels
    video_path = tmp_path / "test_color.mp4"
    height, width = 100, 100
    fps = 1

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Frame 1: Pure red (BGR: [0, 0, 255])
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:, :, 2] = 255  # Red channel in BGR
    writer.write(frame1)

    # Frame 2: Pure green (BGR: [0, 255, 0])
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2[:, :, 1] = 255  # Green channel
    writer.write(frame2)

    # Frame 3: Pure blue (BGR: [255, 0, 0])
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3[:, :, 0] = 255  # Blue channel in BGR
    writer.write(frame3)

    writer.release()

    # Test reading with OpenCV backend
    backend = MediaVideo(str(video_path), plugin="opencv", keep_open=False)

    # Test single frame reading (with tolerance for compression artifacts)
    frame = backend._read_frame(0)
    # Should be red in RGB (255, 0, 0) - allow for compression artifacts
    assert frame[50, 50, 0] > 240  # R channel should be high
    assert frame[50, 50, 1] < 20  # G channel should be low
    assert frame[50, 50, 2] < 20  # B channel should be low

    frame = backend._read_frame(1)
    # Should be green in RGB (0, 255, 0)
    assert frame[50, 50, 0] < 20  # R channel should be low
    assert frame[50, 50, 1] > 240  # G channel should be high
    assert frame[50, 50, 2] < 20  # B channel should be low

    frame = backend._read_frame(2)
    # Should be blue in RGB (0, 0, 255)
    assert frame[50, 50, 0] < 20  # R channel should be low
    assert frame[50, 50, 1] < 20  # G channel should be low
    assert frame[50, 50, 2] > 240  # B channel should be high

    # Test batch frame reading
    frames = backend._read_frames([0, 1, 2])
    # Frame 0: red (with tolerance)
    assert (
        frames[0, 50, 50, 0] > 240
        and frames[0, 50, 50, 1] < 20
        and frames[0, 50, 50, 2] < 20
    )
    # Frame 1: green
    assert (
        frames[1, 50, 50, 0] < 20
        and frames[1, 50, 50, 1] > 240
        and frames[1, 50, 50, 2] < 20
    )
    # Frame 2: blue
    assert (
        frames[2, 50, 50, 0] < 20
        and frames[2, 50, 50, 1] < 20
        and frames[2, 50, 50, 2] > 240
    )


def test_opencv_keep_open_reader_initialization(tmp_path):
    """Test that OpenCV backend correctly initializes reader with keep_open=True."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("OpenCV not installed")

    # Create a simple test video
    video_path = tmp_path / "test_keep_open.mp4"
    height, width = 50, 50
    fps = 1
    num_frames = 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create frames with different intensities for identification
        frame = np.full((height, width, 3), i * 50, dtype=np.uint8)
        writer.write(frame)

    writer.release()

    # Test with keep_open=True
    backend = MediaVideo(str(video_path), plugin="opencv", keep_open=True)

    # Initially no open reader
    assert backend._open_reader is None

    # First read should create the reader
    frame1 = backend._read_frame(0)
    assert backend._open_reader is not None
    assert type(backend._open_reader).__name__ == "VideoCapture"
    first_reader = backend._open_reader

    # Subsequent reads should use the same reader
    frame2 = backend._read_frame(1)
    assert backend._open_reader is first_reader  # Same object

    # Test random access with the persistent reader
    frame4 = backend._read_frame(3)
    assert backend._open_reader is first_reader

    # Verify frame content is correct (intensity should match frame index * 50)
    # Allow for compression artifacts
    assert np.mean(frame1) == pytest.approx(0, abs=5)
    assert np.mean(frame2) == pytest.approx(50, abs=5)
    assert np.mean(frame4) == pytest.approx(150, abs=5)

    # Test with keep_open=False for comparison
    backend2 = MediaVideo(str(video_path), plugin="opencv", keep_open=False)

    # Should not maintain an open reader
    assert backend2._open_reader is None
    frame = backend2._read_frame(0)
    assert backend2._open_reader is None  # Reader not kept after read

    # Verify we can still read frames correctly
    assert np.mean(frame) == pytest.approx(0, abs=5)


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
    """Test that stacked TIFF (H,W,T) is handled by TiffVideo backend."""
    backend = VideoBackend.from_filename(stacked_tiff_path)
    assert type(backend) is TiffVideo
    assert backend.format == "HWT"
    assert backend.num_frames == 8

    # This should read frames from the HWT format
    frame = backend.get_frame(0)
    assert frame.shape == (128, 128, 1)  # Single channel per frame

    # Test multiple frames
    frames = backend.get_frames([0, 3, 7])
    assert frames.shape == (3, 128, 128, 1)


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
    """Test the TiffVideo.is_multipage static method."""
    from sleap_io.io.video_reading import TiffVideo

    assert TiffVideo.is_multipage(single_page_tiff_path) is False
    assert TiffVideo.is_multipage(multipage_tiff_path) is True
    assert TiffVideo.is_multipage(stacked_tiff_path) is False

    # Test with non-existent file
    assert TiffVideo.is_multipage("non_existent.tif") is False


def test_tiff_format_detection(
    single_page_tiff_path, multipage_tiff_path, stacked_tiff_path
):
    """Test the TiffVideo.detect_format static method."""
    from sleap_io.io.video_reading import TiffVideo

    # Test single page detection
    format_type, metadata = TiffVideo.detect_format(single_page_tiff_path)
    assert format_type == "single_frame"
    assert metadata["shape"] == (128, 128)  # Grayscale

    # Test multi-page detection
    format_type, metadata = TiffVideo.detect_format(multipage_tiff_path)
    assert format_type == "multi_page"
    assert len(metadata["shape"]) in (2, 3)  # Could be (H, W) or (H, W, C)

    # Test stacked/multi-channel detection
    format_type, metadata = TiffVideo.detect_format(stacked_tiff_path)
    assert format_type == "rank3_video"  # Detected as HWT (square frames)
    assert metadata["shape"] == (128, 128, 8)
    assert metadata.get("format") == "HWT"
    assert metadata["n_frames"] == 8  # T dimension


def test_tiff_rank3_format_detection():
    """Test TiffVideo._detect_rank3_format static method."""
    from sleap_io.io.video_reading import TiffVideo

    # Test HWC format (single frame with channels)
    format_type, metadata = TiffVideo._detect_rank3_format((480, 640, 3))
    assert format_type == "single_frame"
    assert metadata["format"] == "HWC"

    # Test THW format (when last two dims are equal)
    format_type, metadata = TiffVideo._detect_rank3_format((10, 128, 128))
    assert format_type == "rank3_video"
    assert metadata["format"] == "THW"
    assert metadata["n_frames"] == 10
    assert metadata["height"] == 128
    assert metadata["width"] == 128

    # Test HWT format (square frames)
    format_type, metadata = TiffVideo._detect_rank3_format((128, 128, 10))
    assert format_type == "rank3_video"
    assert metadata["format"] == "HWT"
    assert metadata["height"] == 128
    assert metadata["width"] == 128
    assert metadata["n_frames"] == 10

    # Test HWT format (non-square)
    format_type, metadata = TiffVideo._detect_rank3_format((480, 640, 10))
    assert format_type == "rank3_video"
    assert metadata["format"] == "HWT"
    assert metadata["height"] == 480
    assert metadata["width"] == 640
    assert metadata["n_frames"] == 10


def test_tiff_rank4_format_detection():
    """Test TiffVideo._detect_rank4_format static method."""
    from sleap_io.io.video_reading import TiffVideo

    # Test CHWT format
    format_type, metadata = TiffVideo._detect_rank4_format((3, 480, 640, 10))
    assert format_type == "rank4_video"
    assert metadata["format"] == "CHWT"
    assert metadata["channels"] == 3
    assert metadata["height"] == 480
    assert metadata["width"] == 640
    assert metadata["n_frames"] == 10

    # Test THWC format
    format_type, metadata = TiffVideo._detect_rank4_format((10, 480, 640, 3))
    assert format_type == "rank4_video"
    assert metadata["format"] == "THWC"
    assert metadata["n_frames"] == 10
    assert metadata["height"] == 480
    assert metadata["width"] == 640
    assert metadata["channels"] == 3

    # Test ambiguous case (defaults to THWC)
    format_type, metadata = TiffVideo._detect_rank4_format((10, 480, 640, 10))
    assert format_type == "rank4_video"
    assert metadata["format"] == "THWC"


def test_tiff_format_detection_edge_cases(tmp_path):
    """Test TiffVideo.detect_format edge cases."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Test exception handling
    format_type, metadata = TiffVideo.detect_format("non_existent_file.tif")
    assert format_type == "single_frame"
    assert metadata["shape"] is None

    # Create a rank-4 test file
    rank4_path = tmp_path / "rank4.tif"
    rank4_data = np.zeros((10, 128, 128, 3), dtype=np.uint8)
    iio.imwrite(rank4_path, rank4_data)

    format_type, metadata = TiffVideo.detect_format(str(rank4_path))
    assert format_type == "rank4_video"
    assert metadata["format"] == "THWC"

    # imageio might save 5D as multi-page, so let's just check the other edge case
    # is covered


def test_tiff_thw_format_reading(tmp_path):
    """Test reading THW format TIFF files."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Create a THW format TIFF (10 frames of 50x50)
    thw_path = tmp_path / "thw.tif"
    thw_data = np.random.randint(0, 255, (10, 50, 50), dtype=np.uint8)
    iio.imwrite(thw_path, thw_data)

    # Create TiffVideo with explicit THW format
    video = TiffVideo(str(thw_path), format="THW")
    assert video.num_frames == 10

    # Test single frame reading
    frame = video._read_frame(0)
    assert frame.shape == (50, 50, 1)
    np.testing.assert_array_equal(frame[:, :, 0], thw_data[0])

    # Test multiple frame reading
    frames = video._read_frames([0, 5, 9])
    assert frames.shape == (3, 50, 50, 1)
    np.testing.assert_array_equal(frames[0, :, :, 0], thw_data[0])
    np.testing.assert_array_equal(frames[1, :, :, 0], thw_data[5])
    np.testing.assert_array_equal(frames[2, :, :, 0], thw_data[9])


def test_tiff_rank4_format_reading(tmp_path):
    """Test reading rank-4 format TIFF files."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Test THWC format
    thwc_path = tmp_path / "thwc.tif"
    thwc_data = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    iio.imwrite(thwc_path, thwc_data)

    video = TiffVideo(str(thwc_path), format="THWC")
    assert video.num_frames == 5

    frame = video._read_frame(2)
    assert frame.shape == (64, 64, 3)
    np.testing.assert_array_equal(frame, thwc_data[2])

    frames = video._read_frames([0, 2, 4])
    assert frames.shape == (3, 64, 64, 3)
    np.testing.assert_array_equal(frames, thwc_data[[0, 2, 4]])

    # Note: imageio may not preserve 4D arrays as expected, so we'll skip CHWT test
    # and just test the format error handling

    # Test unknown format error
    video_bad = TiffVideo(str(thwc_path), format="UNKNOWN")
    with pytest.raises(ValueError, match="Unknown format"):
        video_bad._read_frame(0)
    with pytest.raises(ValueError, match="Unknown format"):
        video_bad._read_frames([0, 1])


def test_tiff_chwt_format_with_mocking(tmp_path, monkeypatch):
    """Test CHWT format reading using mocking since imageio doesn't preserve 4D."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Create mock CHWT data (3 channels, 64 height, 64 width, 5 time points)
    chwt_data = np.random.randint(0, 255, (3, 64, 64, 5), dtype=np.uint8)
    chwt_path = tmp_path / "chwt.tif"

    # Mock iio.imread to return our CHWT data
    def mock_imread(filename, index=None):
        if index is not None:
            # For multi-page check, only return data for index 0
            if index > 0:
                raise IndexError("index out of range")
            # Return the full 4D array even with index (simulating single-page TIFF)
            return chwt_data
        # Return full 4D array
        return chwt_data

    monkeypatch.setattr("imageio.v3.imread", mock_imread)

    # Test CHWT format detection
    format_type, metadata = TiffVideo.detect_format(str(chwt_path))
    assert format_type == "rank4_video"
    assert metadata["format"] == "CHWT"
    assert metadata["channels"] == 3
    assert metadata["height"] == 64
    assert metadata["width"] == 64
    assert metadata["n_frames"] == 5

    # Test CHWT format reading
    video = TiffVideo(str(chwt_path), format="CHWT")

    # Test single frame reading
    frame = video._read_frame(2)
    assert frame.shape == (64, 64, 3)
    expected_frame = np.moveaxis(chwt_data[:, :, :, 2], 0, -1)  # CHW -> HWC
    np.testing.assert_array_equal(frame, expected_frame)

    # Test multiple frame reading
    frames = video._read_frames([0, 2, 4])
    assert frames.shape == (3, 64, 64, 3)
    expected_frames = np.moveaxis(chwt_data[:, :, :, [0, 2, 4]], -1, 0)  # CHWT -> TCHW
    expected_frames = np.moveaxis(expected_frames, 1, -1)  # TCHW -> THWC
    np.testing.assert_array_equal(frames, expected_frames)


def test_tiff_attrs_post_init(tmp_path, monkeypatch):
    """Test TiffVideo.__attrs_post_init__ auto-detection."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Create a rank-3 HWT format file
    hwt_path = tmp_path / "hwt_auto.tif"
    hwt_data = np.random.randint(0, 255, (128, 128, 8), dtype=np.uint8)
    iio.imwrite(hwt_path, hwt_data)

    # Create TiffVideo without specifying format (triggers __attrs_post_init__)
    video = TiffVideo(str(hwt_path))
    assert video.format == "HWT"  # Should auto-detect HWT format

    # Test with a rank-4 file
    thwc_path = tmp_path / "thwc_auto.tif"
    thwc_data = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    iio.imwrite(thwc_path, thwc_data)

    video2 = TiffVideo(str(thwc_path))
    assert video2.format == "THWC"  # Should auto-detect THWC format

    # Test with multi-page detection
    def mock_imread_multipage(filename, index=None):
        if index == 1:
            return np.zeros((64, 64), dtype=np.uint8)
        elif index == 0:
            return np.zeros((64, 64), dtype=np.uint8)
        else:
            raise IndexError("index out of range")

    monkeypatch.setattr("imageio.v3.imread", mock_imread_multipage)

    video3 = TiffVideo(str(tmp_path / "multipage.tif"))
    assert video3.format == "multi_page"

    # Test with single_frame detection (falls back to multi_page)
    def mock_imread_single(filename, index=None):
        if index is not None:
            if index == 0:
                return np.zeros((64, 64, 3), dtype=np.uint8)  # HWC single frame
            else:
                raise IndexError("single page")
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr("imageio.v3.imread", mock_imread_single)

    video4 = TiffVideo(str(tmp_path / "single.tif"))
    assert video4.format == "multi_page"  # Falls back to multi_page for single_frame


def test_tiff_detect_format_edge_cases_extended(tmp_path, monkeypatch):
    """Test TiffVideo.detect_format with unusual ndim values."""
    import numpy as np

    from sleap_io.io.video_reading import TiffVideo

    # Test with 5D array (unusual ndim)
    def mock_imread_5d(filename, index=None):
        if index is not None:
            if index == 0:
                return np.zeros((2, 3, 64, 64, 10), dtype=np.uint8)
            else:
                raise IndexError("single page")
        return np.zeros((2, 3, 64, 64, 10), dtype=np.uint8)  # 5D array

    monkeypatch.setattr("imageio.v3.imread", mock_imread_5d)

    format_type, metadata = TiffVideo.detect_format(str(tmp_path / "5d.tif"))
    assert format_type == "single_frame"  # Falls back to single_frame for unusual ndim
    assert metadata["shape"] == (2, 3, 64, 64, 10)

    # Test with 1D array
    def mock_imread_1d(filename, index=None):
        if index is not None:
            if index == 0:
                return np.zeros(100, dtype=np.uint8)
            else:
                raise IndexError("single page")
        return np.zeros(100, dtype=np.uint8)  # 1D array

    monkeypatch.setattr("imageio.v3.imread", mock_imread_1d)

    format_type, metadata = TiffVideo.detect_format(str(tmp_path / "1d.tif"))
    assert format_type == "single_frame"
    assert metadata["shape"] == (100,)


def test_plugin_name_normalization():
    """Test plugin name normalization with various aliases."""
    from sleap_io.io.video_reading import normalize_plugin_name

    # Test opencv aliases
    assert normalize_plugin_name("opencv") == "opencv"
    assert normalize_plugin_name("OpenCV") == "opencv"
    assert normalize_plugin_name("cv") == "opencv"
    assert normalize_plugin_name("cv2") == "opencv"
    assert normalize_plugin_name("CV2") == "opencv"
    assert normalize_plugin_name("ocv") == "opencv"

    # Test FFMPEG aliases
    assert normalize_plugin_name("FFMPEG") == "FFMPEG"
    assert normalize_plugin_name("ffmpeg") == "FFMPEG"
    assert normalize_plugin_name("imageio-ffmpeg") == "FFMPEG"
    assert normalize_plugin_name("imageio_ffmpeg") == "FFMPEG"

    # Test pyav aliases
    assert normalize_plugin_name("pyav") == "pyav"
    assert normalize_plugin_name("PyAV") == "pyav"
    assert normalize_plugin_name("av") == "pyav"
    assert normalize_plugin_name("AV") == "pyav"

    # Test invalid plugin
    with pytest.raises(ValueError, match="Unknown plugin"):
        normalize_plugin_name("invalid_plugin")


def test_image_plugin_name_normalization():
    """Test image plugin name normalization with various aliases."""
    from sleap_io.io.video_reading import normalize_image_plugin_name

    # Test opencv aliases
    assert normalize_image_plugin_name("opencv") == "opencv"
    assert normalize_image_plugin_name("OpenCV") == "opencv"
    assert normalize_image_plugin_name("cv") == "opencv"
    assert normalize_image_plugin_name("cv2") == "opencv"
    assert normalize_image_plugin_name("CV2") == "opencv"
    assert normalize_image_plugin_name("ocv") == "opencv"

    # Test imageio aliases
    assert normalize_image_plugin_name("imageio") == "imageio"
    assert normalize_image_plugin_name("iio") == "imageio"

    # Test invalid plugin
    with pytest.raises(ValueError, match="Unknown image plugin"):
        normalize_image_plugin_name("invalid_plugin")

    # Test invalid plugin that's valid for video but not images
    with pytest.raises(ValueError, match="Unknown image plugin"):
        normalize_image_plugin_name("pyav")


def test_global_default_plugin():
    """Test global default plugin functionality."""
    import sleap_io as sio

    # Test initial state
    assert sio.get_default_video_plugin() is None

    # Test setting default
    sio.set_default_video_plugin("opencv")
    assert sio.get_default_video_plugin() == "opencv"

    # Test setting with alias
    sio.set_default_video_plugin("cv2")
    assert sio.get_default_video_plugin() == "opencv"

    # Test clearing default
    sio.set_default_video_plugin(None)
    assert sio.get_default_video_plugin() is None

    # Test invalid plugin
    with pytest.raises(ValueError):
        sio.set_default_video_plugin("invalid")


def test_media_video_with_plugin(centered_pair_low_quality_path):
    """Test MediaVideo creation with explicit plugin."""
    import sleap_io as sio

    # Clear any global default
    sio.set_default_video_plugin(None)

    # Test with explicit plugin
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="opencv"
    )
    assert isinstance(backend, MediaVideo)
    assert backend.plugin == "opencv"

    # Test with plugin alias
    backend = VideoBackend.from_filename(centered_pair_low_quality_path, plugin="cv2")
    assert backend.plugin == "opencv"

    # Test with different plugin
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="FFMPEG"
    )
    assert backend.plugin == "FFMPEG"


def test_media_video_uses_global_default(centered_pair_low_quality_path):
    """Test MediaVideo uses global default when no plugin specified."""
    import sleap_io as sio

    # Set global default
    sio.set_default_video_plugin("FFMPEG")

    # Create without explicit plugin
    backend = VideoBackend.from_filename(centered_pair_low_quality_path)
    assert isinstance(backend, MediaVideo)
    assert backend.plugin == "FFMPEG"

    # Reset to avoid affecting other tests
    sio.set_default_video_plugin(None)


@pytest.mark.parametrize("plugin", ["opencv", "imageio"])
def test_image_video_plugin_consistency(centered_pair_frame_paths, plugin):
    """Test ImageVideo with different plugins produces consistent output."""
    import sys

    # Skip if opencv not available
    if plugin == "opencv" and "cv2" not in sys.modules:
        pytest.skip("OpenCV not available")

    # Create backend with specified plugin
    backend = ImageVideo(centered_pair_frame_paths, plugin=plugin)
    assert backend.plugin == plugin

    # Read a frame
    frame = backend[0]
    assert frame.ndim == 3  # Should always be 3D (H, W, C)

    # Verify both plugins return the same frames
    backend_opencv = ImageVideo(centered_pair_frame_paths, plugin="opencv")
    backend_imageio = ImageVideo(centered_pair_frame_paths, plugin="imageio")

    frame_opencv = backend_opencv[0]
    frame_imageio = backend_imageio[0]

    # Should be identical
    np.testing.assert_array_equal(frame_opencv, frame_imageio)

    # If image is color (3 channels), verify BGR->RGB conversion for OpenCV
    if cv2 is not None and frame.shape[-1] == 3:
        # Read with OpenCV directly (will be BGR for color images)
        bgr_frame = cv2.imread(centered_pair_frame_paths[0])
        if bgr_frame is not None and bgr_frame.ndim == 3:
            bgr_to_rgb = bgr_frame[..., ::-1]
            # Our plugin should match the converted version
            np.testing.assert_array_equal(frame_opencv, bgr_to_rgb)


def test_image_video_default_plugin(centered_pair_frame_paths):
    """Test ImageVideo respects global default image plugin."""
    import sleap_io as sio

    original = sio.get_default_image_plugin()
    try:
        # Set global default to opencv
        sio.set_default_image_plugin("opencv")
        backend = ImageVideo(centered_pair_frame_paths)
        assert backend.plugin == "opencv"

        # Set global default to imageio
        sio.set_default_image_plugin("imageio")
        backend = ImageVideo(centered_pair_frame_paths)
        assert backend.plugin == "imageio"
    finally:
        # Restore original default
        sio.set_default_image_plugin(original)


def test_image_video_explicit_plugin_overrides_default(centered_pair_frame_paths):
    """Test that explicit plugin parameter overrides global default."""
    import sleap_io as sio

    original = sio.get_default_image_plugin()
    try:
        # Set global default to opencv
        sio.set_default_image_plugin("opencv")

        # But explicitly request imageio
        backend = ImageVideo(centered_pair_frame_paths, plugin="imageio")
        assert backend.plugin == "imageio"

        # And vice versa
        sio.set_default_image_plugin("imageio")
        backend = ImageVideo(centered_pair_frame_paths, plugin="opencv")
        assert backend.plugin == "opencv"
    finally:
        # Restore original default
        sio.set_default_image_plugin(original)


def test_image_video_plugin_with_grayscale(centered_pair_frame_paths):
    """Test ImageVideo plugin works correctly with grayscale images."""
    import sys

    # Skip if opencv not available
    if "cv2" not in sys.modules:
        pytest.skip("OpenCV not available")

    # Test that both plugins handle images correctly (grayscale or color)
    backend_opencv = ImageVideo(centered_pair_frame_paths, plugin="opencv")
    backend_imageio = ImageVideo(centered_pair_frame_paths, plugin="imageio")

    frame_opencv = backend_opencv[0]
    frame_imageio = backend_imageio[0]

    # Both should return identical frames
    np.testing.assert_array_equal(frame_opencv, frame_imageio)
    assert frame_opencv.ndim == 3  # Always 3D (H, W, C)
    assert frame_opencv.shape[-1] in (1, 3)  # Grayscale or RGB


def test_image_video_default_plugin_without_opencv(
    centered_pair_frame_paths, monkeypatch
):
    """Test ImageVideo defaults to imageio when opencv not available."""
    import sys

    # Mock sys.modules to simulate opencv not being available
    if "cv2" in sys.modules:
        monkeypatch.delitem(sys.modules, "cv2")

    # Clear any global default
    import sleap_io as sio

    original_default = sio.get_default_image_plugin()
    try:
        sio.set_default_image_plugin(None)

        # Create ImageVideo without specifying plugin
        backend = ImageVideo(centered_pair_frame_paths)

        # Should default to imageio since opencv is not available
        assert backend.plugin == "imageio"
    finally:
        # Restore
        sio.set_default_image_plugin(original_default)


def test_get_available_backends():
    """Test backend availability detection."""
    import sleap_io as sio

    # Get available backends
    video_backends = sio.get_available_video_backends()
    assert isinstance(video_backends, list)
    # At minimum, we should have some backends if imageio modules are loaded
    # But we can't guarantee which ones without knowing the environment

    image_backends = sio.get_available_image_backends()
    assert isinstance(image_backends, list)
    # imageio should always be available (core dependency)
    assert "imageio" in image_backends


def test_installation_instructions():
    """Test installation instruction helper."""
    import sleap_io as sio

    # Test specific plugin instructions for video
    opencv_instructions = sio.get_installation_instructions("opencv")
    assert "pip install sleap-io[opencv]" in opencv_instructions

    ffmpeg_instructions = sio.get_installation_instructions("FFMPEG")
    assert "Included by default" in ffmpeg_instructions

    pyav_instructions = sio.get_installation_instructions("pyav")
    assert "pip install sleap-io[pyav]" in pyav_instructions

    # Test with alias
    cv2_instructions = sio.get_installation_instructions("cv2")
    assert "pip install sleap-io[opencv]" in cv2_instructions

    # Test all plugins (default)
    all_instructions = sio.get_installation_instructions()
    assert "opencv" in all_instructions
    assert "FFMPEG" in all_instructions
    assert "pyav" in all_instructions
    assert "pip install sleap-io" in all_instructions

    # Test image backend instructions
    opencv_img_instructions = sio.get_installation_instructions("opencv", "image")
    assert "pip install sleap-io[opencv]" in opencv_img_instructions

    imageio_instructions = sio.get_installation_instructions("imageio", "image")
    assert "Already installed" in imageio_instructions

    # Test all image plugins
    all_img_instructions = sio.get_installation_instructions(None, "image")
    assert "opencv" in all_img_instructions
    assert "imageio" in all_img_instructions


def test_preferred_backend_warning(centered_pair_low_quality_path, monkeypatch):
    """Test warning when preferred backend not available."""
    import sys
    import warnings

    import sleap_io as sio

    # Save original state
    original = sio.get_default_video_plugin()

    try:
        # Mock sys.modules to simulate opencv not being available
        if "cv2" in sys.modules:
            monkeypatch.delitem(sys.modules, "cv2")

        # Refresh backend availability tracking
        from sleap_io.io import video_reading

        video_reading._AVAILABLE_VIDEO_BACKENDS["opencv"] = False

        # Set preferred backend to opencv (which we mocked as unavailable)
        sio.set_default_video_plugin("opencv")

        # Try to create a video backend
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a warning and fall back to auto-detection
            backend = VideoBackend.from_filename(centered_pair_low_quality_path)

            # Check that a warning was raised
            assert len(w) > 0
            warning_message = str(w[0].message)
            assert "not available" in warning_message
            assert "opencv" in warning_message

            # Should have fallen back to another backend
            assert isinstance(backend, MediaVideo)
            # Should NOT be opencv since it's unavailable
            assert backend.plugin != "opencv"

    finally:
        # Restore original state
        sio.set_default_video_plugin(original)


def test_no_video_backends_error(monkeypatch):
    """Test error message when no video backends are available."""
    import sys

    from sleap_io.io import video_reading

    # Save original backend availability state
    original_backends = video_reading._AVAILABLE_VIDEO_BACKENDS.copy()

    try:
        # Remove all video backend modules from sys.modules
        for module in ["cv2", "imageio_ffmpeg", "av"]:
            if module in sys.modules:
                monkeypatch.delitem(sys.modules, module)

        # Update availability tracking to reflect no backends
        video_reading._AVAILABLE_VIDEO_BACKENDS = {
            "opencv": False,
            "FFMPEG": False,
            "pyav": False,
        }

        # Try to create a MediaVideo which should fail with helpful error
        with pytest.raises(ImportError) as exc_info:
            MediaVideo(filename="test.mp4")

        # Check error message contains helpful installation instructions
        error_msg = str(exc_info.value)
        assert "No video backend plugins are available" in error_msg
        assert "bundled imageio-ffmpeg should be available" in error_msg
        assert "pip install sleap-io[opencv]" in error_msg
        assert "pip install sleap-io[pyav]" in error_msg
        assert "io.sleap.ai" in error_msg

    finally:
        # Restore original state
        video_reading._AVAILABLE_VIDEO_BACKENDS = original_backends


def test_no_image_backends_warning(centered_pair_frame_paths, monkeypatch):
    """Test warning when preferred image backend is not available."""
    import sys
    import warnings

    import sleap_io as sio
    from sleap_io.io import video_reading

    # Save original state
    original = sio.get_default_image_plugin()
    original_backends = video_reading._AVAILABLE_IMAGE_BACKENDS.copy()

    try:
        # Remove opencv from sys.modules
        if "cv2" in sys.modules:
            monkeypatch.delitem(sys.modules, "cv2")

        # Update availability tracking
        video_reading._AVAILABLE_IMAGE_BACKENDS = {
            "opencv": False,
            "imageio": True,
        }

        # Set preferred backend to opencv (which we mocked as unavailable)
        sio.set_default_image_plugin("opencv")

        # Try to create an ImageVideo
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a warning and fall back to imageio
            backend = ImageVideo(centered_pair_frame_paths)

            # Check that a warning was raised
            assert len(w) > 0
            warning_message = str(w[0].message)
            assert "not available" in warning_message
            assert "opencv" in warning_message

            # Should have fallen back to imageio
            assert backend.plugin == "imageio"

    finally:
        # Restore original state
        sio.set_default_image_plugin(original)
        video_reading._AVAILABLE_IMAGE_BACKENDS = original_backends
