"""Tests for methods in the sleap_io.io.video_reading file."""

import copy
import pickle
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest
from numpy.testing import assert_equal
from pytest_httpserver import HTTPServer

import sleap_io as sio
from sleap_io.io.video_reading import (
    CropVideoBackend,
    HDF5Video,
    ImageVideo,
    MediaVideo,
    TiffVideo,
    VideoBackend,
)
from sleap_io.transform.frame import crop_frame

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


def test_hdf5video_get_frame_raw_bytes(slp_minimal_pkg):
    """Test get_frame_raw_bytes() method of HDF5Video for fast path copying."""
    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) is HDF5Video
    assert backend.has_embedded_images

    # Get raw bytes for frame 0
    raw_bytes = backend.get_frame_raw_bytes(0)

    # Should return int8 numpy array with PNG bytes
    assert raw_bytes is not None
    assert isinstance(raw_bytes, np.ndarray)
    assert raw_bytes.dtype == np.int8

    # Verify PNG magic bytes (137, 80, 78, 71 = \x89PNG)
    # These are stored as int8, so 137 becomes -119
    png_magic = raw_bytes[:4].view(np.uint8)
    assert list(png_magic) == [137, 80, 78, 71]

    # Raw bytes should decode to the same image as normal read
    decoded_via_backend = backend[0]
    import cv2

    decoded_from_raw = cv2.imdecode(raw_bytes.view(np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded_from_raw.ndim == 2:
        decoded_from_raw = np.expand_dims(decoded_from_raw, axis=-1)
    assert decoded_via_backend.shape == decoded_from_raw.shape
    np.testing.assert_array_equal(decoded_via_backend, decoded_from_raw)


def test_hdf5video_get_frame_raw_bytes_unavailable_frame(slp_minimal_pkg):
    """Test get_frame_raw_bytes() returns None for unavailable frames."""
    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) is HDF5Video

    # Frame 999 doesn't exist (only frame 0 is embedded)
    raw_bytes = backend.get_frame_raw_bytes(999)
    assert raw_bytes is None


def test_hdf5video_get_frame_raw_bytes_non_embedded(tmpdir):
    """Test get_frame_raw_bytes() returns None for non-embedded HDF5 videos."""
    # Create an HDF5 file with raw image data (not encoded)
    h5_path = str(tmpdir / "raw_video.h5")
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("video", data=np.zeros((5, 64, 64, 1), dtype=np.uint8))
        ds.attrs["format"] = "hdf5"  # Raw format, not encoded

    backend = HDF5Video(filename=h5_path, dataset="video")
    assert not backend.has_embedded_images

    # Should return None because images are not encoded
    raw_bytes = backend.get_frame_raw_bytes(0)
    assert raw_bytes is None


def test_hdf5video_get_frame_raw_bytes_keep_open_false(slp_minimal_pkg):
    """Test get_frame_raw_bytes() works with keep_open=False."""
    backend = HDF5Video.from_filename(slp_minimal_pkg, keep_open=False)
    assert type(backend) is HDF5Video
    assert backend.keep_open is False

    # Get raw bytes for frame 0 - should work with keep_open=False
    raw_bytes = backend.get_frame_raw_bytes(0)

    # Should still return valid PNG bytes
    assert raw_bytes is not None
    assert isinstance(raw_bytes, np.ndarray)

    # Verify PNG magic bytes
    png_magic = raw_bytes[:4].view(np.uint8)
    assert list(png_magic) == [137, 80, 78, 71]


def test_hdf5video_pickle_drops_reader(slp_minimal_pkg):
    """Pickling an HDF5Video drops the open reader and stays readable."""
    backend = VideoBackend.from_filename(slp_minimal_pkg)
    assert type(backend) is HDF5Video

    # Force the cached reader handle to be populated by reading a frame.
    _ = backend[0]
    assert backend._open_reader is not None

    restored = pickle.loads(pickle.dumps(backend))
    assert type(restored) is HDF5Video

    # The unpicklable h5py handle must not survive the round-trip.
    assert restored._open_reader is None
    assert restored._url_file is None

    # Shape (and lazy reopening) still works after unpickling.
    assert restored.shape == backend.shape
    assert_equal(restored[0], backend[0])


def test_hdf5video_url_reads_frames(httpserver, slp_minimal_pkg):
    """An HDF5Video with a URL filename reads frames via fsspec over HTTP.

    The fixture is served with a plain 200 (no Range/206 support) to confirm
    small files load via a full read.
    """
    file_bytes = Path(slp_minimal_pkg).read_bytes()
    httpserver.expect_request("/labels.pkg.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.pkg.slp")

    # Construct the HDF5Video directly against the URL and exercise _open_h5.
    # Use the "download" stream mode (full read into memory) so it works against
    # a server that only returns 200 with no Range support.
    backend = HDF5Video(filename=url, dataset=None, keep_open=False)
    object.__setattr__(backend, "_url_stream_mode", "download")
    object.__setattr__(backend, "_url_file", None)
    assert type(backend) is HDF5Video

    # Re-run heuristic detection over the URL now that download mode is set so
    # the embedded image format/dataset are read from the remote file.
    backend.__attrs_post_init__()
    assert backend.dataset == "video0/video"
    assert backend.has_embedded_images

    # _open_h5() should open the remote file and yield a usable handle.
    with backend._open_h5() as f:
        assert "video0/video" in f

    # Reading the first frame should pull bytes over HTTP and decode them.
    frame = backend[0]
    assert frame.shape == (384, 384, 1)

    # Compare against a local read of the same fixture for parity.
    local = HDF5Video(filename=slp_minimal_pkg, dataset="video0/video", keep_open=False)
    assert_equal(frame, local[0])


def test_hdf5video_post_init_closes_probe_handle_local(slp_minimal_pkg):
    """`__attrs_post_init__` does not leak the probe HDF5 handle (local).

    Re-running the heuristic probe must leave no open h5py file handle behind:
    the probe handle is closed in a `finally` block regardless of early returns.
    """
    backend = HDF5Video(filename=slp_minimal_pkg, dataset=None, keep_open=False)
    # The post-init probe ran during construction; it must not have stashed an
    # open file handle (local files have no cached `_url_file`).
    assert backend._url_file is None
    # Re-running the probe explicitly must also leave nothing dangling.
    backend.__attrs_post_init__()
    assert backend._url_file is None
    # Opening a fresh handle and checking the file object closes cleanly proves
    # the probe did not hold the file open exclusively.
    with h5py.File(slp_minimal_pkg, "r") as f:
        assert "video0/video" in f


def test_hdf5video_post_init_releases_probe_url_file(httpserver, slp_minimal_pkg):
    """`__attrs_post_init__` releases a URL file-like opened solely to probe.

    A probe-time open must not leak the cached fsspec `_url_file`, so that a
    later (possibly authenticated) read reopens cleanly rather than reusing the
    probe handle.
    """
    file_bytes = Path(slp_minimal_pkg).read_bytes()
    httpserver.expect_request("/labels.pkg.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.pkg.slp")

    backend = HDF5Video(filename=url, dataset=None, keep_open=False)
    object.__setattr__(backend, "_url_stream_mode", "download")
    object.__setattr__(backend, "_url_file", None)

    backend.__attrs_post_init__()
    # The probe detected the embedded dataset...
    assert backend.dataset == "video0/video"
    # ...but did not leave the cached fsspec file-like dangling.
    assert backend._url_file is None


def test_hdf5video_close_releases_url_file(httpserver, slp_minimal_pkg):
    """`HDF5Video.close()` closes and drops the cached fsspec URL file-like."""
    file_bytes = Path(slp_minimal_pkg).read_bytes()
    httpserver.expect_request("/labels.pkg.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.pkg.slp")

    backend = HDF5Video(
        filename=url, dataset=None, keep_open=True, url_stream_mode="download"
    )
    _ = backend[0]
    assert backend._url_file is not None
    assert backend._open_reader is not None
    url_file = backend._url_file

    backend.close()
    assert backend._url_file is None
    assert backend._open_reader is None
    # The fsspec/BytesIO handle is deterministically closed, not left to GC.
    assert url_file.closed is True


def test_hdf5video_close_idempotent(httpserver, slp_minimal_pkg):
    """`HDF5Video.close()` is safe to call twice (no raise, stays released)."""
    file_bytes = Path(slp_minimal_pkg).read_bytes()
    httpserver.expect_request("/labels.pkg.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.pkg.slp")

    backend = HDF5Video(
        filename=url, dataset=None, keep_open=True, url_stream_mode="download"
    )
    _ = backend[0]
    backend.close()
    backend.close()  # second call must not raise
    assert backend._url_file is None
    assert backend._open_reader is None


def test_hdf5video_close_local_releases_reader(slp_minimal_pkg):
    """`HDF5Video.close()` releases the local reader; reopens lazily."""
    backend = HDF5Video(filename=slp_minimal_pkg, dataset="video0/video")
    _ = backend[0]
    assert backend._open_reader is not None
    assert backend._url_file is None  # local files have no URL file-like

    backend.close()
    assert backend._open_reader is None
    assert backend._url_file is None
    # The reader is lazily reopened on the next read.
    assert backend[0].shape == (384, 384, 1)


def test_hdf5video_close_preserves_url_context(httpserver, slp_minimal_pkg):
    """`close()` releases handles but keeps the URL auth/stream context.

    The auth headers and stream mode must survive a close so a reopened read
    stays authenticated.
    """
    file_bytes = Path(slp_minimal_pkg).read_bytes()
    httpserver.expect_request("/labels.pkg.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.pkg.slp")

    backend = HDF5Video(
        filename=url,
        dataset=None,
        keep_open=True,
        url_headers={"Authorization": "Bearer x"},
        url_stream_mode="download",
    )
    _ = backend[0]
    backend.close()
    assert backend._url_headers == {"Authorization": "Bearer x"}
    assert backend._url_stream_mode == "download"


def test_mediavideo_close_releases_reader(centered_pair_low_quality_path):
    """`VideoBackend.close()` releases a MediaVideo reader (close or release)."""
    backend = VideoBackend.from_filename(centered_pair_low_quality_path, keep_open=True)
    assert type(backend) is MediaVideo
    _ = backend[0]
    assert backend._open_reader is not None
    reader = backend._open_reader

    backend.close()
    assert backend._open_reader is None
    # An OpenCV VideoCapture is genuinely released (no longer opened); an
    # imageio reader is closed. The duck-typed closer handles both.
    if hasattr(reader, "isOpened"):
        assert reader.isOpened() is False
    # Reopens lazily on the next read.
    assert backend[0] is not None

    # A subsequent read still works (reopens the URL lazily).
    frame = backend[0]
    assert frame.shape == (384, 384, 1)


def test_hdf5video_get_frame_raw_bytes_fixed_length(tmpdir):
    """Test get_frame_raw_bytes() with fixed-length dataset (strips trailing zeros)."""
    import cv2

    # Create a test image and encode it to PNG
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", test_img)
    png_bytes = encoded.flatten().astype(np.int8)

    # Create HDF5 with fixed-length dataset (padded with zeros)
    h5_path = str(tmpdir / "fixed_length.h5")
    max_size = len(png_bytes) + 1000  # Add padding space
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("video", shape=(1, max_size), dtype=np.int8)
        ds[0, : len(png_bytes)] = png_bytes  # Rest is zeros (padding)
        ds.attrs["format"] = "png"
        ds.attrs["imgformat"] = "png"

    # Read back with get_frame_raw_bytes - should strip trailing zeros
    backend = HDF5Video(filename=h5_path, dataset="video", keep_open=False)
    assert backend.has_embedded_images is True

    raw_bytes = backend.get_frame_raw_bytes(0)
    assert raw_bytes is not None

    # Should have stripped trailing zeros - length should match original
    assert len(raw_bytes) == len(png_bytes)
    np.testing.assert_array_equal(raw_bytes, png_bytes)


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


def test_mediavideo_fps(centered_pair_low_quality_path):
    """Test FPS property on MediaVideo backend."""
    # Test with FFMPEG backend
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="FFMPEG"
    )
    assert isinstance(backend, MediaVideo)
    assert backend.fps == 15.0

    # Test with OpenCV backend
    backend = VideoBackend.from_filename(
        centered_pair_low_quality_path, plugin="opencv"
    )
    assert isinstance(backend, MediaVideo)
    assert backend.fps == 15.0

    # Test that FPS can be overridden
    backend.fps = 30.0
    assert backend.fps == 30.0

    # Test that None resets to reading from file
    backend.fps = None
    assert backend.fps == 15.0


def test_fps_setter_validation():
    """Test that FPS setter validates input."""
    backend = ImageVideo.__new__(ImageVideo)
    backend._fps = None

    # Valid values
    backend.fps = 30.0
    assert backend.fps == 30.0

    backend.fps = 0.5
    assert backend.fps == 0.5

    backend.fps = None
    assert backend.fps is None

    # Invalid values
    with pytest.raises(ValueError, match="FPS must be positive"):
        backend.fps = 0

    with pytest.raises(ValueError, match="FPS must be positive"):
        backend.fps = -10.0


def test_imagevideo_fps(centered_pair_frame_paths):
    """Test FPS property on ImageVideo backend."""
    backend = VideoBackend.from_filename(centered_pair_frame_paths)
    assert isinstance(backend, ImageVideo)

    # ImageVideo has no inherent FPS
    assert backend.fps is None

    # Can set FPS explicitly
    backend.fps = 25.0
    assert backend.fps == 25.0


def test_hdf5video_fps_from_attrs(tmp_path):
    """Test FPS property on HDF5Video with fps attribute."""
    # Create HDF5 file with fps attribute
    h5_path = tmp_path / "test_fps.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("video0")
        ds = grp.create_dataset("video", shape=(5, 100, 100, 3), dtype="uint8")
        ds.attrs["format"] = "hdf5"
        ds.attrs["fps"] = 24.0
        ds.attrs["height"] = 100
        ds.attrs["width"] = 100
        ds.attrs["channels"] = 3

    backend = VideoBackend.from_filename(h5_path, dataset="video0/video")
    assert isinstance(backend, HDF5Video)
    assert backend.fps == 24.0


def test_hdf5video_fps_explicit(tmp_path, centered_pair_low_quality_path):
    """Test explicit FPS setting on HDF5Video."""
    # Create simple HDF5 video without fps attribute
    source = VideoBackend.from_filename(centered_pair_low_quality_path)
    imgs = source[:3]

    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("images", data=imgs)

    backend = VideoBackend.from_filename(h5_path)
    assert isinstance(backend, HDF5Video)
    assert backend.fps is None

    # Set FPS explicitly
    backend.fps = 30.0
    assert backend.fps == 30.0


def test_tiffvideo_fps(tmp_path):
    """Test FPS property on TiffVideo backend."""
    # Create multi-page TIFF
    tiff_path = tmp_path / "test.tif"
    frames = np.random.randint(0, 255, (5, 100, 100), dtype=np.uint8)
    iio.imwrite(tiff_path, frames)

    backend = VideoBackend.from_filename(tiff_path)
    assert isinstance(backend, TiffVideo)

    # TiffVideo has no inherent FPS
    assert backend.fps is None

    # Can set FPS explicitly
    backend.fps = 10.0
    assert backend.fps == 10.0


# ---------------------------------------------------------------------------
# Remote (URL) media video loading via pyav.
#
# av.open / imageio's pyav plugin stream the full body via a plain GET (no
# Range requests), so a simple ``respond_with_data`` 200 server is sufficient
# to back these tests with a real fixture video's bytes.
# ---------------------------------------------------------------------------


def _serve_video(server: HTTPServer, path: str, route: str) -> str:
    """Serve a local video file's bytes at ``route`` and return its URL.

    Args:
        server: A running ``pytest_httpserver`` server.
        path: Local path to the fixture video to serve.
        route: The URL route to bind (e.g. ``"/video.mp4"``).

    Returns:
        The full URL the served bytes are reachable at.
    """
    data = Path(path).read_bytes()
    server.expect_request(route).respond_with_data(data, content_type="video/mp4")
    return server.url_for(route)


def test_from_filename_url_routes_to_mediavideo(
    httpserver, centered_pair_low_quality_path
):
    """A media-video URL is routed to MediaVideo with the pyav plugin."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    backend = VideoBackend.from_filename(url)
    assert type(backend) is MediaVideo
    assert backend.filename == url
    # URLs default to the pyav plugin (validated/normalized on the backend).
    assert backend.plugin == "pyav"


def test_from_filename_url_respects_explicit_plugin(
    httpserver, centered_pair_low_quality_path
):
    """An explicit ``plugin`` on a URL is honored (not overridden by the default)."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    backend = VideoBackend.from_filename(url, plugin="pyav")
    assert type(backend) is MediaVideo
    assert backend.plugin == "pyav"
    assert backend[0].shape == (384, 384, 1)


def test_from_filename_url_with_query_string_matches_extension(
    httpserver, centered_pair_low_quality_path
):
    """A URL carrying a ``?query`` still matches the mp4 extension and reads."""
    base = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    url = base + "?token=secret&x=1"
    backend = VideoBackend.from_filename(url)
    assert type(backend) is MediaVideo
    assert backend[0].shape == (384, 384, 1)


def test_load_video_url_reads_first_frame(httpserver, centered_pair_low_quality_path):
    """A frame read over http matches the frame read from the local file."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")

    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)

    local_frame = local[0]
    remote_frame = remote[0]
    assert remote_frame.shape == local_frame.shape
    assert remote_frame.dtype == local_frame.dtype
    assert_equal(remote_frame, local_frame)


def test_load_video_url_num_frames(httpserver, centered_pair_low_quality_path):
    """num_frames over http (via pyav improps) matches the local value."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)
    assert remote.backend.num_frames == local.backend.num_frames == 1100


def test_load_video_url_img_shape(httpserver, centered_pair_low_quality_path):
    """img_shape over http (via a pyav test-frame read) matches the local value."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)
    assert remote.backend.img_shape == local.backend.img_shape == (384, 384, 1)


def test_load_video_url_fps(httpserver, centered_pair_low_quality_path):
    """FPS over http (via av.open) matches the local value."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)
    assert remote.backend.fps == local.backend.fps == 15.0


def test_load_video_url_full_shape(httpserver, centered_pair_low_quality_path):
    """The full (frames, h, w, c) shape over http matches the local value."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)
    assert remote.shape == local.shape == (1100, 384, 384, 1)


def test_load_video_url_get_frames(httpserver, centered_pair_low_quality_path):
    """Reading multiple frames over http matches the local frames."""
    url = _serve_video(httpserver, centered_pair_low_quality_path, "/video.mp4")
    local = sio.load_video(centered_pair_low_quality_path)
    remote = sio.load_video(url)
    assert_equal(remote[:3], local[:3])


def test_remote_fps_is_cached(httpserver, centered_pair_low_quality_path):
    """Reading ``fps`` on a remote video downloads once, then is cached.

    Regression: ``MediaVideo.fps`` previously re-read container metadata on every
    access, and ``av.open`` over http streams the full body each time (no Range
    requests), so repeated ``fps`` access re-downloaded the whole video.
    """
    data = Path(centered_pair_low_quality_path).read_bytes()
    get_count = {"n": 0}

    def handler(request):
        get_count["n"] += 1
        from werkzeug.wrappers import Response

        return Response(data, content_type="video/mp4")

    httpserver.expect_request("/video.mp4").respond_with_handler(handler)
    url = httpserver.url_for("/video.mp4")

    video = sio.load_video(url)
    assert video.backend.fps == 15.0
    downloads_after_first = get_count["n"]
    assert downloads_after_first == 1

    # A second access (and any helper that reads fps repeatedly) is free.
    assert video.backend.fps == 15.0
    assert video.frame_to_seconds(15) == 1.0
    assert get_count["n"] == downloads_after_first
    assert video.backend._fps == 15.0


def test_from_filename_cloud_scheme_video_raises(centered_pair_low_quality_path):
    """A cloud-scheme media URL is rejected before reaching the decoder.

    Only http/https remote video is supported; cloud schemes (s3/gs/...) are
    recognized as remote by ``_is_url`` but must not be handed to ``av.open``.
    """
    for url in ("s3://bucket/video.mp4", "gs://bucket/clip.mov"):
        with pytest.raises(NotImplementedError, match="http/https"):
            VideoBackend.from_filename(url)


def test_from_filename_cloud_scheme_video_does_not_leak_credentials():
    """The cloud-scheme rejection redacts credentials in its error message."""
    url = "s3://AKIA:secretkey@bucket/video.mp4?X-Amz-Security-Token=topsecret"
    with pytest.raises(NotImplementedError) as exc_info:
        VideoBackend.from_filename(url)
    message = str(exc_info.value)
    assert "secretkey" not in message
    assert "topsecret" not in message


def test_remote_video_open_error_redacts_url():
    """A failed open of a tokenized remote video redacts the URL in the error.

    Regression: ``Video.open`` raised ``FileNotFoundError`` with the raw URL,
    leaking ``?token=`` credentials into tracebacks/logs.
    """
    url = "https://host.invalid/video.mp4?token=supersecret"
    video = sio.load_video(url)
    with pytest.raises(FileNotFoundError) as exc_info:
        video[0]
    message = str(exc_info.value)
    assert "supersecret" not in message
    assert "token=%2A%2A%2A" in message


def test_is_pyav_available_when_installed():
    """``_is_pyav_available`` reports True in the test environment (av installed).

    The complementary False branch only fires for an install lacking the
    ``[pyav]`` extra; it is marked ``# pragma: no cover`` rather than forced via
    monkeypatching, since ``av`` is a required dependency of the test suite.
    """
    from sleap_io.io.video_reading import _is_pyav_available

    assert _is_pyav_available() is True


# ---------------------------------------------------------------------------
# CropVideoBackend
# ---------------------------------------------------------------------------


def _make_chunked_hdf5(path, data, input_format="channels_last", chunks=None):
    """Write a raw rank-4 HDF5 dataset and return an HDF5Video over it.

    Args:
        path: Destination file path.
        data: Rank-4 array to store (on-disk layout matching ``input_format``).
        input_format: Either "channels_last" or "channels_first".
        chunks: Chunk shape passed to ``create_dataset`` (None = whole-frame).

    Returns:
        An ``HDF5Video`` opened over the written dataset.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("video", data=data, chunks=chunks)
    return HDF5Video(filename=str(path), dataset="video", input_format=input_format)


def test_crop_backend_basic_shape(centered_pair_low_quality_path):
    """Cropped img_shape/shape/len/num_frames report the cropped view."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    crop = (10, 20, 110, 120)
    cb = CropVideoBackend.wrap(inner, crop)

    assert isinstance(cb, VideoBackend)
    assert cb.num_frames == inner.num_frames
    assert cb.img_shape == (100, 100, 1)
    assert cb.shape == (inner.num_frames, 100, 100, 1)
    assert len(cb) == inner.num_frames
    assert cb.filename == centered_pair_low_quality_path


def test_crop_backend_byte_parity(centered_pair_low_quality_path):
    """Cropped frame is byte-identical to crop_frame(full, rect)."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    crop = (30, 40, 130, 150)
    cb = CropVideoBackend.wrap(inner, crop)

    full = inner.get_frame(0)
    ref = crop_frame(full, crop)
    assert_equal(cb[0], ref)


def test_crop_backend_oob_negative_pads(centered_pair_low_quality_path):
    """Negative-origin crop pads with the fill value."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    crop = (-10, -5, 20, 25)
    cb = CropVideoBackend.wrap(inner, crop, fill=7)

    full = inner.get_frame(0)
    ref = crop_frame(full, crop, fill=7)
    assert cb[0].shape == (30, 30, 1)
    assert_equal(cb[0], ref)


def test_crop_backend_oob_oversized_pads(centered_pair_low_quality_path):
    """Oversized crop (beyond source bounds) pads with the fill value."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    crop = (360, 360, 420, 420)
    cb = CropVideoBackend.wrap(inner, crop, fill=3)

    full = inner.get_frame(0)
    ref = crop_frame(full, crop, fill=3)
    assert cb[0].shape == (60, 60, 1)
    assert_equal(cb[0], ref)


def test_crop_backend_grayscale_full_frame_detection(
    centered_pair_low_quality_path,
):
    """Grayscale is detected on the UNCROPPED frame, even for a 1px-wide crop."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    cb = CropVideoBackend.wrap(inner, (10, 10, 11, 60))  # 1px wide
    # Construction is lazy: grayscale is unresolved until first access (no decode).
    assert cb.grayscale is None
    # img_shape resolves grayscale on the inner's full frame and reports the crop.
    assert cb.img_shape == (50, 1, 1)
    assert cb.grayscale is True
    # detect_grayscale resolves from the inner full frame, not the crop.
    assert cb.detect_grayscale() is True
    # read_test_frame returns the UNCROPPED inner frame.
    assert cb.read_test_frame().shape[0] == inner.read_test_frame().shape[0]


def test_crop_backend_getitem_scalar_list_slice(
    centered_pair_low_quality_path,
):
    """Scalar/list/slice __getitem__ all return the cropped view."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    cb = CropVideoBackend.wrap(inner, (0, 0, 50, 50))

    assert cb[0].shape == (50, 50, 1)
    assert cb[[0, 1, 2]].shape == (3, 50, 50, 1)
    assert cb[0:3].shape == (3, 50, 50, 1)

    full = inner.get_frames([0, 1, 2])
    ref = np.stack([crop_frame(f, (0, 0, 50, 50)) for f in full], axis=0)
    assert_equal(cb[[0, 1, 2]], ref)


def test_crop_backend_has_frame_delegates(centered_pair_low_quality_path):
    """has_frame delegates to the inner backend."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    cb = CropVideoBackend.wrap(inner, (0, 0, 50, 50))
    assert cb.has_frame(0) is True
    assert cb.has_frame(10**9) is False


@pytest.mark.parametrize("crop", [(5, 5, 55, 65), (-5, -5, 45, 45)])
def test_crop_backend_batched_image_video_parity(centered_pair_frame_paths, crop):
    """Batched crop over an ImageVideo (base-default _read_frames) is byte-parity.

    The in-bounds rect exercises the fast slice+copy path; the OOB rect exercises the
    np.stack(crop_frame) padded fallback that the MediaVideo batched test never hits.
    ImageVideo uses the base-default _read_frames (applies grayscale early), so this
    also guards the D-114 idempotent-grayscale invariant.
    """
    inner = VideoBackend.from_filename(centered_pair_frame_paths)
    assert isinstance(inner, ImageVideo)
    cb = CropVideoBackend.wrap(inner, crop, fill=7)
    full = inner.get_frames([0, 1, 2])
    ref = np.stack([crop_frame(f, crop, fill=7) for f in full], axis=0)
    out = cb[[0, 1, 2]]
    assert out.dtype != object
    assert_equal(out, ref)


def test_crop_backend_deepcopy_roundtrip(centered_pair_low_quality_path):
    """Deepcopy preserves crop/fill and frame parity."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path, keep_open=False)
    cb = CropVideoBackend.wrap(inner, (5, 6, 55, 66), fill=4)
    ref = cb[0].copy()

    cc = copy.deepcopy(cb)
    assert cc.crop == (5, 6, 55, 66)
    assert cc.fill == 4
    assert cc.owns_inner is True
    assert_equal(cc[0], ref)


def test_crop_backend_pickle_roundtrip(centered_pair_low_quality_path):
    """Pickle preserves crop/fill/filename and frame parity."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path, keep_open=False)
    cb = CropVideoBackend.wrap(inner, (5, 6, 55, 66), fill=4)
    ref = cb[0].copy()

    pk = pickle.loads(pickle.dumps(cb))
    assert pk.crop == (5, 6, 55, 66)
    assert pk.fill == 4
    assert pk.filename == centered_pair_low_quality_path
    assert_equal(pk[0], ref)


def test_crop_backend_identity_equality(centered_pair_low_quality_path):
    """Equality is by object identity (eq=False), robust to cache state."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    a = CropVideoBackend.wrap(inner, (0, 0, 50, 50))
    b = CropVideoBackend.wrap(inner, (0, 0, 50, 50))
    _ = a.shape  # populate a's cached shape only
    assert a == a
    assert a != b  # distinct instances despite identical crop/fill


def test_crop_backend_channels_first_parity(tmp_path):
    """channels_first synthetic HDF5 crop matches crop_frame byte-for-byte."""
    n, h, w, c = 3, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    disk = np.transpose(data, (0, 3, 2, 1)).copy()  # (N, C, W, H)
    inner = _make_chunked_hdf5(
        tmp_path / "cf.h5", disk, "channels_first", chunks=(1, c, 8, 8)
    )
    cb = CropVideoBackend.wrap(inner, (5, 6, 25, 26))

    ref = crop_frame(inner._read_frame(0), (5, 6, 25, 26))
    assert_equal(cb[0], ref)


def test_crop_backend_flatten_byte_parity(centered_pair_low_quality_path):
    """crop-of-crop FLATTENS (inner not a crop) with byte-parity to nested."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    c1 = CropVideoBackend.wrap(inner, (10, 10, 110, 110))
    c2 = CropVideoBackend.wrap(c1, (5, 5, 40, 40))

    # Flattened: inner is the original (never a crop), crop is composed.
    assert not isinstance(c2.inner, CropVideoBackend)
    assert c2.crop == (15, 15, 50, 50)

    full = inner.get_frame(0)
    nested = crop_frame(crop_frame(full, (10, 10, 110, 110)), (5, 5, 40, 40))
    assert_equal(c2[0], nested)


def test_crop_backend_nest_when_outer_exceeds_inner(
    centered_pair_low_quality_path,
):
    """F-2: outer crop exceeding the inner frame must NEST (byte-parity)."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    c1 = CropVideoBackend.wrap(inner, (10, 10, 110, 110))  # 100x100 frame
    c2 = CropVideoBackend.wrap(c1, (-1, -1, 40, 40))  # exceeds [0,100]x[0,100]

    assert isinstance(c2.inner, CropVideoBackend)

    full = inner.get_frame(0)
    nested = crop_frame(crop_frame(full, (10, 10, 110, 110)), (-1, -1, 40, 40))
    assert_equal(c2[0], nested)


def test_crop_backend_nest_when_fill_mismatch(centered_pair_low_quality_path):
    """Crop-of-crop with mismatched fills must NEST (byte-parity)."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    c1 = CropVideoBackend.wrap(inner, (10, 10, 110, 110), fill=0)
    c2 = CropVideoBackend.wrap(c1, (-2, -2, 40, 40), fill=9)

    assert isinstance(c2.inner, CropVideoBackend)

    full = inner.get_frame(0)
    nested = crop_frame(
        crop_frame(full, (10, 10, 110, 110), fill=0), (-2, -2, 40, 40), fill=9
    )
    assert_equal(c2[0], nested)


def test_crop_backend_owns_inner_false_does_not_close(tmp_path):
    """owns_inner=False: close does NOT close the (shared) inner."""
    data = np.arange(4 * 16 * 16 * 3, dtype=np.uint8).reshape(4, 16, 16, 3)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 4, 4, 3)
    )
    _ = inner._read_frame(0)
    assert inner._open_reader is not None

    cb = CropVideoBackend.wrap(inner, (2, 2, 12, 12), owns_inner=False)
    cb.close()
    assert inner._open_reader is not None  # sibling-safe: inner stays open


def test_crop_backend_owns_inner_true_closes(tmp_path):
    """owns_inner=True (default): close cascades to the inner."""
    data = np.arange(4 * 16 * 16 * 3, dtype=np.uint8).reshape(4, 16, 16, 3)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 4, 4, 3)
    )
    _ = inner._read_frame(0)
    assert inner._open_reader is not None

    cb = CropVideoBackend.wrap(inner, (2, 2, 12, 12))
    cb.close()
    assert inner._open_reader is None


def test_crop_backend_dataset_delegation(slp_minimal_pkg):
    """dataset/input_format delegate to the inner backend."""
    inner = VideoBackend.from_filename(slp_minimal_pkg)
    cb = CropVideoBackend.wrap(inner, (5, 5, 55, 55))
    assert cb.dataset == inner.dataset
    assert cb.input_format == inner.input_format


def test_crop_backend_to_crop_to_source_coords(centered_pair_low_quality_path):
    """to_crop_coords / to_source_coords translate by (x1, y1) and invert."""
    inner = VideoBackend.from_filename(centered_pair_low_quality_path)
    cb = CropVideoBackend.wrap(inner, (10, 20, 110, 120))

    pts = np.array([[15.0, 25.0], [np.nan, np.nan], [110.0, 120.0]])
    cropped = cb.to_crop_coords(pts)
    assert_equal(cropped[0], np.array([5.0, 5.0]))
    assert np.isnan(cropped[1]).all()
    # Round-trip back to source coordinates.
    back = cb.to_source_coords(cropped)
    assert_equal(back[0], pts[0])
    assert np.isnan(back[1]).all()
    assert_equal(back[2], pts[2])


# ---------------------------------------------------------------------------
# HDF5 pushdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crop",
    [(5, 5, 30, 30), (-3, -4, 12, 11), (40, 30, 60, 47), (-4, -4, 47, 47)],
)
def test_hdf5_read_crop_channels_last_parity(tmp_path, crop):
    """read_crop byte-parity with crop_frame (channels_last, sub-frame chunks).

    Covers inside, off-each-edge, and straddle cases. Crops are chosen so the
    reference ``crop_frame`` has a non-empty valid source region on both axes
    (it does not support a crop wholly beyond the frame on one axis); the
    fully-outside pushdown case is exercised separately in
    :func:`test_hdf5_pushdown_fully_outside_all_fill`.
    """
    n, h, w, c = 3, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "cl.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    assert inner._can_push_crop is True

    ref = crop_frame(inner._read_frame(0), crop, fill=9)
    out = inner.read_crop(0, crop, fill=9)
    assert out is not None
    assert out.dtype == data.dtype
    assert_equal(out, ref)


def test_hdf5_pushdown_fully_outside_all_fill(tmp_path):
    """A crop wholly outside the frame pushes down to an all-fill array."""
    n, h, w, c = 2, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    crop = (-30, -30, -10, -10)  # entirely above/left of the frame
    out = inner.read_crop(0, crop, fill=4)
    assert out is not None
    assert_equal(out, np.full((20, 20, c), 4, dtype=data.dtype))


@pytest.mark.parametrize(
    "crop",
    [(5, 6, 30, 31), (-3, -4, 12, 11), (60, 40, 90, 70)],
)
def test_hdf5_read_crop_channels_first_parity(tmp_path, crop):
    """read_crop byte-parity with crop_frame (channels_first, sub-frame chunks)."""
    n, h, w, c = 3, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    disk = np.transpose(data, (0, 3, 2, 1)).copy()  # (N, C, W, H)
    inner = _make_chunked_hdf5(
        tmp_path / "cf.h5", disk, "channels_first", chunks=(1, c, 16, 16)
    )
    assert inner._can_push_crop is True

    ref = crop_frame(inner._read_frame(0), crop, fill=9)
    out = inner.read_crop(0, crop, fill=9)
    assert out is not None
    assert_equal(out, ref)


def test_hdf5_read_crops_batched_parity(tmp_path):
    """Batched read_crops byte-parity with stacked per-frame crop_frame."""
    n, h, w, c = 4, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "cl.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    crop = (5, 5, 30, 30)
    out = inner.read_crops([0, 1, 2], crop, fill=2)
    ref = np.stack(
        [crop_frame(inner._read_frame(i), crop, fill=2) for i in range(3)], axis=0
    )
    assert out is not None
    assert_equal(out, ref)


def test_hdf5_pushdown_whole_frame_chunk_falls_back(tmp_path):
    """Whole-frame chunking disables pushdown (dataset-level gate)."""
    n, h, w, c = 2, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "whole.h5", data, "channels_last", chunks=(1, h, w, c)
    )
    assert inner._can_push_crop is False
    assert inner.read_crop(0, (5, 5, 20, 20)) is None


def test_hdf5_pushdown_unchunked_falls_back(tmp_path):
    """Contiguous (unchunked) datasets disable pushdown."""
    n, h, w, c = 2, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "contig.h5", data, "channels_last", chunks=None
    )
    assert inner._can_push_crop is False
    assert inner.read_crop(0, (5, 5, 20, 20)) is None


def test_hdf5_pushdown_per_call_gate_full_span_falls_back(tmp_path):
    """A crop as large as the chunk-spanned frame falls back (per-call gate)."""
    n, h, w, c = 2, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    assert inner._can_push_crop is True
    # A crop covering the entire frame spans every chunk -> no benefit -> None.
    assert inner.read_crop(0, (0, 0, w, h)) is None


def test_hdf5_read_crops_per_call_gate_full_span_falls_back(tmp_path):
    """Batched read_crops returns a clean None (not an object array) on gate decline.

    Regression: a full-span crop makes every per-frame read_crop return None; the
    batched path must return None (so the wrapper falls back to full decode + crop),
    never ``np.stack([None, None])`` which yields a corrupt object array.
    """
    n, h, w, c = 3, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    assert inner._can_push_crop is True
    assert inner.read_crops([0, 1, 2], (0, 0, w, h)) is None
    # keep_open=False reopen branch must also fall back cleanly.
    inner_ko = _make_chunked_hdf5(
        tmp_path / "v2.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    inner_ko.keep_open = False
    assert inner_ko.read_crops([0, 1], (0, 0, w, h)) is None


def test_crop_backend_batched_full_span_fallback_parity(tmp_path):
    """Wrapper batched read over a full-span crop returns a real array, not object."""
    n, h, w, c = 3, 32, 40, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    cb = CropVideoBackend.wrap(inner, (0, 0, w, h))  # no-op crop, spans all chunks
    out = cb[[0, 1, 2]]
    assert out.dtype != object
    assert out.dtype == data.dtype
    ref = np.stack(
        [crop_frame(inner._read_frame(i), (0, 0, w, h)) for i in range(3)], axis=0
    )
    assert_equal(out, ref[..., [0]] if cb.grayscale else ref)


def test_hdf5_pushdown_oob_parity(tmp_path):
    """Out-of-bounds pushdown crop pads exactly as crop_frame."""
    n, h, w, c = 2, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    inner = _make_chunked_hdf5(
        tmp_path / "v.h5", data, "channels_last", chunks=(1, 16, 16, c)
    )
    crop = (-8, -8, 8, 8)
    out = inner.read_crop(0, crop, fill=5)
    ref = crop_frame(inner._read_frame(0), crop, fill=5)
    assert out is not None
    assert out.shape == (16, 16, c)
    assert_equal(out, ref)


def test_hdf5_pushdown_keep_open_false_parity(tmp_path):
    """Pushdown works with keep_open=False (reopens per call)."""
    n, h, w, c = 2, 48, 64, 3
    data = np.arange(n * h * w * c, dtype=np.uint8).reshape(n, h, w, c)
    with h5py.File(tmp_path / "v.h5", "w") as f:
        f.create_dataset("video", data=data, chunks=(1, 16, 16, c))
    inner = HDF5Video(filename=str(tmp_path / "v.h5"), dataset="video", keep_open=False)
    crop = (5, 5, 30, 30)
    out = inner.read_crop(0, crop)
    ref = crop_frame(inner._read_frame(0), crop)
    assert out is not None
    assert_equal(out, ref)
    out_b = inner.read_crops([0, 1], crop)
    assert out_b is not None
    assert out_b.shape == (2, 25, 25, c)


def test_hdf5_pushdown_embedded_falls_back(slp_minimal_pkg):
    """Embedded/frame_map inner disables pushdown; crop still works via fallback."""
    inner = VideoBackend.from_filename(slp_minimal_pkg)
    assert inner.frame_map  # embedded subset with a frame map
    assert inner._can_push_crop is False

    fidx = list(inner.frame_map.keys())[0]
    assert inner.read_crop(fidx, (5, 5, 30, 30)) is None
    assert inner.read_crops([fidx], (5, 5, 30, 30)) is None

    # The crop view still works (full decode + crop_frame), shape/grayscale safe.
    cb = CropVideoBackend.wrap(inner, (5, 5, 55, 55))
    assert cb.img_shape == (50, 50, 1)
    assert cb.grayscale is True
    ref = crop_frame(inner._read_frame(fidx), (5, 5, 55, 55))
    assert_equal(cb[fidx], ref[..., [0]])
