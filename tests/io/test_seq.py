"""Tests for the Norpix .seq video backend."""

import struct
from pathlib import Path

import numpy as np
import pytest

from sleap_io.io.seq import _HEADER_SIZE, _MAGIC, SeqHeader, SeqIndex, SeqVideo
from sleap_io.io.video_reading import VideoBackend


def _write_seq_header(
    f,
    width=64,
    height=48,
    image_format=100,
    num_frames=5,
    fps=30.0,
    version=5,
    bit_depth=8,
    bit_depth_real=8,
    image_size_bytes=None,
    true_image_size=None,
):
    """Write a minimal valid .seq header to a binary file handle.

    Args:
        f: Open binary file handle at position 0.
        width: Frame width.
        height: Frame height.
        image_format: Codec identifier (100=monoraw, 200=raw, 102=monojpg, etc).
        num_frames: Number of frames.
        fps: Frame rate.
        version: Format version (>= 5 for 8-byte timestamps).
        bit_depth: Total bit depth.
        bit_depth_real: Bits per channel.
        image_size_bytes: Frame data size. Auto-computed if None.
        true_image_size: Frame stride including timestamps. Auto-computed if None.
    """
    if image_size_bytes is None:
        image_size_bytes = width * height * (bit_depth // bit_depth_real)

    ts_size = 8 if version >= 5 else 6
    if true_image_size is None:
        true_image_size = image_size_bytes + ts_size

    header = bytearray(_HEADER_SIZE)

    # Magic (bytes 0-3)
    struct.pack_into("<I", header, 0, _MAGIC)

    # Name (bytes 4-23): 10 uint16 chars
    name = "TestSeq"
    for i, ch in enumerate(name[:10]):
        struct.pack_into("<H", header, 4 + i * 2, ord(ch))

    # Version and header size (bytes 28-35)
    struct.pack_into("<iI", header, 28, version, _HEADER_SIZE)

    # Description (bytes 36-547): 256 uint16 chars
    desc = "test"
    for i, ch in enumerate(desc[:256]):
        struct.pack_into("<H", header, 36 + i * 2, ord(ch))

    # 9 uint32 fields (bytes 548-583)
    struct.pack_into(
        "<9I",
        header,
        548,
        width,
        height,
        bit_depth,
        bit_depth_real,
        image_size_bytes,
        image_format,
        num_frames,
        0,  # reserved
        true_image_size,
    )

    # FPS (bytes 584-591)
    struct.pack_into("<d", header, 584, fps)

    f.write(header)


def _write_timestamp(f, seconds, ms, us=0, version=5):
    """Write a per-frame timestamp.

    Args:
        f: Open binary file handle.
        seconds: Seconds since epoch (uint32).
        ms: Milliseconds (uint16).
        us: Microseconds (uint16, only for version >= 5).
        version: Format version.
    """
    f.write(struct.pack("<I", seconds))
    f.write(struct.pack("<H", ms))
    if version >= 5:
        f.write(struct.pack("<H", us))


@pytest.fixture
def seq_video_path(tmp_path):
    """Create a minimal monoraw (uncompressed grayscale) .seq file with 5 frames."""
    path = tmp_path / "test_mono.seq"
    width, height = 64, 48
    n_frames = 5

    with open(path, "wb") as f:
        _write_seq_header(f, width=width, height=height, num_frames=n_frames)

        base_ts = 1700000000  # fixed epoch timestamp
        for i in range(n_frames):
            # Each frame has a unique fill value
            frame = np.full((height, width), fill_value=(i + 1) * 40, dtype=np.uint8)
            f.write(frame.tobytes())
            _write_timestamp(f, base_ts + i, ms=0, us=0)

    return path


@pytest.fixture
def seq_video_color_path(tmp_path):
    """Create a raw BGR (uncompressed color) .seq file with 5 frames."""
    path = tmp_path / "test_color.seq"
    width, height = 64, 48
    n_frames = 5

    with open(path, "wb") as f:
        _write_seq_header(
            f,
            width=width,
            height=height,
            image_format=200,  # raw BGR
            num_frames=n_frames,
            bit_depth=24,
            bit_depth_real=8,
        )

        base_ts = 1700000000
        for i in range(n_frames):
            # BGR frame: B=i*10, G=i*20, R=i*30
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i + 1) * 10  # B
            frame[:, :, 1] = (i + 1) * 20  # G
            frame[:, :, 2] = (i + 1) * 30  # R
            f.write(frame.tobytes())
            _write_timestamp(f, base_ts + i, ms=500, us=0)

    return path


@pytest.fixture
def seq_video_compressed_path(tmp_path):
    """Create a monojpg (JPEG compressed grayscale) .seq file with 5 frames."""
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "test_compressed.seq"
    width, height = 64, 48
    n_frames = 5

    # We need to compute frame data first to know total sizes
    frame_data_list = []
    for i in range(n_frames):
        frame = np.full((height, width), fill_value=(i + 1) * 40, dtype=np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        frame_data_list.append(buf.getvalue())

    with open(path, "wb") as f:
        _write_seq_header(
            f,
            width=width,
            height=height,
            image_format=102,  # monojpg
            num_frames=n_frames,
            # For compressed, image_size_bytes and true_image_size are not used
            # for seeking (we scan the file), but set them to something reasonable.
            image_size_bytes=width * height,
            true_image_size=width * height + 8,
        )

        base_ts = 1700000000
        for i in range(n_frames):
            jpeg_bytes = frame_data_list[i]
            # 4-byte size prefix (includes the 4 bytes of the size field itself)
            f.write(struct.pack("<I", len(jpeg_bytes) + 4))
            f.write(jpeg_bytes)
            _write_timestamp(f, base_ts + i, ms=0, us=0)

    return path


@pytest.fixture
def seq_video_png_path(tmp_path):
    """Create a monopng (PNG compressed grayscale) .seq file with 5 frames."""
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "test_png.seq"
    width, height = 64, 48
    n_frames = 5

    frame_data_list = []
    for i in range(n_frames):
        frame = np.full((height, width), fill_value=(i + 1) * 40, dtype=np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_data_list.append(buf.getvalue())

    with open(path, "wb") as f:
        _write_seq_header(
            f,
            width=width,
            height=height,
            image_format=1,  # monopng
            num_frames=n_frames,
            image_size_bytes=width * height,
            true_image_size=width * height + 8,
        )

        base_ts = 1700000000
        for i in range(n_frames):
            png_bytes = frame_data_list[i]
            f.write(struct.pack("<I", len(png_bytes) + 4))
            f.write(png_bytes)
            _write_timestamp(f, base_ts + i, ms=0, us=0)

    return path


@pytest.fixture
def seq_video_color_jpg_path(tmp_path):
    """Create a color JPEG compressed .seq file with 5 frames."""
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "test_color_jpg.seq"
    width, height = 64, 48
    n_frames = 5

    frame_data_list = []
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i + 1) * 30  # R
        frame[:, :, 1] = (i + 1) * 20  # G
        frame[:, :, 2] = (i + 1) * 10  # B
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        frame_data_list.append(buf.getvalue())

    with open(path, "wb") as f:
        _write_seq_header(
            f,
            width=width,
            height=height,
            image_format=201,  # jpg (color)
            num_frames=n_frames,
            bit_depth=24,
            bit_depth_real=8,
            image_size_bytes=width * height * 3,
            true_image_size=width * height * 3 + 8,
        )

        base_ts = 1700000000
        for i in range(n_frames):
            jpeg_bytes = frame_data_list[i]
            f.write(struct.pack("<I", len(jpeg_bytes) + 4))
            f.write(jpeg_bytes)
            _write_timestamp(f, base_ts + i, ms=0, us=0)

    return path


def test_seq_header_parsing(seq_video_path):
    """Test that header fields are correctly parsed from a .seq file."""
    with open(seq_video_path, "rb") as f:
        header = SeqHeader.from_file(f)

    assert header.magic == _MAGIC
    assert header.width == 64
    assert header.height == 48
    assert header.num_frames == 5
    assert header.bit_depth == 8
    assert header.bit_depth_real == 8
    assert header.image_format == 100
    assert header.codec_name == "monoraw"
    assert header.is_compressed is False
    assert header.num_channels == 1
    assert header.name == "TestSeq"
    assert header.fps == 30.0


def test_seq_header_invalid_magic(tmp_path):
    """Test that invalid magic number raises ValueError."""
    path = tmp_path / "bad_magic.seq"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0xDEAD))
        f.write(b"\x00" * (_HEADER_SIZE - 4))

    with pytest.raises(ValueError, match="Invalid .seq magic"):
        SeqVideo(str(path))


def test_seq_header_too_small(tmp_path):
    """Test that a file too small for a header raises ValueError."""
    path = tmp_path / "tiny.seq"
    with open(path, "wb") as f:
        f.write(b"\x00" * 100)

    with pytest.raises(ValueError, match="too small"):
        SeqVideo(str(path))


def test_seq_video_uncompressed(seq_video_path):
    """Test reading uncompressed grayscale frames."""
    backend = SeqVideo(str(seq_video_path))
    assert backend.num_frames == 5

    for i in range(5):
        frame = backend._read_frame(i)
        assert frame.shape == (48, 64, 1)
        assert frame.dtype == np.uint8
        assert np.all(frame == (i + 1) * 40)

    backend.close()


def test_seq_video_color(seq_video_color_path):
    """Test reading uncompressed BGR frames are converted to RGB."""
    backend = SeqVideo(str(seq_video_color_path))
    assert backend.num_frames == 5

    frame = backend._read_frame(0)
    assert frame.shape == (48, 64, 3)
    # Original BGR was B=10, G=20, R=30 → RGB should be R=30, G=20, B=10
    assert frame[0, 0, 0] == 30  # R
    assert frame[0, 0, 1] == 20  # G
    assert frame[0, 0, 2] == 10  # B

    backend.close()


def test_seq_video_compressed(seq_video_compressed_path):
    """Test reading JPEG compressed frames."""
    backend = SeqVideo(str(seq_video_compressed_path))
    assert backend.num_frames == 5

    frame = backend._read_frame(0)
    assert frame.shape == (48, 64, 1)
    assert frame.dtype == np.uint8
    # JPEG is lossy, so check approximate value
    assert np.mean(frame) == pytest.approx(40, abs=5)

    backend.close()


def test_seq_video_png(seq_video_png_path):
    """Test reading PNG compressed frames."""
    backend = SeqVideo(str(seq_video_png_path))
    assert backend.num_frames == 5

    frame = backend._read_frame(0)
    assert frame.shape == (48, 64, 1)
    assert frame.dtype == np.uint8
    # PNG is lossless
    assert np.all(frame == 40)

    backend.close()


def test_seq_video_color_jpg(seq_video_color_jpg_path):
    """Test reading color JPEG compressed frames."""
    backend = SeqVideo(str(seq_video_color_jpg_path))
    assert backend.num_frames == 5

    frame = backend._read_frame(0)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    # JPEG is lossy, check approximate RGB values
    assert np.mean(frame[:, :, 0]) == pytest.approx(30, abs=5)  # R
    assert np.mean(frame[:, :, 1]) == pytest.approx(20, abs=5)  # G
    assert np.mean(frame[:, :, 2]) == pytest.approx(10, abs=5)  # B

    backend.close()


def test_seq_video_from_filename(seq_video_path):
    """Test that VideoBackend.from_filename returns SeqVideo for .seq files."""
    backend = VideoBackend.from_filename(str(seq_video_path))
    assert type(backend) is SeqVideo
    assert backend.num_frames == 5

    backend.close()


def test_seq_video_shape(seq_video_path):
    """Test shape property."""
    backend = SeqVideo(str(seq_video_path))
    assert backend.shape == (5, 48, 64, 1)
    assert backend.img_shape == (48, 64, 1)

    backend.close()


def test_seq_video_shape_color(seq_video_color_path):
    """Test shape property for color video."""
    backend = SeqVideo(str(seq_video_color_path))
    assert backend.img_shape == (48, 64, 3)
    assert backend.shape == (5, 48, 64, 3)

    backend.close()


def test_seq_video_fps(seq_video_path):
    """Test FPS property is computed from timestamps."""
    backend = SeqVideo(str(seq_video_path))
    # Timestamps are 1 second apart, so FPS should be ~1.0
    assert backend.fps == pytest.approx(1.0, abs=0.1)

    backend.close()


def test_seq_video_fps_setter(seq_video_path):
    """Test FPS can be set explicitly."""
    backend = SeqVideo(str(seq_video_path))
    backend.fps = 60.0
    assert backend.fps == 60.0

    with pytest.raises(ValueError, match="positive"):
        backend.fps = -1.0

    backend.close()


def test_seq_video_grayscale_detection(seq_video_path):
    """Test grayscale auto-detection for mono video."""
    backend = SeqVideo(str(seq_video_path), grayscale=None)
    # Access a frame through get_frame to trigger auto-detection
    frame = backend.get_frame(0)
    assert backend.grayscale is True
    assert frame.shape == (48, 64, 1)

    backend.close()


def test_seq_video_slicing(seq_video_path):
    """Test numpy-style indexing with scalars, slices, and negative indices."""
    backend = SeqVideo(str(seq_video_path))

    # Single frame
    frame = backend[0]
    assert frame.shape == (48, 64, 1)

    # Slice
    frames = backend[0:3]
    assert frames.shape == (3, 48, 64, 1)

    # Negative index
    frame_last = backend[-1]
    assert frame_last.shape == (48, 64, 1)
    # Last frame has fill value 5*40=200
    assert np.all(frame_last == 200)

    backend.close()


def test_seq_video_index_cache(seq_video_compressed_path):
    """Test that .seq-index.json is created for compressed files."""
    cache_path = Path(str(seq_video_compressed_path)).with_suffix(".seq-index.json")

    # Cache is created during SeqVideo init
    backend = SeqVideo(str(seq_video_compressed_path))
    assert cache_path.exists()

    # Load it back and verify
    index = SeqIndex.load(cache_path)
    assert index.num_frames == 5
    assert len(index.offsets) == 5
    assert index.offsets[0] == _HEADER_SIZE

    backend.close()

    # Delete cache and reopen — it should be rebuilt
    cache_path.unlink()
    assert not cache_path.exists()
    backend2 = SeqVideo(str(seq_video_compressed_path))
    assert cache_path.exists()
    assert backend2.num_frames == 5

    backend2.close()


def test_seq_video_timestamps(seq_video_path):
    """Test timestamp retrieval."""
    backend = SeqVideo(str(seq_video_path))

    # First frame timestamp
    ts0 = backend.get_timestamp(0)
    assert ts0 == pytest.approx(1700000000.0, abs=1.0)

    # Last frame (negative index)
    ts_last = backend.get_timestamp(-1)
    assert ts_last == pytest.approx(1700000004.0, abs=1.0)

    # All timestamps
    ts_all = backend.get_timestamps()
    assert ts_all.shape == (5,)
    assert ts_all.dtype == np.float64
    # Timestamps should be 1 second apart
    diffs = np.diff(ts_all)
    assert np.allclose(diffs, 1.0, atol=0.01)

    backend.close()


def test_seq_video_timestamps_out_of_range(seq_video_path):
    """Test that out-of-range timestamp access raises IndexError."""
    backend = SeqVideo(str(seq_video_path))

    with pytest.raises(IndexError):
        backend.get_timestamp(100)

    with pytest.raises(IndexError):
        backend.get_timestamp(-100)

    backend.close()


@pytest.mark.parametrize("keep_open", [False, True])
def test_seq_video_keep_open(seq_video_path, keep_open):
    """Test that keep_open parameter works correctly."""
    backend = SeqVideo(str(seq_video_path), keep_open=keep_open)

    # Read multiple frames to exercise open/close logic
    for i in range(5):
        frame = backend._read_frame(i)
        assert frame.shape == (48, 64, 1)

    if keep_open:
        assert backend._file_handle is not None
        assert not backend._file_handle.closed
    else:
        # File handle should not be persistently open
        assert backend._file_handle is None or backend._file_handle.closed

    backend.close()


def test_seq_video_bayer_codec_error(tmp_path):
    """Test that Bayer codecs raise NotImplementedError."""
    path = tmp_path / "bayer.seq"
    with open(path, "wb") as f:
        _write_seq_header(f, image_format=101, num_frames=1)  # brgb8
        # Write a dummy frame and timestamp
        f.write(b"\x00" * (64 * 48))
        _write_timestamp(f, 1700000000, ms=0)

    with pytest.raises(NotImplementedError, match="Bayer"):
        SeqVideo(str(path))


def test_seq_video_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        SeqVideo("/nonexistent/path/video.seq")


def test_seq_index_uncompressed():
    """Test SeqIndex.build_uncompressed produces correct offsets."""
    header = SeqHeader(
        width=64,
        height=48,
        num_frames=3,
        true_image_size=64 * 48 + 8,
        version=5,
    )
    index = SeqIndex.build_uncompressed(header)
    assert index.num_frames == 3
    assert index.offsets[0] == _HEADER_SIZE
    expected_stride = 64 * 48 + 8
    assert index.offsets[1] == _HEADER_SIZE + expected_stride
    assert index.offsets[2] == _HEADER_SIZE + 2 * expected_stride


def test_seq_index_save_load(tmp_path):
    """Test SeqIndex save/load round-trip."""
    index = SeqIndex(
        offsets=[1024, 5000, 9000],
        num_frames=3,
        timestamp_size=8,
    )
    path = tmp_path / "test-index.json"
    index.save(path)

    loaded = SeqIndex.load(path)
    assert loaded.num_frames == 3
    assert loaded.offsets == [1024, 5000, 9000]
    assert loaded.timestamp_size == 8


def test_seq_index_frame_offset_out_of_range():
    """Test that out-of-range frame access raises IndexError."""
    index = SeqIndex(offsets=[1024, 2048], num_frames=2)

    with pytest.raises(IndexError):
        index.frame_offset(2)

    with pytest.raises(IndexError):
        index.frame_offset(-1)


def test_seq_video_header_property(seq_video_path):
    """Test that the header property exposes the parsed header."""
    backend = SeqVideo(str(seq_video_path))
    assert backend.header.width == 64
    assert backend.header.height == 48
    assert backend.header.codec_name == "monoraw"

    backend.close()


def test_seq_video_close(seq_video_path):
    """Test explicit close and double-close safety."""
    backend = SeqVideo(str(seq_video_path), keep_open=True)
    assert backend._file_handle is not None

    backend.close()
    assert backend._file_handle is None

    # Double close should not raise
    backend.close()


def test_seq_video_len(seq_video_path):
    """Test __len__ returns number of frames."""
    backend = SeqVideo(str(seq_video_path))
    assert len(backend) == 5
    backend.close()


def test_seq_video_version4_6byte_timestamps(tmp_path):
    """Test reading a version 4 .seq file with 6-byte timestamps (no microseconds)."""
    path = tmp_path / "test_v4.seq"
    width, height = 64, 48
    n_frames = 5

    with open(path, "wb") as f:
        _write_seq_header(f, width=width, height=height, num_frames=n_frames, version=4)

        base_ts = 1700000000
        for i in range(n_frames):
            frame = np.full((height, width), fill_value=(i + 1) * 40, dtype=np.uint8)
            f.write(frame.tobytes())
            _write_timestamp(f, base_ts + i, ms=500, version=4)

    backend = SeqVideo(str(path))
    assert backend.num_frames == 5

    # Verify frames read correctly
    frame = backend._read_frame(0)
    assert frame.shape == (48, 64, 1)
    assert np.all(frame == 40)

    # Verify timestamps (seconds + ms, no microseconds)
    ts0 = backend.get_timestamp(0)
    assert ts0 == pytest.approx(1700000000.5, abs=0.01)

    ts_all = backend.get_timestamps()
    assert ts_all.shape == (5,)
    diffs = np.diff(ts_all)
    assert np.allclose(diffs, 1.0, atol=0.01)

    backend.close()
