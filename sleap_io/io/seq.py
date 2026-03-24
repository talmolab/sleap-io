"""Backend for reading Norpix .seq video files.

The .seq format is used by StreamPix / Norpix for high-speed video recording in
behavioral neuroscience. This module provides a `SeqVideo` backend that integrates
with sleap-io's `VideoBackend` interface for seamless access to .seq files.

Format overview:
    - 1024-byte binary header (little-endian, magic 0xFEED)
    - Supports uncompressed (raw grayscale/BGR) and compressed (JPEG, PNG) codecs
    - Per-frame timestamps (seconds + milliseconds + optional microseconds)
    - Compressed formats use variable-length frames requiring a seek index

Reference implementation by Ann Kennedy:
    https://gist.github.com/talmo/6d577dccb01a6eb739a6d61c973f41cd
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import attrs
import numpy as np
import simplejson as json

from sleap_io.io.video_reading import VideoBackend

# Image format codec mapping
_IMAGE_FORMAT_CODES = {
    100: "monoraw",  # Grayscale uncompressed
    200: "raw",  # Color BGR uncompressed
    101: "brgb8",  # Bayer pattern raw
    102: "monojpg",  # Grayscale JPEG compressed
    201: "jpg",  # Color JPEG compressed
    103: "jbrgb",  # Bayer JPEG compressed
    1: "monopng",  # Grayscale PNG compressed
    2: "png",  # Color PNG compressed
}

_COMPRESSED_CODECS = {"monojpg", "jpg", "jbrgb", "monopng", "png"}
_BAYER_CODECS = {"brgb8", "jbrgb"}

_HEADER_SIZE = 1024
_MAGIC = 0xFEED


@dataclass
class SeqHeader:
    """Parsed header of a Norpix .seq file.

    Attributes:
        magic: Magic number (must be 0xFEED).
        name: Sequence name string.
        version: Format version number.
        header_size: Size of the header in bytes (always 1024).
        description: User-provided description string.
        width: Frame width in pixels.
        height: Frame height in pixels.
        bit_depth: Total bit depth (e.g., 8 for mono, 24 for color).
        bit_depth_real: Bits per channel (e.g., 8).
        image_size_bytes: Size of a single uncompressed frame in bytes.
        image_format: Numeric codec identifier.
        num_frames: Number of frames declared in the header.
        true_image_size: Stride between frames for uncompressed formats.
        fps: Frame rate from the header.
        codec: Codec string identifier (e.g., "imageFormat100").
    """

    magic: int = _MAGIC
    name: str = "Norpix seq"
    version: int = 0
    header_size: int = _HEADER_SIZE
    description: str = ""
    width: int = 0
    height: int = 0
    bit_depth: int = 8
    bit_depth_real: int = 8
    image_size_bytes: int = 0
    image_format: int = 100
    num_frames: int = 0
    true_image_size: int = 0
    fps: float = 30.0
    codec: str = ""

    @property
    def codec_name(self) -> str:
        """Human-readable codec name."""
        return _IMAGE_FORMAT_CODES.get(
            self.image_format, f"unknown({self.image_format})"
        )

    @property
    def is_compressed(self) -> bool:
        """Whether frames use variable-length compression."""
        return self.codec_name in _COMPRESSED_CODECS

    @property
    def num_channels(self) -> int:
        """Number of color channels."""
        return self.bit_depth // (self.bit_depth_real or 8)

    @classmethod
    def from_file(cls, f) -> SeqHeader:
        """Read and parse the 1024-byte header from an open file handle.

        Args:
            f: Open binary file handle positioned at the start of the file.

        Returns:
            Parsed SeqHeader instance.

        Raises:
            ValueError: If the file is too small or has an invalid magic number.
        """
        f.seek(0)
        raw = f.read(_HEADER_SIZE)

        if len(raw) < _HEADER_SIZE:
            raise ValueError("File too small to contain a valid .seq header")

        # Magic number (bytes 0-3)
        magic = struct.unpack_from("<I", raw, 0)[0]
        if magic != _MAGIC:
            raise ValueError(
                f"Invalid .seq magic: 0x{magic:08X} (expected 0x{_MAGIC:08X})"
            )

        # Name string (bytes 4-23, 10 uint16 chars)
        name_chars = struct.unpack_from("<10H", raw, 4)
        name = "".join(chr(c) for c in name_chars if 0 < c < 128).strip()

        # Version and header size (bytes 28-35)
        version, header_size = struct.unpack_from("<iI", raw, 28)

        # Description (bytes 36-547, 256 uint16 chars)
        desc_chars = struct.unpack_from("<256H", raw, 36)
        description = "".join(chr(c) for c in desc_chars if 0 < c < 128).strip()

        # 9 uint32 fields (bytes 548-583)
        fields = struct.unpack_from("<9I", raw, 548)
        width = fields[0]
        height = fields[1]
        bit_depth = fields[2]
        bit_depth_real = fields[3]
        image_size_bytes = fields[4]
        image_format = fields[5]
        num_frames = fields[6]
        true_image_size = fields[8]

        # Frame rate (bytes 584-591)
        fps = struct.unpack_from("<d", raw, 584)[0]

        codec = f"imageFormat{image_format:03d}"

        return cls(
            magic=magic,
            name=name,
            version=version,
            header_size=header_size,
            description=description,
            width=width,
            height=height,
            bit_depth=bit_depth,
            bit_depth_real=bit_depth_real,
            image_size_bytes=image_size_bytes,
            image_format=image_format,
            num_frames=num_frames,
            true_image_size=true_image_size,
            fps=fps,
            codec=codec,
        )


@dataclass
class SeqIndex:
    """Frame seek index for a .seq file.

    For uncompressed formats, frame offsets are computed analytically from the
    header. For compressed formats, the file must be scanned to build the index,
    which is then cached as a JSON file alongside the .seq file.

    Attributes:
        offsets: Byte offset for each frame in the file.
        num_frames: Number of indexed frames.
        timestamp_size: Size of per-frame timestamp in bytes (6 or 8).
    """

    offsets: list[int] = field(default_factory=list)
    num_frames: int = 0
    timestamp_size: int = 8

    def frame_offset(self, frame: int) -> int:
        """Get the byte offset for a given frame number.

        Args:
            frame: Zero-based frame index.

        Returns:
            Byte offset of the frame in the file.

        Raises:
            IndexError: If frame index is out of range.
        """
        if frame < 0 or frame >= self.num_frames:
            raise IndexError(f"Frame {frame} out of range [0, {self.num_frames})")
        return self.offsets[frame]

    def save(self, path: str | Path) -> None:
        """Save the seek index to a JSON file.

        Args:
            path: Path to write the JSON index file.
        """
        data = {
            "num_frames": self.num_frames,
            "timestamp_size": self.timestamp_size,
            "offsets": self.offsets,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> SeqIndex:
        """Load a seek index from a JSON file.

        Args:
            path: Path to the JSON index file.

        Returns:
            Loaded SeqIndex instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(
            offsets=data["offsets"],
            num_frames=data["num_frames"],
            timestamp_size=data.get("timestamp_size", 8),
        )

    @classmethod
    def build_uncompressed(cls, header: SeqHeader) -> SeqIndex:
        """Build index for uncompressed formats (constant frame stride).

        Args:
            header: Parsed SeqHeader.

        Returns:
            SeqIndex with analytically computed offsets.
        """
        offsets = [
            _HEADER_SIZE + i * header.true_image_size for i in range(header.num_frames)
        ]
        return cls(
            offsets=offsets,
            num_frames=header.num_frames,
            timestamp_size=8 if header.version >= 5 else 6,
        )

    @classmethod
    def build_compressed(cls, f, header: SeqHeader) -> SeqIndex:
        """Build index for compressed formats by scanning the file.

        Compressed frames have variable sizes, so the file must be scanned
        sequentially to locate each frame boundary.

        Args:
            f: Open binary file handle.
            header: Parsed SeqHeader.

        Returns:
            SeqIndex with scanned offsets.
        """
        file_size = f.seek(0, 2)
        n_max = header.num_frames if header.num_frames > 0 else 10_000_000
        ts_size = 8 if header.version >= 5 else 6
        extra = None

        JPEG_SOI = b"\xff\xd8"
        PNG_SIG = b"\x89\x50"

        offsets = [_HEADER_SIZE]

        for i in range(1, n_max):
            prev = offsets[i - 1]
            f.seek(prev)

            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break

            frame_size = struct.unpack("<I", size_bytes)[0]
            if frame_size == 0 or frame_size > file_size:
                break

            if extra is not None:
                next_offset = prev + frame_size + extra
            else:
                search_start = prev + frame_size + ts_size
                found = False

                for pad in range(0, 32, 2):
                    candidate = search_start + pad
                    if candidate + 6 > file_size:
                        break

                    f.seek(candidate)
                    probe = f.read(6)
                    if len(probe) < 6:
                        break

                    cand_size = struct.unpack("<I", probe[:4])[0]
                    cand_magic = probe[4:6]

                    if 0 < cand_size < file_size and cand_magic in (
                        JPEG_SOI,
                        PNG_SIG,
                    ):
                        extra = ts_size + pad
                        next_offset = candidate
                        found = True
                        break

                if not found:
                    break

            if next_offset >= file_size:
                break

            f.seek(next_offset)
            check = f.read(6)
            if len(check) < 6:
                break

            check_size = struct.unpack("<I", check[:4])[0]
            if check_size == 0 or check_size > file_size:
                break

            offsets.append(next_offset)

        return cls(
            offsets=offsets,
            num_frames=len(offsets),
            timestamp_size=ts_size,
        )


@attrs.define
class SeqVideo(VideoBackend):
    """Video backend for reading Norpix .seq files.

    This backend supports reading .seq files produced by StreamPix / Norpix,
    commonly used for high-speed video recording in behavioral neuroscience.

    Supported codecs:
        - monoraw (100): Grayscale uncompressed
        - raw (200): Color BGR uncompressed (converted to RGB)
        - monojpg (102): Grayscale JPEG compressed
        - jpg (201): Color JPEG compressed
        - monopng (1): Grayscale PNG compressed
        - png (2): Color PNG compressed

    Attributes:
        filename: Path to the .seq file.
        grayscale: Whether to force grayscale. If None, autodetect on first frame.
        keep_open: Whether to keep the file handle open between reads.
    """

    EXTS = ("seq",)

    _header: SeqHeader | None = attrs.field(
        default=None, alias="_header", repr=False, eq=False
    )
    _index: SeqIndex | None = attrs.field(
        default=None, alias="_index", repr=False, eq=False
    )
    _file_handle: object | None = attrs.field(
        default=None, alias="_file_handle", repr=False, eq=False
    )

    def __attrs_post_init__(self):
        """Parse header, build seek index, and compute FPS."""
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        f = open(path, "rb")
        try:
            self._header = SeqHeader.from_file(f)

            if self._header.codec_name in _BAYER_CODECS:
                raise NotImplementedError(
                    f"Bayer codec '{self._header.codec_name}' is not supported. "
                    f"Convert the .seq file to a standard format first."
                )

            self._index = self._load_or_build_index(f)
            self._recompute_fps(f)
        except Exception:
            f.close()
            raise

        if self.keep_open:
            self._file_handle = f
        else:
            f.close()

    def _get_file_handle(self):
        """Get an open file handle, opening one if necessary.

        Returns:
            Open binary file handle.
        """
        if self._file_handle is not None and not self._file_handle.closed:
            return self._file_handle
        f = open(self.filename, "rb")
        if self.keep_open:
            self._file_handle = f
        return f

    def _maybe_close(self, f):
        """Close the file handle if keep_open is False.

        Args:
            f: File handle to potentially close.
        """
        if not self.keep_open and f is not self._file_handle:
            f.close()

    def _load_or_build_index(self, f) -> SeqIndex:
        """Load cached index or build one by scanning the file.

        Args:
            f: Open binary file handle.

        Returns:
            SeqIndex for frame access.
        """
        if not self._header.is_compressed:
            return SeqIndex.build_uncompressed(self._header)

        cache_path = Path(self.filename).with_suffix(".seq-index.json")

        if cache_path.exists():
            try:
                idx = SeqIndex.load(cache_path)
                if idx.num_frames > 0:
                    return idx
            except (json.JSONDecodeError, KeyError):
                pass

        idx = SeqIndex.build_compressed(f, self._header)

        try:
            idx.save(cache_path)
        except OSError:
            pass

        return idx

    def _recompute_fps(self, f) -> None:
        """Recompute FPS from actual frame timestamps.

        Uses the first 100 frames to compute a robust median-filtered FPS
        estimate from inter-frame intervals.

        Args:
            f: Open binary file handle.
        """
        try:
            n = min(100, self._index.num_frames)
            if n < 2:
                self._fps = self._header.fps if self._header.fps >= 1.0 else None
                return

            ts = np.array([self._read_timestamp(f, i) for i in range(n)])
            ds = np.diff(ts)
            median_ds = np.median(ds)
            # 5ms tolerance — tuned for high-speed video (>100 fps); for slow
            # framerates where jitter exceeds this, falls back to header FPS.
            ds = ds[np.abs(ds - median_ds) < 0.005]

            if len(ds) > 0:
                computed = 1.0 / np.mean(ds)
                if np.isfinite(computed) and computed >= 1.0:
                    self._fps = float(computed)
                    return
        except Exception:
            pass

        self._fps = self._header.fps if self._header.fps >= 1.0 else None

    def _read_raw_frame(self, f, frame_idx: int) -> tuple[bytes, int]:
        """Read raw frame bytes and return data with position after data.

        Args:
            f: Open binary file handle.
            frame_idx: Zero-based frame index.

        Returns:
            Tuple of (raw_data_bytes, file_position_after_data).
        """
        offset = self._index.frame_offset(frame_idx)
        f.seek(offset)

        if self._header.is_compressed:
            nbytes = struct.unpack("<I", f.read(4))[0]
            data = f.read(nbytes - 4)
            return data, f.tell()
        else:
            data = f.read(self._header.image_size_bytes)
            return data, f.tell()

    def _read_timestamp(self, f, frame_idx: int) -> float:
        """Read the timestamp for a single frame.

        Args:
            f: Open binary file handle.
            frame_idx: Zero-based frame index.

        Returns:
            Timestamp as seconds since epoch (float64).
        """
        _, pos_after_data = self._read_raw_frame(f, frame_idx)
        f.seek(pos_after_data)

        ts_sec = struct.unpack("<I", f.read(4))[0]
        ts_ms = struct.unpack("<H", f.read(2))[0]
        result = ts_sec + ts_ms / 1000.0

        if self._index.timestamp_size == 8:
            ts_us = struct.unpack("<H", f.read(2))[0]
            result += ts_us / 1_000_000.0

        return result

    def _decode_frame(self, data: bytes) -> np.ndarray:
        """Decode raw frame bytes into a numpy image array.

        Args:
            data: Raw frame bytes.

        Returns:
            Decoded frame as numpy array of shape (height, width, channels).

        Raises:
            ValueError: If the codec is unsupported.
        """
        codec = self._header.codec_name
        h, w = self._header.height, self._header.width
        nch = self._header.num_channels

        if codec in ("monoraw", "raw"):
            arr = np.frombuffer(data, dtype=np.uint8)
            if nch == 1:
                return arr[: h * w].reshape(h, w, 1)
            else:
                arr = arr[: h * w * nch].reshape(h, w, nch)
                # BGR -> RGB
                return arr[:, :, ::-1].copy()

        elif codec in ("monojpg", "jpg", "monopng", "png"):
            try:
                from PIL import Image
            except ImportError:
                raise ImportError(
                    f"Pillow is required to decode {codec} frames in .seq files. "
                    f"Install with: pip install Pillow"
                )
            img = Image.open(io.BytesIO(data))
            arr = np.array(img)
            if arr.ndim == 2:
                return arr[:, :, np.newaxis]
            return arr

        else:
            raise ValueError(f"Unsupported .seq codec: {codec}")

    # --- VideoBackend interface ---

    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        return self._index.num_frames

    @property
    def img_shape(self) -> tuple[int, int, int]:
        """Shape of a single frame as (height, width, channels)."""
        h = self._header.height
        w = self._header.width
        nch = self._header.num_channels
        if self.grayscale is True or nch == 1:
            return (h, w, 1)
        return (h, w, 3)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame from the .seq file.

        Args:
            frame_idx: Zero-based frame index. Negative indices are supported.

        Returns:
            Frame as numpy array of shape (height, width, channels).
        """
        if frame_idx < 0:
            frame_idx = self._index.num_frames + frame_idx

        f = self._get_file_handle()
        try:
            data, _ = self._read_raw_frame(f, frame_idx)
            return self._decode_frame(data)
        finally:
            self._maybe_close(f)

    # --- Seq-specific public methods ---

    def get_timestamp(self, frame_idx: int) -> float:
        """Get the timestamp for a single frame.

        Args:
            frame_idx: Zero-based frame index. Negative indices are supported.

        Returns:
            Timestamp as seconds since epoch (float64).

        Raises:
            IndexError: If frame index is out of range.
        """
        if frame_idx < 0:
            frame_idx = self.num_frames + frame_idx
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame {frame_idx} out of range [0, {self.num_frames})")

        f = self._get_file_handle()
        try:
            return self._read_timestamp(f, frame_idx)
        finally:
            self._maybe_close(f)

    def get_timestamps(self) -> np.ndarray:
        """Get all frame timestamps as an array.

        Returns:
            Array of timestamps as float64 (seconds since epoch).
        """
        f = self._get_file_handle()
        try:
            return np.array(
                [self._read_timestamp(f, i) for i in range(self.num_frames)]
            )
        finally:
            self._maybe_close(f)

    @property
    def header(self) -> SeqHeader:
        """The parsed .seq file header."""
        return self._header

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        """Clean up file handle on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
