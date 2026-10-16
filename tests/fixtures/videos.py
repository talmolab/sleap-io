"""Fixtures for video and media files."""

import numpy as np
import pytest

import sleap_io

# Frames of the synthetic AV1 video: a 24 px bright square on a dark background,
# moving right by this many pixels per frame.
AV1_VIDEO_FRAMES = 16
AV1_SQUARE_STEP = 6
AV1_SQUARE_SIZE = 24


@pytest.fixture
def centered_pair_low_quality_path():
    """Path to a video with two flies in the center."""
    return "tests/data/videos/centered_pair_low_quality.mp4"


@pytest.fixture
def centered_pair_low_quality_video(centered_pair_low_quality_path):
    """Video with two flies in the center."""
    return sleap_io.Video.from_filename(centered_pair_low_quality_path)


@pytest.fixture
def centered_pair_frame_paths():
    """Paths to three frames of a video with two flies in the center."""
    return [
        "tests/data/videos/imgs/img.00.jpg",
        "tests/data/videos/imgs/img.01.jpg",
        "tests/data/videos/imgs/img.02.jpg",
    ]


@pytest.fixture
def single_page_tiff_path():
    """Path to a single-page TIFF file.

    Returns path to a 128x128 grayscale TIFF containing a single frame
    with a white square on black background.
    """
    return "tests/data/tiff/single_page.tif"


@pytest.fixture
def multipage_tiff_path():
    """Path to a multi-page TIFF file.

    Returns path to an 8-page TIFF where each page is 128x128 grayscale.
    Each frame contains a white square that moves diagonally across frames,
    and a frame indicator in the top-left corner.
    """
    return "tests/data/tiff/multipage.tif"


@pytest.fixture
def stacked_tiff_path():
    """Path to a stacked TIFF file with frames as channels.

    Returns path to a TIFF with shape (128, 128, 8) where the 8 frames
    are stored as channels in a single image. Each channel contains the
    same moving square pattern as the multi-page variant.
    """
    return "tests/data/tiff/stacked_as_channels.tif"


@pytest.fixture
def tiff_image_sequence_path():
    """Path to a directory containing a sequence of TIFF files.

    Returns path to a directory with 8 individual TIFF files named
    frame_000.tif through frame_007.tif. Each file is 128x128 grayscale
    with the same moving square pattern as other TIFF fixtures.
    """
    return "tests/data/tiff/image_sequence"


@pytest.fixture
def small_robot_path():
    """Path to a small 3-frame robot video for testing RGB/BGR consistency."""
    return "tests/data/videos/small_robot_3_frame.mp4"


@pytest.fixture
def small_robot_video(small_robot_path):
    """Small 3-frame robot video for testing RGB/BGR consistency."""
    return sleap_io.Video.from_filename(small_robot_path)


@pytest.fixture
def av1_video_path(tmp_path):
    """Path to a small synthetic AV1 video (128x128, 16 frames, keyframe every 8).

    Frame `i` shows a bright 24 px square whose left edge is at x = 4 + 6 * i. The
    video is encoded at test time with pyav's AV1 encoder (SVT-AV1 or libaom), so the
    test is skipped when pyav or both encoders are missing.
    """
    av = pytest.importorskip("av")
    encoder = None
    for name in ("libsvtav1", "libaom-av1"):
        try:
            av.codec.Codec(name, "w")
        except Exception:
            continue
        encoder = name
        break
    if encoder is None:
        pytest.skip("pyav has no AV1 encoder")

    path = tmp_path / "av1.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(encoder, rate=10)
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 8
        for i in range(AV1_VIDEO_FRAMES):
            img = np.full((128, 128), 16, dtype=np.uint8)
            x0 = 4 + AV1_SQUARE_STEP * i
            img[52 : 52 + AV1_SQUARE_SIZE, x0 : x0 + AV1_SQUARE_SIZE] = 235
            frame = av.VideoFrame.from_ndarray(img, format="gray")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.as_posix()
