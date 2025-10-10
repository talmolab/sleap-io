"""Fixtures for video and media files."""

import pytest

import sleap_io


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
