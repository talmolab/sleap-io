"""Fixtures that return paths to .slp files."""

import pytest


@pytest.fixture
def slp_typical():
    """Typical SLP file including."""
    return "tests/data/slp/typical.slp"


@pytest.fixture
def slp_simple_skel():
    """SLP file missing the py/reduce in the skeleton dict."""
    return "tests/data/slp/reduce.slp"


@pytest.fixture
def slp_minimal():
    """SLP project with minimal real data."""
    return "tests/data/slp/minimal_instance.slp"


@pytest.fixture
def slp_minimal_pkg():
    """SLP project with minimal real data and embedded images."""
    return "tests/data/slp/minimal_instance.pkg.slp"


@pytest.fixture
def centered_pair():
    """Example with predicted instances from multiple tracks and a single video.

    This project:
    - Has 1 grayscale video with 1100 frames, cropped to 384x384 with 2 flies
    - Has a 24 node skeleton with edges and symmetries
    - Has 0 user instances and 2274 predicted instances
    - Has 2 correct tracks and 25 extraneous tracks
    """
    return "tests/data/slp/centered_pair_predictions.slp"


@pytest.fixture
def slp_predictions_with_provenance():
    """The slp file generated with the colab tutorial and sleap version 1.2.7."""
    return "tests/data/slp/predictions_1.2.7_provenance_and_tracking.slp"


@pytest.fixture
def slp_real_data():
    """A real data example containing predicted and user instances.

    This project:
    - Was generated using SLEAP v1.3.1 on an M1 Mac
    - Contains 1 video (centered_pair_low_quality.mp4)
    - Has a 2 node skeleton of "head" and "abdomen"
    - Has 5 labeled frames
    - Has 10 suggested frames, of which 7 have predictions
    - Has 2 frames with user instances created from predictions:
        - frame_idx 220 has 1 user instance from prediction and 1 predicted instance
        - frame_idx 770 has 2 user instances from predictions
    - Does not have tracks

    Note: There are two versions of these labels, one with absolute paths and one with
    relative paths. The relative paths version is used here:

        >> sio.load_slp("tests/data/slp/labels.v002.rel_paths.slp").video.filename
        "tests/data/videos/centered_pair_low_quality.mp4"
        >> sio.load_slp("tests/data/slp/labels.v002.slp").video.filename
        "/Users/talmo/sleap-io/tests/data/videos/centered_pair_low_quality.mp4"
    """
    return "tests/data/slp/labels.v002.rel_paths.slp"


@pytest.fixture
def slp_imgvideo():
    """SLP project with a single image video."""
    return "tests/data/slp/imgvideo.slp"
