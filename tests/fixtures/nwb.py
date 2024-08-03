"""Fixtures that return paths to .nwb files."""
import pytest

@pytest.fixture
def minimal_instance_pkg():
    """NWB file with a single instance. The video frame is not saved as an image."""
    return "tests/data/nwb/minimal_instance.pkg.nwb"

@pytest.fixture
def centered_pair_no_training():
    """NWB file converted from .slp in the GUI without NWB training data."""
    return "tests/data/nwb/centered_pair_predictions_no_training.nwb"
