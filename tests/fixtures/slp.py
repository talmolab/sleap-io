"""Fixtures that return paths to .slp files."""
import pytest


@pytest.fixture
def slp_typical():
    """Typical SLP file including  `PredictedInstance`, `Instance`, `Track` and `Skeleton` objects."""
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
def slp_predictions():
    """A more complex example containing predicted instances from multiple tracks and a single video"""
    return "tests/data/slp/centered_pair_predictions.slp"


@pytest.fixture
def slp_predictions_with_provenance():
    """The slp file generated with the collab tutorial and sleap version 1.27"""
    return "tests/data/slp/predictions_1.2.7_provenance_and_tracking.slp"
