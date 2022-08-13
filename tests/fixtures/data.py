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
