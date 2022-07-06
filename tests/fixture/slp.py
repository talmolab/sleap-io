import pytest


@pytest.fixture
def slp_file1():
    # Typical SLP file including Predicted and normal `Instance` objects, tracks, and
    # skeletons etc.
    return "typical.slp"


@pytest.fixture
def slp_file2():
    # SLP file missing the py/reduce in the skeleton dict
    return "reduce.slp"
