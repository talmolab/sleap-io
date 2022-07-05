import pytest


@pytest.fixture
def slp_file1():
    # Typical slp file including Predicted and normal `Instance` objects, tracks, and
    # skeletons etc.
    return "typical.slp"
