import pytest


@pytest.fixture
def slp_file1():
    """Typical SLP file including  `PredictedInstance`, `Instance`, `Track` and `Skeleton` objects.

    Returns: filepath to SLP file
    """
    return "typical.slp"


@pytest.fixture
def slp_file2():
    """SLP file missing the py/reduce in the skeleton dict.

    Returns: filepath to SLP file
    """
    return "reduce.slp"
