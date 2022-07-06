import pytest


@pytest.fixture
def slp_file1():
    """Typical SLP file including  `PredictedInstance`, `Instance`, `Track` and `Skeleton` objects.

    Returns: filepath to SLP file
    """
    return "tests/data/slp/typical.slp"


@pytest.fixture
def slp_file2():
    """SLP file missing the py/reduce in the skeleton dict.

    Returns: filepath to SLP file
    """
    return "tests/data/slp/reduce.slp"
