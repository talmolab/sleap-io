"""Fixtures that return paths to `.nwb` files."""

import pytest


@pytest.fixture
def minimal_instance_nwb():
    """NWB file with a single instance."""
    return "tests/data/nwb/minimal_instance.nwb"


@pytest.fixture
def minimal_instance_pkg_nwb():
    """NWB .pkg file with a single instance."""
    return "tests/data/nwb/minimal_instance.pkg.nwb"


@pytest.fixture
def labels_v002_nwb():
    """NWB file with labels saved as a dataset."""
    return "tests/data/nwb/labels.v002.nwb"


@pytest.fixture
def typical_nwb():
    """Typical NWB file."""
    return "tests/data/nwb/typical.nwb"
