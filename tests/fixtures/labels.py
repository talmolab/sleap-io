"""Fixtures that return `Labels` objects."""

import pytest
import sleap_io


@pytest.fixture
def labels_predictions(centered_pair):
    """Labels object containing predicted instances, multiple tracks and a single video."""
    return sleap_io.load_slp(centered_pair)
