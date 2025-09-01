"""Fixtures for LEAP .mat files."""

import pytest


@pytest.fixture
def leap_labels_mat():
    """Path to LEAP labels.mat file."""
    return "tests/data/leap/labels.mat"
