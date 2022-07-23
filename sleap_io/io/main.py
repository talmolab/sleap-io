"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations
from sleap_io.io import slp
from sleap_io import Labels


def load_slp(path: str) -> Labels:
    """Load a SLEAP dataset.

    Args:
        path: Path to a SLEAP labels file (.slp)..

    Returns:
        The dataset as a `Labels` object.
    """
    return slp.read_labels(path)
