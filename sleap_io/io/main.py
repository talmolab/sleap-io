"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations
from sleap_io.io import slp
from sleap_io import Labels
from pathlib import Path


def load_labels(path: str) -> Labels:
    """Load a dataset of labeled frames.

    Args:
        path: Path to a labels file or folder.

    Returns:
        The imported `Labels` that was read in.
    """
    if Path(path).suffix.lower() == ".slp":
        return slp.read_labels(path)
    else:
        raise ValueError("Extension or format not recognized.")
