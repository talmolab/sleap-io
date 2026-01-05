"""In-memory serialization codecs for SLEAP Labels objects.

This package provides flexible conversion between Labels objects and various
in-memory representations:

- **DataFrames**: Multiple formats (multi_index, points, instances, frames) with
  pandas/polars support
- **Dictionaries**: JSON-serializable primitive dictionaries
- **NumPy**: Via Labels.numpy() and Labels.from_numpy() with enhanced flexibility

The codecs package is designed for in-memory serialization, separate from disk I/O
operations in the `sleap_io.io` package. This separation allows for:

1. **Reusability**: Common serialization code shared across I/O backends
2. **Flexibility**: Work with Labels in different formats without touching disk
3. **Composability**: Chain codecs (e.g., Labels → DataFrame → CSV)

Examples:
    Convert to DataFrame for analysis:

    >>> from sleap_io import load_file
    >>> from sleap_io.codecs import to_dataframe
    >>> labels = load_file("predictions.slp")
    >>> df = to_dataframe(labels, format="instances")
    >>> df.groupby("track")["nose.x"].mean()

    Round-trip through dict:

    >>> from sleap_io.codecs import to_dict
    >>> d = to_dict(labels)
    >>> import json
    >>> json.dumps(d)  # Fully JSON-serializable!

    Use with I/O backends:

    >>> df = to_dataframe(labels, format="points")
    >>> df.to_csv("predictions.csv")
"""

from sleap_io.codecs.dataframe import (
    DataFrameFormat,
    from_dataframe,
    to_dataframe,
    to_dataframe_iter,
)
from sleap_io.codecs.dictionary import from_dict, to_dict
from sleap_io.codecs.numpy import from_numpy, to_numpy

__all__ = [
    # DataFrame codec
    "DataFrameFormat",
    "to_dataframe",
    "to_dataframe_iter",
    "from_dataframe",
    # Dictionary codec
    "to_dict",
    "from_dict",
    # NumPy codec
    "to_numpy",
    "from_numpy",
]
