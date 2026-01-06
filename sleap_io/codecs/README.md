# SLEAP-IO Codecs

The `sleap_io.codecs` package provides flexible in-memory serialization for SLEAP `Labels` objects. This package enables conversion between Labels and various representations including DataFrames, dictionaries, and NumPy arrays.

## Features

- **DataFrame Codec**: Convert Labels to/from pandas or polars DataFrames with multiple layout formats
- **Dictionary Codec**: Convert Labels to JSON-serializable dictionaries
- **NumPy Codec**: Enhanced numpy array conversion with flexible options

## Quick Start

```python
from sleap_io import load_file
from sleap_io.codecs import to_dataframe, to_dict, to_numpy

# Load labels
labels = load_file("predictions.slp")

# Convert to DataFrame
df = to_dataframe(labels, format="points")
print(df.head())

# Convert to dictionary (JSON-serializable)
d = to_dict(labels)

# Convert to numpy array
arr = to_numpy(labels, return_confidence=True)
```

## DataFrame Formats

The DataFrame codec supports four different formats optimized for different use cases:

### Points Format (`format="points"`)

One row per point. Most normalized, best for filtering and analysis.

```python
df = to_dataframe(labels, format="points")
# Columns: frame_idx, track, track_score, instance_score, node, x, y, score
```

**Best for**: Filtering specific nodes, computing statistics per node, plotting trajectories

### Instances Format (`format="instances"`)

One row per instance. Denormalized with columns for each node's coordinates.

```python
df = to_dataframe(labels, format="instances")
# Columns: frame_idx, track, track_score, score, nose.x, nose.y, nose.score, tail.x, tail.y, ...
```

**Best for**: Machine learning features, instance-level analysis, exporting for other tools

### Frames Format (`format="frames"`)

One row per frame with all instances multiplexed across columns (wide format).

```python
df = to_dataframe(labels, format="frames")
# Columns: frame_idx, inst0.track, inst0.track_score, inst0.score, inst0.nose.x, inst0.nose.y, ...

# With track names as column prefixes:
df = to_dataframe(labels, format="frames", instance_id="track")
# Columns: frame_idx, mouse1.track_score, mouse1.score, mouse1.nose.x, mouse2.nose.x, ...
```

**Best for**: Trajectory analysis, tracking metrics, time-series analysis

### Multi-Index Format (`format="multi_index"`)

Hierarchical column structure with frame as index.

```python
df = to_dataframe(labels, format="multi_index")
# Columns: MultiIndex with levels (instance, node, coord)
# e.g., (inst0, nose, x), (inst0, nose, y), (inst0, tail, x), ...

# With track names:
df = to_dataframe(labels, format="multi_index", instance_id="track")
# e.g., (mouse1, nose, x), (mouse2, nose, x), ...
```

**Best for**: Compatibility with NWB workflows, hierarchical data analysis

## New Parameters

### instance_id (frames and multi_index formats)

Controls how instance columns are named:

```python
# Use inst0, inst1, inst2, ... (default)
df = to_dataframe(labels, format="frames", instance_id="index")

# Use track names (e.g., mouse1, mouse2)
df = to_dataframe(labels, format="frames", instance_id="track")
```

### untracked (with instance_id="track")

Behavior when encountering untracked instances:

```python
# Raise error if any instance lacks a track (default)
df = to_dataframe(labels, format="frames", instance_id="track", untracked="error")

# Skip untracked instances silently
df = to_dataframe(labels, format="frames", instance_id="track", untracked="ignore")
```

## Options

### Filtering Data

```python
# Filter by video
df = to_dataframe(labels, format="points", video=0)

# Include only predicted instances
df = to_dataframe(
    labels,
    format="points",
    include_user_instances=False,
    include_predicted_instances=True
)

# Exclude confidence scores
df = to_dataframe(labels, format="points", include_score=False)

# Exclude metadata columns
df = to_dataframe(labels, format="points", include_metadata=False)
```

### Backend Selection

```python
# Use pandas (default)
df = to_dataframe(labels, format="points", backend="pandas")

# Use polars (requires: pip install polars)
df = to_dataframe(labels, format="points", backend="polars")
```

## Decoding (from_dataframe)

All formats are now supported for decoding back to Labels:

```python
from sleap_io.codecs import from_dataframe

# Decode points format
labels = from_dataframe(df, format="points", video=video, skeleton=skeleton)

# Decode instances format
labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

# Decode frames format (wide format)
labels = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

# Decode multi_index format
labels = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)
```

## Dictionary Codec

Convert Labels to a fully JSON-serializable dictionary:

```python
from sleap_io.codecs import to_dict
import json

# Convert to dict
d = to_dict(labels)

# Save as JSON
with open("labels.json", "w") as f:
    json.dump(d, f, indent=2)

# Filter by video
d = to_dict(labels, video=0)

# Skip empty frames
d = to_dict(labels, skip_empty_frames=True)
```

## NumPy Codec

Enhanced numpy array conversion:

```python
from sleap_io.codecs import to_numpy, from_numpy

# Convert to numpy (also available as labels.numpy())
arr = to_numpy(labels, return_confidence=True)
# Shape: (n_frames, n_tracks, n_nodes, 3)  # x, y, score

# Create Labels from numpy
labels = from_numpy(
    arr,
    video=video,
    skeleton=skeleton,
    tracks=[Track("track1"), Track("track2")],
    return_confidence=True
)
```

## Column Naming Conventions

### Points Format
- `frame_idx`: Frame index (int)
- `node`: Node/keypoint name (str)
- `x`, `y`: Coordinates (float)
- `track`: Track name or None (str)
- `track_score`: Tracking confidence score (float or None)
- `instance_score`: Instance detection score (float or None)
- `score`: Per-point confidence score (float)

### Instances Format
- `frame_idx`: Frame index (int)
- `track`: Track name or None (str)
- `track_score`: Tracking confidence score (float or None)
- `score`: Instance detection score (float or None)
- `{node}.x`, `{node}.y`: Per-node coordinates (float)
- `{node}.score`: Per-node confidence score (float)

### Frames Format
- `frame_idx`: Frame index (int)
- `{inst}.track`: Track name for instance (str or None)
- `{inst}.track_score`: Tracking confidence score (float or None)
- `{inst}.score`: Instance detection score (float or None)
- `{inst}.{node}.x`, `{inst}.{node}.y`: Per-node coordinates (float)
- `{inst}.{node}.score`: Per-node confidence score (float)

Where `{inst}` is either `inst0`, `inst1`, ... or track names like `mouse1`, `mouse2` depending on `instance_id` parameter.

## Notes

- All DataFrame formats support both encoding (Labels -> DataFrame) and decoding (DataFrame -> Labels)
- Uses dot separator (`.`) for hierarchical column names: `nose.x`, `inst0.nose.x`
- Dictionary format is lossless and preserves all information
- NumPy format is optimized for tracking data and assumes single skeleton per project
- Alternative column names (`node_name`, `track_name`) are accepted when decoding for interoperability with external tools
