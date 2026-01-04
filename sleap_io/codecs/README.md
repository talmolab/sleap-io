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
# Columns: frame_idx, video_path, track_name, node_name, x, y, score
```

**Best for**: Filtering specific nodes, computing statistics per node, plotting trajectories

### Instances Format (`format="instances"`)

One row per instance. Denormalized with columns for each node's coordinates.

```python
df = to_dataframe(labels, format="instances")
# Columns: frame_idx, track_name, nose_x, nose_y, tail_x, tail_y, ...
```

**Best for**: Machine learning features, instance-level analysis, exporting for other tools

### Frames Format (`format="frames"`)

One row per frame-track combination. Similar to instances but organized by tracks.

```python
df = to_dataframe(labels, format="frames")
# Indexed by (frame_idx, track_idx)
```

**Best for**: Trajectory analysis, tracking metrics, time-series analysis

### Multi-Index Format (`format="multi_index"`)

Hierarchical column structure. Similar to NWB format.

```python
df = to_dataframe(labels, format="multi_index")
# Columns: MultiIndex with levels (video_path, skeleton_name, track_name, node_name, coord)
```

**Best for**: Compatibility with NWB workflows, hierarchical data analysis

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

### Dictionary Structure

```python
{
    "version": "1.0.0",
    "skeletons": [
        {
            "name": "skeleton1",
            "nodes": ["node1", "node2", ...],
            "edges": [[0, 1], ...],  # Node index pairs
            "symmetries": [[0, 1], ...]  # Optional
        }
    ],
    "videos": [...],
    "tracks": [...],
    "labeled_frames": [
        {
            "frame_idx": 0,
            "video_idx": 0,
            "instances": [
                {
                    "type": "predicted_instance",
                    "skeleton_idx": 0,
                    "track_idx": 0,
                    "points": [
                        {"x": 1.0, "y": 2.0, "visible": true, "complete": true},
                        ...
                    ],
                    "score": 0.95
                }
            ]
        }
    ],
    "suggestions": [...],
    "provenance": {...}
}
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

# Or use flexible defaults
labels = from_numpy(
    arr,
    video=video,
    skeleton=skeleton,
    track_names=["mouse1", "mouse2"]
)
```

## Integration with I/O

Codecs can be used with I/O operations:

```python
from sleap_io import load_file
from sleap_io.codecs import to_dataframe

# Load, convert, and save to CSV
labels = load_file("predictions.slp")
df = to_dataframe(labels, format="instances")
df.to_csv("predictions.csv", index=False)

# Or use pandas' various export formats
df.to_excel("predictions.xlsx")
df.to_parquet("predictions.parquet")
df.to_feather("predictions.feather")
```

## Examples

### Export tracking data for analysis

```python
from sleap_io import load_file
from sleap_io.codecs import to_dataframe

labels = load_file("tracking_results.slp")

# Get trajectory data
df = to_dataframe(labels, format="frames")

# Compute velocity per track
df["velocity"] = df.groupby("track_name")[["nose_x", "nose_y"]].diff().abs().sum(axis=1)

# Export for further analysis
df.to_csv("trajectories_with_velocity.csv")
```

### Convert to dictionary for web API

```python
from sleap_io import load_file
from sleap_io.codecs import to_dict
import json

labels = load_file("predictions.slp")

# Convert to dict and serialize
data = to_dict(labels, skip_empty_frames=True)

# Send as JSON response
json_response = json.dumps(data)
```

### Prepare data for machine learning

```python
from sleap_io import load_file
from sleap_io.codecs import to_dataframe

labels = load_file("training_data.slp")

# Get instances format with all node coordinates
df = to_dataframe(labels, format="instances", include_score=False)

# Extract feature matrix
feature_cols = [col for col in df.columns if col.endswith(("_x", "_y"))]
X = df[feature_cols].values

# Extract labels (e.g., track IDs)
y = df["track_name"].values
```

## API Reference

See the module docstrings for complete API documentation:

- `sleap_io.codecs.dataframe.to_dataframe()` - Convert to DataFrame
- `sleap_io.codecs.dictionary.to_dict()` - Convert to dictionary
- `sleap_io.codecs.numpy.to_numpy()` - Convert to numpy array
- `sleap_io.codecs.numpy.from_numpy()` - Create Labels from numpy array

## Notes

- All codecs prioritize encoding (Labels → format) over decoding
- Decoding support will be added in future releases
- DataFrame formats may be lossy (e.g., some metadata not preserved in certain formats)
- Dictionary format is lossless and preserves all information
- NumPy format is optimized for tracking data and assumes single skeleton per project
