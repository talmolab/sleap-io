# Codecs

Convert pose tracking data to dictionaries, NumPy arrays, and DataFrames for analysis and interoperability.

## What are codecs?

Codecs perform **in-memory format conversion** between `Labels` objects and common data representations:

| Codec | Output | Use Case |
|-------|--------|----------|
| Dictionary | `dict` | JSON export, serialization, inspection |
| NumPy | `np.ndarray` | Numerical analysis, ML pipelines |
| DataFrame | `pd.DataFrame` / `pl.DataFrame` | Tabular analysis, visualization |

Codecs are distinct from I/O operations: they convert data structures in memory rather than reading/writing files. This separation enables flexible workflows:

```python
# Codec + I/O composition
df = labels.to_dataframe()     # Codec: Labels → DataFrame
df.to_csv("poses.csv")         # I/O: DataFrame → disk
```

## Quick start

```python
import sleap_io as sio

# Load pose tracking data
labels = sio.load_file("predictions.slp")

# Convert to dictionary (JSON-serializable)
data = labels.to_dict()

# Convert to NumPy array
# Shape: (n_frames, n_tracks, n_nodes, 2)
tracks = labels.numpy()

# Convert to pandas DataFrame
df = labels.to_dataframe()
```

Each codec also supports decoding (reconstruction from the converted format):

```python
from sleap_io.codecs import from_dict, from_dataframe, from_numpy

# Reconstruct Labels from dictionary
labels = from_dict(data)

# Reconstruct from DataFrame (requires video/skeleton context)
labels = from_dataframe(df, video=video, skeleton=skeleton)

# Reconstruct from NumPy array
labels = from_numpy(tracks, video=video, skeleton=skeleton)
```

---

## Dictionary codec

The dictionary codec converts `Labels` to a JSON-serializable Python dictionary with complete structural metadata.

### Encoding with `to_dict()`

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
data = labels.to_dict()
```

**Output structure:**

```python
{
    "version": "1.0.0",
    "skeletons": [
        {
            "name": "fly",
            "nodes": ["head", "thorax", "abdomen"],
            "edges": [[0, 1], [1, 2]]
        }
    ],
    "videos": [
        {"filename": "video.mp4", "shape": [1000, 1024, 1024, 1]}
    ],
    "tracks": [
        {"name": "fly_1"}
    ],
    "labeled_frames": [
        {
            "frame_idx": 0,
            "video_idx": 0,
            "instances": [
                {
                    "type": "predicted_instance",
                    "skeleton_idx": 0,
                    "points": [
                        {"x": 100.0, "y": 200.0, "visible": true, "complete": false},
                        {"x": 150.0, "y": 250.0, "visible": true, "complete": false},
                        {"x": 200.0, "y": 300.0, "visible": true, "complete": false}
                    ],
                    "track_idx": 0,
                    "score": 0.91
                }
            ]
        }
    ],
    "suggestions": [],
    "provenance": {}
}
```

The dictionary uses index-based references (`skeleton_idx`, `video_idx`, `track_idx`) for compactness. All values are JSON-serializable primitives.

**Parameters:**

- `video`: Filter to a specific video (by object or index)
- `skip_empty_frames`: Exclude frames with no instances

```python
# Export only frames with instances from the first video
data = labels.to_dict(video=0, skip_empty_frames=True)
```

### Decoding with `from_dict()`

```python
from sleap_io.codecs import from_dict

# Reconstruct Labels from dictionary
labels = from_dict(data)
```

The decoder reconstructs the full object graph including skeletons, videos, tracks, and instances.

!!! note "Limitations"
    - Video backends are not restored (videos are created with filename only)
    - The `from_predicted` relationship is indicated but not fully linked

### JSON export workflow

```python
import json

# Export to JSON file
data = labels.to_dict()
with open("labels.json", "w") as f:
    json.dump(data, f, indent=2)

# Import from JSON file
with open("labels.json", "r") as f:
    data = json.load(f)
labels = from_dict(data)
```

See also: [`to_dict()`](#sleap_io.codecs.dictionary.to_dict), [`from_dict()`](#sleap_io.codecs.dictionary.from_dict)

---

## NumPy codec

The NumPy codec converts `Labels` to a 4D array of track coordinates, suitable for numerical analysis and ML pipelines.

### Encoding with `to_numpy()` / `Labels.numpy()`

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
tracks = labels.numpy()

print(tracks.shape)
# (n_frames, n_tracks, n_nodes, 2)
```

**Output array:**

```
Shape: (2, 2, 2, 2)
       (n_frames=2, n_tracks=2, n_nodes=2, coords=2)

Frame 0, Track 0:
[[100. 200.]    # head: x=100, y=200
 [150. 250.]]   # tail: x=150, y=250

Frame 0, Track 1:
[[300. 400.]    # head: x=300, y=400
 [350. 450.]]   # tail: x=350, y=450
```

Missing points are represented as `np.nan`.

**Include confidence scores:**

```python
tracks = labels.numpy(return_confidence=True)
print(tracks.shape)
# (n_frames, n_tracks, n_nodes, 3)  # x, y, score

# Frame 0, Track 0 with scores:
# [[100.   200.     0.95]    # head: x, y, score
#  [150.   250.     0.92]]   # tail: x, y, score
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video` | `None` (first) | Video to convert |
| `untracked` | `False` | Include untracked instances |
| `return_confidence` | `False` | Include point scores (3rd coord) |
| `user_instances` | `True` | Include user-labeled instances |
| `predicted_instances` | `True` | Include predicted instances |

```python
# Get only predicted instances with confidence scores
tracks = labels.numpy(
    user_instances=False,
    predicted_instances=True,
    return_confidence=True
)
```

### Decoding with `from_numpy()` / `Labels.from_numpy()`

```python
from sleap_io.codecs import from_numpy
import numpy as np

# Create array: 100 frames, 2 tracks, 3 nodes, xy coordinates
tracks = np.random.rand(100, 2, 3, 2) * 500

# Reconstruct Labels (requires skeleton and video)
labels = from_numpy(
    tracks,
    video=video,
    skeleton=skeleton,
    track_names=["animal_1", "animal_2"]
)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `tracks_array` | 4D array of shape `(frames, tracks, nodes, 2 or 3)` |
| `video` | Video object for the labels |
| `skeleton` | Skeleton defining node structure |
| `tracks` | List of Track objects (or auto-created) |
| `track_names` | Names for auto-created tracks |
| `first_frame` | Starting frame index (default: 0) |
| `return_confidence` | Whether array contains scores |

See also: [`to_numpy()`](#sleap_io.codecs.numpy.to_numpy), [`from_numpy()`](#sleap_io.codecs.numpy.from_numpy)

---

## DataFrame codec

The DataFrame codec converts `Labels` to tabular format with multiple layout options, supporting both pandas and polars backends.

### Encoding with `to_dataframe()` / `Labels.to_dataframe()`

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
df = labels.to_dataframe()
```

Four output formats are available:

| Format | Structure | Use Case |
|--------|-----------|----------|
| `points` | One row per point | Normalized, round-trip compatible |
| `instances` | One row per instance | Feature extraction, ML |
| `frames` | One row per frame | Trajectory analysis |
| `multi_index` | Hierarchical columns | NWB-style, pivot tables |

### Points format (default)

One row per (frame, instance, node) combination—the most normalized representation.

```python
df = labels.to_dataframe(format="points")
```

```
   frame_idx     node      x      y  track  track_score  instance_score  score
0          0     head  100.0  200.0   None          NaN             NaN    NaN
1          0   thorax  150.0  250.0   None          NaN             NaN    NaN
2          0  abdomen  200.0  300.0   None          NaN             NaN    NaN
3          0     head  105.0  205.0  fly_1         0.98            0.91   0.95
4          0   thorax  155.0  255.0  fly_1         0.98            0.91   0.92
5          0  abdomen  205.0  305.0  fly_1         0.98            0.91   0.88
```

**Columns:**

- `frame_idx`: Frame number
- `node`: Body part name
- `x`, `y`: Point coordinates
- `track`: Track name (or `None` for untracked)
- `track_score`: Tracking confidence
- `instance_score`: Instance-level prediction score
- `score`: Per-point confidence score

This format supports full round-trip decoding with [`from_dataframe()`](#decoding-with-from_dataframe).

### Instances format

One row per instance—coordinates spread across columns.

```python
df = labels.to_dataframe(format="instances")
```

```
   frame_idx  track  track_score  score  head.x  head.y  head.score  tail.x  tail.y  tail.score
0          0  fly_1         0.98   0.91   100.0   200.0        0.95   150.0   250.0        0.92
1          0  fly_2         0.95   0.86   300.0   400.0        0.88   350.0   450.0        0.85
```

Each node has columns `{node}.x`, `{node}.y`, and `{node}.score`. This format is useful for computing per-instance features.

### Frames format

One row per frame—all instances multiplexed across columns.

```python
df = labels.to_dataframe(format="frames")
```

**With `instance_id="index"` (default):**

```
   frame_idx  inst0.track  inst0.score  inst0.head.x  inst0.head.y  ...  inst1.track  inst1.head.x  ...
0          0        fly_1         0.91         100.0         200.0  ...        fly_2         300.0  ...
```

**With `instance_id="track"`:**

```python
df = labels.to_dataframe(format="frames", instance_id="track")
```

```
   frame_idx  fly_1.score  fly_1.head.x  fly_1.head.y  ...  fly_2.score  fly_2.head.x  ...
0          0         0.91         100.0         200.0  ...         0.86         300.0  ...
```

The `instance_id` parameter controls column naming:

- `"index"`: Use positional prefixes (`inst0`, `inst1`, ...)
- `"track"`: Use track names as prefixes

!!! warning "Track mode requirements"
    Using `instance_id="track"` requires all instances to have tracks assigned. Use `untracked="ignore"` to skip untracked instances, or `untracked="error"` (default) to raise an error.

### Multi-index format

Hierarchical column structure with frame as row index.

```python
df = labels.to_dataframe(format="multi_index")
```

```
           inst0                                    inst1
           track track_score score   head           track track_score score   head
                                        x      y                                  x      y
frame_idx
0          fly_1        0.98  0.91  100.0  200.0    fly_2        0.95  0.86  300.0  400.0
```

This format is compatible with NWB pose estimation conventions and works well with pandas pivot operations.

### Decoding with `from_dataframe()`

Reconstruct `Labels` from a DataFrame:

```python
from sleap_io.codecs import from_dataframe

# Decode points format
labels = from_dataframe(df, video=video, skeleton=skeleton, format="points")
```

All four formats support decoding:

```python
# Decode instances format
labels = from_dataframe(df_instances, video=video, skeleton=skeleton, format="instances")

# Decode frames format
labels = from_dataframe(df_frames, video=video, skeleton=skeleton, format="frames")

# Decode multi_index format
labels = from_dataframe(df_multi, video=video, skeleton=skeleton, format="multi_index")
```

!!! tip "Skeleton inference"
    If not provided, the skeleton is inferred from column names (node names extracted from `{node}.x` pattern).

### Common parameters

All `to_dataframe()` calls support these options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video` | `None` | Filter to specific video |
| `include_metadata` | `True` | Include track/score columns |
| `include_score` | `True` | Include confidence scores |
| `include_user_instances` | `True` | Include user-labeled instances |
| `include_predicted_instances` | `True` | Include predictions |
| `video_id` | `"path"` | How to represent videos |
| `include_video` | `None` | Force video column on/off |
| `backend` | `"pandas"` | Output type (`"pandas"` or `"polars"`) |

**Video representation options (`video_id`):**

| Value | Column | Example |
|-------|--------|---------|
| `"path"` | `video_path` | `"/data/video.mp4"` |
| `"index"` | `video_idx` | `0` |
| `"name"` | `video_path` | `"video.mp4"` |
| `"object"` | `video` | `<Video>` |

See also: [`to_dataframe()`](#sleap_io.codecs.dataframe.to_dataframe), [`from_dataframe()`](#sleap_io.codecs.dataframe.from_dataframe)

---

## Working with large datasets

For datasets with millions of frames, use streaming conversion to avoid memory issues.

### Chunked iteration with `to_dataframe_iter()`

```python
from sleap_io.codecs import to_dataframe_iter

# Iterate in chunks of 10,000 rows
for chunk in to_dataframe_iter(labels, format="points", chunk_size=10000):
    # Process each chunk
    process(chunk)
```

**Output:**

```
Chunk 1: 10000 rows
Chunk 2: 10000 rows
Chunk 3: 5000 rows  # Final partial chunk
```

The `chunk_size` parameter specifies rows per chunk:

- `points` format: One point per row
- `instances` format: One instance per row
- `frames` / `multi_index` format: One frame per row

### Streaming to disk

```python
# Stream to Parquet files
for i, chunk in enumerate(to_dataframe_iter(labels, chunk_size=100000)):
    chunk.to_parquet(f"poses_part{i:04d}.parquet")
```

### Memory-efficient CSV export

```python
first_chunk = True
for chunk in to_dataframe_iter(labels, chunk_size=50000):
    chunk.to_csv(
        "poses.csv",
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )
    first_chunk = False
```

### Labels wrapper method

The iterator is also available on `Labels`:

```python
for chunk in labels.to_dataframe_iter(chunk_size=10000):
    process(chunk)
```

See also: [`to_dataframe_iter()`](#sleap_io.codecs.dataframe.to_dataframe_iter)

---

## Choosing a backend

The DataFrame codec supports both pandas and polars backends.

### Pandas (default)

```python
df = labels.to_dataframe(backend="pandas")
# Returns: pandas.DataFrame
```

Pandas is the default and works out of the box.

### Polars

```python
df = labels.to_dataframe(backend="polars")
# Returns: polars.DataFrame
```

```
shape: (6, 8)
┌───────────┬─────────┬───────┬───────┬───────┬─────────────┬────────────────┬───────┐
│ frame_idx ┆ node    ┆ x     ┆ y     ┆ track ┆ track_score ┆ instance_score ┆ score │
│ ---       ┆ ---     ┆ ---   ┆ ---   ┆ ---   ┆ ---         ┆ ---            ┆ ---   │
│ i64       ┆ str     ┆ f64   ┆ f64   ┆ str   ┆ f64         ┆ f64            ┆ f64   │
╞═══════════╪═════════╪═══════╪═══════╪═══════╪═════════════╪════════════════╪═══════╡
│ 0         ┆ head    ┆ 100.0 ┆ 200.0 ┆ fly_1 ┆ 0.98        ┆ 0.91           ┆ 0.95  │
│ 0         ┆ thorax  ┆ 150.0 ┆ 250.0 ┆ fly_1 ┆ 0.98        ┆ 0.91           ┆ 0.92  │
│ ...       ┆ ...     ┆ ...   ┆ ...   ┆ ...   ┆ ...         ┆ ...            ┆ ...   │
└───────────┴─────────┴───────┴───────┴───────┴─────────────┴────────────────┴───────┘
```

!!! info "Installation"
    Polars is optional. Install with:
    ```bash
    pip install sleap-io[polars]
    ```

### When to use polars

| Scenario | Recommendation |
|----------|----------------|
| Interactive analysis | pandas (broader ecosystem) |
| Large datasets | polars (faster, lower memory) |
| Streaming to Parquet | polars (native Arrow support) |
| Integration with ML libraries | pandas (wider compatibility) |

Performance comparison (10,000 frames, 3 tracks, 10 nodes):

| Format | Native Polars | Pandas→Polars | Speedup |
|--------|--------------|---------------|---------|
| points | 504ms | 544ms | 1.08x |
| instances | 387ms | 438ms | 1.13x |
| frames | 436ms | 477ms | 1.09x |
| multi_index | 426ms | 550ms | 1.29x |

The native polars backend constructs DataFrames directly without pandas conversion overhead.

---

## Common patterns

### Export to CSV

```python
# Simple export
df = labels.to_dataframe(format="points")
df.to_csv("poses.csv", index=False)

# With specific columns
df = labels.to_dataframe(format="instances", include_score=False)
df.to_csv("poses_no_scores.csv", index=False)
```

### Export to Parquet

```python
# Pandas
df = labels.to_dataframe()
df.to_parquet("poses.parquet")

# Polars (more efficient for large files)
df = labels.to_dataframe(backend="polars")
df.write_parquet("poses.parquet")
```

### Filter before conversion

```python
# Only predicted instances from first video
df = labels.to_dataframe(
    video=0,
    include_user_instances=False,
    include_predicted_instances=True
)
```

### Analyze trajectories

```python
df = labels.to_dataframe(format="points")

# Mean position per track
df.groupby("track")[["x", "y"]].mean()

# Velocity calculation
df_sorted = df.sort_values(["track", "node", "frame_idx"])
df_sorted["vx"] = df_sorted.groupby(["track", "node"])["x"].diff()
df_sorted["vy"] = df_sorted.groupby(["track", "node"])["y"].diff()
```

### Round-trip editing

```python
# Convert to DataFrame
df = labels.to_dataframe(format="points")

# Edit (e.g., apply calibration)
df["x"] = df["x"] * scale_x + offset_x
df["y"] = df["y"] * scale_y + offset_y

# Convert back
labels_calibrated = from_dataframe(
    df,
    video=labels.videos[0],
    skeleton=labels.skeletons[0],
    format="points"
)
```

---

## Troubleshooting

### Polars not installed

```
ImportError: Polars backend requires polars to be installed
```

**Solution:** Install polars with `pip install sleap-io[polars]` or `pip install polars`.

### Untracked instances in track mode

```
ValueError: Instance has no track assigned (use untracked='ignore' to skip)
```

**Solution:** Either assign tracks to all instances, or use `untracked="ignore"`:

```python
df = labels.to_dataframe(format="frames", instance_id="track", untracked="ignore")
```

### Missing video/skeleton in decoding

```
ValueError: video is required when DataFrame does not contain video column
```

**Solution:** Provide the required context:

```python
labels = from_dataframe(df, video=video, skeleton=skeleton)
```

### Array dimension mismatch in `from_numpy()`

```
ValueError: tracks_array must be 4-dimensional (frames, tracks, nodes, coords)
```

**Solution:** Ensure your array has shape `(n_frames, n_tracks, n_nodes, 2)` or `(n_frames, n_tracks, n_nodes, 3)` if including confidence scores.

---

## API reference

### DataFrameFormat

::: sleap_io.codecs.DataFrameFormat
    options:
        heading_level: 4

### to_dict

::: sleap_io.codecs.dictionary.to_dict
    options:
        heading_level: 4

### from_dict

::: sleap_io.codecs.dictionary.from_dict
    options:
        heading_level: 4

### to_numpy

::: sleap_io.codecs.numpy.to_numpy
    options:
        heading_level: 4

### from_numpy

::: sleap_io.codecs.numpy.from_numpy
    options:
        heading_level: 4

### to_dataframe

::: sleap_io.codecs.dataframe.to_dataframe
    options:
        heading_level: 4

### from_dataframe

::: sleap_io.codecs.dataframe.from_dataframe
    options:
        heading_level: 4

### to_dataframe_iter

::: sleap_io.codecs.dataframe.to_dataframe_iter
    options:
        heading_level: 4
