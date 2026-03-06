# Analysis HDF5 Format

The SLEAP Analysis HDF5 format is a portable format for exporting pose tracking predictions as dense numpy arrays. This format is designed for easy loading in MATLAB and Python analysis pipelines.

## Overview

Analysis HDF5 files contain:

- **Pose coordinates** as dense 4D arrays
- **Confidence scores** for points, instances, and tracking
- **Track occupancy** indicating which frames have valid data
- **Metadata** for skeleton, video, and format information

## HDF5 Layout

```
analysis.h5
├── tracks                    # Dataset: Pose coordinates (4D array)
│   ├── @dims                 # Attribute: Dimension names (JSON list)
│   └── ...                   # Coordinates with NaN for missing data
│
├── track_occupancy           # Dataset: Track presence per frame (2D bool)
│   └── @dims                 # Attribute: Dimension names
│
├── point_scores              # Dataset: Per-point confidence (3D array)
│   └── @dims
│
├── instance_scores           # Dataset: Per-instance confidence (2D array)
│   └── @dims
│
├── tracking_scores           # Dataset: Tracking confidence (2D array)
│   └── @dims
│
├── track_names               # Dataset: Track name strings
├── node_names                # Dataset: Node/keypoint name strings
├── video_path                # Dataset: Source video path
│
├── @format                   # Attribute: "analysis" (format identifier)
├── @sleap_io_version         # Attribute: Format version (e.g., "1.0")
├── @preset                   # Attribute: Axis ordering preset
├── @provenance               # Attribute: Source file provenance (JSON)
├── @skeleton_name            # Attribute: Skeleton name
├── @skeleton_edges           # Attribute: Edge list (JSON)
├── @skeleton_symmetries      # Attribute: Symmetry pairs (JSON)
└── @labels_path              # Attribute: Original labels file path
```

## Axis Ordering Presets

The `preset` parameter controls the axis ordering of arrays. This is critical for compatibility with different analysis environments.

### `matlab` Preset (Default)

Optimized for MATLAB's column-major memory layout. Compatible with SLEAP's original analysis export.

| Dataset | Shape | Dimensions |
|---------|-------|------------|
| `tracks` | `(n_tracks, 2, n_nodes, n_frames)` | `["track", "xy", "node", "frame"]` |
| `track_occupancy` | `(n_frames, n_tracks)` | `["frame", "track"]` |
| `point_scores` | `(n_tracks, n_nodes, n_frames)` | `["track", "node", "frame"]` |
| `instance_scores` | `(n_tracks, n_frames)` | `["track", "frame"]` |
| `tracking_scores` | `(n_tracks, n_frames)` | `["track", "frame"]` |

**MATLAB usage:**

```matlab
data = h5read('analysis.h5', '/tracks');
% Access frame 10, track 1, node 3: data(1, :, 3, 10) -> [x, y]

occupancy = h5read('analysis.h5', '/track_occupancy');
% Check if track 1 is present in frame 10: occupancy(10, 1)
```

### `standard` Preset

Python-native ordering with frame as the first axis for intuitive indexing.

| Dataset | Shape | Dimensions |
|---------|-------|------------|
| `tracks` | `(n_frames, n_tracks, n_nodes, 2)` | `["frame", "track", "node", "xy"]` |
| `track_occupancy` | `(n_frames, n_tracks)` | `["frame", "track"]` |
| `point_scores` | `(n_frames, n_tracks, n_nodes)` | `["frame", "track", "node"]` |
| `instance_scores` | `(n_frames, n_tracks)` | `["frame", "track"]` |
| `tracking_scores` | `(n_frames, n_tracks)` | `["frame", "track"]` |

**Python usage:**

```python
import h5py

with h5py.File('analysis.h5', 'r') as f:
    tracks = f['tracks'][:]
    # Access frame 10, track 1, node 3: tracks[10, 1, 3, :] -> [x, y]

    occupancy = f['track_occupancy'][:]
    # Check if track 1 is present in frame 10: occupancy[10, 1]
```

!!! note "track_occupancy ordering"
    The `track_occupancy` dataset always has shape `(n_frames, n_tracks)` regardless of preset. This matches SLEAP's original behavior where `track_occupancy` was stored with frames first.

## Datasets

### `tracks`

Dense array of pose coordinates. Missing data is represented as `NaN`.

| Property | Value |
|----------|-------|
| Dtype | `float64` |
| Compression | gzip |
| Shape | Depends on preset (see above) |

**Coordinate system:** Pixel center at (0, 0). X increases rightward, Y increases downward.

### `track_occupancy`

Boolean array indicating which tracks have valid data per frame.

| Property | Value |
|----------|-------|
| Dtype | `bool` |
| Shape | `(n_frames, n_tracks)` |

### `point_scores`

Per-point confidence scores from the pose estimation model.

| Property | Value |
|----------|-------|
| Dtype | `float32` |
| Range | 0.0 to 1.0 |
| Missing | `NaN` |

### `instance_scores`

Per-instance confidence scores (average of point scores).

| Property | Value |
|----------|-------|
| Dtype | `float32` |
| Range | 0.0 to 1.0 |
| Missing | `NaN` |

### `tracking_scores`

Per-instance tracking confidence (from identity tracking models).

| Property | Value |
|----------|-------|
| Dtype | `float32` |
| Range | 0.0 to 1.0 |
| Default | 0.0 for user instances |

### `track_names`

Array of track name strings.

| Property | Value |
|----------|-------|
| Dtype | Variable-length string |
| Length | `n_tracks` |

### `node_names`

Array of skeleton node/keypoint names.

| Property | Value |
|----------|-------|
| Dtype | Variable-length string |
| Length | `n_nodes` |

### `video_path`

Source video file path.

| Property | Value |
|----------|-------|
| Dtype | String |

## Attributes

### File-Level Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `format` | string | Always `"analysis"` |
| `sleap_io_version` | string | Format version (e.g., `"1.0"`) |
| `preset` | string | Axis ordering: `"matlab"`, `"standard"`, or `"custom"` |
| `provenance` | JSON string | Source file and creation metadata |
| `skeleton_name` | string | Skeleton name |
| `skeleton_edges` | JSON string | Edge list as `[[src, dst], ...]` |
| `skeleton_symmetries` | JSON string | Symmetry pairs as `[["left", "right"], ...]` |
| `labels_path` | string | Original labels file path (optional) |

### Dataset Attributes

Each dataset has a `dims` attribute containing a JSON-encoded list of dimension names:

```python
>>> f['tracks'].attrs['dims']
b'["track", "xy", "node", "frame"]'  # matlab preset
```

## Track Filtering

The `min_occupancy` parameter filters tracks with low occupancy:

```python
import sleap_io as sio

# Keep all non-empty tracks (default)
sio.save_analysis_h5(labels, "all.h5", min_occupancy=0.0)

# Keep only tracks present in >50% of frames
sio.save_analysis_h5(labels, "filtered.h5", min_occupancy=0.5)
```

Occupancy is calculated as: `frames_with_track / total_frames`

## Custom Axis Ordering

For advanced use cases, you can specify explicit dimension positions:

```python
sio.save_analysis_h5(
    labels,
    "custom.h5",
    frame_dim=0,   # Frame is first axis
    track_dim=1,   # Track is second axis
    node_dim=2,    # Node is third axis
    xy_dim=3,      # XY is fourth axis
)
```

!!! warning "Mutually exclusive"
    You cannot use both `preset` and explicit dimension parameters.

## Reading Analysis HDF5

### With sleap-io

```python
import sleap_io as sio

# Load as Labels object
labels = sio.load_analysis_h5("analysis.h5")

# Access as numpy arrays
poses = labels.numpy()
```

### With h5py

```python
import h5py
import json

with h5py.File('analysis.h5', 'r') as f:
    # Read data
    tracks = f['tracks'][:]
    occupancy = f['track_occupancy'][:]

    # Read metadata
    preset = f.attrs['preset']
    dims = json.loads(f['tracks'].attrs['dims'])
    node_names = [n.decode() for n in f['node_names'][:]]
    track_names = [t.decode() for t in f['track_names'][:]]

    print(f"Preset: {preset}")
    print(f"Dimensions: {dims}")
    print(f"Nodes: {node_names}")
    print(f"Tracks: {track_names}")
```

### With MATLAB

```matlab
% Read data
tracks = h5read('analysis.h5', '/tracks');
occupancy = h5read('analysis.h5', '/track_occupancy');

% Read metadata
preset = h5readatt('analysis.h5', '/', 'preset');
node_names = h5read('analysis.h5', '/node_names');
track_names = h5read('analysis.h5', '/track_names');

% Get xy coordinates for frame 100, track 1, all nodes
frame_idx = 100;
track_idx = 1;
xy = squeeze(tracks(track_idx, :, :, frame_idx));  % Shape: (2, n_nodes)
```

## CLI Usage

```bash
# Export to Analysis HDF5 (default matlab preset)
sio export predictions.slp -o analysis.h5

# Use standard (Python-native) ordering
sio export predictions.slp -o analysis.h5 --h5-dim-order standard

# Filter tracks by occupancy
sio export predictions.slp -o filtered.h5 --min-occupancy 0.5
```

## API Reference

::: sleap_io.io.main.load_analysis_h5
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.main.save_analysis_h5
    options:
      heading_level: 3
      show_root_toc_entry: false
