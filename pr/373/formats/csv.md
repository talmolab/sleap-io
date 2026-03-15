# CSV Format

sleap-io supports multiple CSV formats for exporting pose tracking data. Each format has different structures suited for various analysis workflows.

## Supported Formats

| Format | Description | Best For |
|--------|-------------|----------|
| `sleap` | SLEAP Analysis CSV | Native SLEAP exports, one row per instance |
| `dlc` | DeepLabCut format | DLC compatibility, multi-header structure |
| `points` | One row per point | Database imports, normalized data |
| `instances` | One row per instance | Analysis pipelines |
| `frames` | One row per frame | Time series analysis, wide format |

## Format Details

### SLEAP Format (`sleap`)

The default SLEAP Analysis CSV format produces one row per instance.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `track` | string | Track name (empty if untracked) |
| `frame_idx` | int | Frame index in video |
| `instance.score` | float | Instance confidence score |
| `{node}.x` | float | X coordinate for node |
| `{node}.y` | float | Y coordinate for node |
| `{node}.score` | float | Point confidence score |

**Example:**

```csv
track,frame_idx,instance.score,nose.x,nose.y,nose.score,tail.x,tail.y,tail.score
mouse_1,0,0.95,100.5,200.3,0.98,150.2,250.1,0.92
mouse_1,1,0.94,102.1,198.5,0.97,148.8,248.3,0.91
mouse_2,0,0.93,300.2,400.1,0.96,350.5,450.8,0.89
```

### DeepLabCut Format (`dlc`)

Multi-header format compatible with DeepLabCut analysis tools.

**Structure:**

- Row 1: Scorer name (repeated)
- Row 2: Body part names (repeated for x, y, likelihood)
- Row 3: Coordinate type (x, y, likelihood)
- Data rows: One per frame

**Example:**

```csv
scorer,MyModel,MyModel,MyModel,MyModel,MyModel,MyModel
bodyparts,nose,nose,nose,tail,tail,tail
coords,x,y,likelihood,x,y,likelihood
0,100.5,200.3,0.98,150.2,250.1,0.92
1,102.1,198.5,0.97,148.8,248.3,0.91
```

**Multi-animal DLC format** adds an `individuals` row:

```csv
scorer,MyModel,MyModel,MyModel,MyModel,MyModel,MyModel
individuals,mouse_1,mouse_1,mouse_1,mouse_2,mouse_2,mouse_2
bodyparts,nose,nose,nose,nose,nose,nose
coords,x,y,likelihood,x,y,likelihood
0,100.5,200.3,0.98,300.2,400.1,0.96
```

### Points Format (`points`)

The most normalized format with one row per point. Ideal for database imports.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `video_path` | string | Path to video file |
| `frame_idx` | int | Frame index |
| `track` | string | Track name |
| `instance_idx` | int | Instance index within frame |
| `instance_score` | float | Instance confidence |
| `node` | string | Node/keypoint name |
| `x` | float | X coordinate |
| `y` | float | Y coordinate |
| `score` | float | Point confidence |

**Example:**

```csv
video_path,frame_idx,track,instance_idx,instance_score,node,x,y,score
video.mp4,0,mouse_1,0,0.95,nose,100.5,200.3,0.98
video.mp4,0,mouse_1,0,0.95,tail,150.2,250.1,0.92
video.mp4,0,mouse_2,1,0.93,nose,300.2,400.1,0.96
video.mp4,0,mouse_2,1,0.93,tail,350.5,450.8,0.89
```

### Instances Format (`instances`)

One row per instance with all node coordinates as columns.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `video_path` | string | Path to video file |
| `frame_idx` | int | Frame index |
| `track` | string | Track name |
| `instance_idx` | int | Instance index within frame |
| `instance_score` | float | Instance confidence |
| `{node}.x` | float | X coordinate for node |
| `{node}.y` | float | Y coordinate for node |
| `{node}.score` | float | Point confidence for node |

**Example:**

```csv
video_path,frame_idx,track,instance_idx,instance_score,nose.x,nose.y,nose.score,tail.x,tail.y,tail.score
video.mp4,0,mouse_1,0,0.95,100.5,200.3,0.98,150.2,250.1,0.92
video.mp4,0,mouse_2,1,0.93,300.2,400.1,0.96,350.5,450.8,0.89
video.mp4,1,mouse_1,0,0.94,102.1,198.5,0.97,148.8,248.3,0.91
```

### Frames Format (`frames`)

One row per frame with all instances multiplexed into columns. Best for time series analysis.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | int | Frame index |
| `video_path` | string | Path to video file |
| `inst{N}.{node}.x` | float | X coordinate for instance N, node |
| `inst{N}.{node}.y` | float | Y coordinate for instance N, node |
| `inst{N}.{node}.score` | float | Point confidence for instance N, node |

**Example:**

```csv
frame_idx,video_path,inst0.nose.x,inst0.nose.y,inst0.nose.score,inst0.tail.x,inst0.tail.y,inst0.tail.score,inst1.nose.x,inst1.nose.y,inst1.nose.score,inst1.tail.x,inst1.tail.y,inst1.tail.score
0,video.mp4,100.5,200.3,0.98,150.2,250.1,0.92,300.2,400.1,0.96,350.5,450.8,0.89
1,video.mp4,102.1,198.5,0.97,148.8,248.3,0.91,,,,,,
```

!!! note "Empty Frames"
    When `all_frames=True` (or `include_empty=True`), frames without instances are included with NaN values for all coordinates.

## Frame Padding

By default, CSV export only includes frames with instances. To include all frames:

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Include empty frames (padded with NaN)
sio.save_csv(labels, "all_frames.csv", include_empty=True)

# Only frames with instances (sparse)
sio.save_csv(labels, "sparse.csv", include_empty=False)

# Specific frame range
sio.save_csv(labels, "clip.csv", start_frame=100, end_frame=500)
```

## Metadata Sidecar

CSV files cannot store all Labels information (skeleton edges, symmetries, suggestions). To enable full round-trip reconstruction, use `save_metadata=True`:

```python
sio.save_csv(labels, "data.csv", save_metadata=True)
# Creates: data.csv and data.json
```

### Metadata JSON Structure

```json
{
    "format_version": "1.0",
    "csv_format": "frames",
    "skeleton": {
        "name": "Skeleton-0",
        "nodes": ["nose", "head", "neck", "tail"],
        "edges": [[0, 1], [1, 2], [2, 3]],
        "symmetries": []
    },
    "videos": [
        {
            "filename": "video.mp4",
            "shape": [1000, 480, 640, 3]
        }
    ],
    "tracks": ["mouse_1", "mouse_2"],
    "suggestions": [],
    "provenance": {
        "source_file": "labels.slp",
        "sleap_io_version": "0.6.0"
    }
}
```

| Field | Description |
|-------|-------------|
| `format_version` | Metadata format version |
| `csv_format` | CSV format used (`sleap`, `dlc`, `points`, etc.) |
| `skeleton` | Full skeleton definition with nodes, edges, symmetries |
| `videos` | Video metadata (filename, shape) |
| `tracks` | Track names in order |
| `suggestions` | Suggested frame indices |
| `provenance` | Source file and version information |

## Loading CSV Files

```python
import sleap_io as sio

# Load CSV (auto-detects format)
labels = sio.load_csv("data.csv")

# If metadata sidecar exists, it's loaded automatically
# data.json provides skeleton edges, symmetries, etc.
```

## CLI Usage

```bash
# Export to CSV
sio export predictions.slp -o analysis.csv

# Specify format
sio export predictions.slp -o dlc.csv --csv-format dlc --scorer MyModel

# Include all frames
sio export predictions.slp -o all.csv --empty-frames

# Frame range
sio export predictions.slp -o clip.csv --start 100 --end 500

# With metadata sidecar
sio export predictions.slp -o data.csv --save-metadata
```

## API Reference

::: sleap_io.io.main.load_csv
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.main.save_csv
    options:
      heading_level: 3
      show_root_toc_entry: false
