# SLP File Format

The `.slp` file format is SLEAP's native format for storing pose tracking data. It is built on top of [HDF5](https://www.hdfgroup.org/solutions/hdf5/), a hierarchical data format designed for storing and organizing large amounts of scientific data.

SLP files can contain:

- **Video metadata** and references to source video files
- **Embedded images** with configurable encoding (PNG, JPEG, or raw arrays)
- **Skeleton definitions** with nodes, edges, and symmetries
- **Labeled frames** with user annotations and model predictions
- **Tracks** for identity tracking across frames
- **Suggestions** for frames to label
- **Recording sessions** for multi-camera setups

## HDF5 Layout

SLP files have the following hierarchical structure:

```
file.slp
├── /metadata                    # Group: Format and skeleton metadata
│   ├── @format_id               # Attribute: Format version (float, e.g., 1.4)
│   └── @json                    # Attribute: JSON metadata string
│
├── /videos_json                 # Dataset: Video metadata (variable-length bytes)
├── /tracks_json                 # Dataset: Track metadata (variable-length bytes)
├── /suggestions_json            # Dataset: Suggestions (variable-length bytes, optional)
├── /sessions_json               # Dataset: Recording sessions (variable-length bytes, optional)
│
├── /frames                      # Dataset: Labeled frame metadata (structured array)
├── /instances                   # Dataset: Instance metadata (structured array)
├── /points                      # Dataset: User-labeled points (structured array)
├── /pred_points                 # Dataset: Predicted points (structured array)
│
└── /video{N}/                   # Group: Per-video embedded data (one per video)
    ├── /video                   # Dataset: Embedded image data
    │   ├── @format              # Attribute: "png", "jpg", or "hdf5"
    │   ├── @channel_order       # Attribute: "RGB" or "BGR"
    │   ├── @frames              # Attribute: Total frames in source video
    │   ├── @height              # Attribute: Frame height
    │   ├── @width               # Attribute: Frame width
    │   ├── @channels            # Attribute: Number of channels
    │   └── @fps                 # Attribute: Frames per second (optional)
    ├── /frame_numbers           # Dataset: Embedded frame indices (int array)
    └── /source_video/           # Group: Source video metadata
        └── @json                # Attribute: JSON with source video info
```

### Core Datasets

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `points` | `(N,)` | structured | User-labeled point coordinates |
| `pred_points` | `(N,)` | structured | Predicted point coordinates with scores |
| `instances` | `(N,)` | structured | Instance metadata linking to points |
| `frames` | `(N,)` | structured | Frame metadata linking to instances |

### Metadata Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| `videos_json` | `bytes[]` | JSON array of video metadata |
| `tracks_json` | `bytes[]` | JSON array of track definitions |
| `suggestions_json` | `bytes[]` | JSON array of suggested frames (optional) |
| `sessions_json` | `bytes[]` | JSON array of recording sessions (optional) |

## Videos

Video metadata is stored in the `/videos_json` dataset as an array of JSON strings. Each video entry contains:

```json
{
    "filename": "path/to/video.mp4",
    "backend": {
        "type": "MediaVideo",
        "shape": [1000, 480, 640, 3],
        "filename": "path/to/video.mp4",
        "grayscale": false,
        "bgr": true,
        "fps": 30.0
    },
    "source_video": null
}
```

### Backend Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `MediaVideo` | Standard video files (mp4, avi, mov, etc.) | `filename`, `bgr`, `fps` |
| `HDF5Video` | Embedded frames in HDF5 | `dataset`, `input_format`, `has_embedded_images` |
| `ImageVideo` | Image sequences | `filename`, `filenames` (list) |
| `TiffVideo` | TIFF stacks | `filename`, `keep_open` |

### Source Video Lineage

Videos can have a `source_video` field that tracks the original video when frames are embedded:

```json
{
    "filename": ".",
    "backend": {
        "type": "HDF5Video",
        "dataset": "video0/video"
    },
    "source_video": {
        "filename": "original.mp4",
        "backend": { ... }
    }
}
```

This creates a chain of provenance, allowing the original video to be restored when extracting embedded data.

## Embedded Images

Frames can be embedded directly in SLP files for portability. Embedded frames are stored in `/video{N}/` groups.

### Encoding Formats

| Format | Storage | Compression | Notes |
|--------|---------|-------------|-------|
| `png` | `int8[]` | Lossless PNG | Default, best quality |
| `jpg` | `int8[]` | Lossy JPEG | Smaller files, some quality loss |
| `hdf5` | Raw array | Optional gzip | No encoding overhead, large files |

### Frame Selection Modes

When saving with embedded frames, the `embed` parameter controls which frames to include:

| Mode | Description |
|------|-------------|
| `None` | Re-embed existing embedded frames |
| `True` / `"all"` | All labeled frames and suggestions |
| `"user"` | Only user-labeled frames |
| `"suggestions"` | Only suggested frames |
| `"user+suggestions"` | Both user and suggested frames |
| `"source"` | No embedding, restore source video |
| `list[(Video, int)]` | Custom list of (video, frame_idx) pairs |

### Channel Order

The `channel_order` attribute (introduced in format 1.4) tracks the color channel ordering:

- **OpenCV encoding**: BGR (`"BGR"`)
- **imageio encoding**: RGB (`"RGB"`)
- **Raw HDF5 arrays**: RGB (`"RGB"`)

This ensures correct color reproduction when reading embedded frames.

## Skeletons

Skeleton definitions are stored in the `/metadata` group's `json` attribute. The metadata JSON contains both a global node list and skeleton definitions that reference nodes by index.

### Metadata JSON Structure

```json
{
    "version": "2.0.0",
    "skeletons": [
        {
            "directed": true,
            "graph": {"name": "Skeleton-0", "num_edges_inserted": 5},
            "links": [
                {"source": 0, "target": 1, "type": {"py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [1]}]}},
                {"source": 1, "target": 2, "type": {"py/id": 1}}
            ],
            "nodes": [{"id": 0}, {"id": 1}, {"id": 2}]
        }
    ],
    "nodes": [
        {"name": "head", "weight": 1.0},
        {"name": "thorax", "weight": 1.0},
        {"name": "abdomen", "weight": 1.0}
    ]
}
```

### Edge Types

Edges are encoded with a type field using Python's pickle-style encoding:

| Type ID | Meaning | Description |
|---------|---------|-------------|
| `1` | Regular edge | Skeletal connection between nodes |
| `2` | Symmetry edge | Bilateral symmetry relationship |

The first occurrence of each type uses the full `py/reduce` encoding; subsequent occurrences use `py/id` references.

### Symmetries

Symmetries define bilateral relationships (e.g., left/right body parts). They are stored as edges with type `2`:

```json
{
    "source": 3,
    "target": 4,
    "type": {"py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [2]}]}
}
```

!!! note "Symmetry Deduplication"
    Legacy SLEAP files may store symmetries bidirectionally. The decoder automatically deduplicates them.

## Instances

Instances represent individual animals or objects in a frame. They are stored in the `/instances` dataset as a structured array.

### Instance Dtype

```python
instance_dtype = np.dtype([
    ("instance_id", "i8"),        # Unique instance identifier
    ("instance_type", "u1"),      # 0=USER, 1=PREDICTED
    ("frame_id", "u8"),           # Index into frames dataset
    ("skeleton", "u4"),           # Index into skeletons list
    ("track", "i4"),              # Index into tracks list (-1 if untracked)
    ("from_predicted", "i8"),     # Parent prediction ID (-1 if none)
    ("score", "f4"),              # Prediction score (0.0 for user)
    ("point_id_start", "u8"),     # Start index in points array
    ("point_id_end", "u8"),       # End index (exclusive)
    ("tracking_score", "f4"),     # Tracking confidence (format >= 1.2)
])
```

### Instance Types

| Type | Value | Description |
|------|-------|-------------|
| `USER` | `0` | User-labeled annotation |
| `PREDICTED` | `1` | Model prediction |

### Instance Linking

The `from_predicted` field links user instances to their source predictions, enabling tracking of corrections made to model outputs.

## Points

Point coordinates are stored in separate datasets for user-labeled and predicted instances.

### User Points (`/points`)

```python
point_dtype = np.dtype([
    ("x", "f8"),        # X coordinate
    ("y", "f8"),        # Y coordinate
    ("visible", "?"),   # Is point visible
    ("complete", "?"),  # Is point marked complete
])
```

### Predicted Points (`/pred_points`)

```python
predicted_point_dtype = np.dtype([
    ("x", "f8"),        # X coordinate
    ("y", "f8"),        # Y coordinate
    ("visible", "?"),   # Is point visible
    ("complete", "?"),  # Is point marked complete
    ("score", "f8"),    # Prediction confidence
])
```

### Coordinate System

!!! warning "Coordinate System Change"
    Format 1.1 changed the coordinate system from pixel corner to pixel center.

| Format | Origin | Notes |
|--------|--------|-------|
| < 1.1 | Top-left corner of pixel at (0, 0) | Legacy |
| >= 1.1 | Center of pixel at (0, 0) | Current |

When reading format < 1.1 files, the reader applies a -0.5 offset to convert coordinates.

## Labeled Frames

Labeled frames are stored in the `/frames` dataset, linking video frames to their instances.

### Frame Dtype

```python
frame_dtype = np.dtype([
    ("frame_id", "u8"),           # Unique frame identifier
    ("video", "u4"),              # Video index or sparse video ID
    ("frame_idx", "u8"),          # Frame index within video
    ("instance_id_start", "u8"),  # Start index in instances array
    ("instance_id_end", "u8"),    # End index (exclusive)
])
```

### Video ID Mapping

Modern SLP files use sequential video indices (0, 1, 2, ...), but legacy files may contain sparse video IDs derived from the embedded video group names (e.g., 0, 15, 29). The reader handles both cases transparently.

## Lazy Loading

For large SLP files with hundreds of thousands of frames, sleap-io provides a lazy loading mode that defers object creation until needed.

### Architecture

```
Labels (lazy mode)
├── LazyDataStore
│   ├── frames_data (numpy array)
│   ├── instances_data (numpy array)
│   ├── points_data (numpy array)
│   └── pred_points_data (numpy array)
└── LazyFrameList
    └── Materializes frames on-demand
```

### Performance Benefits

| Operation | Eager | Lazy | Speedup |
|-----------|-------|------|---------|
| Load file | ~0.5s | ~0.005s | **~100x** |
| Load + numpy() | ~0.9s | ~0.4s | **~2x** |
| Full iteration | ~0.0002s | ~0.4s | Eager faster |

*Benchmarks on 18,000 frames with ~40,000 instances.*

### When to Use Lazy Loading

**Recommended for:**

- Converting to NumPy arrays (`labels.numpy()`)
- Saving to another file without modifications
- Accessing a small subset of frames
- Quick metadata inspection

**Not recommended for:**

- Iterating over all frames (eager is faster)
- Modifying data (must materialize first)
- Multiple passes over the data

### Fast Paths

Lazy labels support optimized code paths:

1. **numpy() conversion**: Builds arrays directly from raw HDF5 data without creating Python objects
2. **Saving**: Copies raw arrays directly without materialization (when `embed` is `None`, `False`, or `"source"`)
3. **Metadata queries**: Properties like `n_user_instances`, `n_pred_instances` use O(1) array operations

### Usage

```python
import sleap_io as sio

# Load lazily
labels = sio.load_slp("predictions.slp", lazy=True)

# Check lazy state
print(labels.is_lazy)  # True

# Fast numpy conversion
poses = labels.numpy()

# Materialize when modifications needed
labels = labels.materialize()
labels.append(new_frame)  # Now works
```

## Version History

The SLP format has evolved through several versions, tracked by the `format_id` attribute in `/metadata`.

### Format 1.0

Initial release format.

### Format 1.1

**Coordinate system change**: Changed from top-left pixel corner at (0, 0) to pixel center at (0, 0).

- Reading: Applies -0.5 offset to coordinates from older files
- Writing: Always uses new coordinate system

### Format 1.2

**Added tracking_score field** to instance dtype.

- Instance dtype expanded from 9 to 10 fields
- `tracking_score` stores tracking confidence for multi-animal workflows
- Reading: Defaults to 0.0 for older files

### Format 1.3

Minor handling improvements for tracking_score (no schema change from 1.2).

### Format 1.4 (Current)

**Added channel_order attribute** to embedded video datasets.

- Tracks RGB vs BGR channel ordering for embedded images
- Ensures correct color reproduction across different encoding backends
- Reading: Defaults to RGB if attribute missing

## API

### High-Level Functions

::: sleap_io.io.main.load_slp

::: sleap_io.io.main.save_slp

### Core Module

::: sleap_io.io.slp.read_labels

::: sleap_io.io.slp.write_labels

### Video I/O

::: sleap_io.io.slp.read_videos

::: sleap_io.io.slp.write_videos

::: sleap_io.io.slp.embed_videos

### Skeleton I/O

::: sleap_io.io.slp.read_skeletons

::: sleap_io.io.slp.serialize_skeletons

### Instance I/O

::: sleap_io.io.slp.read_instances

::: sleap_io.io.slp.read_points

::: sleap_io.io.slp.read_pred_points

### Lazy Loading

::: sleap_io.io.slp_lazy.LazyDataStore
    options:
      show_source: false
      members:
        - materialize_frame
        - materialize_all
        - to_numpy
        - get_user_frame_indices

::: sleap_io.io.slp_lazy.LazyFrameList
    options:
      show_source: false
