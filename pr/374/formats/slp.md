# SLP File Format

The `.slp` file format is SLEAP's native format for storing pose tracking data. It is built on top of [HDF5](https://www.hdfgroup.org/solutions/hdf5/), a hierarchical data format designed for storing and organizing large amounts of scientific data.

SLP files can contain:

- **Video metadata** and references to source video files ([`Video`][sleap_io.Video])
- **Embedded images** with configurable encoding (PNG, JPEG, or raw arrays)
- **Skeleton definitions** with nodes, edges, and symmetries ([`Skeleton`][sleap_io.Skeleton])
- **Labeled frames** with user annotations and model predictions ([`LabeledFrame`][sleap_io.LabeledFrame])
- **Tracks** for identity tracking across frames ([`Track`][sleap_io.Track])
- **Suggestions** for frames to label ([`SuggestionFrame`][sleap_io.SuggestionFrame])
- **Recording sessions** for multi-camera setups ([`RecordingSession`][sleap_io.RecordingSession])
- **Regions of interest** (ROIs) with vector geometry ([`ROI`][sleap_io.ROI])
- **Segmentation masks** with run-length encoding ([`SegmentationMask`][sleap_io.SegmentationMask])

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
├── /negative_frames             # Dataset: Negative frame markers (optional)
│
├── /rois                        # Dataset: ROI metadata (structured array, optional)
│   ├── @categories              # Attribute: JSON array of category strings
│   ├── @names                   # Attribute: JSON array of name strings
│   └── @sources                 # Attribute: JSON array of source strings
├── /roi_wkb                     # Dataset: Packed WKB geometry bytes (uint8 array)
│
├── /masks                       # Dataset: Mask metadata (structured array, optional)
│   ├── @categories              # Attribute: JSON array of category strings
│   ├── @names                   # Attribute: JSON array of name strings
│   └── @sources                 # Attribute: JSON array of source strings
├── /mask_rle                    # Dataset: Packed RLE bytes (uint8 array)
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

[`Skeleton`][sleap_io.Skeleton] definitions are stored in the `/metadata` group's `json` attribute. The metadata JSON contains both a global node list and skeleton definitions that reference [`Node`][sleap_io.Node]s by index.

### JSON Format (SLP Files)

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

[`Edge`][sleap_io.Edge]s are encoded with a type field using Python's pickle-style encoding:

| Type ID | Meaning | Description |
|---------|---------|-------------|
| `1` | Regular edge | Skeletal connection between nodes |
| `2` | Symmetry edge | Bilateral symmetry relationship |

The first occurrence of each type uses the full `py/reduce` encoding; subsequent occurrences use `py/id` references.

### Symmetries

[`Symmetry`][sleap_io.Symmetry] relationships define bilateral pairings (e.g., left/right body parts). They are stored as edges with type `2`:

```json
{
    "source": 3,
    "target": 4,
    "type": {"py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [2]}]}
}
```

!!! note "Symmetry Deduplication"
    Legacy SLEAP files may store symmetries bidirectionally. The decoder automatically deduplicates them.

### YAML Format

In addition to the JSON format stored in SLP files, sleap-io supports a simplified YAML format for [`Skeleton`][sleap_io.Skeleton] definitions. This format is more human-readable and easier to edit manually.

!!! info "Preferred Format"
    The YAML format is the preferred format for skeleton definitions in sleap-nn and new tooling. The JSON format will be maintained for backwards compatibility with existing SLP files.

#### Structure

```yaml
skeleton_name:
  nodes:
    - name: head
    - name: thorax
    - name: abdomen
  edges:
    - source:
        name: head
      destination:
        name: thorax
    - source:
        name: thorax
      destination:
        name: abdomen
  symmetries:
    - - name: left_wing
      - name: right_wing
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `list[dict]` | List of [`Node`][sleap_io.Node] definitions with `name` key |
| `edges` | `list[dict]` | List of [`Edge`][sleap_io.Edge] definitions with `source` and `destination` |
| `symmetries` | `list[list]` | List of [`Symmetry`][sleap_io.Symmetry] pairs, each as a list of two node references |

#### Multiple Skeletons

The YAML format supports multiple skeletons in a single file, with skeleton names as top-level keys:

```yaml
fly:
  nodes:
    - name: head
    - name: thorax
  edges:
    - source: { name: head }
      destination: { name: thorax }
  symmetries: []

mouse:
  nodes:
    - name: nose
    - name: spine
  edges:
    - source: { name: nose }
      destination: { name: spine }
  symmetries: []
```

#### API Functions

Use the following functions to work with YAML skeletons:

- [`encode_yaml_skeleton`][sleap_io.io.skeleton.encode_yaml_skeleton] - Encode skeleton(s) to YAML string
- [`decode_yaml_skeleton`][sleap_io.io.skeleton.decode_yaml_skeleton] - Decode skeleton(s) from YAML string or file

## Metadata

The `/metadata` group stores format information and serialized metadata about the labels.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `format_id` | `float` | Format version (e.g., `1.4`) |
| `json` | `bytes` | JSON-encoded metadata string |

### JSON Structure

The `json` attribute contains a JSON object with the following fields:

```json
{
    "version": "2.0.0",
    "skeletons": [...],
    "nodes": [...],
    "videos": [],
    "tracks": [],
    "suggestions": [],
    "negative_anchors": {},
    "provenance": {
        "sleap_version": "1.3.4",
        "filename": "labels.slp"
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | `string` | SLEAP software version |
| `skeletons` | `array` | Skeleton graph definitions (see [Skeletons](#skeletons)) |
| `nodes` | `array` | Node definitions with names and weights |
| `videos` | `array` | Empty (stored in `/videos_json` dataset) |
| `tracks` | `array` | Empty (stored in `/tracks_json` dataset) |
| `suggestions` | `array` | Empty (stored in `/suggestions_json` dataset) |
| `negative_anchors` | `object` | Negative sample anchors for training |
| `provenance` | `object` | File origin and creation metadata |

### Provenance

The `provenance` field tracks the origin and history of the labels file:

| Key | Type | Description |
|-----|------|-------------|
| `sleap_version` | `string` | SLEAP version that created the file |
| `filename` | `string` | Original filename |
| *custom* | `any` | Additional user-defined provenance data |

!!! note "Custom Provenance"
    The provenance dictionary can contain arbitrary key-value pairs for tracking custom metadata like model training runs, data sources, or processing history.

## Tracks

[`Track`][sleap_io.Track]s enable identity tracking of individual animals across frames. Track metadata is stored in the `/tracks_json` dataset as an array of JSON strings.

### JSON Structure

Each track is stored as a two-element JSON array:

```json
[0, "track_name"]
```

| Index | Type | Description |
|-------|------|-------------|
| `0` | `int` | Spawned frame index (reserved, currently always `0`) |
| `1` | `string` | Track name for identification |

### Example

```json
[0, "mouse_1"]
[0, "mouse_2"]
[0, "female"]
```

### Instance Linking

[`Instance`][sleap_io.Instance]s reference tracks by index in the `/instances` dataset:

- `track = 0` → First track in `/tracks_json`
- `track = 1` → Second track in `/tracks_json`
- `track = -1` → Untracked instance

!!! note "Track Identity"
    Tracks are compared by object identity, not name. Two tracks with the same name are considered different unless they are the same object. This allows multiple tracks to share a name if needed.

## Suggestions

Suggestions indicate frames that should be labeled, typically generated by active learning algorithms or manual selection. Suggestion metadata is stored in the `/suggestions_json` dataset.

### JSON Structure

Each suggestion is stored as a JSON object:

```json
{
    "video": "0",
    "frame_idx": 42,
    "group": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `video` | `string` | Video index (as string) |
| `frame_idx` | `int` | Frame index within the video |
| `group` | `int` | Suggestion group ID (default: `0`) |

### Groups

The `group` field enables organizing suggestions into batches, useful for:

- Separating suggestions by generation method
- Tracking labeling progress across multiple sessions
- Grouping frames by difficulty or priority

### Optional Dataset

The `/suggestions_json` dataset is optional. Files without suggestions will not contain this dataset, and the reader returns an empty list when it's missing.

## Sessions

Recording sessions store multi-camera calibration data and synchronized frame groups. Session metadata is stored in the `/sessions_json` dataset.

### JSON Structure

Each session is stored as a JSON object:

```json
{
    "calibration": {
        "cam_0": {
            "name": "Camera 1",
            "size": [1080, 1920],
            "matrix": [[...], [...], [...]],
            "distortions": [...],
            "rotation": [...],
            "translation": [...]
        },
        "cam_1": {...},
        "metadata": {}
    },
    "camcorder_to_video_idx_map": {
        "0": 0,
        "1": 1
    },
    "frame_group_dicts": [...]
}
```

### Calibration

Camera calibration data is stored per camera with these fields:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `name` | `string` | - | Camera identifier |
| `size` | `int[]` | `(2,)` | Image dimensions `[height, width]` |
| `matrix` | `float[][]` | `(3, 3)` | Intrinsic camera matrix |
| `distortions` | `float[]` | `(5,)` | Radial-tangential distortion coefficients `[k1, k2, p1, p2, k3]` |
| `rotation` | `float[]` | `(3,)` | Rotation vector (axis-angle representation) |
| `translation` | `float[]` | `(3,)` | Translation vector |

### Camera-Video Mapping

The `camcorder_to_video_idx_map` object maps camera indices to video indices in `/videos_json`:

```json
{
    "0": 0,
    "1": 1
}
```

This links each camera in the calibration to its corresponding video.

### Frame Groups

Frame groups synchronize labeled frames across multiple cameras at the same time point. Each frame group contains:

- A frame index identifying the synchronized time point
- Instance groups linking instances across camera views
- References to `LabeledFrame` objects by index

### Optional Dataset

The `/sessions_json` dataset is optional. Files without multi-camera sessions will not contain this dataset.

## Instances

[`Instance`][sleap_io.Instance]s represent individual animals or objects in a frame. They are stored in the `/instances` dataset as a structured array.

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

| Type | Value | Data Model Class | Description |
|------|-------|------------------|-------------|
| `USER` | `0` | [`Instance`][sleap_io.Instance] | User-labeled annotation |
| `PREDICTED` | `1` | [`PredictedInstance`][sleap_io.PredictedInstance] | Model prediction |

### Instance Linking

The `from_predicted` field links user [`Instance`][sleap_io.Instance]s to their source [`PredictedInstance`][sleap_io.PredictedInstance]s, enabling tracking of corrections made to model outputs.

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

[`LabeledFrame`][sleap_io.LabeledFrame]s are stored in the `/frames` dataset, linking video frames to their instances.

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

## Negative Frames

Negative frames are frames explicitly marked as containing no instances (pure background). They are valuable for training, helping models learn what backgrounds look like without any animals present. Negative frames are distinct from "empty frames" which had instances that were deleted.

### Storage

Negative frames are stored in two places:

1. **`/frames` dataset**: Like all labeled frames, negative frames have a row in the `/frames` dataset. They have `instance_id_start == instance_id_end` (empty instance range).

2. **`/negative_frames` dataset**: A sidecar dataset that marks which empty frames are intentionally negative vs accidentally empty.

### Negative Frames Dtype

```python
negative_frames_dtype = np.dtype([
    ("video_id", "u4"),   # Sparse video ID (same as in /frames)
    ("frame_idx", "u8"),  # Frame index within video
])
```

### Example

```python
# A file with two negative frames
negative_frames = [
    (0, 42),   # Video 0, frame 42 is negative
    (0, 100),  # Video 0, frame 100 is negative
]
```

### Data Model Integration

When loading SLP files, the `is_negative` attribute is set on [`LabeledFrame`][sleap_io.LabeledFrame] objects:

```python
import sleap_io as sio

labels = sio.load_slp("labels.slp")

# Access negative frames
negative = labels.negative_frames  # List of LabeledFrames with is_negative=True

# Check if a frame is negative
for lf in labels:
    if lf.is_negative:
        print(f"Frame {lf.frame_idx} is a negative frame")

# Negative frames are included in user_labeled_frames for training export
user_frames = labels.user_labeled_frames  # Includes negative frames
```

### clean() Behavior

When calling [`Labels.clean(frames=True)`][sleap_io.Labels.clean], negative frames are preserved even though they have no instances. Only non-negative empty frames are removed.

### Optional Dataset

The `/negative_frames` dataset is optional. Files without negative frames will not contain this dataset, and all frames will have `is_negative=False`.

## Lazy Loading

For large SLP files with hundreds of thousands of frames, sleap-io provides a lazy loading mode that defers [`Labels`][sleap_io.Labels] object creation until needed.

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

## Bounding Boxes

[`BoundingBox`][sleap_io.BoundingBox] annotations store axis-aligned or oriented bounding boxes for object detection and tracking workflows. Bounding box support was introduced in format 1.7.

### BoundingBox Dataset

Bounding box data is stored in a single structured array dataset:

- `/bboxes`: Structured array containing bounding box geometry, metadata, and class information

### BoundingBox Dtype

```python
bbox_dtype = np.dtype([
    ("x_center", "f8"),     # Center x-coordinate in pixels
    ("y_center", "f8"),     # Center y-coordinate in pixels
    ("width", "f8"),        # Box width in pixels
    ("height", "f8"),       # Box height in pixels
    ("angle", "f8"),        # Rotation angle in radians (0 = axis-aligned)
    ("video", "i4"),        # Video index (-1 if none)
    ("frame_idx", "i8"),    # Frame index (-1 if none)
    ("track", "i4"),        # Track index (-1 if none)
    ("instance", "i4"),     # Instance index (-1 if none)
    ("is_predicted", "u1"), # 0 = UserBoundingBox, 1 = PredictedBoundingBox
    ("score", "f4"),        # Confidence score (NaN for user bboxes)
])
```

### User vs Predicted

- `is_predicted = 0`: [`UserBoundingBox`][sleap_io.UserBoundingBox] — human-annotated
- `is_predicted = 1`: [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox] — model-predicted, `score` contains the confidence value

### String Metadata

String metadata follows the same pattern as ROIs. JSON-encoded HDF5 attributes on the `/bboxes` dataset store `categories`, `names`, and `sources` arrays.

### Optional Dataset

The `/bboxes` dataset is only written when the [`Labels`][sleap_io.Labels] object contains bounding boxes. On read, a missing dataset defaults to an empty list.

### Migration from Format 1.5/1.6

When reading older files without a `/bboxes` dataset, any ROIs with axis-aligned rectangular geometry (`is_bbox = True`) are automatically migrated to [`UserBoundingBox`][sleap_io.UserBoundingBox] objects in `Labels.bboxes`. The migrated ROIs are removed from `Labels.rois`.

## Regions of Interest (ROIs)

[`ROI`][sleap_io.ROI]s store vector geometry annotations such as polygons and other shapes. ROI support was introduced in format 1.5.

### ROI Datasets

ROI data is stored across two datasets:

- `/rois`: Structured array containing ROI metadata and byte offsets into the geometry data
- `/roi_wkb`: Packed `uint8` array of [WKB (Well-Known Binary)](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary) geometry bytes

Each ROI's geometry is stored as a WKB blob in `/roi_wkb`, with `/rois` providing the byte range via `wkb_start` and `wkb_end` offsets.

### ROI Dtype

```python
roi_dtype = np.dtype([
    ("annotation_type", "u1"),  # Legacy field, always written as 0
    ("video", "i4"),            # Video index (-1 if none)
    ("frame_idx", "i8"),        # Frame index (-1 for static ROIs)
    ("track", "i4"),            # Track index (-1 if none)
    ("score", "f4"),            # Legacy field, always written as NaN
    ("wkb_start", "u8"),       # Start byte offset into /roi_wkb
    ("wkb_end", "u8"),         # End byte offset into /roi_wkb
    ("instance", "i4"),        # Instance index (-1 if none) (Format 1.6+)
])
```

!!! note "Legacy fields"
    The `annotation_type` and `score` columns are retained in the on-disk dtype
    for backward compatibility with older readers but are no longer used. Writers
    always set `annotation_type = 0` and `score = NaN`. Use the `category` string
    attribute for semantic classification and [`BoundingBox`][sleap_io.BoundingBox]
    for detection annotations with scores.

### String Metadata

String metadata fields (`categories`, `names`, `sources`) are stored as JSON-encoded HDF5 attributes on the `/rois` dataset:

```python
# Example attribute values
rois_dataset.attrs["categories"]  # '["arena", "nest"]'
rois_dataset.attrs["names"]       # '["arena_boundary", "nest_region"]'
rois_dataset.attrs["sources"]     # '["manual", "model_v2"]'
```

Each attribute is a JSON array with one entry per ROI, corresponding by index.

### Static vs Temporal ROIs

- **Static ROIs**: `frame_idx = -1`. Apply globally (e.g., arena boundaries).
- **Temporal ROIs**: `frame_idx >= 0`. Associated with a specific frame in a video.

### Optional Dataset

The `/rois` and `/roi_wkb` datasets are only written when the [`Labels`][sleap_io.Labels] object contains ROIs. On read, missing datasets default to empty lists.

## Segmentation Masks

[`SegmentationMask`][sleap_io.SegmentationMask]s store raster binary masks using run-length encoding (RLE). Mask support was introduced in format 1.5.

### Mask Datasets

Mask data is stored across two datasets:

- `/masks`: Structured array containing mask metadata and byte offsets into the RLE data
- `/mask_rle`: Packed `uint8` array of RLE-encoded mask bytes

The RLE encoding stores `uint32` run-length counts packed as little-endian `uint8` bytes. Each mask's RLE data is located in `/mask_rle` at the byte range specified by `rle_start` and `rle_end`.

### Mask Dtype

```python
mask_dtype = np.dtype([
    ("height", "u4"),           # Mask height in pixels
    ("width", "u4"),            # Mask width in pixels
    ("annotation_type", "u1"),  # Legacy field, always written as 2
    ("video", "i4"),            # Video index (-1 if none)
    ("frame_idx", "i8"),        # Frame index (-1 for static masks)
    ("track", "i4"),            # Track index (-1 if none)
    ("score", "f4"),            # Legacy field, always written as NaN
    ("rle_start", "u8"),       # Start byte offset into /mask_rle
    ("rle_end", "u8"),         # End byte offset into /mask_rle
])
```

!!! note "Legacy fields"
    As with ROIs, `annotation_type` and `score` are retained for backward
    compatibility but ignored on read. Writers always set
    `annotation_type = 2` (SEGMENTATION) and `score = NaN`.

### String Metadata

String metadata follows the same pattern as ROIs. JSON-encoded HDF5 attributes on the `/masks` dataset store `categories`, `names`, and `sources` arrays.

### Optional Dataset

The `/masks` and `/mask_rle` datasets are only written when the [`Labels`][sleap_io.Labels] object contains masks. On read, missing datasets default to empty lists.

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

### Format 1.4

**Added channel_order attribute** to embedded video datasets.

- Tracks RGB vs BGR channel ordering for embedded images
- Ensures correct color reproduction across different encoding backends
- Reading: Defaults to RGB if attribute missing

### Format 1.5

**Added ROI and segmentation mask support.**

- New datasets: `/rois`, `/roi_wkb` for vector geometry (WKB-encoded)
- New datasets: `/masks`, `/mask_rle` for binary masks (RLE-encoded)
- String metadata stored as JSON HDF5 attributes (`categories`, `names`, `sources`)
- Backward compatible: datasets only written when non-empty, missing datasets default to empty lists on read
- Requires `shapely>=2.0` for geometry operations

### Format 1.6

**Added ROI-instance association.**

- Added `instance` field (`i4`) to the `/rois` dtype for linking ROIs to specific instances
- ROI instance associations are persisted via instance index

### Format 1.7 (Current)

**Added bounding box support.**

- New dataset: `/bboxes` for first-class bounding box annotations
- Supports axis-aligned and oriented (rotated) bounding boxes
- User/predicted distinction via `is_predicted` flag and `score` field
- Migration on read: rectangular ROIs from older files are automatically converted to [`BoundingBox`][sleap_io.BoundingBox] objects
- `annotation_type` and `score` fields on `/rois` and `/masks` are now legacy (always written as constants)

## API

### High-Level Functions

::: sleap_io.io.main.load_slp
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.main.save_slp
    options:
      heading_level: 4
      show_root_toc_entry: false

### Core Module

::: sleap_io.io.slp.read_labels
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.write_labels
    options:
      heading_level: 4
      show_root_toc_entry: false

### Video I/O

::: sleap_io.io.slp.read_videos
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.write_videos
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.embed_videos
    options:
      heading_level: 4
      show_root_toc_entry: false

### Skeleton I/O

::: sleap_io.io.slp.read_skeletons
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.serialize_skeletons
    options:
      heading_level: 4
      show_root_toc_entry: false

### Instance I/O

::: sleap_io.io.slp.read_instances
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.read_points
    options:
      heading_level: 4
      show_root_toc_entry: false

::: sleap_io.io.slp.read_pred_points
    options:
      heading_level: 4
      show_root_toc_entry: false

### Lazy Loading

::: sleap_io.io.slp_lazy.LazyDataStore
    options:
      heading_level: 4
      show_root_toc_entry: false
      show_source: false
      members:
        - materialize_frame
        - materialize_all
        - to_numpy
        - get_user_frame_indices

::: sleap_io.io.slp_lazy.LazyFrameList
    options:
      heading_level: 4
      show_root_toc_entry: false
      show_source: false
