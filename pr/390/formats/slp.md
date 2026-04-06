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
- **Bounding boxes** for object detection and tracking ([`BoundingBox`][sleap_io.BoundingBox])
- **Regions of interest** (ROIs) with vector geometry ([`ROI`][sleap_io.ROI])
- **Segmentation masks** with run-length encoding ([`SegmentationMask`][sleap_io.SegmentationMask])
- **Label images** for dense per-pixel instance segmentation ([`LabelImage`][sleap_io.LabelImage])

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
├── /identities_json             # Dataset: Identity metadata (variable-length bytes, optional)
│
├── /frames                      # Dataset: Labeled frame metadata (structured array)
├── /instances                   # Dataset: Instance metadata (structured array)
├── /points                      # Dataset: User-labeled points (structured array)
├── /pred_points                 # Dataset: Predicted points (structured array)
├── /negative_frames             # Dataset: Negative frame markers (optional)
│
├── /bboxes/                     # Group: Columnar bounding box storage (Format 2.0+)
│   ├── x1                       # Dataset: float64 top-left x
│   ├── y1                       # Dataset: float64 top-left y
│   ├── x2                       # Dataset: float64 bottom-right x
│   ├── y2                       # Dataset: float64 bottom-right y
│   ├── angle                    # Dataset: float64 rotation angle (radians)
│   ├── video                    # Dataset: int32 video index
│   ├── frame_idx                # Dataset: int64 frame index
│   ├── track                    # Dataset: int32 track index
│   ├── instance                 # Dataset: int32 instance index
│   ├── is_predicted             # Dataset: uint8 (0=user, 1=predicted)
│   ├── score                    # Dataset: float32 confidence score
│   ├── category                 # Dataset: vlen str category labels
│   ├── name                     # Dataset: vlen str name labels
│   └── source                   # Dataset: vlen str source labels
│
├── /rois                        # Dataset: ROI metadata (structured array, optional)
│   ├── @categories              # Attribute: JSON array (legacy fallback)
│   ├── @names                   # Attribute: JSON array (legacy fallback)
│   └── @sources                 # Attribute: JSON array (legacy fallback)
├── /roi_wkb                     # Dataset: Packed WKB geometry bytes (uint8 array)
├── /roi_categories              # Dataset: vlen string, one per ROI (Format 1.9+)
├── /roi_names                   # Dataset: vlen string, one per ROI (Format 1.9+)
├── /roi_sources                 # Dataset: vlen string, one per ROI (Format 1.9+)
│
├── /masks                       # Dataset: Mask metadata (structured array, optional)
│   ├── @categories              # Attribute: JSON array (legacy fallback)
│   ├── @names                   # Attribute: JSON array (legacy fallback)
│   └── @sources                 # Attribute: JSON array (legacy fallback)
├── /mask_rle                    # Dataset: Packed RLE bytes (uint8 array)
├── /mask_categories             # Dataset: vlen string, one per mask (Format 1.9+)
├── /mask_names                  # Dataset: vlen string, one per mask (Format 1.9+)
├── /mask_sources                # Dataset: vlen string, one per mask (Format 1.9+)
│
├── /mask_score_map_index        # Dataset: Score map index for masks (Format 1.9+)
├── /mask_score_maps             # Dataset: Packed score map data for masks (Format 1.9+)
├── /label_image_score_map_index # Dataset: Score map index for label images (Format 1.9+)
├── /label_image_score_maps      # Dataset: Packed score map data for label images (Format 1.9+)
│
├── /label_images                # Dataset: Label image metadata (Format 1.8+)
├── /label_image_objects         # Dataset: Per-object metadata (Format 1.8+)
├── /label_image_data            # Dataset: Packed pixel data, zlib-compressed (Format 1.8+)
├── /label_image_sources         # Dataset: vlen string, one per label image (Format 1.9+)
├── /label_image_obj_categories  # Dataset: vlen string, one per object (Format 1.9+)
├── /label_image_obj_names       # Dataset: vlen string, one per object (Format 1.9+)
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
| `identities_json` | `bytes[]` | JSON array of identity definitions (optional) |

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

| Key | Type | Set by | Description |
|-----|------|--------|-------------|
| `sleap_version` | `string` | SLEAP | SLEAP version that created the file |
| `filename` | `string` | `load_slp` | Original filename (set on load) |
| `source_labels` | `string` | `split` / `extract` | Path to parent labels file |
| `merge_history` | `array` | `merge` | Records of merge operations (timestamp, source, strategy) |
| *custom* | `any` | user | Additional user-defined provenance data |

!!! note "Custom Provenance"
    The provenance dictionary can contain arbitrary key-value pairs for
    tracking custom metadata. Values must be JSON-serializable. Path objects
    are auto-converted to strings on save.

!!! example "Recording segmentation model parameters"
    When using segmentation tools like Cellpose, record the model parameters
    in provenance for reproducibility:

    ```python
    import sleap_io as sio

    labels = sio.Labels(label_images=label_images)
    labels.provenance["segmentation_model"] = "cellpose"
    labels.provenance["cellpose_model_type"] = "cyto3"
    labels.provenance["cellpose_diameter"] = 30
    labels.provenance["cellpose_cellprob_threshold"] = 0.0
    labels.save("segmentation.slp")

    # Later, verify parameters:
    loaded = sio.load_slp("segmentation.slp")
    print(loaded.provenance["cellpose_diameter"])  # 30
    ```

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

## Identities

[`Identity`][sleap_io.Identity] objects represent ground-truth animal identities that persist across sessions and videos. Identity metadata is stored in the `/identities_json` dataset as an array of JSON strings.

### JSON Structure

Each identity is stored as a JSON object:

```json
{
    "name": "mouse_A",
    "color": "#e6194b"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Human-readable identity name |
| `color` | `string` | Optional hex color for visualization (omitted if null) |
| *custom* | `any` | Additional metadata fields |

### Instance Group Linking

`InstanceGroup`s reference identities by index via the `identity_idx` field in the session JSON:

```json
{
    "camcorder_to_lf_and_inst_idx_map": {...},
    "identity_idx": 0
}
```

The index corresponds to the position in `/identities_json`.

### Optional Dataset

The `/identities_json` dataset is only written when the [`Labels`][sleap_io.Labels] object contains identities. On read, a missing dataset defaults to an empty list.

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

Instance groups may also contain 3D reconstruction data:

- `points` — triangulated 3D keypoint coordinates (list of `[x, y, z]`)
- `instance_3d_score` — instance-level confidence score for the 3D reconstruction
- `instance_3d_point_scores` — per-keypoint confidence scores (when using `PredictedInstance3D`)
- `identity_idx` — index into `/identities_json` linking this group to an `Identity`

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

[`BoundingBox`][sleap_io.BoundingBox] annotations store axis-aligned or oriented bounding boxes for object detection and tracking workflows. Bounding box support was introduced in format 1.7. Format 2.0+ uses columnar storage under the `/bboxes/` HDF5 group.

### Columnar Datasets (Format 2.0+)

Bounding box data is stored as individual datasets within the `/bboxes/` HDF5 group:

| Dataset | Dtype | Description |
|---------|-------|-------------|
| `x1` | `float64` | Top-left x-coordinate in pixels |
| `y1` | `float64` | Top-left y-coordinate in pixels |
| `x2` | `float64` | Bottom-right x-coordinate in pixels |
| `y2` | `float64` | Bottom-right y-coordinate in pixels |
| `angle` | `float64` | Rotation angle in radians (0 = axis-aligned) |
| `video` | `int32` | Video index (-1 if none) |
| `frame_idx` | `int64` | Frame index (-1 if none) |
| `track` | `int32` | Track index (-1 if none) |
| `instance` | `int32` | Instance index (-1 if none) |
| `is_predicted` | `uint8` | 0 = UserBoundingBox, 1 = PredictedBoundingBox |
| `score` | `float32` | Confidence score (NaN for user bboxes) |
| `category` | vlen `str` | Category label per bounding box |
| `name` | vlen `str` | Name label per bounding box |
| `source` | vlen `str` | Source label per bounding box |

### User vs Predicted

- `is_predicted = 0`: [`UserBoundingBox`][sleap_io.UserBoundingBox] -- human-annotated
- `is_predicted = 1`: [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox] -- model-predicted, `score` contains the confidence value

### String Metadata

String metadata (`category`, `name`, `source`) is stored as vlen string datasets within the `/bboxes/` group, one entry per bounding box.

### Optional Dataset

The `/bboxes/` group is only written when the [`Labels`][sleap_io.Labels] object contains bounding boxes. On read, a missing group defaults to an empty list.

!!! note "Legacy format (1.7--1.9)"
    Older SLP files store bounding boxes as a single structured array dataset (`/bboxes`)
    with `x_center`, `y_center`, `width`, `height` columns and JSON-encoded string
    attributes. The reader auto-detects whether `/bboxes` is a group (format 2.0+) or a
    dataset (legacy) and handles both transparently.

### Migration from Format 1.5/1.6

When reading older files without a `/bboxes` dataset or group, any ROIs with axis-aligned rectangular geometry (`is_bbox = True`) are automatically migrated to [`UserBoundingBox`][sleap_io.UserBoundingBox] objects in `Labels.bboxes`. The migrated ROIs are removed from `Labels.rois`.

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
    ("is_predicted", "u1"),     # 0 = UserROI, 1 = PredictedROI (Format 1.9+)
    ("score", "f4"),            # Confidence score (NaN for user ROIs) (Format 1.9+)
    ("wkb_start", "u8"),       # Start byte offset into /roi_wkb
    ("wkb_end", "u8"),         # End byte offset into /roi_wkb
    ("instance", "i4"),        # Instance index (-1 if none) (Format 1.6+)
])
```

!!! note "Legacy and predicted fields"
    The `annotation_type` column is retained in the on-disk dtype for backward
    compatibility with older readers but is no longer used. Writers always set
    `annotation_type = 0`. Use the `category` string metadata for semantic
    classification and [`BoundingBox`][sleap_io.BoundingBox] for detection
    annotations.

    The `is_predicted` field distinguishes [`UserROI`][sleap_io.UserROI] (`0`)
    from [`PredictedROI`][sleap_io.PredictedROI] (`1`). The `score` field stores
    the confidence value for predicted ROIs (NaN for user ROIs).

### String Metadata

Format 1.9+ stores ROI string metadata as vlen HDF5 string datasets at the root level:

- `/roi_categories`: One category string per ROI
- `/roi_names`: One name string per ROI
- `/roi_sources`: One source string per ROI

The reader checks for these datasets first. For pre-1.9 files, it falls back to JSON-encoded HDF5 attributes on the `/rois` dataset (`@categories`, `@names`, `@sources`).

```python
# Format 1.9+ (vlen string datasets)
f["/roi_categories"]  # ["arena", "nest"]
f["/roi_names"]       # ["arena_boundary", "nest_region"]
f["/roi_sources"]     # ["manual", "model_v2"]

# Pre-1.9 legacy (JSON attributes on /rois dataset)
rois_dataset.attrs["categories"]  # '["arena", "nest"]'
rois_dataset.attrs["names"]       # '["arena_boundary", "nest_region"]'
rois_dataset.attrs["sources"]     # '["manual", "model_v2"]'
```

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
    ("instance", "i4"),         # Instance index (-1 if none) (Format 1.9+)
    ("is_predicted", "u1"),     # 0 = UserSegmentationMask, 1 = Predicted (Format 1.9+)
    ("score", "f4"),            # Confidence score (NaN for user masks) (Format 1.9+)
    ("rle_start", "u8"),       # Start byte offset into /mask_rle
    ("rle_end", "u8"),         # End byte offset into /mask_rle
])
```

!!! note "Legacy and predicted fields"
    The `annotation_type` column is retained for backward compatibility but
    ignored on read. Writers always set `annotation_type = 2` (SEGMENTATION).

    The `is_predicted` field distinguishes `UserSegmentationMask` (`0`) from
    `PredictedSegmentationMask` (`1`). The `score` field stores the confidence
    value for predicted masks (NaN for user masks).

### String Metadata

Format 1.9+ stores mask string metadata as vlen HDF5 string datasets at the root level:

- `/mask_categories`: One category string per mask
- `/mask_names`: One name string per mask
- `/mask_sources`: One source string per mask

The reader checks for these datasets first. For pre-1.9 files, it falls back to JSON-encoded HDF5 attributes on the `/masks` dataset (`@categories`, `@names`, `@sources`).

### Optional Dataset

The `/masks` and `/mask_rle` datasets are only written when the [`Labels`][sleap_io.Labels] object contains masks. On read, missing datasets default to empty lists.

## Score Map Datasets

Score maps store per-pixel confidence values for predicted segmentation masks and label images. These datasets are only written when `PredictedSegmentationMask` or `PredictedLabelImage` objects have a `score_map` set.

### Mask Score Maps

Mask score maps are stored across two datasets:

- `/mask_score_map_index`: Structured array indexing into the packed data
- `/mask_score_maps`: Packed `uint8` array of zlib-compressed `float32` score map data

#### Index Dtype

```python
mask_score_map_index_dtype = np.dtype([
    ("mask_idx", "u4"),     # Index into /masks dataset
    ("data_start", "u8"),   # Start byte offset into /mask_score_maps
    ("data_end", "u8"),     # End byte offset into /mask_score_maps
    ("height", "u4"),       # Score map height in pixels
    ("width", "u4"),        # Score map width in pixels
])
```

### Label Image Score Maps

Label image score maps follow the same structure:

- `/label_image_score_map_index`: Structured array indexing into the packed data
- `/label_image_score_maps`: Packed `uint8` array of zlib-compressed `float32` score map data

#### Index Dtype

```python
label_image_score_map_index_dtype = np.dtype([
    ("li_idx", "u4"),       # Index into /label_images dataset
    ("data_start", "u8"),   # Start byte offset into /label_image_score_maps
    ("data_end", "u8"),     # End byte offset into /label_image_score_maps
    ("height", "u4"),       # Score map height in pixels
    ("width", "u4"),        # Score map width in pixels
])
```

### Data Format

Score map pixel data is stored as `float32` arrays, compressed with zlib, and packed into a single `uint8` byte array. Each score map's compressed bytes are located at the byte range `[data_start, data_end)` in the corresponding packed dataset.

### Optional Datasets

The score map datasets are only written when at least one predicted mask or label image has a `score_map` set. On read, missing datasets are silently skipped.

## Label Images

[`LabelImage`][sleap_io.LabelImage]s store dense per-pixel instance segmentation data, where each pixel is assigned an integer label corresponding to an object. Label image support was introduced in format 1.8.

### Label Image Datasets

Label image data is stored across three datasets:

- `/label_images`: Structured array containing label image metadata
- `/label_image_objects`: Structured array containing per-object metadata
- `/label_image_data`: Packed `uint8` array of zlib-compressed pixel data

### Label Image Dtype

```python
label_image_dtype = np.dtype([
    ("video", "i4"),          # Video index (-1 if none)
    ("frame_idx", "i8"),      # Frame index
    ("height", "u4"),         # Image height in pixels
    ("width", "u4"),          # Image width in pixels
    ("n_objects", "u4"),      # Number of objects in this label image
    ("objects_start", "u4"),  # Start index into /label_image_objects
    ("data_start", "u8"),     # Start byte offset into /label_image_data
    ("data_end", "u8"),       # End byte offset into /label_image_data
    ("is_predicted", "u1"),   # 0 = UserLabelImage, 1 = Predicted (Format 1.9+)
    ("score", "f4"),          # Confidence score (NaN for user) (Format 1.9+)
])
```

### Objects Dtype

Each object within a label image is described by a row in `/label_image_objects`:

```python
label_image_object_dtype = np.dtype([
    ("label_id", "i4"),    # Pixel label value in the image data
    ("track", "i4"),       # Track index (-1 if none)
    ("instance", "i4"),    # Instance index (-1 if none)
    ("score", "f4"),       # Per-object confidence score (Format 1.9+)
])
```

### Pixel Data

The pixel data for each label image is stored as an `int32` array, compressed with zlib, and packed into `/label_image_data` as `uint8` bytes. Each label image's compressed bytes are at the byte range `[data_start, data_end)`.

### String Metadata

Format 1.9+ stores label image string metadata as vlen HDF5 string datasets:

- `/label_image_sources`: One source string per label image
- `/label_image_obj_categories`: One category string per object
- `/label_image_obj_names`: One name string per object

### Optional Datasets

The label image datasets are only written when the [`Labels`][sleap_io.Labels] object contains label images. On read, missing datasets default to empty lists.

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

### Format 1.7

**Added bounding box support.**

- New dataset: `/bboxes` for first-class bounding box annotations
- Supports axis-aligned and oriented (rotated) bounding boxes
- User/predicted distinction via `is_predicted` flag and `score` field
- Migration on read: rectangular ROIs from older files are automatically converted to [`BoundingBox`][sleap_io.BoundingBox] objects
- `annotation_type` and `score` fields on `/rois` and `/masks` are now legacy (always written as constants)

### Format 1.8

**Added label image support.**

- New datasets: `/label_images`, `/label_image_objects`, `/label_image_data` for per-pixel segmentation annotations
- First-class `LabelImage` type for instance segmentation workflows

### Format 1.9

**Added identity, Instance3D, and predicted variant support.**

- New dataset: `/identities_json` for ground-truth animal identities
- Extended `InstanceGroup` serialization with `identity_idx`, `instance_3d_score`, and `instance_3d_point_scores` fields
- `Identity` objects persist across sessions and videos, distinct from per-video `Track`s
- `Instance3D` and `PredictedInstance3D` provide structured 3D keypoint storage
- Added `is_predicted` (`u1`) and updated `score` fields to ROI, mask, and label image dtypes for predicted variant support (`PredictedROI`, `PredictedSegmentationMask`, `PredictedLabelImage`)
- Added `instance` (`i4`) field to mask dtype for mask-instance association
- Migrated ROI and mask string metadata from JSON attributes to vlen HDF5 string datasets (`/roi_categories`, `/roi_names`, `/roi_sources`, `/mask_categories`, `/mask_names`, `/mask_sources`)
- Added label image string metadata datasets (`/label_image_sources`, `/label_image_obj_categories`, `/label_image_obj_names`)
- Added score map datasets (`/mask_score_map_index`, `/mask_score_maps`, `/label_image_score_map_index`, `/label_image_score_maps`)
- Backward compatible: new fields are optional, old readers skip unknown keys via metadata pass-through

### Format 2.0 (Current)

**Columnar bounding box storage.**

- `/bboxes` changed from a structured array dataset to an HDF5 group with columnar datasets
- Bounding box coordinates use `x1`/`y1`/`x2`/`y2` (top-left/bottom-right) representation instead of `x_center`/`y_center`/`width`/`height`
- String metadata (`category`, `name`, `source`) stored as vlen string datasets within the group
- The reader auto-detects whether `/bboxes` is a group (format 2.0+) or a dataset (legacy 1.7--1.9) and handles both transparently

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
