# Examples

This page provides practical examples for common tasks with sleap-io. Each example includes working code that you can copy and adapt for your needs.

!!! info "Prerequisites"

    All examples assume you have sleap-io installed. See the **[Installation Guide](install.md)** for options including `uv`, `pip`, and development setup.

    Quick start with [`uv`](https://docs.astral.sh/uv/) (recommended):
    ```bash
    # Run any example script directly (no install needed)
    uv run --with sleap-io example.py
    ```

    This automatically handles dependencies without needing to manage environments.

    Most examples use `import sleap_io as sio` for brevity.

## Basics

### Create labels from raw data

Build a complete [`Labels`][sleap_io.Labels] dataset programmatically from a [`Skeleton`][sleap_io.Skeleton], [`Video`][sleap_io.Video], and [`Instance`][sleap_io.Instance] objects.

```python title="create_labels.py" linenums="1"
import sleap_io as sio
import numpy as np

# Create skeleton
skeleton = sio.Skeleton(
    nodes=["head", "thorax", "abdomen"],
    edges=[("head", "thorax"), ("thorax", "abdomen")]
)

# Create video
video = sio.load_video("test.mp4")

# Create instance from numpy array
instance = sio.Instance.from_numpy(
    points=np.array([
        [10.2, 20.4],  # head
        [5.8, 15.1],   # thorax
        [0.3, 10.6],   # abdomen
    ]),
    skeleton=skeleton
)

# Create labeled frame
lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])

# Create labels
labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

# Save
labels.save("labels.slp")
```

??? tip "Creating predicted instances"
    To create predictions with confidence scores, include scores as the third
    column of the points array:
    ```python
    predicted = sio.PredictedInstance.from_numpy(
        np.array([
            [10.2, 20.4, 0.9],   # head (x, y, score)
            [5.8, 15.1, 0.8],    # thorax
            [0.3, 10.6, 0.7],    # abdomen
        ]),
        skeleton=skeleton,
        score=0.85,  # instance-level confidence
    )
    ```

!!! note "See also"
    - [Model](model/index.md): Complete data model documentation
    - [`Labels`](model/labels.md#sleap_io.Labels): Labels container class
    - [`Instance`](model/poses.md#sleap_io.Instance): Instance class for manual annotations
    - [`PredictedInstance`](model/poses.md#sleap_io.PredictedInstance): Instance class for predictions

### Convert labels to raw arrays

Extract pose data as NumPy arrays for analysis or visualization using [`Labels.numpy()`][sleap_io.Labels.numpy].

```python title="labels_to_numpy.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("tests/data/slp/centered_pair_predictions.slp")

# Convert predictions to point coordinates in a single array
trx = labels.numpy()
n_frames, n_tracks, n_nodes, xy = trx.shape
assert xy == 2  # x and y coordinates

# Convert to array with confidence scores appended
trx_with_scores = labels.numpy(return_confidence=True)
n_frames, n_tracks, n_nodes, xy_score = trx_with_scores.shape 
assert xy_score == 3  # x, y, and confidence score
```

??? example "Expected output shapes"
    For a dataset with 100 frames, 2 tracks, and 3 nodes:
    
    - Without scores: `(100, 2, 3, 2)` 
    - With scores: `(100, 2, 3, 3)`

!!! note "See also"
    [`Labels.numpy`](model/labels.md#sleap_io.Labels.numpy): Full documentation of array conversion options

## Annotation types

Each new v0.7.0 annotation type (bounding boxes, centroids, identities, and 3-D instances) has a `User*` variant for ground-truth annotations and a `Predicted*` variant carrying confidence scores. All of them live inside [`LabeledFrame`][sleap_io.LabeledFrame] and are appended with the same type-dispatching [`LabeledFrame.append`][sleap_io.LabeledFrame.append].

### Bounding box detections

Create, inspect, and convert `BoundingBox` annotations for detection and tracking workflows.

```python title="bbox_basics.py" linenums="1"
import sleap_io as sio

# Construct from corners or from top-left + size
bbox = sio.UserBoundingBox(x1=10, y1=20, x2=100, y2=200, category="mouse")
bbox = sio.UserBoundingBox.from_xywh(10, 20, 90, 180)  # same box

# Query geometric properties
bbox.xyxy         # (10.0, 20.0, 100.0, 200.0)
bbox.xywh         # (10.0, 20.0, 90.0, 180.0)
bbox.area         # 16200.0
bbox.centroid_xy  # (55.0, 110.0)

# Predictions carry a score (and optional tracking_score)
pred_bbox = sio.PredictedBoundingBox(
    x1=10, y1=20, x2=100, y2=200, score=0.95, tracking_score=0.88,
)

# Convert to related annotation types
roi = bbox.to_roi()
mask = bbox.to_mask(height=480, width=640)
```

Attach a box to a frame and save it into SLP (format v2.0+) — bboxes also round-trip through COCO detection, Ultralytics detection, GeoJSON, and JABS when the dataset carries the metadata those formats require.

```python title="bbox_save.py" linenums="1"
import sleap_io as sio

video = sio.Video.from_filename("clip.mp4", open=False)
lf = sio.LabeledFrame(video=video, frame_idx=0)
lf.append(sio.UserBoundingBox(x1=10, y1=20, x2=100, y2=200, category="mouse"))

labels = sio.Labels(labeled_frames=[lf], videos=[video])
labels.save("boxes.slp")
```

!!! note "See also"
    - [Regions: Bounding boxes](model/regions.md#bounding-boxes) — full API and metadata fields
    - [Formats: COCO](formats/index.md#coco-format-json) and [Ultralytics](formats/index.md#ultralytics-yolo-format) for detection round-trips

### Centroid tracking and TrackMate import

`Centroid` annotations are lightweight `(x, y)` points used by TrackMate, SORT, ByteTrack, and similar trackers. They interconvert with full pose [`Instance`][sleap_io.Instance] objects.

```python title="centroid_basics.py" linenums="1"
import sleap_io as sio

track = sio.Track(name="mouse_A")

# User and predicted variants
c = sio.UserCentroid(x=100.5, y=200.3, track=track)
p = sio.PredictedCentroid(x=50.0, y=60.0, score=0.95, source="trackmate")

# Convert a pose Instance into a single-point centroid
centroid = sio.Centroid.from_instance(pose_instance, method="center_of_mass")

# And back: produces a single-node Instance on CENTROID_SKELETON
inst = centroid.to_instance()

# Convenience accessors on full pose Instances
xy = pose_instance.centroid_xy        # (x, y) tuple
c2 = pose_instance.to_centroid()      # shortcut for Centroid.from_instance(pose_instance)
```

Import TrackMate `*_spots.csv` exports straight into a `Labels` object — either with the explicit loader or by letting `sio.load_file` auto-detect the format from the CSV header:

```python title="load_trackmate.py" linenums="1"
import sleap_io as sio

# Explicit loader (use when you also want to pass edges or other options)
labels = sio.load_trackmate("spots.csv", video="experiment.tif")

# Auto-detect from CSV headers (falls back to DLC and generic CSV readers)
labels = sio.load_file("spots.csv")

print(len(labels.centroids), "centroid detections")
```

!!! note "See also"
    - [Regions: Centroids](model/regions.md#centroids) — full API
    - [Formats: TrackMate](formats/trackmate.md) — spots/edges/tracks CSV layout

### Cross-session identity and 3-D instances

[`Identity`][sleap_io.Identity] identifies the same animal across multiple recording sessions, distinct from per-video [`Track`][sleap_io.Track]. [`Instance3D`][sleap_io.Instance3D] / [`PredictedInstance3D`][sleap_io.PredictedInstance3D] carry structured triangulated keypoints, and [`InstanceGroup`][sleap_io.InstanceGroup] ties multi-view 2-D instances together with their 3-D reconstruction.

```python title="identity_and_3d.py" linenums="1"
import numpy as np
import sleap_io as sio

skel = sio.Skeleton(nodes=["head", "tail"], edges=[("head", "tail")])

# Persistent per-animal identity (color is free-form and optional)
mouse_a = sio.Identity(name="mouse_A", color="#e6194b")

# 3-D keypoint storage, aligned to a skeleton
pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
inst_3d = sio.Instance3D(points=pts, skeleton=skel)
pred_3d = sio.PredictedInstance3D(
    points=pts, skeleton=skel, point_scores=np.array([0.9, 0.8]),
)

# Group per-camera 2-D detections with the reconstructed 3-D instance
cam1 = sio.Camera(name="cam1")
cam2 = sio.Camera(name="cam2")
inst_2d_cam1 = sio.Instance.from_numpy(np.zeros((2, 2)), skeleton=skel)
inst_2d_cam2 = sio.Instance.from_numpy(np.zeros((2, 2)), skeleton=skel)
group = sio.InstanceGroup(
    instance_by_camera={cam1: inst_2d_cam1, cam2: inst_2d_cam2},
    instance_3d=inst_3d,
    identity=mouse_a,
)

# Identities and recording sessions hang off Labels directly
labels = sio.Labels(identities=[mouse_a])
```

Identities and `Instance3D` both persist into SLP format v1.9+, so round-tripping between sleap-io and sleap-io.js / luc3d is lossless.

!!! note "See also"
    [3D model](model/3d.md): `Camera`, `CameraGroup`, `RecordingSession`, `FrameGroup`, `InstanceGroup`, `Identity`, `Instance3D`.

### Reading and writing GeoJSON ROIs

ROIs are serializable to the `movement`-compatible GeoJSON format used by Shapely, GeoPandas, QGIS, and QuPath. Each ROI also implements the Python [`__geo_interface__`](https://gist.github.com/sgillies/2217756) protocol so it plugs straight into any `geopandas.GeoDataFrame`.

```python title="geojson_rois.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("labels.slp")

# Dump every ROI (static and frame-level) to GeoJSON
sio.save_geojson(labels.rois, "regions.geojson")

# Load an externally drawn region set
rois = sio.load_geojson("regions.geojson")
first = rois[0]
first.__geo_interface__   # standard GeoJSON Feature dict

# Universal loader auto-detects the .geojson extension
labels = sio.load_file("regions.geojson")
```

!!! note "See also"
    - [Formats: GeoJSON](formats/index.md#geojson-format-geojson) — schema and movement-library interop
    - [Regions: Static vs. temporal ROIs](model/regions.md#static-vs-temporal-rois)

### Adding annotations to frames (type dispatch)

`LabeledFrame.append()` routes each annotation to the correct per-type list based on its runtime type. The same method handles every annotation kind — you never have to touch `lf.bboxes`, `lf.centroids`, etc. by hand for appends.

```python title="frame_append_dispatch.py" linenums="1"
import numpy as np
import sleap_io as sio
from shapely.geometry import box

skel = sio.Skeleton(nodes=["head"], edges=[])
video = sio.Video.from_filename("clip.mp4", open=False)
lf = sio.LabeledFrame(video=video, frame_idx=0)

# Each append() routes by type:
lf.append(sio.Instance.from_numpy(np.zeros((1, 2)), skeleton=skel))  # → lf.instances
lf.append(sio.UserBoundingBox(x1=0, y1=0, x2=10, y2=10))             # → lf.bboxes
lf.append(sio.UserCentroid(x=5, y=5))                                # → lf.centroids
lf.append(sio.UserSegmentationMask.from_numpy(np.zeros((8, 8), bool)))  # → lf.masks
lf.append(sio.UserLabelImage.from_numpy(np.zeros((8, 8), int)))      # → lf.label_images
lf.append(sio.UserROI(geometry=box(0, 0, 10, 10)))                   # → lf.rois

# Per-frame accessors are plain lists you can iterate or index:
print(len(lf.instances), len(lf.bboxes), len(lf.centroids))
print(len(lf.masks), len(lf.label_images), len(lf.rois))

# Flat read-only views across the whole Labels object
labels = sio.Labels(labeled_frames=[lf], videos=[video])
labels.bboxes      # every bbox across every frame
labels.centroids   # every centroid across every frame
labels.static_rois # video-level ROIs (separate from frame-level lf.rois)
```

!!! note "See also"
    [Regions → Working with annotations in frames](model/regions.md#working-with-annotations-in-frames) for more context on the flat-vs-nested views.

### Fast O(1) frame and track lookups

[`Labels.get_frame`][sleap_io.Labels.get_frame] and [`Labels.get_track_annotations`][sleap_io.Labels.get_track_annotations] replace O(n) linear scans with O(1) dictionary lookups backed by lazy indices. Use them when iterating over large datasets or cross-referencing by frame / track.

```python title="fast_lookups.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
video = labels.videos[0]
track = labels.tracks[0]

# O(1) frame lookup (returns None if the frame is missing)
lf = labels.get_frame(video, frame_idx=42)
if lf is not None:
    print(len(lf.instances), "instances on frame 42")

# O(1) per-track annotation view (sorted by frame_idx)
annotations = labels.get_track_annotations(video, track)
for ann in annotations[:5]:
    print(ann.frame_idx, type(ann).__name__)

# Existing convenience methods now use the same indices internally
labels.find(video, frame_idx=42)                    # O(1)
labels.get_centroids(video=video, frame_idx=0)      # O(1)
labels.get_bboxes(video=video, frame_idx=0)         # O(1)

# If you mutate labels.labeled_frames in place, force a rebuild
labels.labeled_frames.append(sio.LabeledFrame(video=video, frame_idx=99))
labels.reindex()
```

!!! tip "When to call `reindex()`"
    The indices rebuild automatically when you use the public `Labels` APIs. You only need `reindex()` after mutating `labels.labeled_frames` or the annotation lists in place (which bypasses the invalidation hooks).

### Resolving a video by path or foreign instance

`Video` objects compare by identity, so a `Video` you create yourself (e.g. with `sio.load_video`) is *not* recognized by `Labels` lookups even if it points at the same file as one already in the project. [`Labels.match_video`][sleap_io.Labels.match_video] resolves a foreign `Video` — or a plain filename — to the canonical instance, and `find`, `extract`, `__getitem__`, `numpy`, and the `get_*` family all canonicalize their `video` argument through it automatically.

```python title="resolve_video.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# A freshly loaded Video is a different object than the one in the project.
foreign = sio.load_video("predictions_video.mp4")

# These all work now (previously returned [] / raised IndexError):
labels.find(foreign)                      # by foreign Video instance
labels.find("predictions_video.mp4")      # by filename
labels["predictions_video.mp4"]           # __getitem__ by filename
labels.extract([("predictions_video.mp4", 0)])

# Resolve explicitly to the canonical Video stored on the project:
canonical = labels.match_video(foreign)   # -> Video, or None if no match
```

!!! note "Matching strategy"
    `match_video` uses a tiered cascade by default: a definitive match (same underlying file, or identical path) wins, and only if none is found does it fall back to basename matching. Ambiguous matches raise `ValueError`. Pass `method=` (`"path"`, `"basename"`, `"content"`, ...) for explicit control.

## Format conversion

### Load and save in different formats

Convert between supported formats with automatic format detection using [`load_file()`][sleap_io.load_file] and [`Labels.save()`][sleap_io.Labels.save].

```python title="format_conversion.py" linenums="1"
import sleap_io as sio

# Load from SLEAP file
labels = sio.load_file("predictions.slp")

# Save to NWB file
labels.save("predictions.nwb")
```

!!! tip
    sleap-io automatically detects the format from the file extension. Supported formats include `.slp`, `.nwb`, `.json` (COCO/Label Studio), `.labelstudio.json`, `.h5` (JABS), and `.mat` (LEAP). Use `format="coco"` to explicitly save as COCO format.

!!! note "See also"
    - [`Labels.save`](model/labels.md#sleap_io.Labels.save): Save method with format options
    - [Formats](formats/): Complete list of supported formats

### Working with NWB files

[Neurodata Without Borders (NWB)](https://www.nwb.org/) provides a standardized format for neurophysiology data. sleap-io offers comprehensive NWB support with automatic format detection.

```python title="nwb_basic.py" linenums="1"
import sleap_io as sio

# Load any NWB file - automatically detects if it contains
# annotations (PoseTraining) or predictions (PoseEstimation)
labels = sio.load_nwb("pose_data.nwb")

# Save with automatic format detection
# Uses "annotations" if data has user labels, "predictions" otherwise
sio.save_nwb(labels, "output.nwb")

# Force specific format
sio.save_nwb(labels, "training.nwb", nwb_format="annotations")
sio.save_nwb(labels, "inference.nwb", nwb_format="predictions")

# Export with embedded video frames for sharing complete datasets
sio.save_nwb(labels, "dataset_export.nwb", nwb_format="annotations_export")
```

!!! info "Format auto-detection"
    The harmonization layer automatically determines the appropriate format:

    - **Annotations**: Used when data contains user-labeled instances (training data)
    - **Predictions**: Used when data contains only predicted instances (inference results)
    - **Annotations Export**: Use explicitly to create self-contained files with embedded video frames

### Save training data with rich metadata

Include detailed experimental metadata when saving training annotations.

```python title="nwb_metadata.py" linenums="1"
import sleap_io as sio

labels = sio.load_file("labels.slp")

# Basic save with default metadata
sio.save_nwb(labels, "training_data.nwb", nwb_format="annotations")
```

??? tip "Advanced: custom NWB metadata (internal API)"
    The public `save_nwb()` uses default metadata. To customize session
    descriptions, subject info, and other NWB fields, use the internal
    `save_labels()` function. This is not part of the public API and may
    change between versions.

    ```python
    from sleap_io.io.nwb_annotations import save_labels

    save_labels(
        labels,
        "training_data.nwb",
        session_description="Mouse skilled reaching task - training dataset",
        identifier="mouse_01_session_03_annotations",
        session_start_time="2024-01-15T09:30:00",
        annotator="John Doe",
        nwb_kwargs={
            "session_id": "session_003",
            "experimenter": ["John Doe", "Jane Smith"],
            "lab": "Motor Control Lab",
            "institution": "University of Example",
            "experiment_description": "Skilled reaching task with food pellet reward",
            "subject": {
                "subject_id": "mouse_01",
                "age": "P90",
                "sex": "M",
                "species": "Mus musculus",
                "strain": "C57BL/6J",
                "weight": "25g",
            },
        },
    )
    ```

!!! tip "Metadata best practices"
    Include as much metadata as possible for reproducibility:

    - Experimental protocol details
    - Subject information
    - Recording conditions
    - Annotator identity for tracking labeling provenance

### Export dataset with embedded videos

Create self-contained NWB files with video frames for sharing complete datasets.
The simplest approach uses the public API:

```python title="nwb_export.py" linenums="1"
import sleap_io as sio

labels = sio.load_file("labels.slp")

# Export annotations with embedded video frames to NWB
sio.save_nwb(labels, "dataset_export.nwb", nwb_format="annotations_export")
```

??? tip "Advanced: fine-grained export control"
    For more control over the export process (e.g., custom filenames, multi-subject
    support), use the internal export functions directly. These are not part of the
    public API and may change between versions.

    ```python
    from sleap_io.io.nwb_annotations import export_labels, export_labeled_frames

    # Export annotations + MJPEG video + frame map to a directory
    export_labels(
        labels,
        output_dir="export/",
        mjpeg_filename="annotated_frames.avi",
        frame_map_filename="frame_map.json",
        nwb_filename="pose_training.nwb",
        clean=True,  # Remove empty frames and predictions before export
    )

    # Or export just labeled frames with provenance tracking
    frame_map = export_labeled_frames(
        labels,
        frame_map_path="export/frame_map.json",
        mjpeg_path="export/labeled_frames.avi",
        nwb_path="export/labeled_frames.nwb",
    )
    print(f"Exported {frame_map.total_frames} frames")
    ```

!!! info "Export contents"

    - **NWB file**: Annotations in PoseTraining format with skeleton definition
    - **MJPEG video**: Labeled frames re-encoded as a seekable MJPEG video
    - **Frame map JSON**: Provenance metadata tracking which frames came from which source videos

### Convert between NWB and other formats

Use NWB as an interchange format between different pose tracking tools.

```python title="nwb_conversion.py" linenums="1"
import sleap_io as sio

# Load from DeepLabCut
dlc_data = sio.load_file("dlc_predictions.h5")

# Save as NWB predictions
sio.save_nwb(dlc_data, "dlc_in_nwb.nwb", nwb_format="predictions")

# Load SLEAP training data
sleap_labels = sio.load_file("training.slp")

# Export as NWB with videos for sharing
sio.save_nwb(sleap_labels, "training_export.nwb", nwb_format="annotations_export")

# Convert NWB back to SLEAP format
nwb_labels = sio.load_nwb("training_export.nwb")
nwb_labels.save("converted.slp")
```

!!! tip "Format preservation"
    NWB format preserves:

    - Complete skeleton structure with node names
    - Track identities
    - Confidence scores
    - User vs predicted instance types
    - Video metadata (when using `annotations_export`)

!!! note "See also"
    - [NWB Format Documentation](formats/#nwb-format-nwb): Complete NWB format reference
    - [`load_nwb`](formats/#sleap_io.load_nwb): NWB loading function
    - [`save_nwb`](formats/#sleap_io.save_nwb): NWB saving function with format options

### Convert to Ultralytics YOLO format

Export your dataset for use with Ultralytics YOLO models.

```python title="ultralytics_export.py" linenums="1"
import sleap_io as sio

# Load source labels
labels = sio.load_file("labels.v001.slp")

# Create train/val splits and export as YOLO format
labels_set = labels.make_training_splits(n_train=0.8, n_val=0.2, seed=42)
labels_set.save("yolo_dataset/", format="ultralytics")

# Or export from existing split files
file_dict = {
    "train": "path/to/train.slp",
    "val": "path/to/val.slp",
}
labels_set = sio.load_labels_set(file_dict)
labels_set.save("yolo_dataset/", format="ultralytics")
```

!!! info "YOLO export structure"
    The exported dataset will have the standard YOLO directory structure with train/val splits, images, and label files.

!!! note "See also"
    - [`LabelsSet`](model/labels.md#sleap_io.LabelsSet): LabelsSet class documentation
    - [`load_labels_set`](formats/#sleap_io.load_labels_set): Loading function for label sets

### Export to COCO format

Export your dataset for use with mmpose, CVAT, and other COCO-compatible tools.

```python title="coco_export.py" linenums="1"
import sleap_io as sio

# Load source labels
labels = sio.load_file("labels.slp")

# Export to COCO format
sio.save_coco(labels, "annotations.json")

# Or use save_file with auto-detection from .json extension
labels.save("annotations.json", format="coco")

# Customize export with options
sio.save_coco(
    labels,
    "annotations_binary.json",
    visibility_encoding="binary",  # Use binary (0/1) instead of ternary (0/1/2)
    image_filenames=["frame_001.jpg", "frame_002.jpg", ...]  # Custom filenames
)
```

!!! info "mmpose compatibility"
    The COCO export is fully compatible with [mmpose](https://github.com/open-mmlab/mmpose) and includes:

    - Required `bbox` field computed from visible keypoints
    - `area` field for bounding box area
    - `iscrowd` field for standard compliance
    - Track IDs via `attributes.object_id` (CVAT-compatible)
    - 1-based skeleton edge indexing
    - Support for both binary (0/1) and ternary (0/1/2) visibility encodings

!!! tip "Use cases"
    COCO export is ideal for:

    - Training pose estimation models with mmpose
    - Annotating data in CVAT and importing to SLEAP
    - Sharing datasets with the broader computer vision community
    - Integration with COCO-compatible evaluation tools

!!! note "See also"
    - [`save_coco`](formats/#sleap_io.save_coco): Full COCO export documentation
    - [`load_coco`](formats/#sleap_io.load_coco): COCO import documentation
    - [COCO Format](formats/#coco-format-json): COCO format details

## Loading from URLs

Load `.slp` files directly from HTTP/HTTPS and cloud-storage URLs. Reads are
lazy and range-based: for the default streaming mode only the bytes actually
needed (typically a small fraction of the file) are pulled over the network.

### Quickstart

[`load_slp`][sleap_io.load_slp] (and the universal
[`load_file`][sleap_io.load_file]) accept a URL anywhere a local path is
accepted:

```python
import sleap_io as sio

# Just works for http/https out of the box
labels = sio.load_slp("https://example.com/path/labels.slp")

# load_file also accepts URLs (dispatches by extension)
labels = sio.load_file("https://example.com/path/labels.slp")
```

`pkg.slp` files with embedded frames work too — the embedded `HDF5Video`
backends reopen the remote file lazily when you read frames, reusing the same
streaming configuration.

!!! info "What is and isn't supported"
    URL loading currently covers `.slp`/`.pkg.slp` (including through the
    universal [`load_file`][sleap_io.load_file]) and remote *media video* over
    `http`/`https` (see [Remote video](#remote-video) below). Other *labels*
    formats (NWB, COCO, Label Studio, JABS, DLC, TrackMate, LEAP, GeoJSON,
    Ultralytics) are not yet implemented over URLs and raise
    `NotImplementedError`; download the file locally first. These are tracked
    as follow-ups.

### Remote video

[`load_video`][sleap_io.load_video] (and [`load_file`][sleap_io.load_file] for
video extensions) can read a media video directly from an `http`/`https` URL:

```python
import sleap_io as sio

# Reads frames lazily over the network; needs the [pyav] extra
video = sio.load_video("https://example.com/path/video.mp4")
frame = video[0]  # frames are decoded on demand
```

Supported container extensions are the same as for local media videos
(`mp4`, `avi`, `mov`, `mj2`, `mkv`). Only `http` and `https` URLs are accepted
for video — cloud schemes (`s3://`, `gs://`, …) are not supported for video
loading. The URL's query string and fragment are ignored when detecting the
extension, so pre-signed/tokenized URLs like
`https://host/video.mp4?token=...` route correctly.

Remote video requires the `pyav` extra, which is selected automatically as the
backend for URLs:

```bash
pip install "sleap-io[pyav]"   # remote video support (provides `av`)
```

If the `av` package is missing, `load_video(url)` raises an `ImportError` with
the install hint above.

!!! danger "Security: remote video hands untrusted data to FFmpeg"
    Decoding a remote video streams bytes from the URL into FFmpeg (via pyav).
    FFmpeg's demuxers and decoders are a large, historically
    vulnerability-prone attack surface (the CVE history for media parsers is
    extensive), so a malicious URL or stream can attempt to exploit the
    decoder running in your process.

    To limit risk, sleap-io only ever passes `http`/`https` URLs through to the
    decoder (no other schemes, no shell/protocol indirection). You should:

    - **Load remote video only from sources you trust.** Treat an arbitrary
      third-party URL the same as running untrusted code.
    - **Sandbox untrusted inputs.** If you must decode video from an untrusted
      source, do it in an isolated environment (container/VM with no
      credentials, restricted network, and a non-privileged user) and keep
      FFmpeg/pyav up to date.

### Google Drive

Google Drive file share links are recognized and resolved to a direct download
automatically, so you can pass a Drive URL straight to
[`load_slp`][sleap_io.load_slp] or [`load_file`][sleap_io.load_file]:

```python
import sleap_io as sio

# Any of these Drive share-link shapes work:
labels = sio.load_slp("https://drive.google.com/file/d/<FILE_ID>/view")
labels = sio.load_slp("https://drive.google.com/uc?id=<FILE_ID>&export=download")
labels = sio.load_slp("https://drive.google.com/open?id=<FILE_ID>")

# load_file resolves the Drive link, sniffs the content, and routes it:
labels = sio.load_file("https://drive.google.com/file/d/<FILE_ID>/view")
```

The file must be shared as **"Anyone with the link"** (no sign-in required).
Because Drive download links carry no file extension and reject the `HEAD`/range
requests that lazy streaming relies on, a Drive file is **fully downloaded into
memory** during resolution (the `stream_mode`/cache keyword arguments do not
apply). The two-hop "can't scan for viruses" confirmation page that Drive serves
for larger files is handled transparently.

Some limitations:

- **Folder links are not supported** — pass a single-file share link
  (`…/file/d/<FILE_ID>/view`), not a `…/drive/folders/<ID>` URL. A folder URL
  raises a `ValueError`.
- **Drive videos are not supported** — `load_video(<drive url>)` raises a
  `NotImplementedError`. Download the video file first, then load it locally.
- **Quota / permission errors** — if Drive returns its "too many users have
  viewed or downloaded this file recently" page, a `RemoteIOError` is raised;
  retry later or re-check the file's sharing settings.

For `load_file`, the format is detected from the downloaded bytes (which costs
one extra fetch). Pass an explicit `format=` (e.g. `format="slp"`) to skip the
detection download.

### Supported schemes and install matrix

| Scheme | Requires | Notes |
|--------|----------|-------|
| `http`, `https` | nothing extra | Works with a plain `pip install sleap-io` |
| `s3` | `sleap-io[cloud]` | Amazon S3 (via `s3fs`) |
| `gs`, `gcs` | `sleap-io[cloud]` | Google Cloud Storage (via `gcsfs`) |
| `az`, `abfs` | `sleap-io[cloud]` | Azure Blob / ADLS (via `adlfs`) |

```bash
pip install sleap-io               # http/https only
pip install "sleap-io[cloud]"      # + s3, gs/gcs, az/abfs
pip install "sleap-io[all]"        # everything (cloud schemes included)
```

```python
# Cloud scheme (requires sleap-io[cloud])
labels = sio.load_slp("s3://my-bucket/path/labels.slp")
```

!!! warning "Missing cloud extra"
    Using a cloud scheme without the `[cloud]` extra raises an `ImportError`
    whose message names the missing package and the
    `pip install 'sleap-io[cloud]'` install hint.

### Streaming modes

Control how bytes are fetched with the `stream_mode` keyword argument:

| `stream_mode` | Backing strategy | Memory | Disk cache | Revalidation | Best for |
|---------------|------------------|--------|------------|--------------|----------|
| `"auto"` (default) | fsspec `blockcache` | Low (LRU of `max_blocks`) | None | n/a | One-off lazy reads, low memory |
| `"blockcache"` | fsspec `blockcache` | Low | None | n/a | Same as `auto` (explicit) |
| `"cache"` | fsspec `simplecache` | Whole file on disk | Persistent | None | Repeated opens of an immutable file |
| `"filecache"` | fsspec `filecache` | Whole file on disk | Persistent | ETag / `Last-Modified` after `cache_expiry` | Repeated opens of a file that may change |
| `"download"` | Full read into memory | Whole file in RAM | None | n/a | Small files, ephemeral environments |

```python
# Default: lazy range reads, low memory
labels = sio.load_slp("https://example.com/labels.slp")

# Persistent on-disk cache with daily ETag revalidation
labels = sio.load_slp(
    "https://example.com/labels.slp",
    stream_mode="filecache",
    cache_storage="~/.cache/sleap-io",
    cache_expiry=86400,  # revalidate after a day
)

# Ephemeral full download into memory (no disk cache)
labels = sio.load_slp("https://example.com/labels.slp", stream_mode="download")
```

The `"auto"`/`"blockcache"` reads can be tuned with `block_size` (range block
size, default 1 MiB) and `max_blocks` (in-memory LRU cap per open file,
default 32 → ~32 MiB per file).

### Authentication

Pass HTTP headers (such as a bearer token) with `headers=`:

```python
labels = sio.load_slp(
    "https://my-org.example/private/labels.slp",
    headers={"Authorization": "Bearer <token>"},
)
```

!!! warning "Headers are stripped on cross-origin redirect"
    For security, sensitive headers (`Authorization`, `Cookie`,
    `Proxy-Authorization`) are **dropped automatically** if the request is
    redirected to a different origin (scheme/host/port). This prevents leaking
    credentials to a third-party host and is intentional — if a download
    redirects cross-origin (e.g. to a pre-signed CDN URL), put the credentials
    in the redirect target's query string rather than in `headers`.

    Cloud schemes (`s3://`, `gs://`, …) use their own per-provider credential
    chains (environment variables, credential files, instance metadata) rather
    than `headers=`.

### Caching: location, override, and clearing

For `"cache"` and `"filecache"` modes, downloaded files live in the directory
you pass as `cache_storage=`. sleap-io writes a small marker file there so it
can later identify and clean up only its own cache files.

To clear the cache, call [`clear_remote_cache`][sleap_io.clear_remote_cache]
with the **same** `cache_storage` you loaded with:

```python
import sleap_io as sio

# Delete every sleap-io cache file in the directory
sio.clear_remote_cache(cache_storage="~/.cache/sleap-io")

# Or only files older than an hour (3600 seconds)
sio.clear_remote_cache(cache_storage="~/.cache/sleap-io", older_than=3600)
```

!!! note "An explicit `cache_storage` is required to clear the cache"
    `clear_remote_cache` only operates on a directory that contains the
    sleap-io marker file, and it only deletes files matching fsspec's
    cache-key naming pattern — so it will never touch unrelated files even if
    you point it at a shared directory. It refuses to run on a directory with
    no marker, or on forbidden paths like `/` or `$HOME`. Because fsspec's
    *default* cache directory is a per-process temporary location that cannot
    be cleared reliably, you must pass the explicit `cache_storage` you used
    when loading.

### CI and ephemeral environments

In CI or other short-lived environments, prefer the default `stream_mode="auto"`
(no persistent cache to manage), or scope a per-run cache to a temporary
directory you control:

```python
import os
import sleap_io as sio

cache_dir = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "sleap-io-cache")
labels = sio.load_slp(url, stream_mode="filecache", cache_storage=cache_dir)
```

### Troubleshooting

- **`RemoteIOError`** — raised for HTTP-level failures (404 not found, 416 range
  past end of file, 5xx after retries, connection errors, timeouts). The
  exception carries a `status` (HTTP code or `None`) and a credential-redacted
  `url` so tokens never leak into logs or tracebacks. See
  [`RemoteIOError`][sleap_io.RemoteIOError].
- **`ImportError` for cloud schemes** — install the cloud adapters with
  `pip install 'sleap-io[cloud]'` (covers `s3`, `gs`/`gcs`, `az`/`abfs`).
- **`RuntimeWarning` about `aiohttp`** — remote loading needs
  `aiohttp >= 3.13.5` for safe cross-origin header stripping. If you see this
  warning, upgrade with `pip install --upgrade 'aiohttp>=3.13.5'`.
- **`ImportError` from `load_video(url)`** — remote video loading needs the
  `pyav` extra; install it with `pip install 'sleap-io[pyav]'`. Only `http`/
  `https` URLs are supported for video. See [Remote video](#remote-video) for
  the security considerations of decoding untrusted remote video.
- **Google Drive errors** — a `ValueError` means the link is a folder (pass a
  `…/file/d/<ID>/view` file link) or the file ID could not be parsed; a
  `RemoteIOError` mentioning a quota/permission page means the file is not shared
  publicly or Drive is rate-limiting downloads. See [Google Drive](#google-drive).

!!! note "See also"
    - [`load_slp`][sleap_io.load_slp]: Full URL keyword-argument reference
    - [`load_file`][sleap_io.load_file]: Universal loader with URL sniffing
    - [`load_video`][sleap_io.load_video]: Loads remote media video over http/https
    - [`clear_remote_cache`][sleap_io.clear_remote_cache]: Cache cleanup helper
    - [`RemoteIOError`][sleap_io.RemoteIOError]: Remote I/O error surface
    - [SLP Format](formats/slp.md): The on-disk `.slp` layout that URL loading streams

## Editing labels data

### Fix video paths

Update file paths when moving projects between systems.

```python title="fix_paths.py" linenums="1"
import sleap_io as sio

# Load labels without trying to open the video files
labels = sio.load_file("labels.v001.slp", open_videos=False)

# Fix paths using prefix replacement
labels.replace_filenames(prefix_map={
    "D:/data/sleap_projects": "/home/user/sleap_projects",
    "C:/Users/sleaper/Desktop/test": "/home/user/sleap_projects",
})

# Save labels with updated paths
labels.save("labels.v002.slp")
```

!!! warning "Path separators"
    The prefix map handles path separators automatically, but be consistent with forward slashes (`/`) for cross-platform compatibility.

!!! tip
    Use `open_videos=False` when loading to avoid errors from missing videos at the old paths.

!!! note "See also"
    [`Labels.replace_filenames`](model/labels.md#sleap_io.Labels.replace_filenames): Additional path manipulation options

### Copy labels

Create deep copies of labels with control over video backend behavior.

```python title="copy_labels.py" linenums="1"
import sleap_io as sio

labels = sio.load_file("labels.slp")

# Default: preserves each video's current open_backend setting
labels_copy = labels.copy()

# Prevent file handles from being created (useful for batch processing)
labels_copy = labels.copy(open_videos=False)

# Force all videos to auto-open when frames are accessed
labels_copy = labels.copy(open_videos=True)

# Filtering done separately (cleaner separation of concerns)
labels_copy = labels.copy()
labels_copy.remove_predictions()
labels_copy.suggestions = []
```

!!! tip "Non-mutating save"
    By default, save operations don't mutate the original `Labels` object:
    ```python
    # Original labels are NOT modified (safer)
    labels.save("output.pkg.slp", embed="user")
    assert labels.videos[0].filename != "output.pkg.slp"  # Still points to original

    # With embed_inplace=True: original labels ARE modified (faster)
    labels.save("output.pkg.slp", embed="user", embed_inplace=True)
    ```

!!! note "See also"
    [`Labels.copy`](model/labels.md#sleap_io.Labels.copy): Full documentation of copy options

### Replace skeleton

Change the skeleton structure while preserving existing annotations.

```python title="replace_skeleton.py" linenums="1"
import sleap_io as sio

# Load existing labels with skeleton nodes: "head", "trunk", "tti"
labels = sio.load_file("labels.slp")

# Create a new skeleton with different nodes
new_skeleton = sio.Skeleton(["HEAD", "CENTROID", "TAIL_BASE", "TAIL_TIP"])

# Replace skeleton with node correspondence mapping
labels.replace_skeleton(
    new_skeleton,
    node_map={
        "head": "HEAD",
        "trunk": "CENTROID",
        "tti": "TAIL_BASE"
        # "TAIL_TIP" will have NaN values since there's no correspondence
    }
)

# Save with the new skeleton format
labels.save("labels_with_new_skeleton.slp")
```

!!! warning
    Nodes without correspondence in the `node_map` will have NaN values in the resulting instances.

!!! tip
    This is particularly useful when converting between different annotation tools or skeleton conventions.

!!! note "See also"
    [`Labels.replace_skeleton`](model/labels.md#sleap_io.Labels.replace_skeleton): Additional skeleton manipulation options

### Update from numpy

Work with pose data as NumPy arrays for filtering or analysis.

```python title="numpy_filtering.py" linenums="1"
import sleap_io as sio
import numpy as np

labels = sio.load_file("predictions.slp")

# Convert to array — shape: (n_frames, n_tracks, n_nodes, 2)
# In v0.7.0, n_frames == len(video); frames past the last labeled frame are NaN.
trx = labels.numpy()

# Apply temporal smoothing along the frame axis (axis=0)
# This moving average operates independently per track/node/coordinate
kernel = np.ones(5) / 5
trx_smoothed = np.apply_along_axis(
    lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=trx
)

# Update the labels with smoothed coordinates
labels.update_from_numpy(trx_smoothed)

# Save the smoothed version
labels.save("predictions.smoothed.slp")
```

!!! tip "Advanced filtering with movement"
    For more sophisticated temporal filtering (Kalman, median, Savitzky-Golay),
    check out the [`movement`](https://movement.neuroinformatics.dev/) library
    which provides purpose-built tools for pose trajectory processing.

!!! warning
    When updating from numpy, the array shape must match the original data structure exactly.

!!! note "See also"
    - [`Labels.numpy`](model/labels.md#sleap_io.Labels.numpy): Array conversion options
    - [`Labels.update_from_numpy`](model/labels.md#sleap_io.Labels.update_from_numpy): Updating labels from arrays
    - [`movement`](https://movement.neuroinformatics.dev/): Advanced pose processing library

## Exporting labels

### Save labels with embedded images

Create self-contained label files with embedded video frames.

```python title="embed_images.py" linenums="1"
import sleap_io as sio

# Load source labels
labels = sio.load_file("labels.v001.slp")

# Save with embedded images for frames with user labeled data and suggested frames
labels.save("labels.v001.pkg.slp", embed="user+suggestions")
```

!!! info "Embedding options"

    - `"user"`: Only frames with manual annotations
    - `"user+suggestions"`: Manual annotations plus suggested frames
    - `"all"`: All frames with any labels (including predictions)
    - `"source"`: Embed source video if labels were loaded from embedded data

!!! note "See also"
    [`Labels.save`](model/labels.md#sleap_io.Labels.save): Complete save options including embedding

### Advanced embedding options

**Progress callback for GUI integration:**

```python title="embed_with_progress.py" linenums="1"
import sleap_io as sio

labels = sio.load_file("labels.slp")

def on_progress(current, total):
    print(f"Embedding frame {current}/{total}")
    return True  # Return False to cancel

labels.save("labels.pkg.slp", embed="user", progress_callback=on_progress)
```

**Cancellation support** (e.g., for GUI integration where a user can click "Cancel"):

```python title="embed_with_cancel.py" linenums="1"
import sleap_io as sio
from sleap_io.io.slp import ExportCancelled

labels = sio.load_file("labels.slp")

cancelled = False  # Set to True from another thread/signal to cancel

def on_progress(current, total):
    return not cancelled  # Return False to cancel the export

try:
    labels.save("output.pkg.slp", embed="user", progress_callback=on_progress)
except ExportCancelled:
    print("Export was cancelled")
```

**Control video embedding for videos without labels:**

```python title="embed_control.py" linenums="1"
# Default: all videos converted to embedded references (portable)
labels.save("output.pkg.slp", embed="user")

# Selective: only embed specific frames, keep other videos as external paths
labels.save("output.pkg.slp", embed="user", embed_all_videos=False)
```

### Trim labels and video

Extract a subset of frames with corresponding labels.

```python title="trim_video.py" linenums="1"
import sleap_io as sio

# Load existing data
labels = sio.load_file("labels.slp")

# Create a new labels file with frames 1000-2000 from video 0
clip = labels.trim("clip.slp", list(range(1_000, 2_000)), video=0)

# The new file contains:
# - A trimmed video saved as "clip.mp4"
# - Labels with adjusted frame indices
```

!!! tip
    The `trim` method automatically:

    - Creates a new video file with only the specified frames
    - Adjusts frame indices in the labels to match the new video
    - Preserves all instance data and tracks

!!! note "See also"
    [`Labels.trim`](model/labels.md#sleap_io.Labels.trim): Full trim method documentation

### Make training/validation/test splits

Split your dataset for machine learning workflows.

```python title="make_splits.py" linenums="1"
import sleap_io as sio

# Load source labels
labels = sio.load_file("labels.v001.slp")

# Make splits and export with embedded images
labels.make_training_splits(
    n_train=0.8,
    n_val=0.1,
    n_test=0.1,
    save_dir="split1",
    seed=42
)

# Splits are saved as self-contained SLP package files
labels_train = sio.load_file("split1/train.pkg.slp")
labels_val = sio.load_file("split1/val.pkg.slp")
labels_test = sio.load_file("split1/test.pkg.slp")

# Or get splits as a LabelsSet for programmatic access
labels_set = labels.make_training_splits(n_train=0.8, n_val=0.1, n_test=0.1)
train_labels = labels_set["train"]
val_labels = labels_set["val"]
test_labels = labels_set["test"]
```

!!! info
    The `.pkg.slp` extension indicates a self-contained package with embedded images, making the splits portable and shareable.

!!! note "See also"
    - [`Labels.make_training_splits`](model/labels.md#sleap_io.Labels.make_training_splits): Full documentation of splitting options
    - [`LabelsSet`](model/labels.md#sleap_io.LabelsSet): LabelsSet class for working with split datasets

## Video operations

### Read video data

Load and access video frames directly.

```python title="read_video.py" linenums="1"
import sleap_io as sio

video = sio.load_video("test.mp4")
n_frames, height, width, channels = video.shape

frame = video[0]  # Get first frame
height, width, channels = frame.shape

# Access specific frames
middle_frame = video[n_frames // 2]
last_frame = video[-1]
```

!!! info
    Video backends are optional. Install the backend you need:

    ```bash
    # imageio-ffmpeg is included by default
    ```

    To check which backends are available or get installation help:

    ```python
    import sleap_io as sio
    print(sio.get_available_video_backends())
    print(sio.get_installation_instructions())
    ```

!!! note "See also"
    - [`sio.load_video`](formats/#sleap_io.load_video): Video loading function
    - [`Video`](model/video.md#sleap_io.Video): Video class documentation

### Norpix `.seq` video

Load Norpix StreamPix `.seq` files the same way as any other video. The Norpix backend exposes per-frame timestamps and an auto-computed FPS derived from the timestamp stream, which is handy for high-speed behavioral recordings where the header FPS is often wrong.

```python title="read_seq.py" linenums="1"
import sleap_io as sio

video = sio.load_video("recording.seq")
frame = video[0]

# Inspect backend-specific metadata
seq = video.backend
print(seq.fps)              # auto-computed from per-frame timestamps
print(seq.get_timestamps()[0])  # seconds-since-epoch of the first frame
```

Supported codecs: raw mono, raw RGB, JPEG, and PNG. The `sio show` CLI prints Norpix-specific header info (codec, bit depth, description, FPS), and `sio reencode recording.seq -o recording.mp4` converts to MP4 via the Python path.

!!! note "See also"
    [Video backends](model/video.md#backends) — how `SeqVideo` fits alongside `MediaVideo`, `HDF5Video`, `ImageVideo`, and `TiffVideo`.

### Re-encode video

Fix video seeking issues by re-encoding with optimal settings.

```python title="reencode_video.py" linenums="1"
import sleap_io as sio

sio.save_video(sio.load_video("input.mp4"), "output.mp4")
```

!!! info "Why re-encode?"
    Some video formats are not readily seekable at frame-level accuracy. Re-encoding with default settings ensures reliable seeking with minimal quality loss.

!!! note "See also"
    [`save_video`](formats/#sleap_io.save_video): Video saving options and codec settings

### Switch video and image backends

Control which backend is used for video reading and embedded frame encoding.

#### Video reading backends

Choose which backend to use when loading videos with `sio.load_video()`.

```python title="video_backends.py" linenums="1"
import sleap_io as sio

# Set default video backend for reading video files
sio.set_default_video_plugin("opencv")
video = sio.load_video("test.mp4")

# Or use imageio-ffmpeg (bundled, always available)
sio.set_default_video_plugin("FFMPEG")
video = sio.load_video("test.mp4")

# Check current default
print(sio.get_default_video_plugin())  # "FFMPEG"
```

!!! info "Video backend trade-offs"

    | Backend | Install | Speed | Notes |
    |---------|---------|-------|-------|
    | **FFMPEG** (`FFMPEG`) | Bundled (always available) | Moderate | Default. Most reliable, best seeking accuracy |
    | **OpenCV** (`opencv`) | `pip install sleap-io[opencv]` | Fastest | May have platform-specific issues |
    | **PyAV** (`pyav`) | `pip install sleap-io[pyav]` | Fast | Alternative FFMPEG wrapper |

Choose which backend to use when encoding frames in `.pkg.slp` files with `sio.save_slp()`.

```python title="image_backends.py" linenums="1"
import sleap_io as sio

# Load labels
labels = sio.load_slp("labels.slp")

# Set default image encoding backend
sio.set_default_image_plugin("opencv")

# Save with embedded frames using OpenCV encoding
labels.save("labels.pkg.slp", embed="all")

# Or specify plugin directly in save call
labels.save("labels.pkg.slp", embed="all", plugin="imageio")

# Check current default
print(sio.get_default_image_plugin())  # "opencv"
```

!!! tip "Automatic RGB/BGR conversion"
    When loading `.pkg.slp` files, sleap-io automatically handles RGB/BGR channel order conversions between different encoding and decoding backends. Frames will always load in RGB order regardless of which plugin was used for encoding vs decoding.

!!! info "Image backend options"

    | Backend | Install | Notes |
    |---------|---------|-------|
    | **imageio** (`imageio`) | Bundled (always available) | Default. Encodes in RGB channel order |
    | **OpenCV** (`opencv`) | `pip install sleap-io[opencv]` | Faster encoding. Encodes in BGR internally |

    RGB/BGR conversion is handled automatically — frames always load in RGB regardless of which backend was used for encoding.

!!! note "Plugin vs backend terminology"
    - **Video plugins**: Used by `sio.load_video()` for reading video files (`opencv`, `FFMPEG`, `pyav`)
    - **Image plugins**: Used by `sio.save_slp()` for encoding embedded frames in `.pkg.slp` files (`opencv`, `imageio`)
    - Both can be set via `set_default_*_plugin()` functions

!!! note "See also"
    - [`set_default_video_plugin`](formats/#sleap_io.set_default_video_plugin): Set video reading backend
    - [`set_default_image_plugin`](formats/#sleap_io.set_default_image_plugin): Set image encoding backend
    - [`get_default_video_plugin`](formats/#sleap_io.get_default_video_plugin): Get current video backend
    - [`get_default_image_plugin`](formats/#sleap_io.get_default_image_plugin): Get current image backend

## Segmentation

Work with dense per-pixel instance segmentation data (e.g., from [Cellpose](https://www.cellpose.org/) or [StarDist](https://github.com/stardist/stardist)).

### Import segmentation masks to SLP

Convert a `(T, H, W)` integer mask array into an SLP file with object metadata.

```python title="segmentation_to_slp.py" linenums="1"
import numpy as np
import sleap_io as sio

# Load masks from segmentation tool output — (n_frames, height, width) int32
# 0 = background, 1+ = object IDs per frame
masks = np.load("cellpose_masks.npy")  # e.g., shape (100, 512, 512)

# Load the source video
video = sio.load_video("microscopy.tif")  # shape: (n_frames, height, width, channels)

# Convert to PredictedLabelImages with shared tracks across frames
label_images = sio.PredictedLabelImage.from_stack(
    masks, video=video, source="cellpose:nuclei",
    create_tracks=True, score=1.0,
)

# Build Labels — label_images are distributed to LabeledFrames automatically
labels = sio.Labels(label_images=label_images, videos=[video])
labels.save("segmentation.slp")
```

??? tip "Custom tracks and categories per frame"
    For more control, use `from_numpy()` per frame with explicit track/category mappings:
    ```python
    tracks = {1: sio.Track(name="cell_A"), 2: sio.Track(name="cell_B")}
    li = sio.PredictedLabelImage.from_numpy(
        masks[0], video=video, frame_idx=0,
        tracks=tracks, categories={1: "neuron", 2: "glia"},
        source="cellpose:nuclei", score=1.0,
    )
    ```

!!! info "How track sharing works"
    `from_stack(create_tracks=True)` maps each unique integer label to a `Track` object
    shared across all frames. If cell 5 appears in frames 0, 3, and 7, it gets the same
    `Track` in all three. This is useful when IDs are consistent across frames (e.g.,
    after tracking). Without `create_tracks`, objects have no cross-frame identity.

!!! note "See also"
    - [Regions: Label images](model/regions.md#label-images): Full label image data model documentation
    - [`PredictedLabelImage.from_stack`](model/regions.md#sleap_io.PredictedLabelImage.from_stack): Stack conversion API
    - [`PredictedLabelImage.from_numpy`](model/regions.md#sleap_io.PredictedLabelImage.from_numpy): Per-frame conversion API

### Import per-object binary masks to SLP

Convert per-object binary masks from tools like [SAM](https://github.com/facebookresearch/sam2) or Mask R-CNN into an SLP file. Unlike `from_stack` (which takes a pre-composited integer array), `from_binary_masks` takes individual boolean masks per object.

```python title="binary_masks_to_slp.py" linenums="1"
import numpy as np
import sleap_io as sio

# Per-object binary masks from SAM — (n_objects, height, width) bool array
# Each mask[i] is a binary mask for one detected object in a single frame
sam_masks = np.load("sam_masks.npy")  # e.g., shape (5, 512, 512) = 5 objects
object_scores = [0.95, 0.92, 0.88, 0.85, 0.80]

# Load the source video (e.g., a single-frame or multi-frame TIFF)
video = sio.load_video("microscopy.tif")  # shape: (n_frames, height, width, channels)

# Create a PredictedLabelImage with tracks and per-object scores
li = sio.PredictedLabelImage.from_binary_masks(
    sam_masks,
    create_tracks=True,          # auto-create one Track per mask
    scores=object_scores,        # per-object confidence → Info.score
    score=0.9,                   # image-level confidence
    source="sam",
)

# Add the label image to a Labels dataset via a LabeledFrame
lf = sio.LabeledFrame(video=video, frame_idx=0)
lf.append(li)
labels = sio.Labels(labeled_frames=[lf])
labels.save("sam_output.slp")
```

!!! tip "When to use which import method"
    - **`from_binary_masks`**: You have per-object boolean masks for a single frame — shape `(n_objects, H, W)` (SAM, Mask R-CNN).
    - **`from_stack`**: You have a `(n_frames, H, W)` integer array where each pixel value is an object ID (Cellpose, StarDist).
    - **`from_numpy`**: You have a single `(H, W)` integer array for one frame.

!!! note "See also"
    - [Regions: From binary masks](model/regions.md#from-binary-masks): Full `from_binary_masks` documentation
    - [`PredictedLabelImage.from_binary_masks`](model/regions.md#sleap_io.PredictedLabelImage.from_binary_masks): API reference

### Stream large segmentation results to SLP

For datasets too large to hold in memory, write frames one at a time with constant memory usage.

```python title="streaming_segmentation.py" linenums="1"
import numpy as np
import sleap_io as sio

# Load source video for frame data and metadata
video = sio.load_video("microscopy.tif")  # shape: (n_frames, height, width, channels)

# Stream frames to SLP — file is created lazily on first add()
# video is optional: associates all label images with this video in the SLP file
with sio.LabelImageWriter("output.slp", video=video) as writer:
    for frame_idx in range(len(video)):
        # Read the frame and run your segmentation model
        mask = run_segmentation(video[frame_idx])  # returns (H, W) int32

        li = sio.PredictedLabelImage.from_numpy(
            mask,
            source="cellpose:nuclei", score=1.0,
        )
        writer.add(li)
# File finalized and closed on context exit
```

!!! info "Tracking across frames"
    By default, each frame's objects are independent — no cross-frame identity.
    To link objects across frames, pass a shared dict as ``tracks`` with
    ``create_tracks=True``:

    ```python
    shared_tracks = {}  # accumulates {label_id: Track} across frames

    li = sio.PredictedLabelImage.from_numpy(
        mask,
        tracks=shared_tracks, create_tracks=True,  # reuse existing, create new
        source="cellpose:nuclei", score=1.0,
    )
    ```

    Existing entries in the dict are reused and new label IDs get fresh ``Track``
    objects added in place. This gives cross-frame identity without requiring all
    data in memory. The writer auto-collects new tracks from each ``add()`` call.

!!! tip "Memory and performance"
    The writer uses the chunked HDF5 format with gzip compression. Only one frame's
    compressed data is in memory at a time. A 42-frame (592x608) dataset compresses from
    ~60 MB raw to ~0.6 MB in the SLP file.

!!! note "See also"
    - [Regions: Streaming writes](model/regions.md#streaming-writes): Full `LabelImageWriter` documentation
    - [`LabelImageWriter`](model/regions.md#sleap_io.LabelImageWriter): API reference

### Extract segmentation data from SLP

Load segmentation data and convert back to numpy arrays, TIFF files, or individual masks.

```python title="extract_segmentation.py" linenums="1"
import numpy as np
import sleap_io as sio

# Load — pixel data is lazy (metadata queries don't decompress)
labels = sio.load_slp("segmentation.slp")

# Access label images through frames (annotations are nested in LabeledFrames)
lf = labels[0]  # first labeled frame
print(lf.frame_idx, len(lf.label_images))  # no decompression yet

# Inspect a single label image
li = lf.label_images[0]
print(li.n_objects, li.tracks, li.categories)  # still no decompression

# Extract all frames as (n_frames, height, width) numpy array
# .data triggers lazy decompression for each frame
all_label_images = labels.label_images  # flattened view across all frames
all_masks = np.stack([li.data for li in all_label_images])

# Export as TIFF stack (with sidecar metadata JSON)
sio.save_label_images("masks.tif", all_label_images, stack=True)

# Decompose one frame into per-object binary SegmentationMasks
for mask in li.to_masks():
    print(f"{mask.category}: {mask.area} pixels")
```

??? example "Expected output shapes"
    For a dataset with 42 frames at 592x608 with 21 objects:

    - `all_masks.shape`: `(42, 592, 608)`, dtype `int32`
    - `li.to_masks()`: list of 21 [`SegmentationMask`](model/regions.md#segmentation-masks) objects
    - Each mask: boolean array `(592, 608)` for one object

!!! note "See also"
    - [Regions: Lazy loading](model/regions.md#lazy-loading): How lazy pixel data access works
    - [TIFF Format](formats/tiff.md): TIFF label image I/O details
    - [`LabelImage.to_masks`](model/regions.md#sleap_io.LabelImage.to_masks): Decomposition API

### Merge segmentation results

Combine label images from multiple SLP files into one (e.g., after parallel
batch processing). This uses [`merge_label_images()`][sleap_io.merge_label_images],
a specialized function that copies compressed HDF5 chunks directly between files
without decompressing pixel data — much faster than loading everything into
memory with [`Labels.merge()`](merging.md).

```python title="merge_segmentation.py" linenums="1"
import sleap_io as sio

# Merge batch results at the file level (no decompression needed)
merged = sio.merge_label_images(
    ["batch_0.slp", "batch_1.slp", "batch_2.slp"],
    "all_frames.slp",
)
print(f"Merged: {len(merged.label_images)} total frames")
```

!!! info "Why not `Labels.merge()`?"
    [`Labels.merge()`](merging.md) loads all data into Python objects, which is
    necessary for matching and resolving conflicts between keypoint annotations.
    For label images — which are typically non-overlapping frame batches from
    parallel segmentation — `merge_label_images()` skips all of that and
    concatenates the compressed pixel chunks directly. This makes it I/O-bound
    rather than CPU-bound.

!!! note "See also"
    - [Merging: Label images](merging.md#merging-label-images): Full merge documentation
    - [`merge_label_images`](model/regions.md#sleap_io.merge_label_images): API reference

### Parallel segmentation pipeline

Combine the streaming writer and the zero-decompression merger to build a parallel batch segmentation pipeline. Each worker streams its own shard with [`LabelImageWriter`][sleap_io.LabelImageWriter], and a final [`merge_label_images`][sleap_io.merge_label_images] call concatenates the shards without decompressing a single pixel.

```python title="parallel_segmentation.py" linenums="1"
import sleap_io as sio

def run_shard(video_path: str, frame_range: range, shard_path: str) -> None:
    """Segment a frame range and stream the results to one SLP shard."""
    video = sio.load_video(video_path)
    with sio.LabelImageWriter(shard_path, video=video) as writer:
        for frame_idx in frame_range:
            mask = run_segmentation(video[frame_idx])   # (H, W) int32
            li = sio.PredictedLabelImage.from_numpy(
                mask, source="cellpose:nuclei", score=1.0,
            )
            writer.add(li)

# Launch one worker per shard (e.g., with multiprocessing, joblib, or a job
# scheduler) so every worker streams to its own file.
run_shard("microscopy.tif", range(0, 500), "shard_0.slp")
run_shard("microscopy.tif", range(500, 1000), "shard_1.slp")
run_shard("microscopy.tif", range(1000, 1500), "shard_2.slp")

# Concatenate the shards into the final dataset in O(shards), without
# decompressing any pixel chunks.
merged = sio.merge_label_images(
    ["shard_0.slp", "shard_1.slp", "shard_2.slp"],
    "all_frames.slp",
)
print(f"Merged: {len(merged.label_images)} frames")
```

!!! tip "Why this is fast"
    Workers run in parallel and produce contiguous SLP shards that each hold only
    one frame's worth of compressed data in memory at a time. `merge_label_images`
    then copies compressed HDF5 chunks byte-for-byte between files — no
    decompression, no conflict resolution, no Python-level object reconstruction.

## Rendering

Create videos and images with pose overlays for visualization and publication.

```python title="render_poses.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Render full video (MP4 with skeleton overlays at source resolution)
labels.render("output.mp4")

# Fast preview for iteration (0.25x resolution, faster encoding)
labels.render("preview.mp4", preset="preview")

# Single frame to PNG image
sio.render_image(labels[0], "frame.png")
```

```bash title="CLI"
sio render -i predictions.slp -o output.mp4
sio render -i predictions.slp --preset preview
sio render -i predictions.slp --lf 0  # Single frame
```

### Segmentation overlay rendering

Render pose predictions on top of a segmentation mask (or render masks on bare images, no labels file needed). The overlay pipeline accepts 2-D label images, 3-D label stacks, or a directory of per-frame TIFFs.

```python title="segmentation_overlay.py" linenums="1"
import numpy as np
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
lf = labels[0]

# Overlay a (H, W) integer label image on a single labeled frame
label_mask = np.load("frame0_mask.npy")  # (H, W) int
img = sio.render_image(
    lf,
    overlay=label_mask,
    overlay_alpha=0.4,
    overlay_outline=True,
)

# Overlay-only mode (no poses) — renders masks on an arbitrary image
frame = sio.load_video("clip.mp4")[0]
masked = sio.render_image(image=frame, overlay=label_mask)

# Video rendering with a 3-D label stack (one mask per frame)
stack = np.load("masks_stack.npy")  # (T, H, W) int
sio.render_video(labels, "overlay.mp4", overlay=stack)

# Standalone draw_* helpers compose overlays manually onto any image
canvas = frame.copy()
sio.draw_label_image(canvas, label_mask, alpha=0.4, outline=True)
sio.draw_bboxes(canvas, lf.bboxes)
sio.draw_masks(canvas, [m for m in lf.masks])
sio.draw_rois(canvas, lf.rois, fill_alpha=0.3)
```

```bash title="CLI"
# Overlay segmentation on rendered poses
sio render predictions.slp --overlay masks.tif --overlay-alpha 0.4

# Overlay-only mode: no labels file needed
sio render --images frames/ --overlay masks.tif -o output.mp4
```

!!! note "See also"
    See the **[Rendering Guide](rendering.md)** for complete documentation including color schemes, marker shapes, custom callbacks, and all CLI options.
