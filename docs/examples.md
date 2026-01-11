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

Build a complete labels dataset programmatically.

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
    To create predictions with confidence scores:
    ```python
    predicted_instance = sio.PredictedInstance.from_numpy(
        points=points_array,
        confidence=confidence_array,  # Shape: (n_nodes,)
        skeleton=skeleton
    )
    ```

!!! note "See also"
    - [Model](model.md): Complete data model documentation
    - [`Labels`](model.md#sleap_io.Labels): Labels container class
    - [`Instance`](model.md#sleap_io.Instance): Instance class for manual annotations
    - [`PredictedInstance`](model.md#sleap_io.PredictedInstance): Instance class for predictions

### Convert labels to raw arrays

Extract pose data as NumPy arrays for analysis or visualization.

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
    [`Labels.numpy`](model.md#sleap_io.Labels.numpy): Full documentation of array conversion options

## Format conversion

### Load and save in different formats

Convert between supported formats with automatic format detection.

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
    - [`Labels.save`](model.md#sleap_io.Labels.save): Save method with format options
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
from sleap_io.io.nwb_annotations import save_labels

# Save with comprehensive metadata
save_labels(
    labels,
    "training_data.nwb",
    session_description="Mouse skilled reaching task - training dataset",
    identifier="mouse_01_session_03_annotations",
    session_start_time="2024-01-15T09:30:00",
    annotator="John Doe",
    nwb_kwargs={
        # Session metadata
        "session_id": "session_003",
        "experimenter": ["John Doe", "Jane Smith"],
        "lab": "Motor Control Lab",
        "institution": "University of Example",

        # Experimental details
        "experiment_description": "Skilled reaching task with food pellet reward",
        "protocol": "Protocol 2024-001",
        "surgery": "Cranial window implant over M1",

        # Subject information
        "subject": {
            "subject_id": "mouse_01",
            "age": "P90",
            "sex": "M",
            "species": "Mus musculus",
            "strain": "C57BL/6J",
            "weight": "25g"
        }
    }
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

```python title="nwb_export.py" linenums="1"
from sleap_io.io.nwb_annotations import export_labels, export_labeled_frames

# Method 1: Export complete dataset with all videos
export_labels(
    labels,
    output_dir="export/",
    nwb_filename="complete_dataset.nwb",
    as_training=True,      # Include manual annotations
    include_videos=True,    # Embed all video frames
    include_skeleton=True   # Include skeleton definition
)

# Method 2: Export only frames with labels as a new video
export_labeled_frames(
    labels,
    output_path="labeled_frames.avi",         # MJPEG video output
    labels_output_path="labeled_frames.nwb",  # Corresponding labels
    fps=30.0,                                  # Output frame rate
    scale=1.0                                  # Video scale factor
)

# The export includes a FrameMap JSON file tracking frame origins
import json
with open("labeled_frames.frame_map.json", "r") as f:
    frame_map = json.load(f)
    print(f"Exported {frame_map['total_frames']} frames from {len(frame_map['videos'])} videos")
```

!!! info "Export formats"

    - **Full export**: Includes all video frames, creating large but complete files
    - **Labeled frames only**: Exports just frames with annotations, reducing file size
    - **Frame provenance**: JSON metadata tracks which frames came from which source videos

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
    - [`LabelsSet`](model.md#sleap_io.LabelsSet): LabelsSet class documentation
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
    [`Labels.replace_filenames`](model.md#sleap_io.Labels.replace_filenames): Additional path manipulation options

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
    [`Labels.copy`](model.md#sleap_io.Labels.copy): Full documentation of copy options

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
    [`Labels.replace_skeleton`](model.md#sleap_io.Labels.replace_skeleton): Additional skeleton manipulation options

### Update from numpy

Work with pose data as NumPy arrays for filtering or analysis.

```python title="numpy_filtering.py" linenums="1"
import sleap_io as sio
import numpy as np

labels = sio.load_file("predictions.slp")

# Convert to array of shape (n_frames, n_tracks, n_nodes, xy)
trx = labels.numpy()

# Apply temporal filtering (example: simple moving average)
window_size = 5
trx_filtered = np.convolve(trx.reshape(-1), np.ones(window_size)/window_size, mode='same').reshape(trx.shape)

# Update the labels with filtered data
labels.update_from_numpy(trx_filtered)

# Save the filtered version
labels.save("predictions.filtered.slp")
```

??? tip "Advanced filtering with movement"
    For more sophisticated analysis and filtering, check out the [`movement`](https://movement.neuroinformatics.dev/) library for pose processing.

!!! warning
    When updating from numpy, the array shape must match the original data structure exactly.

!!! note "See also"
    - [`Labels.numpy`](model.md#sleap_io.Labels.numpy): Array conversion options
    - [`Labels.update_from_numpy`](model.md#sleap_io.Labels.update_from_numpy): Updating labels from arrays
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
    [`Labels.save`](model.md#sleap_io.Labels.save): Complete save options including embedding

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

**Cancellation support:**

```python title="embed_with_cancel.py" linenums="1"
from sleap_io.io.slp import ExportCancelled

cancelled = False

def on_progress(current, total):
    return not cancelled  # Return False to cancel

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
    [`Labels.trim`](model.md#sleap_io.Labels.trim): Full trim method documentation

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
    - [`Labels.make_training_splits`](model.md#sleap_io.Labels.make_training_splits): Full documentation of splitting options
    - [`LabelsSet`](model.md#sleap_io.LabelsSet): LabelsSet class for working with split datasets

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
    - [`Video`](model.md#sleap_io.Video): Video class documentation

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

!!! info "Backend trade-offs"

    **OpenCV** (`opencv`):

    - ✅ Generally faster for frame reading
    - ❌ May have compatibility issues on some platforms
    - ❌ Frame seeking may be less accurate for some codecs

    - ✅ Works out of the box (bundled with sleap-io)
    - ✅ More reliable and cross-platform
    - ✅ Better seeking accuracy
    - ✅ Always installed with sleap-io (default)
    - ❌ May be slower than OpenCV

    **PyAV** (`pyav`):

    - ✅ Alternative FFMPEG wrapper with different performance characteristics

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

    **OpenCV** (`opencv`):

    - ✅ Generally faster encoding
    - Encodes in BGR channel order

    - ✅ Always installed with sleap-io (default)
    - ✅ More reliable and cross-platform
    - Encodes in RGB channel order

!!! note "Plugin vs backend terminology"
    - **Video plugins**: Used by `sio.load_video()` for reading video files (`opencv`, `FFMPEG`, `pyav`)
    - **Image plugins**: Used by `sio.save_slp()` for encoding embedded frames (`opencv`, `imageio`)
    - Both can be set via `set_default_*_plugin()` functions

!!! note "See also"
    - [`set_default_video_plugin`](formats/#sleap_io.set_default_video_plugin): Set video reading backend
    - [`set_default_image_plugin`](formats/#sleap_io.set_default_image_plugin): Set image encoding backend
    - [`get_default_video_plugin`](formats/#sleap_io.get_default_video_plugin): Get current video backend
    - [`get_default_image_plugin`](formats/#sleap_io.get_default_image_plugin): Get current image backend

## Rendering

Create videos and images with pose overlays for visualization and publication.

```python title="render_poses.py" linenums="1"
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Render full video
labels.render("output.mp4")

# Fast preview (0.25x resolution)
labels.render("preview.mp4", preset="preview")

# Single frame to image
sio.render_image(labels.labeled_frames[0], "frame.png")
```

```bash title="CLI"
sio render -i predictions.slp -o output.mp4
sio render -i predictions.slp --preset preview
sio render -i predictions.slp --lf 0  # Single frame
```

!!! note "See also"
    See the **[Rendering Guide](rendering.md)** for complete documentation including color schemes, marker shapes, custom callbacks, and all CLI options.
