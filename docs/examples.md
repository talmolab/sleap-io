# Examples

This page provides practical examples for common tasks with sleap-io. Each example includes working code that you can copy and adapt for your needs.

!!! info "Prerequisites"
    
    All examples assume you have sleap-io installed:
    ```bash
    pip install sleap-io
    ```
    
    Most examples use `import sleap_io as sio` for brevity.

## Basic I/O operations

### Load and save in different formats

Convert between supported formats with automatic format detection.

```python title="format_conversion.py" linenums="1" hl_lines="6 9"
import sleap_io as sio

# Load from SLEAP file
labels = sio.load_file("predictions.slp")

# Save to NWB file
labels.save("predictions.nwb")

# The format is automatically detected from the extension
```

!!! tip
    sleap-io automatically detects the format from the file extension. Supported formats include `.slp`, `.nwb`, `.labelstudio.json`, and `.jabs`.

!!! note "See also"
    - [`Labels.save`](model.md#sleap_io.Labels.save) - Save method with format options
    - [Formats](formats.md) - Complete list of supported formats

### Convert labels to raw arrays

Extract pose data as NumPy arrays for analysis or visualization.

```python title="labels_to_numpy.py" linenums="1" hl_lines="6 11"
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
    [`Labels.numpy`](model.md#sleap_io.Labels.numpy) - Full documentation of array conversion options

## Video operations

### Read video data

Load and access video frames directly.

```python title="read_video.py" linenums="1" hl_lines="3 6"
import sleap_io as sio

video = sio.load_video("test.mp4")
n_frames, height, width, channels = video.shape

frame = video[0]  # Get first frame
height, width, channels = frame.shape

# Access specific frames
middle_frame = video[n_frames // 2]
last_frame = video[-1]
```

!!! warning
    Loading videos requires either `opencv-python` or `pyav`. Install with:
    ```bash
    pip install sleap-io[opencv]  # or sleap-io[pyav]
    ```

!!! note "See also"
    - [`sio.load_video`](formats.md#sleap_io.load_video) - Video loading function
    - [`Video`](model.md#sleap_io.Video) - Video class documentation

### Re-encode video

Fix video seeking issues by re-encoding with optimal settings.

```python title="reencode_video.py" linenums="1" hl_lines="3"
import sleap_io as sio

sio.save_video(sio.load_video("input.mp4"), "output.mp4")
```

!!! info "Why re-encode?"
    Some video formats are not readily seekable at frame-level accuracy. Re-encoding with default settings ensures reliable seeking with minimal quality loss.

!!! note "See also"
    [`save_video`](formats.md#sleap_io.save_video) - Video saving options and codec settings

### Trim labels and video

Extract a subset of frames with corresponding labels.

```python title="trim_video.py" linenums="1" hl_lines="7"
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
    [`Labels.trim`](model.md#sleap_io.Labels.trim) - Full trim method documentation

## Data creation

### Create labels from raw data

Build a complete labels dataset programmatically.

```python title="create_labels.py" linenums="1" hl_lines="4-7 12-18 21 24 27"
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
    - [Model](model.md) - Complete data model documentation
    - [`Labels`](model.md#sleap_io.Labels) - Labels container class
    - [`Instance`](model.md#sleap_io.Instance) - Instance class for manual annotations
    - [`PredictedInstance`](model.md#sleap_io.PredictedInstance) - Instance class for predictions

## Dataset management

### Make training/validation/test splits

Split your dataset for machine learning workflows.

=== "Basic splits"
    
    ```python title="make_splits.py" linenums="1" hl_lines="6"
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
    ```

=== "With stratification"
    
    ```python title="stratified_splits.py" linenums="1" hl_lines="7"
    import sleap_io as sio
    
    labels = sio.load_file("labels.v001.slp")
    
    # Stratify by video to ensure each split has diverse data
    labels.make_training_splits(
        n_train=0.8,
        n_val=0.1, 
        n_test=0.1,
        save_dir="stratified_split",
        stratify="video",  # Ensures videos are distributed across splits
        seed=42
    )
    ```

!!! info
    The `.pkg.slp` extension indicates a self-contained package with embedded images, making the splits portable and shareable.

!!! note "See also"
    [`Labels.make_training_splits`](model.md#sleap_io.Labels.make_training_splits) - Full documentation of splitting options

### Working with dataset splits (LabelsSet)

Manage multiple related datasets as a group.

```python title="labels_set.py" linenums="1" hl_lines="6 9-11 14 17 20"
import sleap_io as sio

# Load source labels
labels = sio.load_file("labels.v001.slp")

# Create splits and get them as a LabelsSet
labels_set = labels.make_training_splits(n_train=0.8, n_val=0.1, n_test=0.1)

# Access individual splits
train_labels = labels_set["train"]
val_labels = labels_set["val"] 
test_labels = labels_set["test"]

# Save the entire LabelsSet
labels_set.save("splits/")  # Saves as SLP files by default

# Save as Ultralytics YOLO format
labels_set.save("yolo_dataset/", format="ultralytics")

# Load a LabelsSet from a directory
loaded_set = sio.load_labels_set("splits/")
```

??? example "Loading from specific files"
    ```python
    # Load from custom file paths
    file_dict = {
        "train": "path/to/train.slp",
        "val": "path/to/val.slp",
        "test": "path/to/test.slp"
    }
    loaded_set = sio.load_labels_set(file_dict)
    ```

!!! tip
    LabelsSet is particularly useful when exporting to formats that expect separate train/val/test files, like YOLO.

!!! note "See also"
    - [`LabelsSet`](model.md#sleap_io.LabelsSet) - LabelsSet class documentation
    - [`load_labels_set`](formats.md#sleap_io.load_labels_set) - Loading function for label sets

## Data manipulation

### Fix video paths

Update file paths when moving projects between systems.

```python title="fix_paths.py" linenums="1" hl_lines="3 6-9"
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
    [`Labels.replace_filenames`](model.md#sleap_io.Labels.replace_filenames) - Additional path manipulation options

### Save labels with embedded images

Create self-contained label files with embedded video frames.

=== "User labels only"
    
    ```python title="embed_user.py" linenums="1" hl_lines="6"
    import sleap_io as sio
    
    # Load source labels
    labels = sio.load_file("labels.v001.slp")
    
    # Embed frames with user-labeled data only
    labels.save("labels.v001.pkg.slp", embed="user")
    ```

=== "User + suggestions"
    
    ```python title="embed_suggestions.py" linenums="1" hl_lines="6"
    import sleap_io as sio
    
    # Load source labels
    labels = sio.load_file("labels.v001.slp")
    
    # Embed frames with user labels and suggested frames
    labels.save("labels.v001.pkg.slp", embed="user+suggestions")
    ```

=== "All frames"
    
    ```python title="embed_all.py" linenums="1" hl_lines="6"
    import sleap_io as sio
    
    # Load source labels
    labels = sio.load_file("labels.v001.slp")
    
    # Embed all labeled frames (including predictions)
    labels.save("labels.v001.pkg.slp", embed="all")
    ```

!!! info "Embedding options"
    - `"user"`: Only frames with manual annotations
    - `"user+suggestions"`: Manual annotations plus suggested frames
    - `"all"`: All frames with any labels (including predictions)
    - `"source"`: Embed source video if labels were loaded from embedded data

!!! note "See also"
    [`Labels.save`](model.md#sleap_io.Labels.save) - Complete save options including embedding

### Replace skeleton

Change the skeleton structure while preserving existing annotations.

```python title="replace_skeleton.py" linenums="1" hl_lines="7 10-15"
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
    [`Labels.replace_skeleton`](model.md#sleap_io.Labels.replace_skeleton) - Additional skeleton manipulation options

### Convert to and from numpy arrays

Work with pose data as NumPy arrays for filtering or analysis.

```python title="numpy_filtering.py" linenums="1" hl_lines="6 11 14"
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
    For more sophisticated analysis and filtering:
    ```python
    # Check out the movement library for pose processing
    # https://movement.neuroinformatics.dev/
    import movement
    ```

!!! warning
    When updating from numpy, the array shape must match the original data structure exactly.

!!! note "See also"
    - [`Labels.numpy`](model.md#sleap_io.Labels.numpy) - Array conversion options
    - [`Labels.update_from_numpy`](model.md#sleap_io.Labels.update_from_numpy) - Updating labels from arrays
    - [`movement`](https://movement.neuroinformatics.dev/) - Advanced pose processing library

## Troubleshooting

??? failure "Video not found after loading"
    
    **Problem:** Video files can't be found after loading a project on a different system.
    
    **Solution:** Use `open_videos=False` and fix paths:
    ```python
    labels = sio.load_file("project.slp", open_videos=False)
    labels.replace_filenames(prefix_map={"old/path": "new/path"})
    ```

??? failure "Memory issues with large videos"
    
    **Problem:** Loading large videos consumes too much memory.
    
    **Solution:** Use embedded frames for specific frames only:
    ```python
    # Only embed frames with labels
    labels.save("compact.pkg.slp", embed="user")
    ```

??? failure "Cannot seek to specific frame"
    
    **Problem:** Some video formats don't support accurate frame seeking.
    
    **Solution:** Re-encode the video:
    ```python
    sio.save_video(sio.load_video("problematic.avi"), "fixed.mp4")
    ```