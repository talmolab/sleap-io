# Data formats

sleap-io provides a unified interface for reading and writing pose tracking data across multiple formats. The library automatically detects file formats and provides harmonized I/O operations.

## Universal I/O Functions

::: sleap_io.io.main.load_file

::: sleap_io.io.main.save_file

## Video I/O

::: sleap_io.io.main.load_video

::: sleap_io.io.main.save_video

## Format-Specific Functions

### SLEAP Native Format (.slp)

The native SLEAP format stores complete pose tracking projects including videos, skeletons, and annotations.

!!! tip "Detailed Format Specification"
    For comprehensive documentation of the SLP file format including HDF5 layout, data structures, and version history, see the **[SLP File Format Reference](slp.md)**.

::: sleap_io.io.main.load_slp

::: sleap_io.io.main.save_slp

#### Lazy Loading for Large Files

When working with large SLP files (hundreds of thousands of frames), loading can be slow due to the creation of many Python objects. sleap-io provides a **lazy loading** mode that defers object creation until needed, significantly speeding up common workflows.

##### When to Use Lazy Loading

Lazy loading is recommended when:

- You only need to convert data to NumPy arrays (`labels.numpy()`)
- You're saving to a different file without modifications
- You're accessing a small subset of frames
- You want fast load times for large files

##### Basic Usage

```python
import sleap_io as sio

# Load lazily (up to 90x faster than eager loading!)
labels = sio.load_slp("predictions.slp", lazy=True)

# Check if labels is lazy
print(labels.is_lazy)  # True

# Fast path: convert directly to NumPy (no object creation)
poses = labels.numpy()

# Fast path: save without materialization
sio.save_slp(labels, "copy.slp")
```

##### Accessing Frames

Lazy-loaded `Labels` support standard read operations:

```python
# These work normally (frames materialized on-demand)
print(len(labels))  # Number of frames
first_frame = labels[0]  # Access single frame
last_frame = labels[-1]  # Negative indexing
subset = labels[10:20]  # Slicing

# Iteration (materializes each frame)
for lf in labels:
    print(f"Frame {lf.frame_idx}: {len(lf)} instances")
```

##### Modifying Lazy Labels

Lazy `Labels` are read-only. To make modifications, first materialize:

```python
# This raises RuntimeError
labels.append(new_frame)  # Error: Cannot append on lazy-loaded Labels

# Materialize first to enable modifications
labels = labels.materialize()  # Creates eager copy
labels.append(new_frame)  # Now works
```

##### Performance Comparison

| Operation | Eager | Lazy | Speedup |
|-----------|-------|------|---------|
| Load only | 0.47s | 0.005s | **~90x** |
| Load + numpy() | 0.86s | 0.38s | **~2x** |
| Full iteration | 0.0002s | 0.41s | Eager faster |

*Benchmarks on 18,000 frames with ~40,000 instances.*

Lazy loading excels at avoiding unnecessary work. If you need to iterate over all frames, eager loading is faster.

##### API Reference

**Labels properties and methods for lazy loading:**

- `Labels.is_lazy` - `True` if lazy-loaded
- `Labels.materialize()` - Convert to eager `Labels` (returns self if already eager)
- `Labels.numpy()` - Uses fast path when lazy (no object creation)
- `Labels.to_dataframe()` - Uses fast path when lazy (no object creation)

**Fast statistics (O(1) for lazy-loaded Labels):**

- `Labels.n_user_instances` - Total number of user-labeled instances
- `Labels.n_pred_instances` - Total number of predicted instances
- `Labels.n_frames_per_video()` - Dictionary mapping videos to frame counts
- `Labels.n_instances_per_track()` - Dictionary mapping tracks to instance counts

### NWB Format (.nwb)

[Neurodata Without Borders (NWB)](https://www.nwb.org/) is a standardized format for neurophysiology data. sleap-io provides comprehensive support for both reading and writing pose tracking data in NWB format.

#### Harmonized NWB I/O

The harmonized API automatically detects and routes to the appropriate NWB backend:

::: sleap_io.io.main.load_nwb

::: sleap_io.io.main.save_nwb

#### NWB Format Types

sleap-io supports multiple NWB format types through the `nwb_format` parameter:

- **`"auto"`** (default): Automatically detect based on data content
  - Uses `"annotations"` if data contains user-labeled instances
  - Uses `"predictions"` if data contains only predicted instances

- **`"annotations"`**: Save as PoseTraining format (ndx-pose extension)
  - Stores manual annotations for training data
  - Preserves skeleton structure and node names
  - Includes annotator information

- **`"annotations_export"`**: Export annotations with embedded video frames
  - Creates self-contained NWB file with video data
  - Generates MJPEG video with frame provenance tracking
  - Useful for sharing complete datasets

- **`"predictions"`**: Save as PoseEstimation format (ndx-pose extension)
  - Stores predicted pose data from inference
  - Includes confidence scores
  - Supports multiple animals/tracks

#### Examples

##### Basic NWB Usage

```python
import sleap_io as sio

# Load any NWB file (auto-detects format)
labels = sio.load_nwb("pose_data.nwb")

# Save with auto-detection
sio.save_nwb(labels, "output.nwb")

# Save with specific format
sio.save_nwb(labels, "training.nwb", nwb_format="annotations")
sio.save_nwb(labels, "predictions.nwb", nwb_format="predictions")
```

##### Advanced Annotations API

For more control over NWB training data, use the annotations module directly:

```python
from sleap_io.io.nwb_annotations import save_labels, load_labels

# Save with custom metadata
save_labels(
    labels,
    "training.nwb",
    session_description="Mouse reaching task",
    identifier="mouse_01_session_03",
    annotator="researcher_name",
    nwb_kwargs={
        "session_id": "session_003",
        "experimenter": ["John Doe", "Jane Smith"],
        "lab": "Motor Control Lab",
        "institution": "University",
        "experiment_description": "Skilled reaching behavior"
    }
)

# Load annotations
labels = load_labels("training.nwb")
```

##### Export with Video Frames

```python
from sleap_io.io.nwb_annotations import export_labels, export_labeled_frames

# Export complete dataset with videos
export_labels(
    labels,
    output_dir="export/",
    nwb_filename="dataset_with_videos.nwb",
    as_training=True,  # Include manual annotations
    include_videos=True  # Embed video frames
)

# Export only labeled frames as video
export_labeled_frames(
    labels,
    output_path="labeled_frames.avi",
    labels_output_path="labels.nwb",
    fps=30.0
)
```


##### Multi-Subject Support

For multi-animal experiments, sleap-io supports the [ndx-multisubjects](https://pypi.org/project/ndx-multisubjects/) NWB extension. This links each tracked animal to a proper NWB Subject entry.

```python
from sleap_io.io.nwb_annotations import save_labels

# Basic multi-subject export (uses track names as subject IDs)
save_labels(labels, "output.nwb", use_multisubjects=True)

# With detailed subject metadata
subjects_metadata = [
    {"sex": "M", "species": "Mus musculus", "age": "P30D"},
    {"sex": "F", "species": "Mus musculus", "age": "P45D"},
]
save_labels(
    labels,
    "output.nwb",
    use_multisubjects=True,
    subjects_metadata=subjects_metadata
)
```

!!! info "Subject metadata fields"
    The `subjects_metadata` list should have one entry per track. Each entry can include:

    - `sex`: Subject sex (defaults to "U" for unknown)
    - `species`: Species name (defaults to "unknown")
    - `age`: Age in ISO 8601 duration format (e.g., "P30D" for 30 days)
    - Any other fields supported by NWB Subject

!!! warning "Tracked instances required"
    Multi-subject export requires all instances to have track assignments. A warning is issued if untracked instances are present.

#### NWB Metadata

The NWB format requires certain metadata fields. sleap-io provides sensible defaults:

- **Required fields** (auto-generated if not provided):
  - `session_description`: Defaults to "Processed SLEAP pose data"
  - `identifier`: Auto-generated UUID string
  - `session_start_time`: Current timestamp

- **Optional fields** (via `nwb_kwargs`):
  - `session_id`: Unique session identifier
  - `experimenter`: List of experimenters
  - `lab`: Laboratory name
  - `institution`: Institution name
  - `experiment_description`: Detailed experiment description
  - Any other valid NWB file fields

### JABS Format (.h5)

[JABS](https://github.com/KumarLabJax/JABS-behavior-classifier) (Janelia Automatic Behavior System) format for behavior classification.

::: sleap_io.io.main.load_jabs

::: sleap_io.io.main.save_jabs

### SLEAP Analysis HDF5 Format (.h5)

The SLEAP Analysis HDF5 format is a portable format for exporting pose tracking predictions as dense numpy arrays. This is the format produced by SLEAP's "Export Analysis HDF5" feature, designed for easy loading in MATLAB and Python analysis pipelines.

::: sleap_io.io.main.load_analysis_h5

::: sleap_io.io.main.save_analysis_h5

#### Axis Ordering Presets

The format supports configurable axis ordering via presets:

| Preset | Description | tracks shape |
|--------|-------------|--------------|
| `matlab` (default) | SLEAP-compatible, optimized for MATLAB | `(tracks, 2, nodes, frames)` |
| `standard` | Python-native, intuitive indexing | `(frames, tracks, nodes, 2)` |

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Default (MATLAB-compatible) - matches SLEAP's export
sio.save_analysis_h5(labels, "output.h5")

# Python-native ordering for easier numpy indexing
sio.save_analysis_h5(labels, "output.h5", preset="standard")

# Filter tracks with <50% occupancy
sio.save_analysis_h5(labels, "output.h5", min_occupancy=0.5)

# Load back
loaded = sio.load_analysis_h5("output.h5")
```

#### Self-Documenting Format

Each dataset stores its dimension names in the `dims` HDF5 attribute, making files self-documenting:

```python
import h5py

with h5py.File("output.h5", "r") as f:
    print(f["tracks"].attrs["dims"])  # e.g., '["track", "xy", "node", "frame"]'
    print(f.attrs["preset"])  # "matlab", "standard", or "custom"
```

### Label Studio Format (.json)

[Label Studio](https://labelstud.io/) is a multi-modal annotation platform. Export annotations from Label Studio and load them into SLEAP.

::: sleap_io.io.main.load_labelstudio

::: sleap_io.io.main.save_labelstudio

### DeepLabCut Format (.h5, .csv)

Load predictions from [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), a popular markerless pose estimation tool.

::: sleap_io.io.main.load_dlc

### CSV Format (.csv)

sleap-io provides comprehensive CSV support for reading and writing pose tracking data, enabling interoperability with spreadsheet tools, custom pipelines, and other pose estimation frameworks.

#### Supported CSV Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `sleap` | SLEAP Analysis CSV (default) | Native SLEAP exports, one row per instance |
| `dlc` | DeepLabCut format | DLC compatibility, multi-header structure |
| `points` | One row per point | Most normalized, database-friendly |
| `instances` | One row per instance | Compact, analysis-friendly |
| `frames` | One row per frame | Wide format, all instances in columns |

#### Basic Usage

```python
import sleap_io as sio

# Load CSV (auto-detects format)
labels = sio.load_csv("predictions.csv")

# Save in SLEAP Analysis format (default)
sio.save_csv(labels, "output.csv")

# Save in DLC format
sio.save_csv(labels, "dlc_output.csv", format="dlc", scorer="MyModel")

# Save with metadata for full round-trip support
sio.save_csv(labels, "output.csv", save_metadata=True)
# Creates: output.csv + output.json (metadata)
```

#### Round-Trip with Metadata

CSV files cannot store all Labels information (skeleton edges, symmetries, suggestions). To enable full round-trip reconstruction, use `save_metadata=True`:

```python
# Save with metadata sidecar file
sio.save_csv(labels, "data.csv", save_metadata=True)
# Creates: data.csv and data.json

# Load back with full metadata
labels = sio.load_csv("data.csv")
# Automatically loads data.json if present
```

The metadata JSON file contains:

- Video paths and backend metadata
- Skeleton definitions (nodes, edges, symmetries)
- Track names
- Suggested frames
- Provenance information

#### Format-Specific Examples

##### SLEAP Analysis Format

The default format matches SLEAP's "Export Analysis CSV" output:

```python
# Write SLEAP format
sio.save_csv(labels, "analysis.csv", format="sleap")
```

Output columns: `track, frame_idx, instance.score, {node}.x, {node}.y, {node}.score, ...`

##### DeepLabCut Format

For compatibility with DeepLabCut workflows:

```python
# Write DLC format with custom scorer name
sio.save_csv(labels, "dlc_output.csv", format="dlc", scorer="MyNetwork")

# Multi-animal DLC format (auto-detected from tracks)
sio.save_csv(multi_animal_labels, "multi_dlc.csv", format="dlc")
```

DLC format uses multi-row headers (scorer, bodyparts, coords) and is compatible with DLC's analysis tools.

##### DataFrame Codec Formats

For custom analysis pipelines, use the normalized formats from the DataFrame codec:

```python
# Points format: most normalized (one row per point)
sio.save_csv(labels, "points.csv", format="points")

# Instances format: one row per instance
sio.save_csv(labels, "instances.csv", format="instances")

# Frames format: one row per frame (wide format)
sio.save_csv(labels, "frames.csv", format="frames")
```

::: sleap_io.io.main.load_csv

::: sleap_io.io.main.save_csv

### AlphaTracker Format

Load predictions from [AlphaTracker](https://github.com/yinaanyachukwu/AlphaTracker), a tracking system for socially-housed animals.

::: sleap_io.io.main.load_alphatracker

### LEAP Format (.mat)

Load predictions from [LEAP](https://github.com/talmo/leap), a SLEAP predecessor. Requires `scipy` for .mat file support.

::: sleap_io.io.main.load_leap

### COCO Format (.json)

[COCO](https://cocodataset.org/) (Common Objects in Context) format is widely used in computer vision and pose estimation. sleap-io provides full read and write support, making it compatible with tools like [mmpose](https://github.com/open-mmlab/mmpose), [CVAT](https://www.cvat.ai/), and other COCO-compatible frameworks.

::: sleap_io.io.main.load_coco

::: sleap_io.io.main.save_coco

### Ultralytics YOLO Format

Support for [Ultralytics YOLO](https://docs.ultralytics.com/) pose format.

::: sleap_io.io.main.load_ultralytics

::: sleap_io.io.main.save_ultralytics

## Working with Multiple Datasets

### Load Multiple Files

Load and combine multiple pose tracking files:

::: sleap_io.io.main.load_labels_set

## Skeleton Files

Load and save skeleton definitions separately:

::: sleap_io.io.main.load_skeleton

::: sleap_io.io.main.save_skeleton

## Format Detection

sleap-io automatically detects file formats based on:

1. **File extension**: `.slp`, `.nwb`, `.h5`, `.json`, `.mat`, `.csv`
2. **File content**: For ambiguous extensions like `.h5` (JABS vs DLC) or `.json` (Label Studio vs COCO)
3. **Explicit format**: Pass `format` parameter to override auto-detection

## Format Conversion Examples

### Convert Between Formats

```python
import sleap_io as sio

# Load from any supported format
labels = sio.load_file("data.slp")

# Save to different formats
labels.save("data.nwb")  # NWB format
labels.save("data.labelstudio.json")  # Label Studio
labels.save("data_yolo/")  # Ultralytics YOLO
```

### Batch Conversion

```python
import sleap_io as sio
from pathlib import Path

# Convert all SLEAP files to NWB
for slp_file in Path("data/").glob("*.slp"):
    labels = sio.load_file(slp_file)
    nwb_file = slp_file.with_suffix(".nwb")
    labels.save(nwb_file)
```

### Round-Trip Preservation

Most formats preserve data during round-trip conversion:

```python
import sleap_io as sio

# Load original
labels_original = sio.load_file("data.slp")

# Save and reload
labels_original.save("temp.nwb")
labels_reloaded = sio.load_file("temp.nwb")

# Data is preserved
assert len(labels_original) == len(labels_reloaded)
assert labels_original.skeleton == labels_reloaded.skeleton
```

## Format Limitations

Different formats have varying capabilities:

| Format | Read | Write | Videos | Skeletons | Tracks | Confidence | User/Predicted |
|--------|------|-------|--------|-----------|--------|------------|----------------|
| SLEAP (.slp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NWB (.nwb) | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ | ✅ |
| JABS (.h5) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Analysis HDF5 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Label Studio | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| CSV (.csv) | ✅ | ✅ | ❌ | ✅** | ✅ | ✅ | ❌ |
| DeepLabCut | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| AlphaTracker | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| LEAP (.mat) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| COCO (.json) | ✅ | ✅ | ❌ | ✅ | ✅*** | ❌ | ✅ |
| Ultralytics | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |

*NWB can embed videos with `annotations_export` format
**CSV skeleton edges/symmetries preserved via optional metadata JSON sidecar
***COCO tracks are stored via `attributes.object_id` (CVAT-compatible)

## See Also

- [Data Model](../model.md): Understanding the core data structures
- [Examples](../examples.md): More usage examples and recipes
- [Merging](../merging.md): Combining data from multiple sources