# Data formats

sleap-io provides a unified interface for reading and writing pose tracking data across multiple formats. The library automatically detects file formats and provides harmonized I/O operations.

## Universal I/O Functions

::: sleap_io.load_file

::: sleap_io.save_file

## Video I/O

::: sleap_io.load_video

::: sleap_io.save_video

## Format-Specific Functions

### SLEAP Native Format (.slp)

The native SLEAP format stores complete pose tracking projects including videos, skeletons, and annotations.

::: sleap_io.load_slp

::: sleap_io.save_slp

### NWB Format (.nwb)

[Neurodata Without Borders (NWB)](https://www.nwb.org/) is a standardized format for neurophysiology data. sleap-io provides comprehensive support for both reading and writing pose tracking data in NWB format.

#### Harmonized NWB I/O

The harmonized API automatically detects and routes to the appropriate NWB backend:

::: sleap_io.load_nwb

::: sleap_io.save_nwb

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

::: sleap_io.load_jabs

::: sleap_io.save_jabs

### Label Studio Format (.json)

[Label Studio](https://labelstud.io/) is a multi-modal annotation platform. Export annotations from Label Studio and load them into SLEAP.

::: sleap_io.load_labelstudio

::: sleap_io.save_labelstudio

### DeepLabCut Format (.h5, .csv)

Load predictions from [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), a popular markerless pose estimation tool.

::: sleap_io.load_dlc

### AlphaTracker Format

Load predictions from [AlphaTracker](https://github.com/yinaanyachukwu/AlphaTracker), a tracking system for socially-housed animals.

::: sleap_io.load_alphatracker

### LEAP Format (.mat)

Load predictions from [LEAP](https://github.com/talmo/leap), a SLEAP predecessor. Requires `scipy` for .mat file support.

::: sleap_io.load_leap

### COCO Format (.json)

[COCO](https://cocodataset.org/) (Common Objects in Context) format is widely used in computer vision and pose estimation. sleap-io provides full read and write support, making it compatible with tools like [mmpose](https://github.com/open-mmlab/mmpose), [CVAT](https://www.cvat.ai/), and other COCO-compatible frameworks.

::: sleap_io.load_coco

::: sleap_io.save_coco

### Ultralytics YOLO Format

Support for [Ultralytics YOLO](https://docs.ultralytics.com/) pose format.

::: sleap_io.load_ultralytics

::: sleap_io.save_ultralytics

## Working with Multiple Datasets

### Load Multiple Files

Load and combine multiple pose tracking files:

::: sleap_io.load_labels_set

## Skeleton Files

Load and save skeleton definitions separately:

::: sleap_io.load_skeleton

::: sleap_io.save_skeleton

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
| Label Studio | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| DeepLabCut | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| AlphaTracker | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| LEAP (.mat) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| COCO (.json) | ✅ | ✅ | ❌ | ✅ | ✅** | ❌ | ✅ |
| Ultralytics | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |

*NWB can embed videos with `annotations_export` format
**COCO tracks are stored via `attributes.object_id` (CVAT-compatible)

## See Also

- [Data Model](model.md): Understanding the core data structures
- [Examples](examples.md): More usage examples and recipes
- [Merging](merging.md): Combining data from multiple sources