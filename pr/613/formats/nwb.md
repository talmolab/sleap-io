# NWB Format (.nwb)

[Neurodata Without Borders (NWB)](https://www.nwb.org/) is a standardized format for neurophysiology data. sleap-io provides comprehensive support for both reading and writing pose tracking data in NWB format.

## Harmonized NWB I/O

The harmonized API automatically detects and routes to the appropriate NWB backend:

::: sleap_io.io.main.load_nwb

::: sleap_io.io.main.save_nwb

## NWB Format Types

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

## Examples

### Basic NWB Usage

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

### Advanced Annotations API

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

### Export with Video Frames

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


### Multi-Subject Support

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

## NWB Metadata

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
