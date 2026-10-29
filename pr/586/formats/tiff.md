# TIFF Label Image Format

TIFF files store dense integer label images for instance segmentation. Each pixel value is a non-negative integer: `0` is background, and positive values identify distinct objects. This is the standard output of cell segmentation tools like [Cellpose](https://www.cellpose.org/) and [StarDist](https://github.com/stardist/stardist).

sleap-io reads and writes these label images as [`LabelImage`][sleap_io.LabelImage] objects via the [`tifffile`](https://pypi.org/project/tifffile/) library.

## File Structures

Three layouts are supported:

| Layout | Description | Path argument |
|--------|-------------|---------------|
| **Single TIFF** | One 2D frame | `"frame.tif"` |
| **Multi-page stack** | One page per frame in a single file | `"labels.tif"` |
| **Directory** | One `.tif`/`.tiff` per frame, sorted alphanumerically | `"labels_dir/"` |

## Sidecar Metadata

When writing, a JSON sidecar file is created alongside the TIFF at `{path}.meta.json`. It stores track names and category strings for each label ID so they survive a round trip:

```json
{
  "format": "sleap-io-label-image-meta",
  "version": 3,
  "axes": "YX",
  "objects": {
    "1": {"track": "cell_1", "category": "neuron"},
    "2": {"track": "cell_2", "category": "glia"}
  }
}
```

`axes` is `"YX"` for a single label image and `"TYX"` for a multi-frame stack; the reader uses it as the authoritative layout hint. Optional `scale` and `offset` keys are added when any label image carries a spatial transform.

On read, the sidecar is loaded automatically if present. Without it, tracks are auto-created with the label ID as the name and categories are left empty.

## Reading

```python
import sleap_io as sio

# Single TIFF or multi-page stack
label_images = sio.load_label_images("labels.tif")

# Directory of per-frame TIFFs
label_images = sio.load_label_images("labels_dir/")

# With an associated video
video = sio.load_video("experiment.mp4")
label_images = sio.load_label_images("labels.tif", video=video)

# With explicit track/category mappings (overrides sidecar)
from sleap_io import Track
tracks = {1: Track(name="cell_A"), 2: Track(name="cell_B")}
categories = {1: "neuron", 2: "glia"}
label_images = sio.load_label_images(
    "labels.tif", tracks=tracks, categories=categories
)
```

Each returned [`LabelImage`][sleap_io.LabelImage] corresponds to one frame in the stack, ordered by position (0, 1, 2, ...).

## Writing

```python
import sleap_io as sio

# Write as a multi-page TIFF stack (default)
sio.save_label_images("output.tif", label_images, stack=True)
# Creates: output.tif + output.tif.meta.json

# Write as per-frame files in a directory
sio.save_label_images("output_dir/", label_images, stack=False)
# Creates: output_dir/0.tif, output_dir/1.tif, ... + output_dir.meta.json (sibling of the dir)
```

## Cellpose Workflow Example

A common workflow is to run [Cellpose](https://www.cellpose.org/) on microscopy
data, convert the output masks to `LabelImage` objects, and save them as TIFF or
SLP:

```python
import numpy as np
import sleap_io as sio
from cellpose import models

# Run Cellpose segmentation
model = models.CellposeModel(model_type="nuclei")
masks, flows, styles = model.eval(images, diameter=25)
masks_stack = np.stack(masks)  # (T, H, W) int32

# Convert to LabelImage objects with consistent tracks across frames
video = sio.Video(filename="experiment.tif")
label_images = sio.PredictedLabelImage.from_stack(
    masks_stack,
    source="cellpose:nuclei",
    create_tracks=True,
    score=1.0,
)

# Save as TIFF stack (with sidecar metadata)
sio.save_label_images("cellpose_masks.tif", label_images)

# Or save as SLP (preserves tracks, categories, and provenance)
labeled_frames = []
for i, li in enumerate(label_images):
    lf = sio.LabeledFrame(video=video, frame_idx=i)
    lf.append(li)  # dispatches to lf.label_images
    labeled_frames.append(lf)
labels = sio.Labels(labeled_frames=labeled_frames, videos=[video])
labels.provenance["segmentation_model"] = "cellpose"
labels.provenance["cellpose_diameter"] = 25
labels.save("cellpose_masks.slp")
```

`lf.append(li)` is the idiomatic way to attach a `LabelImage` to a frame — it routes the annotation onto `lf.label_images` via the type-dispatched [`LabeledFrame.append`][sleap_io.LabeledFrame.append]. Constructing `LabeledFrame(..., label_images=[li])` directly is still supported.

The `from_stack()` method ensures that the same `Track` object is shared across
frames for a given label ID, which is essential for consistent tracking and
downstream analysis.

!!! tip "TIFF → SLP follow-ups"
    - [`sio.normalize_label_ids`][sleap_io.normalize_label_ids] rewrites per-frame label IDs so they are globally consistent across a stack — essential when upstream segmentation assigns different IDs in different frames (e.g., Cellpose without tracking). See [Regions → Normalizing label IDs](../model/segmentation.md#normalizing-label-ids).
    - [`sio.merge_label_images`][sleap_io.merge_label_images] concatenates multiple chunked SLP files (e.g., parallel batch segmentation shards) via zero-decompression HDF5 chunk copies. See [Examples → Parallel segmentation pipeline](../examples.md#parallel-segmentation-pipeline).
    - [`sio.LabelImageWriter`][sleap_io.LabelImageWriter] streams `LabelImage` frames one at a time into SLP with constant memory — ideal for TIFF-stack pipelines that don't fit in memory. See [Regions → Streaming writes](../model/segmentation.md#streaming-writes).

!!! note "See also"
    [Formats → COCO Panoptic segmentation](index.md#coco-panoptic-segmentation) — the COCO Panoptic reader/writer uses the same `LabelImage` model, so the same post-processing helpers apply.

## API

::: sleap_io.io.main.load_label_images
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.main.save_label_images
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.tiff.read_label_images
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.tiff.write_label_images
    options:
      heading_level: 3
      show_root_toc_entry: false
