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
  "version": 1,
  "objects": {
    "1": {"track": "cell_1", "category": "neuron"},
    "2": {"track": "cell_2", "category": "glia"}
  }
}
```

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

Each returned [`LabelImage`][sleap_io.LabelImage] has `frame_idx` set to its position in the stack (0, 1, 2, ...).

## Writing

```python
import sleap_io as sio

# Write as a multi-page TIFF stack (default)
sio.save_label_images("output.tif", label_images, stack=True)
# Creates: output.tif + output.tif.meta.json

# Write as per-frame files in a directory
sio.save_label_images("output_dir/", label_images, stack=False)
# Creates: output_dir/0.tif, output_dir/1.tif, ... + output_dir/.meta.json
```

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
