# Segmentation

> **Spatial annotations:** [Centroids](centroids.md) · [Boxes](boxes.md) · [ROIs](rois.md) · [Segmentation](segmentation.md). These types nest per-frame on [`LabeledFrame`](labels.md) — see [Working with annotations in frames](index.md#working-with-annotations-in-frames).

sleap-io represents pixel-level segmentation with two complementary types: [`SegmentationMask`][sleap_io.SegmentationMask] for per-object binary masks (run-length encoded), and [`LabelImage`][sleap_io.LabelImage] for a dense integer label image holding **all** objects of a frame in one array — the standard output of instance-segmentation tools like [Cellpose](https://www.cellpose.org/) and [StarDist](https://github.com/stardist/stardist).

---

## Segmentation masks

A [`SegmentationMask`][sleap_io.SegmentationMask] stores per-pixel binary annotations in a compact
run-length encoded (RLE) format. RLE avoids storing the full raster array,
making masks efficient for storage and serialization while still supporting
fast conversion to and from numpy arrays. `SegmentationMask` is abstract — use
[`UserSegmentationMask`][sleap_io.UserSegmentationMask] or [`PredictedSegmentationMask`][sleap_io.PredictedSegmentationMask].

### From a numpy array

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((480, 640), dtype=bool)
>>> mask_data[100:200, 150:300] = True  # rectangular region
>>> mask = sio.UserSegmentationMask.from_numpy(
...     mask_data,
... )
>>> print(mask.area)
>>> print(mask.height)
>>> print(mask.width)

```

### Decoding back to numpy

The `.data` property decodes the RLE back to a full boolean array:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> mask = sio.UserSegmentationMask.from_numpy(mask_data)
>>> decoded = mask.data
>>> print(decoded.shape)
>>> print(decoded.dtype)
>>> print(decoded.sum())

```

### Bounding box of a mask

The `.bbox` property returns the tightest axis-aligned bounding box containing
all foreground pixels as `(x, y, width, height)`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> mask = sio.UserSegmentationMask.from_numpy(mask_data)
>>> print(mask.bbox)

```

To get a full `BoundingBox` object (with metadata) instead of a raw tuple, use
`.to_bbox()`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> mask = sio.UserSegmentationMask.from_numpy(
...     mask_data, track=sio.Track(name="cell_A"), category="neuron",
... )
>>> bb = mask.to_bbox()
>>> print(bb.xyxy)
>>> print(bb.track.name)
>>> print(bb.centroid_xy)

```

The returned `BoundingBox` inherits track, category, name, instance, and source
from the mask. `PredictedSegmentationMask` produces a `PredictedBoundingBox`
with the mask's score.

### User vs. predicted segmentation masks

[`UserSegmentationMask`][sleap_io.UserSegmentationMask] and [`PredictedSegmentationMask`][sleap_io.PredictedSegmentationMask] distinguish human
annotations from model predictions. `PredictedSegmentationMask` adds `score`
and optional `score_map` fields:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> user_mask = sio.UserSegmentationMask.from_numpy(mask_data)
>>> print(user_mask.is_predicted)
>>> pred_mask = sio.PredictedSegmentationMask.from_numpy(
...     mask_data, score=0.87,
... )
>>> print(pred_mask.score)
>>> print(pred_mask.score_map)  # None unless explicitly set

```

The `score_map` field is an optional dense `float32` array of shape `(H, W)`
providing pixel-level confidence. It is stored separately in the SLP format
using zlib compression to avoid bloating files.

#### Adopting predictions (human-in-the-loop)

`PredictedSegmentationMask.to_user()` converts a prediction into a
`UserSegmentationMask`, the predicted -> human-correct -> retrain round-trip for
masks (mirroring `Instance.from_predicted` for poses). It copies the mask raster
and shared metadata (`track`, `instance`, `name`, `category`, `source`,
`tracking_score`, `scale`, `offset`), drops the prediction-only fields
(`score`, `score_map`), and records the source prediction on `from_predicted`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> pred_mask = sio.PredictedSegmentationMask.from_numpy(mask_data, score=0.87)
>>> user_mask = pred_mask.to_user()
>>> print(user_mask.is_predicted)
>>> print(user_mask.from_predicted is pred_mask)

```

Pass `to_user(link=False)` for an unlinked copy. The `from_predicted` link is
persisted to the SLP format as an index into the saved mask list (mirroring
instance `from_predicted`), so it survives a save/load round-trip as long as the
source prediction is also saved. Files written before this column existed load
it as `None`.

This mirrors the pose flow, where a user `Instance` is created from a
`PredictedInstance` with `from_predicted=` set. To adopt a prediction within a
frame, append the user mask to `frame.masks`; the source prediction stays in the
frame, exactly as predicted poses do, so the "predicted + user → replace"
resolution can happen later at merge time (see [Merging](../merging.md)):

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> video = sio.Video(filename="example.mp4", open_backend=False)
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> pred_mask = sio.PredictedSegmentationMask.from_numpy(mask_data, score=0.87)
>>> frame = sio.LabeledFrame(video=video, frame_idx=0, masks=[pred_mask])
>>> frame.masks.append(pred_mask.to_user())
>>> print(len(frame.masks))  # prediction stays, user mask appended alongside

```

To find predicted masks that have **not** yet been corrected — the segmentation
analogue of `LabeledFrame.unused_predictions` for poses — use
`LabeledFrame.unused_predicted_masks`. A `PredictedSegmentationMask` is treated
as adopted (and excluded) when a `UserSegmentationMask` in the same frame links
to it via `from_predicted` (checked first), or, lacking a link, spatially
overlaps it (bbox-centroid within 5 px). This drives the "retrain only what a
human corrected" workflow:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> video = sio.Video(filename="example.mp4", open_backend=False)
>>> mask_data = np.zeros((100, 100), dtype=bool)
>>> mask_data[20:40, 30:60] = True
>>> pred_a = sio.PredictedSegmentationMask.from_numpy(mask_data, score=0.87)
>>> pred_b = sio.PredictedSegmentationMask.from_numpy(mask_data, score=0.62)
>>> pred_b.offset = (500.0, 500.0)  # a separate prediction elsewhere in the frame
>>> frame = sio.LabeledFrame(video=video, frame_idx=0, masks=[pred_a, pred_b])
>>> frame.masks.append(pred_a.to_user())  # adopt pred_a, leave pred_b
>>> unused = frame.unused_predicted_masks  # only the uncorrected prediction
>>> len(unused), unused[0] is pred_b
(1, True)

```

### Multi-resolution masks

Segmentation masks stored at lower resolution — e.g., from a model that
downsamples inputs, or a cropped tile extracted from a larger image — can carry
spatial metadata so they round-trip losslessly into image-pixel space.

`SegmentationMask.scale` is a `(sx, sy)` scale factor and `offset` is an
`(x, y)` pixel shift. Together they define the transform
`image_coord = mask_coord / scale + offset`. Use the `stride=` convenience
argument to set an isotropic downsample ratio, or pass `offset=` directly:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> half_res = np.zeros((240, 320), dtype=bool)
>>> half_res[60:120, 80:180] = True
>>> mask = sio.UserSegmentationMask.from_numpy(half_res, stride=2)
>>> print(mask.scale)  # (0.5, 0.5) — equivalent to stride=2
>>> print(mask.bbox)   # bbox is returned in full-image coordinates

```

Crop-space masks (e.g., from a detector that processes `(H, W)` tiles) use the
`offset=` kwarg — `mask.bbox` then maps back into image-pixel space without any
extra math on your side:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> crop = np.zeros((50, 50), dtype=bool)
>>> crop[10:30, 15:40] = True
>>> mask = sio.UserSegmentationMask.from_numpy(crop, offset=(100.0, 200.0))
>>> print(mask.offset)
>>> print(mask.bbox)  # offset + extent → image coordinates

```

Call `mask.resampled(target_height, target_width)` to materialize the mask at a
new resolution while preserving its spatial metadata for further transforms.

!!! note "Also on `LabelImage`"
    The same `scale` / `offset` / `resampled()` convention applies to
    [`LabelImage`](#label-images) — a single `LabelImage` can carry a whole
    frame's worth of objects at a downsampled resolution, with the object-id
    metadata mapping back to full-image pixel space via the same transform.

### Metadata fields

Every segmentation mask can carry optional metadata:

| Field            | Type               | Description                                  |
| ---------------- | ------------------ | -------------------------------------------- |
| `track`          | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None`    | Confidence of track identity assignment      |
| `instance`       | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `scale`          | `tuple[float, float]` | `(sx, sy)` spatial scale (default `(1, 1)`) |
| `offset`         | `tuple[float, float]` | `(x, y)` pixel offset (default `(0, 0)`)    |
| `category`       | `str`              | Class label (e.g., `"neuron"`)              |
| `name`           | `str`              | Human-readable name                          |
| `source`         | `str`              | Annotation source identifier                 |

!!! tip "Rendering"
    Use [`sio.draw_masks`][sleap_io.draw_masks] to composite a sequence of segmentation masks onto an image, or include them in [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video] overlays. See [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays).


## Label images

A [`LabelImage`][sleap_io.LabelImage] stores dense per-pixel instance segmentation for a single video
frame, where each pixel value encodes which object occupies that pixel. Unlike a
[`SegmentationMask`][sleap_io.SegmentationMask] — which is a binary mask for a single object — a
`LabelImage` stores **all** objects in one integer array. Background pixels are
`0`, and each positive integer identifies a distinct object. The `objects` dict
maps these IDs to metadata (track, category, name). `LabelImage` is abstract —
use [`UserLabelImage`][sleap_io.UserLabelImage] or [`PredictedLabelImage`][sleap_io.PredictedLabelImage].

### From a numpy array

The `from_numpy` factory method is the easiest way to create a `LabelImage`.
Pass an integer array where each unique positive value is an object:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((128, 128), dtype=np.int32)
>>> data[10:40, 10:40] = 1   # object 1
>>> data[60:90, 60:90] = 2   # object 2
>>> li = sio.UserLabelImage.from_numpy(data)
>>> print(li.n_objects)
>>> print(li.label_ids)

```

By default, `from_numpy` does not create tracks. Set `create_tracks=True` to
auto-create one `Track` per unique label ID, or supply explicit tracks and
categories:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((128, 128), dtype=np.int32)
>>> data[10:40, 10:40] = 1
>>> data[60:90, 60:90] = 2
>>> tracks = [sio.Track(name="cell_A"), sio.Track(name="cell_B")]
>>> li = sio.UserLabelImage.from_numpy(
...     data,
...     tracks=tracks,
...     categories=["neuron", "glia"],
... )
>>> print(li.tracks)
>>> print(li.categories)

```

Lists are positional (`tracks[0]` maps to label 1, etc.). Dict mappings give
explicit control:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((128, 128), dtype=np.int32)
>>> data[10:40, 10:40] = 5
>>> data[60:90, 60:90] = 10
>>> li = sio.UserLabelImage.from_numpy(
...     data,
...     tracks={5: sio.Track(name="A"), 10: sio.Track(name="B")},
...     categories={5: "neuron", 10: "glia"},
... )
>>> print(li.n_objects)

```

### From segmentation masks

Compose a `LabelImage` from existing `SegmentationMask` objects. Each mask
becomes one object with a unique label ID. Track, category, and name are
inherited from each mask's metadata:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask1 = sio.UserSegmentationMask.from_numpy(
...     np.ones((64, 64), dtype=bool), track=sio.Track(name="A"),
... )
>>> mask2 = sio.UserSegmentationMask.from_numpy(
...     np.ones((64, 64), dtype=bool), track=sio.Track(name="B"),
... )
>>> li = sio.UserLabelImage.from_masks([mask1, mask2])
>>> print(li.n_objects)
>>> print(li.tracks)

```

### From binary masks

When you have per-object binary masks — e.g., from SAM, Mask R-CNN, or similar
instance segmentation tools — use `from_binary_masks` to composite them into a
single `LabelImage` without constructing `SegmentationMask` objects first:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_a = np.zeros((64, 64), dtype=bool)
>>> mask_b = np.zeros((64, 64), dtype=bool)
>>> mask_a[10:30, 10:30] = True
>>> mask_b[40:60, 40:60] = True
>>> li = sio.PredictedLabelImage.from_binary_masks(
...     [mask_a, mask_b],
...     tracks=[sio.Track(name="cell_A"), sio.Track(name="cell_B")],
...     categories=["neuron", "glia"],
...     scores=[0.95, 0.87],
...     score=0.9,
... )
>>> print(li.n_objects)
>>> print(li.objects[1].track.name, li.objects[1].score)

```

Accepts a list of `(H, W)` arrays, a stacked `(N, H, W)` array, or a single
`(H, W)` array. Values are cast to bool (nonzero = True). Overlapping pixels
are assigned to the last mask. Use `create_tracks=True` to auto-create tracks
instead of providing them explicitly.

Use `label_ids` to control pixel values explicitly — useful when objects
appear/disappear across frames and you need consistent values per track:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> mask_a = np.zeros((64, 64), dtype=bool)
>>> mask_b = np.zeros((64, 64), dtype=bool)
>>> mask_a[10:30, 10:30] = True
>>> mask_b[40:60, 40:60] = True
>>> li = sio.PredictedLabelImage.from_binary_masks(
...     [mask_a, mask_b],
...     label_ids=[5, 10],
...     scores=[0.95, 0.87],
... )
>>> print(li.label_ids)
[ 5 10]

```

!!! tip "When to use which factory method"
    - **`from_binary_masks`**: Per-object binary masks from SAM, Mask R-CNN, etc.
    - **`from_numpy`**: Pre-composited integer array from Cellpose, StarDist, etc.
    - **`from_masks`**: Existing `SegmentationMask` objects with rich metadata.
    - **`from_stack`**: `(T, H, W)` integer array for multiple frames at once.

### Direct construction

For full control, construct directly with the `data` array and `objects` dict:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.array([[0, 1], [2, 0]], dtype=np.int32)
>>> li = sio.UserLabelImage(
...     data=data,
...     objects={
...         1: sio.LabelImage.Info(
...             track=sio.Track(name="cell_1"), category="neuron",
...         ),
...         2: sio.LabelImage.Info(
...             track=sio.Track(name="cell_2"), category="glia",
...         ),
...     },
... )
>>> print(li.n_objects)

```

### Object metadata

Each non-zero label ID can have a `LabelImage.Info` entry in the `objects` dict:

| Field       | Type                             | Description                                |
| ----------- | -------------------------------- | ------------------------------------------ |
| `track`     | [`Track`](poses.md) `\| None`   | Cross-frame identity                       |
| `category`  | `str`                            | Semantic class label (e.g., `"neuron"`)    |
| `name`      | `str`                            | Human-readable name (e.g., `"cell_042"`)   |
| `instance`  | [`Instance`](poses.md) `\| None`| Linked pose instance                       |
| `score`     | `float \| None`                  | Per-object confidence score                |

Label IDs not present in `objects` are treated as having default (empty)
metadata.

### Querying objects

Index with a `Track` to get the binary mask for that object:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((64, 64), dtype=np.int32)
>>> data[10:30, 10:30] = 1
>>> track = sio.Track(name="cell_A")
>>> li = sio.UserLabelImage.from_numpy(data, tracks={1: track})
>>> mask = li[track]
>>> print(mask.dtype)
>>> print(mask.sum())

```

Test containment, iterate objects, or get a union mask by category:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((64, 64), dtype=np.int32)
>>> data[10:30, 10:30] = 1
>>> track = sio.Track(name="cell_A")
>>> li = sio.UserLabelImage.from_numpy(data, tracks={1: track})
>>> print(track in li)
True
>>> for track, category, mask in li.items():
...     print(f"{track.name}: {category}, {mask.sum()} px")
cell_A: , 400 px

```

| Property      | Type              | Description                                    |
| ------------- | ----------------- | ---------------------------------------------- |
| `n_objects`   | `int`             | Number of unique non-zero labels               |
| `label_ids`   | `np.ndarray`      | Sorted array of non-zero label values          |
| `tracks`      | `list[Track]`     | Tracks with non-None track in objects           |
| `categories`  | `set[str]`        | Unique non-empty category strings               |
| `height`      | `int`             | Image height in pixels                          |
| `width`       | `int`             | Image width in pixels                           |

### Decomposing to SegmentationMasks

A `LabelImage` can be decomposed into per-object binary `SegmentationMask`
objects and reconstructed from them:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((64, 64), dtype=np.int32)
>>> data[10:30, 10:30] = 1
>>> data[40:60, 40:60] = 2
>>> li = sio.UserLabelImage.from_numpy(data)
>>> masks = li.to_masks()
>>> print(len(masks))
>>> print(masks[0].area)
>>> li2 = sio.UserLabelImage.from_masks(masks)
>>> print(li2.n_objects)

```

### Extracting bounding boxes

Extract per-object bounding boxes directly from a `LabelImage` with
`.to_bboxes()`. This is more efficient than decomposing to masks first, since it
only needs the pixel extents (no full mask decode):

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((64, 64), dtype=np.int32)
>>> data[10:30, 10:30] = 1
>>> data[40:60, 40:60] = 2
>>> li = sio.UserLabelImage.from_numpy(
...     data,
...     tracks=[sio.Track(name="cell_A"), sio.Track(name="cell_B")],
...     categories=["neuron", "glia"],
... )
>>> bboxes = li.to_bboxes()
>>> print(len(bboxes))
>>> for bb in bboxes:
...     print(bb.track.name, bb.category, bb.xyxy, bb.centroid_xy)

```

Each `BoundingBox` inherits track, category, name, instance, and source from the
corresponding object. `PredictedLabelImage` produces `PredictedBoundingBox`
objects with per-object scores (falling back to the image-level score when a
per-object score is `None`). Bounding boxes are in image coordinates, respecting
the label image's scale and offset.

### User vs. predicted label images

[`UserLabelImage`][sleap_io.UserLabelImage] and [`PredictedLabelImage`][sleap_io.PredictedLabelImage] distinguish human annotations from
model predictions. `PredictedLabelImage` adds `score` and optional `score_map`
fields, and per-object scores can be set via `LabelImage.Info.score`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> data = np.zeros((128, 128), dtype=np.int32)
>>> data[10:40, 10:40] = 1
>>> data[60:90, 60:90] = 2
>>> pred_li = sio.PredictedLabelImage(
...     data=data,
...     objects={
...         1: sio.LabelImage.Info(category="neuron", score=0.92),
...         2: sio.LabelImage.Info(category="glia", score=0.78),
...     },
...     score=0.88,
... )
>>> print(pred_li.is_predicted)
>>> print(pred_li.score)
>>> print(pred_li.objects[1].score)

```

### From a stack of frames

The `from_stack` factory method converts a `(T, H, W)` array (e.g., direct
Cellpose output) into a list of `LabelImage` objects with consistent `Track`
objects shared across frames:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> masks = np.zeros((3, 64, 64), dtype=np.int32)
>>> masks[0, 10:30, 10:30] = 1
>>> masks[1, 10:30, 10:30] = 1
>>> masks[2, 20:40, 20:40] = 2
>>> label_images = sio.PredictedLabelImage.from_stack(
...     masks, create_tracks=True, score=1.0,
... )
>>> print(len(label_images))
3
>>> print(label_images[0].objects[1].track is label_images[1].objects[1].track)
True

```

### TIFF I/O

Label images can be saved and loaded as TIFF files using the top-level
functions `sio.load_label_images()` and `sio.save_label_images()`. See
the [TIFF Format Reference](../formats/tiff.md) for details on file
structures and sidecar metadata.

### Streaming writes

For large datasets where holding all frames in memory is impractical,
[`LabelImageWriter`][sleap_io.LabelImageWriter] writes label images one at a
time to an SLP file with constant memory usage:

```python
import sleap_io as sio

video = sio.load_video("microscopy.tif")
with sio.LabelImageWriter("output.slp", video=video) as writer:
    for frame_idx, mask in enumerate(segmentation_results):
        li = sio.PredictedLabelImage.from_numpy(
            mask,
            source="cellpose:nuclei", create_tracks=True, score=1.0,
        )
        writer.add(li)
# File is finalized on context exit
```

The writer uses the chunked `(T, H, W)` format (v2.2) with
`write_direct_chunk` for maximum throughput. The HDF5 file and pixel dataset
are created lazily on the first `add()` call. All frames must have the same
`(H, W)` dimensions.

Key features:

- **Constant memory**: Only one frame's compressed data is in memory at a time.
- **Exponential growth**: The dataset starts at `initial_capacity` frames and
  doubles when full, then is trimmed to the actual count on finalize.
- **Score maps**: `PredictedLabelImage` score maps are supported.
- **Batch convenience**: `writer.add_batch(list_of_label_images)` writes
  multiple frames in one call.

### Merging label images

[`merge_label_images()`][sleap_io.merge_label_images] combines label images
from multiple SLP files into one, copying compressed chunks directly without
decompression when possible:

```python
import sleap_io as sio

merged = sio.merge_label_images(
    ["chunk_0.slp", "chunk_1.slp", "chunk_2.slp"],
    "merged.slp",
)
print(len(merged.label_images))  # Total across all sources
```

This is useful for parallelized segmentation workflows where each chunk of
frames is processed independently and the results need to be combined. Videos
are deduplicated by filename and tracks by name. All source files must have the
same frame dimensions `(H, W)`.

### Normalizing label IDs

When label images come from different sources or segmentation runs, the same
Track may have different pixel values in different frames.
[`normalize_label_ids()`][sleap_io.normalize_label_ids] rewrites pixel values
so each Track gets a globally consistent label ID (1, 2, 3, ...) assigned in
order of first appearance:

```python
import sleap_io as sio

labels = sio.load_slp("segmented.slp")
track_map = sio.normalize_label_ids(labels.label_images, by="track")
# Now the same Track always has the same pixel value in every frame.

# Safe to stack into a (T, H, W) array:
import numpy as np

stack = np.stack([li.data for li in labels.label_images])
```

For semantic segmentation, group by category instead — all objects with the same
category string merge into one pixel value per frame:

```python
sio.normalize_label_ids(labels.label_images, by="category")
```

!!! tip "Rendering"
    Use [`sio.draw_label_image`][sleap_io.draw_label_image] to composite a single `LabelImage` onto an arbitrary image, or pass `overlay=label_stack_or_image` to [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video]. See [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays) for the full overlay pipeline.

### Lazy loading

When loading SLP files, label image pixel data is loaded lazily — metadata
(tracks, frame indices, categories) is available immediately, and the actual
pixel array is decompressed only on first `.data` access:

```python
labels = sio.load_slp("large_dataset.slp")

# Metadata queries — no decompression
li = labels.get_label_images(frame_idx=42)[0]
print(li.tracks)      # free
print(li.height)      # free (cached from metadata)

# Pixel data decompressed on first access, then cached
mask = li.data  # decompresses this frame only
```

This keeps memory usage proportional to the number of frames actually accessed
rather than the total dataset size. The underlying HDF5 file handle is managed
by the `Labels` object and can be explicitly closed with `labels.close()`.


## Converting between annotation types

The four spatial annotation types can be converted between each other where
geometrically meaningful:

| From                | To                    | Method                            |
| ------------------- | --------------------- | --------------------------------- |
| `BoundingBox`       | `ROI`                 | `bbox.to_roi()`                   |
| `BoundingBox`       | `SegmentationMask`    | `bbox.to_mask(height, width)`     |
| `ROI`               | `SegmentationMask`    | `roi.to_mask(height, width)`      |
| `SegmentationMask`  | `ROI` (polygon)       | `mask.to_polygon()`               |
| `SegmentationMask`  | `BoundingBox`         | `mask.to_bbox()`                  |
| `PredictedSegmentationMask` | `UserSegmentationMask` | `pred_mask.to_user()`     |
| `LabelImage`        | `list[SegmentationMask]` | `li.to_masks()`                |
| `LabelImage`        | `list[BoundingBox]`   | `li.to_bboxes()`                  |
| `list[SegmentationMask]` | `LabelImage`     | `UserLabelImage.from_masks(masks)` |

All conversions preserve metadata (track, instance, name,
category, source) when applicable. `to_bbox()` and `to_bboxes()` preserve
prediction semantics (`Predicted*` inputs produce `Predicted*` outputs with
scores). Other conversions return `User*` types.

The geometry conversions above project to a different shape, so they carry the
identifying metadata but drop `tracking_score` (the geometry, not the track
assignment, is what's being reinterpreted). `to_user()` is different: it is the
predicted -> user *adoption* path (see
[below](#user-vs-predicted-segmentation-masks)), so it faithfully carries the
full annotation — including `tracking_score`, `scale`, and `offset` — dropping
only the prediction-only fields (`score`, `score_map`) and recording the source
prediction on `from_predicted`.

```pycon
>>> import sleap_io as sio
>>> # BoundingBox -> ROI -> SegmentationMask -> polygon ROI
>>> bbox = sio.UserBoundingBox(
...     x1=40, y1=35, x2=60, y2=65,
... )
>>> roi = bbox.to_roi()
>>> print(roi.area)
>>> mask = roi.to_mask(100, 100)
>>> print(mask.area)
>>> polygon_roi = mask.to_polygon()
>>> print(polygon_roi.area)

```

---

!!! note "See also"

    - **[Centroids](centroids.md)**, **[Boxes](boxes.md)**, **[ROIs](rois.md)** — the other spatial annotation types.
    - **[Labels & Frames](labels.md)**: Accessing masks/label images via `labels.masks`, `labels.label_images`, and `get_masks()` / `get_label_images()`.
    - **[Rendering](../rendering.md)**: Visualizing segmentation overlays on video frames.
    - **[TIFF Format](../formats/tiff.md)**: Reading and writing label images as TIFF with sidecar metadata.
    - **[SLP Format](../formats/slp.md#label-images)**: HDF5 storage layout (blob and chunked formats).
    - **[Merging](../merging.md#merging-label-images)**: Combining label images from multiple SLP files.

---

## API reference

::: sleap_io.SegmentationMask

::: sleap_io.UserSegmentationMask

::: sleap_io.PredictedSegmentationMask

::: sleap_io.LabelImage

::: sleap_io.UserLabelImage

::: sleap_io.PredictedLabelImage

::: sleap_io.LabelImageWriter

::: sleap_io.merge_label_images

::: sleap_io.normalize_label_ids
