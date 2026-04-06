# Regions

Beyond keypoint poses, sleap-io supports spatial annotation types: bounding
boxes, regions of interest (vector polygons), segmentation masks (raster), and
label images (dense instance segmentation). These can be associated with videos,
frames, tracks, and instances, and are stored on the [`Labels`](labels.md) object
via `labels.bboxes`, `labels.rois`, `labels.masks`, and `labels.label_images`.

sleap-io provides four spatial annotation types with different trade-offs:

- **`BoundingBox`** — axis-aligned or rotated rectangles, stored as corner
  coordinates. Has `UserBoundingBox` and `PredictedBoundingBox` subtypes for
  distinguishing human annotations from model outputs.
- **`ROI`** — arbitrary vector geometry via Shapely (polygons, multi-polygons,
  etc.). Can be static (whole video) or per-frame. Has `UserROI` and
  `PredictedROI` subtypes.
- **`SegmentationMask`** — per-pixel binary masks stored as run-length encoding
  for compactness. One mask per object. Has `UserSegmentationMask` and
  `PredictedSegmentationMask` subtypes.
- **`LabelImage`** — dense per-pixel integer label image storing **all** objects
  for a frame in one array. Standard output of instance segmentation tools like
  [Cellpose](https://www.cellpose.org/) and
  [StarDist](https://github.com/stardist/stardist). Has `UserLabelImage` and
  `PredictedLabelImage` subtypes.

All four base classes are **abstract** — use the `User*` or `Predicted*`
subclass to create instances. All four can be associated with a video, frame,
track, and instance. They are stored on [`Labels`](labels.md) via
`labels.bboxes`, `labels.rois`, `labels.masks`, and `labels.label_images`, and
can be converted between each other (bbox -> ROI -> mask, label image <-> masks).

---

## Bounding boxes

A `BoundingBox` represents a rectangular region defined by its corner
coordinates (`x1`, `y1`, `x2`, `y2`) and an optional rotation angle. Bounding
boxes are the primary annotation type for object detection workflows.
`BoundingBox` is abstract — use `UserBoundingBox` or `PredictedBoundingBox`.

### Direct construction

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     video=video, frame_idx=0,
... )
>>> print(bbox.area)
>>> print(bbox.xyxy)
>>> print(bbox.x_center)  # computed property
>>> print(bbox.width)      # computed property

```

The `x_center`, `y_center`, `width`, and `height` fields are available as
read-only computed properties.

### From corner coordinates

The `from_xyxy` factory method creates a bounding box from `(x1, y1, x2, y2)`
corner coordinates:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox2 = sio.UserBoundingBox.from_xyxy(75, 160, 125, 240, video=video, frame_idx=0)
>>> print(bbox2.x_center)
>>> print(bbox2.width)

```

There is also `from_xywh` for `(x, y, width, height)` format where `(x, y)` is
the top-left corner:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox3 = sio.UserBoundingBox.from_xywh(75, 160, 50, 80, video=video, frame_idx=0)
>>> print(bbox3.x_center)
>>> print(bbox3.y_center)

```

### User vs. predicted bounding boxes

`UserBoundingBox` and `PredictedBoundingBox` distinguish human annotations from
model predictions. `PredictedBoundingBox` adds a `score` field for confidence:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> user_bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     video=video, frame_idx=0,
... )
>>> print(user_bbox.is_predicted)
>>> pred_bbox = sio.PredictedBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     video=video, frame_idx=0, score=0.95,
... )
>>> print(pred_bbox.score)
>>> print(pred_bbox.is_predicted)

```

### Rotated bounding boxes

Set `angle` (in radians) to create an oriented bounding box. Rotated boxes
support `corners` and `bounds` but not `xyxy` or `xywh`, since those are only
meaningful for axis-aligned rectangles:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> rotated = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     angle=0.785, video=video, frame_idx=0,
... )
>>> print(rotated.is_rotated)
>>> print(rotated.corners.shape)
>>> print(rotated.bounds)  # axis-aligned extent of the rotated box

```

### Metadata fields

Every bounding box can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `video`     | [`Video`](video.md) `\| None`    | Associated video                             |
| `frame_idx` | `int \| None`      | Frame index within the video                 |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `category`  | `str`              | Class label (e.g., `"mouse"`)                |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | Annotation source identifier                 |

---

## Regions of interest

An `ROI` represents a vector geometry annotation using
[Shapely](https://shapely.readthedocs.io/) geometries. ROIs are suitable for
defining arenas, exclusion zones, or arbitrary spatial regions: anything that
is naturally described by a polygon or set of polygons rather than a simple
rectangle. `ROI` is abstract — use `UserROI` or `PredictedROI`.

### Static vs. temporal ROIs

An ROI is **static** when `frame_idx` is `None`, meaning it applies to all
frames of the video (e.g., an arena boundary). When `frame_idx` is set, the ROI
applies only to that specific frame.

### From polygon coordinates

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_poly = sio.UserROI.from_polygon(
...     [(0, 0), (100, 0), (100, 100), (0, 100)],
...     video=video,
... )
>>> print(roi_poly.area)
>>> print(roi_poly.centroid)

```

### From Shapely geometry

Construct an ROI directly from any Shapely geometry object:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi = sio.UserROI(geometry=box(10, 20, 100, 200), video=video)
>>> print(roi.area)
>>> print(roi.bounds)
>>> print(roi.is_static)

```

### From a BoundingBox object

Any `BoundingBox` can be converted to an `ROI` with `.to_roi()`:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     video=video, frame_idx=0,
... )
>>> roi_from_bbox = bbox.to_roi()
>>> print(roi_from_bbox.area)

```

### Multi-polygon ROIs

For disjoint regions, use `from_multi_polygon`:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_multi = sio.UserROI.from_multi_polygon(
...     [
...         [(0, 0), (10, 0), (10, 10), (0, 10)],
...         [(50, 50), (60, 50), (60, 60), (50, 60)],
...     ],
...     video=video,
... )
>>> print(roi_multi.area)

```

Multi-geometry ROIs can be split into individual ROIs with `.explode()`:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_multi = sio.UserROI.from_multi_polygon(
...     [
...         [(0, 0), (10, 0), (10, 10), (0, 10)],
...         [(50, 50), (60, 50), (60, 60), (50, 60)],
...     ],
...     video=video,
... )
>>> parts = roi_multi.explode()
>>> print(len(parts))
>>> print(parts[0].area)

```

### GeoJSON compatibility

ROIs implement the `__geo_interface__` protocol, making them compatible with
GeoJSON-aware tools:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> roi = sio.UserROI(geometry=box(0, 0, 10, 10))
>>> print(roi.__geo_interface__["type"])

```

### User vs. predicted ROIs

`UserROI` and `PredictedROI` distinguish human annotations from model
predictions. `PredictedROI` adds a `score` field for confidence:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> user_roi = sio.UserROI(
...     geometry=box(10, 20, 100, 200), video=video,
... )
>>> print(user_roi.is_predicted)
>>> pred_roi = sio.PredictedROI(
...     geometry=box(10, 20, 100, 200), video=video, score=0.92,
... )
>>> print(pred_roi.score)
>>> print(pred_roi.is_predicted)

```

---

## Segmentation masks

A `SegmentationMask` stores per-pixel binary annotations in a compact
run-length encoded (RLE) format. RLE avoids storing the full raster array,
making masks efficient for storage and serialization while still supporting
fast conversion to and from numpy arrays. `SegmentationMask` is abstract — use
`UserSegmentationMask` or `PredictedSegmentationMask`.

### From a numpy array

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> mask_data = np.zeros((480, 640), dtype=bool)
>>> mask_data[100:200, 150:300] = True  # rectangular region
>>> mask = sio.UserSegmentationMask.from_numpy(
...     mask_data, video=video, frame_idx=0,
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

### User vs. predicted segmentation masks

`UserSegmentationMask` and `PredictedSegmentationMask` distinguish human
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

---

## Label images

A `LabelImage` stores dense per-pixel instance segmentation for a single video
frame, where each pixel value encodes which object occupies that pixel. Unlike a
`SegmentationMask` — which is a binary mask for a single object — a
`LabelImage` stores **all** objects in one integer array. Background pixels are
`0`, and each positive integer identifies a distinct object. The `objects` dict
maps these IDs to metadata (track, category, name). `LabelImage` is abstract —
use `UserLabelImage` or `PredictedLabelImage`.

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
>>> print(track in li)
>>> for track, category, mask in li.items():
...     print(f"{track.name}: {category}, {mask.sum()} px")

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

### User vs. predicted label images

`UserLabelImage` and `PredictedLabelImage` distinguish human annotations from
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

video = sio.Video("microscopy.tif")
with sio.LabelImageWriter("output.slp", video=video) as writer:
    for frame_idx, mask in enumerate(segmentation_results):
        li = sio.PredictedLabelImage.from_numpy(
            mask, video=video, frame_idx=frame_idx,
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

### Lazy loading

When loading SLP files, label image pixel data is loaded lazily — metadata
(tracks, frame indices, categories) is available immediately, and the actual
pixel array is decompressed only on first `.data` access:

```python
labels = sio.load_slp("large_dataset.slp")

# Metadata queries — no decompression
li = labels.get_label_images(frame_idx=42)[0]
print(li.tracks)      # free
print(li.frame_idx)   # free
print(li.height)      # free (cached from metadata)

# Pixel data decompressed on first access, then cached
mask = li.data  # decompresses this frame only
```

This keeps memory usage proportional to the number of frames actually accessed
rather than the total dataset size. The underlying HDF5 file handle is managed
by the `Labels` object and can be explicitly closed with `labels.close()`.

---

## Conversions

The four spatial annotation types can be converted between each other where
geometrically meaningful:

| From                | To                    | Method                            |
| ------------------- | --------------------- | --------------------------------- |
| `BoundingBox`       | `ROI`                 | `bbox.to_roi()`                   |
| `BoundingBox`       | `SegmentationMask`    | `bbox.to_mask(height, width)`     |
| `ROI`               | `SegmentationMask`    | `roi.to_mask(height, width)`      |
| `SegmentationMask`  | `ROI` (polygon)       | `mask.to_polygon()`               |
| `LabelImage`        | `list[SegmentationMask]` | `li.to_masks()`                |
| `list[SegmentationMask]` | `LabelImage`     | `UserLabelImage.from_masks(masks)` |

All conversions preserve metadata (video, frame_idx, track, instance, name,
category, source) when applicable. Conversions always return `User*` types —
geometric conversion does not preserve prediction semantics.

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> # BoundingBox -> ROI -> SegmentationMask -> polygon ROI
>>> bbox = sio.UserBoundingBox(
...     x1=40, y1=35, x2=60, y2=65,
...     video=video, frame_idx=0,
... )
>>> roi = bbox.to_roi()
>>> print(roi.area)
>>> mask = roi.to_mask(100, 100)
>>> print(mask.area)
>>> polygon_roi = mask.to_polygon()
>>> print(polygon_roi.area)

```

---

## Class diagram

``` mermaid
classDiagram
    class BoundingBox:::regions {
        <<abstract>>
        +float x1
        +float y1
        +float x2
        +float y2
        +float angle
        +to_roi()
        +to_mask()
    }

    class UserBoundingBox:::regions
    class PredictedBoundingBox:::regions {
        +float score
    }

    class ROI:::regions {
        <<abstract>>
        +geometry
        +str name
        +is_static
        +to_mask()
        +explode()
    }

    class UserROI:::regions
    class PredictedROI:::regions {
        +float score
    }

    class SegmentationMask:::regions {
        <<abstract>>
        +rle_counts
        +int height
        +int width
        +to_polygon()
    }

    class UserSegmentationMask:::regions
    class PredictedSegmentationMask:::regions {
        +float score
        +ndarray score_map
    }

    class LabelImage:::regions {
        <<abstract>>
        +ndarray data
        +dict objects
        +to_masks()
        +from_masks()
        +from_numpy()
    }

    class UserLabelImage:::regions
    class PredictedLabelImage:::regions {
        +float score
        +ndarray score_map
    }

    class Labels:::labels {
        +bboxes
        +rois
        +masks
        +label_images
    }

    BoundingBox <|-- UserBoundingBox
    BoundingBox <|-- PredictedBoundingBox
    ROI <|-- UserROI
    ROI <|-- PredictedROI
    SegmentationMask <|-- UserSegmentationMask
    SegmentationMask <|-- PredictedSegmentationMask
    LabelImage <|-- UserLabelImage
    LabelImage <|-- PredictedLabelImage
    BoundingBox --> ROI : to_roi()
    BoundingBox --> SegmentationMask : to_mask()
    ROI --> SegmentationMask : to_mask()
    SegmentationMask --> ROI : to_polygon()
    LabelImage --> SegmentationMask : to_masks()
    SegmentationMask --> LabelImage : from_masks()
    Labels "1" *-- "0..*" BoundingBox
    Labels "1" *-- "0..*" ROI
    Labels "1" *-- "0..*" SegmentationMask
    Labels "1" *-- "0..*" LabelImage

    classDef regions fill:#d32f2f,stroke:#c62828,color:#fff
    classDef labels fill:#43a047,stroke:#2e7d32,color:#fff
```

!!! note "See also"

    - **[Labels & Frames](labels.md)**: Accessing bounding boxes, ROIs, masks, and label images from a `Labels` dataset via `labels.bboxes`, `labels.rois`, `labels.masks`, and `labels.label_images`, including filtered queries with `get_bboxes()`, `get_rois()`, `get_masks()`, and `get_label_images()`.
    - **[Rendering](../rendering.md)**: Visualizing segmentation overlays and bounding boxes on video frames.
    - **[TIFF Format](../formats/tiff.md)**: Reading and writing label images as TIFF files with sidecar metadata.
    - **[SLP Format](../formats/slp.md#label-images)**: HDF5 storage layout for label images (blob and chunked formats).
    - **[Merging](../merging.md#merging-label-images)**: Combining label images from multiple SLP files.

---

## API reference

::: sleap_io.BoundingBox

::: sleap_io.UserBoundingBox

::: sleap_io.PredictedBoundingBox

::: sleap_io.ROI

::: sleap_io.UserROI

::: sleap_io.PredictedROI

::: sleap_io.SegmentationMask

::: sleap_io.UserSegmentationMask

::: sleap_io.PredictedSegmentationMask

::: sleap_io.LabelImage

::: sleap_io.UserLabelImage

::: sleap_io.PredictedLabelImage

::: sleap_io.LabelImageWriter

::: sleap_io.merge_label_images
