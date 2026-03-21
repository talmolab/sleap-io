# Regions

Beyond keypoint poses, sleap-io supports spatial annotation types: bounding
boxes, regions of interest (vector polygons), and segmentation masks (raster).
These can be associated with videos, frames, tracks, and instances, and are
stored on the [`Labels`](labels.md) object via `labels.bboxes`, `labels.rois`, and
`labels.masks`.

sleap-io provides three spatial annotation types with different trade-offs:

- **`BoundingBox`** — axis-aligned or rotated rectangles, stored as center +
  dimensions. Has `UserBoundingBox` and `PredictedBoundingBox` subtypes for
  distinguishing human annotations from model outputs.
- **`ROI`** — arbitrary vector geometry via Shapely (polygons, multi-polygons,
  etc.). Can be static (whole video) or per-frame.
- **`SegmentationMask`** — per-pixel binary masks stored as run-length encoding
  for compactness.

All three can be associated with a video, frame, track, and instance. They are
stored on [`Labels`](labels.md) via `labels.bboxes`, `labels.rois`, and `labels.masks`, and
can be converted between each other (bbox -> ROI -> mask).

---

## Bounding boxes

A `BoundingBox` represents a rectangular region defined by its center
coordinates, dimensions, and an optional rotation angle. Bounding boxes are the
primary annotation type for object detection workflows.

### From center and dimensions

```pycon
>>> import sleap_io as sio
>>> import numpy as np
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox = sio.BoundingBox(
...     x_center=100, y_center=200, width=50, height=80,
...     video=video, frame_idx=0,
... )
>>> print(bbox.area)
>>> print(bbox.xyxy)
>>> print(bbox.corners)

```

### From corner coordinates

The `from_xyxy` factory method creates a bounding box from `(x1, y1, x2, y2)`
corner coordinates:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox2 = sio.BoundingBox.from_xyxy(75, 160, 125, 240, video=video, frame_idx=0)
>>> print(bbox2.x_center)
>>> print(bbox2.width)

```

There is also `from_xywh` for `(x, y, width, height)` format where `(x, y)` is
the top-left corner:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox3 = sio.BoundingBox.from_xywh(75, 160, 50, 80, video=video, frame_idx=0)
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
...     x_center=100, y_center=200, width=50, height=80,
...     video=video, frame_idx=0,
... )
>>> print(user_bbox.is_predicted)
>>> pred_bbox = sio.PredictedBoundingBox(
...     x_center=100, y_center=200, width=50, height=80,
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
>>> rotated = sio.BoundingBox(
...     x_center=100, y_center=200, width=50, height=80,
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
rectangle.

### Static vs. temporal ROIs

An ROI is **static** when `frame_idx` is `None`, meaning it applies to all
frames of the video (e.g., an arena boundary). When `frame_idx` is set, the ROI
applies only to that specific frame.

### From corner coordinates

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi = sio.ROI.from_xyxy(10, 20, 100, 200, video=video)
>>> print(roi.area)
>>> print(roi.bounds)
>>> print(roi.is_static)

```

### From polygon coordinates

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_poly = sio.ROI.from_polygon(
...     [(0, 0), (100, 0), (100, 100), (0, 100)],
...     video=video,
... )
>>> print(roi_poly.area)
>>> print(roi_poly.centroid)

```

### From a bounding box (xywh)

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_bbox = sio.ROI.from_bbox(10, 20, 90, 180, video=video)
>>> print(roi_bbox.area)
>>> print(roi_bbox.is_bbox)

```

### From a BoundingBox object

Any `BoundingBox` can be converted to an `ROI` with `.to_roi()`:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> bbox = sio.BoundingBox(
...     x_center=100, y_center=200, width=50, height=80,
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
>>> roi_multi = sio.ROI.from_multi_polygon(
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
>>> roi_multi = sio.ROI.from_multi_polygon(
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
>>> roi = sio.ROI.from_xyxy(0, 0, 10, 10)
>>> print(roi.__geo_interface__["type"])

```

---

## Segmentation masks

A `SegmentationMask` stores per-pixel binary annotations in a compact
run-length encoded (RLE) format. RLE avoids storing the full raster array,
making masks efficient for storage and serialization while still supporting
fast conversion to and from numpy arrays.

### From a numpy array

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> mask_data = np.zeros((480, 640), dtype=bool)
>>> mask_data[100:200, 150:300] = True  # rectangular region
>>> mask = sio.SegmentationMask.from_numpy(
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
>>> mask = sio.SegmentationMask.from_numpy(mask_data)
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
>>> mask = sio.SegmentationMask.from_numpy(mask_data)
>>> print(mask.bbox)

```

---

## Conversions

The three spatial annotation types can be converted between each other where
geometrically meaningful:

| From                | To                    | Method                            |
| ------------------- | --------------------- | --------------------------------- |
| `BoundingBox`       | `ROI`                 | `bbox.to_roi()`                   |
| `BoundingBox`       | `SegmentationMask`    | `bbox.to_mask(height, width)`     |
| `ROI`               | `SegmentationMask`    | `roi.to_mask(height, width)`      |
| `SegmentationMask`  | `ROI` (polygon)       | `mask.to_polygon()`               |

All conversions preserve metadata (video, frame_idx, track, instance, name,
category, source) when applicable.

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> # BoundingBox -> ROI -> SegmentationMask -> polygon ROI
>>> bbox = sio.BoundingBox(
...     x_center=50, y_center=50, width=20, height=30,
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
        +float x_center
        +float y_center
        +float width
        +float height
        +float angle
        +to_roi()
        +to_mask()
    }

    class UserBoundingBox:::regions
    class PredictedBoundingBox:::regions {
        +float score
    }

    class ROI:::regions {
        +geometry
        +str name
        +is_static
        +to_mask()
        +explode()
    }

    class SegmentationMask:::regions {
        +rle_counts
        +int height
        +int width
        +to_polygon()
    }

    class Labels:::labels {
        +bboxes
        +rois
        +masks
    }

    BoundingBox <|-- UserBoundingBox
    BoundingBox <|-- PredictedBoundingBox
    BoundingBox --> ROI : to_roi()
    BoundingBox --> SegmentationMask : to_mask()
    ROI --> SegmentationMask : to_mask()
    SegmentationMask --> ROI : to_polygon()
    Labels "1" *-- "0..*" BoundingBox
    Labels "1" *-- "0..*" ROI
    Labels "1" *-- "0..*" SegmentationMask

    classDef regions fill:#d32f2f,stroke:#c62828,color:#fff
    classDef labels fill:#43a047,stroke:#2e7d32,color:#fff
```

!!! note "See also"

    - **[Labels & Frames](labels.md)**: Accessing bounding boxes, ROIs, and masks from a `Labels` dataset via `labels.bboxes`, `labels.rois`, and `labels.masks`, including filtered queries with `get_bboxes()`, `get_rois()`, and `get_masks()`.
    - **[Rendering](../rendering.md)**: Visualizing segmentation overlays and bounding boxes on video frames.

---

## API reference

::: sleap_io.BoundingBox

::: sleap_io.UserBoundingBox

::: sleap_io.PredictedBoundingBox

::: sleap_io.ROI

::: sleap_io.SegmentationMask
