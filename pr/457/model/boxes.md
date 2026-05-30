# Bounding boxes

> **Spatial annotations:** [Centroids](centroids.md) · [Boxes](boxes.md) · [ROIs](rois.md) · [Segmentation](segmentation.md). These types nest per-frame on [`LabeledFrame`](labels.md) — see [Working with annotations in frames](index.md#working-with-annotations-in-frames).

A [`BoundingBox`][sleap_io.BoundingBox] represents a rectangular region defined by its corner
coordinates (`x1`, `y1`, `x2`, `y2`) and an optional rotation angle. Bounding
boxes are the primary annotation type for object detection workflows.
`BoundingBox` is abstract — use [`UserBoundingBox`][sleap_io.UserBoundingBox] or [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox].

## Direct construction

```pycon
>>> import sleap_io as sio
>>> bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
... )
>>> print(bbox.area)
>>> print(bbox.xyxy)
>>> print(bbox.x_center)  # computed property
>>> print(bbox.width)      # computed property

```

The `x_center`, `y_center`, `centroid_xy`, `width`, and `height` fields are
available as read-only computed properties.

## From corner coordinates

The `from_xyxy` factory method creates a bounding box from `(x1, y1, x2, y2)`
corner coordinates:

```pycon
>>> import sleap_io as sio
>>> bbox2 = sio.UserBoundingBox.from_xyxy(75, 160, 125, 240)
>>> print(bbox2.x_center)
>>> print(bbox2.width)

```

There is also `from_xywh` for `(x, y, width, height)` format where `(x, y)` is
the top-left corner:

```pycon
>>> import sleap_io as sio
>>> bbox3 = sio.UserBoundingBox.from_xywh(75, 160, 50, 80)
>>> print(bbox3.x_center)
>>> print(bbox3.y_center)

```

## User vs. predicted bounding boxes

[`UserBoundingBox`][sleap_io.UserBoundingBox] and [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox] distinguish human annotations from
model predictions. `PredictedBoundingBox` adds a `score` field for confidence:

```pycon
>>> import sleap_io as sio
>>> user_bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
... )
>>> print(user_bbox.is_predicted)
>>> pred_bbox = sio.PredictedBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     score=0.95,
... )
>>> print(pred_bbox.score)
>>> print(pred_bbox.is_predicted)

```

## Rotated bounding boxes

Set `angle` (in radians) to create an oriented bounding box. Rotated boxes
support `corners` and `bounds` but not `xyxy` or `xywh`, since those are only
meaningful for axis-aligned rectangles:

```pycon
>>> import sleap_io as sio
>>> rotated = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
...     angle=0.785,
... )
>>> print(rotated.is_rotated)
>>> print(rotated.corners.shape)
>>> print(rotated.bounds)  # axis-aligned extent of the rotated box

```

## Metadata fields

Every bounding box can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None` | Confidence of track identity assignment    |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `category`  | `str`              | Class label (e.g., `"mouse"`)                |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | Annotation source identifier                 |

!!! tip "Rendering"
    Use [`sio.draw_bboxes`][sleap_io.draw_bboxes] to composite bounding boxes onto an image, or pass `bboxes` to [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video]. See [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays).

---

!!! note "See also"

    - **[Centroids](centroids.md)**, **[ROIs](rois.md)**, **[Segmentation](segmentation.md)** — the other spatial annotation types.
    - **[Labels & Frames](labels.md)**: Accessing boxes via `labels.bboxes` and filtered queries with `get_bboxes()`.
    - **[Formats: COCO](../formats/index.md#coco-format-json)** and **[Ultralytics YOLO](../formats/index.md#ultralytics-yolo-format)**: Bounding-box detection round-trips.
    - **[Converting between annotation types](segmentation.md#converting-between-annotation-types)**: `bbox.to_roi()`, `bbox.to_mask()`, and more.

---

## API reference

::: sleap_io.BoundingBox

::: sleap_io.UserBoundingBox

::: sleap_io.PredictedBoundingBox
