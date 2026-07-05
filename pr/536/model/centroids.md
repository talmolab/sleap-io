# Centroids

> **Spatial annotations:** [Centroids](centroids.md) · [Boxes](boxes.md) · [ROIs](rois.md) · [Segmentation](segmentation.md). These types nest per-frame on [`LabeledFrame`](labels.md) — see [Working with annotations in frames](index.md#working-with-annotations-in-frames).

A [`Centroid`][sleap_io.Centroid] represents a single point at the center of an object. Centroids
are the primary annotation type for **object detection** workflows where full
pose skeletons are not needed. `Centroid` is abstract — use [`UserCentroid`][sleap_io.UserCentroid] or
[`PredictedCentroid`][sleap_io.PredictedCentroid].

Centroids support optional 3D coordinates (`z`), interconversion with
single-node [`Instance`](poses.md) objects, and the same track/instance metadata as other annotation types.

!!! tip "Importing TrackMate detections"
    `PredictedCentroid(source="trackmate")` is the canonical representation for TrackMate (ImageJ/Fiji) point tracking results. Load spot exports directly with [`sio.load_trackmate`][sleap_io.load_trackmate] or let [`sio.load_file`][sleap_io.load_file] auto-detect the format from the CSV header. See [Formats → TrackMate](../formats/trackmate.md) for the full schema.

## Direct construction

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(
...     x=100.0, y=200.0,
... )
>>> print(centroid.xy)
>>> print(centroid.yx)

```

## From a pose Instance

Create a centroid from an existing pose `Instance` with
[`Centroid.from_pose`][sleap_io.Centroid.from_pose] (equivalently
[`Instance.to_centroid`][sleap_io.Instance.to_centroid]):

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> centroid = sio.UserCentroid.from_pose(inst)
>>> print(centroid.xy)
>>> print(centroid.source)  # records the method used

```

The `method` parameter controls how the center point is computed:

| Method | Behavior | `source` tag |
|--------|----------|--------------|
| `"center_of_mass"` | NaN-ignoring (unweighted) mean of visible points (default) | `"center_of_mass"` |
| `"bbox_center"` | Center of the bounding box of visible points | `"bbox_center"` |
| `"geometric_median"` | Weiszfeld geometric median of visible points (outlier-robust) | `"geometric_median"` |
| `"anchor"` | Coordinates of a specific node (requires `node`) | `"anchor:<node>"` |

For `method="anchor"`, pass `node` (a node name or index). If the anchor node is
occluded, supply a `fallback` method (one of the non-anchor methods) to compute
the centroid from the visible points instead; the `source` tag then records the
chain, e.g. `"anchor:nose->center_of_mass"`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> anchor_c = sio.UserCentroid.from_pose(inst, method="anchor", node="head")
>>> print(anchor_c.xy)
>>> print(anchor_c.source)

```

When there are no visible points (or an occluded anchor with no fallback), the
result is a degenerate centroid (`x = y = nan`); pass `error_on_empty=True` to
raise a `ValueError` instead. The `is_empty` property reports this cheaply:

```pycon
>>> import sleap_io as sio
>>> print(sio.UserCentroid(x=float("nan"), y=float("nan")).is_empty)
True

```

!!! note "Renamed from `from_instance`"
    `Centroid.from_instance()` was renamed to `Centroid.from_pose()` (the pose
    modality is named `pose` everywhere, even though the backing class is
    `Instance`). `from_instance()` is kept as a deprecated alias that forwards to
    `from_pose()` and emits a `DeprecationWarning`.

## Converting back to a pose Instance

A centroid can be converted to a single-node `Instance` with
[`Centroid.to_pose`][sleap_io.Centroid.to_pose] for interoperability with
pose-based workflows:

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(x=30.0, y=40.0)
>>> inst = centroid.to_pose()
>>> print(inst.numpy())

```

!!! note "Renamed from `to_instance`"
    `Centroid.to_instance()` was renamed to `Centroid.to_pose()`.
    `to_instance()` is kept as a deprecated alias that forwards to `to_pose()`
    and emits a `DeprecationWarning`.

## Constructing boxes, ROIs, and masks

A centroid is a single point, so it can seed a fixed-size
[`BoundingBox`](boxes.md), a circular [`ROI`](rois.md), or a rasterized
`SegmentationMask` around itself. These complete the
[conversion matrix](segmentation.md#converting-between-annotation-types) for the
centroid modality:

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(x=30.0, y=40.0)
>>> box = centroid.to_bbox(size=20)                 # 20x20 box centered on the point
>>> print(box.xyxy)
>>> roi = centroid.to_roi(radius=5)                 # disc of radius 5
>>> print(round(roi.area, 2))
>>> mask = centroid.to_mask(100, 100, radius=5)     # rasterized disc
>>> print(mask.area)

```

`to_bbox` requires `size` (a scalar for a square box or `(w, h)`) and accepts
`padding`. `to_roi` and `to_mask` require `radius`. A `PredictedCentroid`
produces predicted outputs carrying its `score`. A degenerate (NaN) centroid
yields an empty target (NaN box / empty-`Polygon` ROI / all-background mask)
unless `error_on_empty=True`.

## User vs. predicted centroids

[`UserCentroid`][sleap_io.UserCentroid] and [`PredictedCentroid`][sleap_io.PredictedCentroid] distinguish human annotations from
model predictions. `PredictedCentroid` adds a `score` field for confidence:

```pycon
>>> import sleap_io as sio
>>> user_c = sio.UserCentroid(x=10, y=20)
>>> print(user_c.is_predicted)
>>> pred_c = sio.PredictedCentroid(x=10, y=20, score=0.95)
>>> print(pred_c.score)
>>> print(pred_c.is_predicted)

```

When using `from_pose()`, the return type matches the input:
`PredictedInstance` produces `PredictedCentroid`, `Instance` produces
`UserCentroid`.

## Metadata fields

Every centroid can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None` | Confidence of track identity assignment    |
| `identity`  | [`Identity`](embedding.md) `\| None` | Global cross-video re-ID identity (mirrors `Instance.identity`); persists via `/identity_links` (`owner_type=2`) |
| `identity_score` | `float \| None` | Confidence of the `identity` assignment      |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `z`         | `float \| None`    | Optional Z-coordinate for 3D data            |
| `category`  | `str`              | Class label (e.g., `"cell"`)                 |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | How the centroid was computed (e.g., `"center_of_mass"`) |

!!! tip "Rendering"
    Centroids compose with pose rendering in [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video] and are listed in [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays). For standalone canvases, pair [`sio.draw_bboxes`][sleap_io.draw_bboxes] with `Centroid.to_pose()` or use the pose drawing helpers directly.

---

!!! note "See also"

    - **[Boxes](boxes.md)**, **[ROIs](rois.md)**, **[Segmentation](segmentation.md)** — the other spatial annotation types.
    - **[Labels & Frames](labels.md)**: Accessing centroids via `labels.centroids` and filtered queries with `get_centroids()`.
    - **[Formats: TrackMate](../formats/trackmate.md)**: Importing point-tracking detections as `PredictedCentroid`.
    - **[Converting between annotation types](segmentation.md#converting-between-annotation-types)**: `centroid.to_pose()`, `centroid.to_bbox()`, `centroid.to_roi()`, `centroid.to_mask()`, and the full modality matrix.

---

## API reference

::: sleap_io.Centroid

::: sleap_io.UserCentroid

::: sleap_io.PredictedCentroid
