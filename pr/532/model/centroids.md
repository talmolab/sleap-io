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

## From an Instance

Create a centroid from an existing pose `Instance` using one of three methods:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> centroid = sio.UserCentroid.from_instance(inst)
>>> print(centroid.xy)

```

The `method` parameter controls how the center point is computed:

| Method | Behavior |
|--------|----------|
| `"center_of_mass"` | Mean of all visible point coordinates (default) |
| `"bbox_center"` | Center of the bounding box of visible points |
| `"anchor"` | Coordinates of a specific node (requires `node` argument) |

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> anchor_c = sio.UserCentroid.from_instance(inst, method="anchor", node="head")
>>> print(anchor_c.xy)

```

## Converting back to an Instance

A centroid can be converted to a single-node `Instance` for interoperability
with pose-based workflows:

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(x=30.0, y=40.0)
>>> inst = centroid.to_instance()
>>> print(inst.numpy())

```

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

When using `from_instance()`, the return type matches the input:
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
    Centroids compose with pose rendering in [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video] and are listed in [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays). For standalone canvases, pair [`sio.draw_bboxes`][sleap_io.draw_bboxes] with `Centroid.to_instance()` or use the pose drawing helpers directly.

---

!!! note "See also"

    - **[Boxes](boxes.md)**, **[ROIs](rois.md)**, **[Segmentation](segmentation.md)** — the other spatial annotation types.
    - **[Labels & Frames](labels.md)**: Accessing centroids via `labels.centroids` and filtered queries with `get_centroids()`.
    - **[Formats: TrackMate](../formats/trackmate.md)**: Importing point-tracking detections as `PredictedCentroid`.

---

## API reference

::: sleap_io.Centroid

::: sleap_io.UserCentroid

::: sleap_io.PredictedCentroid
