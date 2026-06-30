# Regions of interest

> **Spatial annotations:** [Centroids](centroids.md) · [Boxes](boxes.md) · [ROIs](rois.md) · [Segmentation](segmentation.md). These types nest per-frame on [`LabeledFrame`](labels.md) — see [Working with annotations in frames](index.md#working-with-annotations-in-frames).

An [`ROI`][sleap_io.ROI] represents a vector geometry annotation using
[Shapely](https://shapely.readthedocs.io/) geometries. ROIs are suitable for
defining arenas, exclusion zones, or arbitrary spatial regions: anything that
is naturally described by a polygon or set of polygons rather than a simple
rectangle. `ROI` is abstract — use [`UserROI`][sleap_io.UserROI] or [`PredictedROI`][sleap_io.PredictedROI].

## Static vs. temporal ROIs

ROIs can be **static** (applying to all frames of a video) or **frame-bound** (attached to a specific `LabeledFrame`). Static ROIs are stored in `Labels.static_rois` and have a `video` attribute. Frame-bound ROIs are stored on individual `LabeledFrame.rois` lists.

Adding a static ROI to a dataset is a direct list append:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> labels = sio.Labels(videos=[video])
>>> arena = sio.UserROI(geometry=box(10, 10, 100, 100), video=video)
>>> labels.static_rois.append(arena)
>>> print(len(labels.static_rois))

```

Frame-bound ROIs use `LabeledFrame.append(roi)` and participate in the O(1) per-frame accessors (`lf.rois`).

## From polygon coordinates

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi_poly = sio.UserROI.from_polygon(
...     [(0, 0), (100, 0), (100, 100), (0, 100)],
...     video=video,
... )
>>> print(roi_poly.area)
>>> print(roi_poly.centroid_xy)

```

## From Shapely geometry

Construct an ROI directly from any Shapely geometry object:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi = sio.UserROI(geometry=box(10, 20, 100, 200), video=video)
>>> print(roi.area)
>>> print(roi.bounds)

```

## From a BoundingBox object

Any `BoundingBox` can be converted to an `ROI` with `.to_roi()`:

```pycon
>>> import sleap_io as sio
>>> bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
... )
>>> roi_from_bbox = bbox.to_roi()
>>> print(roi_from_bbox.area)

```

## Reducing an ROI to a point or box

An `ROI` participates in the unified
[conversion matrix](segmentation.md#converting-between-annotation-types). Reduce
it to a [`Centroid`](centroids.md) with `to_centroid()`, or fit a
[`BoundingBox`](boxes.md) with `to_bbox()`. Predicted ROIs produce predicted
outputs carrying their `score`:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> roi = sio.UserROI(geometry=box(10, 20, 100, 200))
>>> print(roi.to_centroid().xy)            # geometric centroid
>>> print(roi.to_bbox().xyxy)              # axis-aligned bounds
>>> print(roi.to_bbox(padding=5).xyxy)     # inflated box

```

`to_centroid(representative=True)` uses Shapely's `representative_point()` (a
point guaranteed to lie inside the geometry) instead of the geometric centroid.
`to_bbox(rotated=True)` fits a minimum-area oriented box from the geometry's
`minimum_rotated_rectangle`. Both verbs accept `error_on_empty=False`; an empty
geometry yields a degenerate target unless you pass `error_on_empty=True`. The
companion `is_empty` property reports whether the geometry is empty:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import Polygon, box
>>> print(sio.UserROI(geometry=box(0, 0, 10, 10)).is_empty)
False
>>> print(sio.UserROI(geometry=Polygon()).is_empty)
True

```

## Multi-polygon ROIs

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

## GeoJSON compatibility

ROIs implement the `__geo_interface__` protocol, making them compatible with
GeoJSON-aware tools:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> roi = sio.UserROI(geometry=box(0, 0, 10, 10))
>>> print(roi.__geo_interface__["type"])

```

## User vs. predicted ROIs

[`UserROI`][sleap_io.UserROI] and [`PredictedROI`][sleap_io.PredictedROI] distinguish human annotations from model
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

## Metadata fields

Every ROI can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `video`     | [`Video`](video.md) `\| None` | Associated video — set for static ROIs, `None` for frame-bound ROIs |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None` | Confidence of track identity assignment    |
| `identity`  | [`Identity`](embedding.md) `\| None` | Global cross-video re-ID identity (mirrors `Instance.identity`); persists via `/identity_links` (`owner_type=5`) |
| `identity_score` | `float \| None` | Confidence of the `identity` assignment      |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `category`  | `str`              | Class label (e.g., `"arena"`)                |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | Annotation source identifier                 |

!!! tip "Rendering"
    Use [`sio.draw_rois`][sleap_io.draw_rois] to composite ROIs onto an image, or pass them to [`sio.render_image`][sleap_io.render_image]. The ROI stroke/fill colors follow the same palette options as other overlays. See [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays).

---

!!! note "See also"

    - **[Centroids](centroids.md)**, **[Boxes](boxes.md)**, **[Segmentation](segmentation.md)** — the other spatial annotation types.
    - **[Labels & Frames](labels.md)**: Accessing ROIs via `labels.rois`, video-level `labels.static_rois`, and `get_rois()`.
    - **[Converting between annotation types](segmentation.md#converting-between-annotation-types)**: `roi.to_centroid()`, `roi.to_bbox()`, `roi.to_mask()`, and the full modality matrix.

---

## API reference

::: sleap_io.ROI

::: sleap_io.UserROI

::: sleap_io.PredictedROI
