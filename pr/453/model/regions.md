# Regions

Beyond keypoint poses, sleap-io supports spatial annotation types: centroids,
bounding boxes, regions of interest (vector polygons), segmentation masks
(raster), and label images (dense instance segmentation). Annotations are
**nested per-frame** on [`LabeledFrame`](labels.md) — each frame stores its own
lists of annotations (e.g., `lf.centroids`, `lf.bboxes`, `lf.masks`). The
[`Labels`](labels.md) object provides flattened convenience properties
(`labels.centroids`, `labels.bboxes`, etc.) that aggregate across all frames,
and query methods (e.g., `labels.get_centroids()`) for filtered access.

sleap-io provides five spatial annotation types with different trade-offs:

- **[`Centroid`][sleap_io.Centroid]** — lightweight point annotation representing the center of an
  object with `x`, `y`, and optional `z` coordinates. Supports conversion to
  and from single-node [`Instance`](poses.md) objects. Has [`UserCentroid`][sleap_io.UserCentroid] and
  [`PredictedCentroid`][sleap_io.PredictedCentroid] subtypes.
- **[`BoundingBox`][sleap_io.BoundingBox]** — axis-aligned or rotated rectangles, stored as corner
  coordinates. Has [`UserBoundingBox`][sleap_io.UserBoundingBox] and [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox] subtypes for
  distinguishing human annotations from model outputs.
- **[`ROI`][sleap_io.ROI]** — arbitrary vector geometry via Shapely (polygons, multi-polygons,
  etc.). Can be static (whole video) or per-frame. Has [`UserROI`][sleap_io.UserROI] and
  [`PredictedROI`][sleap_io.PredictedROI] subtypes.
- **[`SegmentationMask`][sleap_io.SegmentationMask]** — per-pixel binary masks stored as run-length encoding
  for compactness. One mask per object. Has [`UserSegmentationMask`][sleap_io.UserSegmentationMask] and
  [`PredictedSegmentationMask`][sleap_io.PredictedSegmentationMask] subtypes.
- **[`LabelImage`][sleap_io.LabelImage]** — dense per-pixel integer label image storing **all** objects
  for a frame in one array. Standard output of instance segmentation tools like
  [Cellpose](https://www.cellpose.org/) and
  [StarDist](https://github.com/stardist/stardist). Has [`UserLabelImage`][sleap_io.UserLabelImage] and
  [`PredictedLabelImage`][sleap_io.PredictedLabelImage] subtypes.

All five base classes are **abstract** — use the `User*` or `Predicted*`
subclass to create instances. All five can be associated with a track and
instance. They are stored on [`LabeledFrame`](labels.md) and
accessible via frame-level attributes or the flattened [`Labels`](labels.md)
properties. The four geometry types can be converted between each other
(bbox -> ROI -> mask, mask -> bbox, label image <-> masks, label image -> bboxes).

---

## Working with annotations in frames

Since annotations are nested in [`LabeledFrame`](labels.md), you add them
directly to a frame's annotation lists. [`LabeledFrame.append`][sleap_io.LabeledFrame.append] dispatches on the runtime type of the annotation and pushes it onto the correct per-type list — you never have to touch `lf.instances`, `lf.bboxes`, `lf.centroids`, `lf.masks`, `lf.label_images`, or `lf.rois` directly.

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> lf = sio.LabeledFrame(video=video, frame_idx=0)
>>> lf.append(sio.UserBoundingBox(x1=10, y1=20, x2=50, y2=60))  # → lf.bboxes
>>> lf.append(sio.UserCentroid(x=100, y=200))                    # → lf.centroids
>>> lf.append(sio.UserSegmentationMask.from_numpy(np.zeros((8, 8), bool)))  # → lf.masks
>>> lf.append(sio.UserLabelImage.from_numpy(np.zeros((8, 8), int)))         # → lf.label_images
>>> lf.append(sio.UserROI(geometry=box(0, 0, 10, 10)))            # → lf.rois
>>> labels = sio.Labels(labeled_frames=[lf])
>>> print(len(labels.centroids), len(labels.bboxes))
>>> print(len(labels.masks), len(labels.label_images), len(labels.rois))

```

The `labels.centroids`, `labels.bboxes`, `labels.masks`, `labels.label_images`,
and `labels.rois` properties return flattened read-only views across all frames. Static, video-level ROIs (with no frame association) live separately on [`Labels.static_rois`][sleap_io.Labels.static_rois] — see [Static vs. temporal ROIs](#static-vs-temporal-rois) below.

---

## Centroids

A [`Centroid`][sleap_io.Centroid] represents a single point at the center of an object. Centroids
are the primary annotation type for **object detection** workflows where full
pose skeletons are not needed. `Centroid` is abstract — use [`UserCentroid`][sleap_io.UserCentroid] or
[`PredictedCentroid`][sleap_io.PredictedCentroid].

Centroids support optional 3D coordinates (`z`), interconversion with
single-node [`Instance`](poses.md) objects, and the same track/instance metadata as other annotation types.

!!! tip "Importing TrackMate detections"
    `PredictedCentroid(source="trackmate")` is the canonical representation for TrackMate (ImageJ/Fiji) point tracking results. Load spot exports directly with [`sio.load_trackmate`][sleap_io.load_trackmate] or let [`sio.load_file`][sleap_io.load_file] auto-detect the format from the CSV header. See [Formats → TrackMate](../formats/trackmate.md) for the full schema.

### Direct construction

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(
...     x=100.0, y=200.0,
... )
>>> print(centroid.xy)
>>> print(centroid.yx)

```

### From an Instance

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

### Converting back to an Instance

A centroid can be converted to a single-node `Instance` for interoperability
with pose-based workflows:

```pycon
>>> import sleap_io as sio
>>> centroid = sio.UserCentroid(x=30.0, y=40.0)
>>> inst = centroid.to_instance()
>>> print(inst.numpy())

```

### User vs. predicted centroids

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

### Metadata fields

Every centroid can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None` | Confidence of track identity assignment    |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `z`         | `float \| None`    | Optional Z-coordinate for 3D data            |
| `category`  | `str`              | Class label (e.g., `"cell"`)                 |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | How the centroid was computed (e.g., `"center_of_mass"`) |

!!! tip "Rendering"
    Centroids compose with pose rendering in [`sio.render_image`][sleap_io.render_image] / [`sio.render_video`][sleap_io.render_video] and are listed in [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays). For standalone canvases, pair [`sio.draw_bboxes`][sleap_io.draw_bboxes] with `Centroid.to_instance()` or use the pose drawing helpers directly.

---

## Bounding boxes

A [`BoundingBox`][sleap_io.BoundingBox] represents a rectangular region defined by its corner
coordinates (`x1`, `y1`, `x2`, `y2`) and an optional rotation angle. Bounding
boxes are the primary annotation type for object detection workflows.
`BoundingBox` is abstract — use [`UserBoundingBox`][sleap_io.UserBoundingBox] or [`PredictedBoundingBox`][sleap_io.PredictedBoundingBox].

### Direct construction

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

### From corner coordinates

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

### User vs. predicted bounding boxes

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

### Rotated bounding boxes

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

### Metadata fields

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

## Regions of interest

An [`ROI`][sleap_io.ROI] represents a vector geometry annotation using
[Shapely](https://shapely.readthedocs.io/) geometries. ROIs are suitable for
defining arenas, exclusion zones, or arbitrary spatial regions: anything that
is naturally described by a polygon or set of polygons rather than a simple
rectangle. `ROI` is abstract — use [`UserROI`][sleap_io.UserROI] or [`PredictedROI`][sleap_io.PredictedROI].

### Static vs. temporal ROIs

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

### From polygon coordinates

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

### From Shapely geometry

Construct an ROI directly from any Shapely geometry object:

```pycon
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> roi = sio.UserROI(geometry=box(10, 20, 100, 200), video=video)
>>> print(roi.area)
>>> print(roi.bounds)

```

### From a BoundingBox object

Any `BoundingBox` can be converted to an `ROI` with `.to_roi()`:

```pycon
>>> import sleap_io as sio
>>> bbox = sio.UserBoundingBox(
...     x1=75, y1=160, x2=125, y2=240,
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

### Metadata fields

Every ROI can carry optional metadata:

| Field       | Type               | Description                                  |
| ----------- | ------------------ | -------------------------------------------- |
| `video`     | [`Video`](video.md) `\| None` | Associated video — set for static ROIs, `None` for frame-bound ROIs |
| `track`     | [`Track`](poses.md) `\| None`    | Tracking identity across frames              |
| `tracking_score` | `float \| None` | Confidence of track identity assignment    |
| `instance`  | [`Instance`](poses.md) `\| None` | Linked pose instance                         |
| `category`  | `str`              | Class label (e.g., `"arena"`)                |
| `name`      | `str`              | Human-readable name                          |
| `source`    | `str`              | Annotation source identifier                 |

!!! tip "Rendering"
    Use [`sio.draw_rois`][sleap_io.draw_rois] to composite ROIs onto an image, or pass them to [`sio.render_image`][sleap_io.render_image]. The ROI stroke/fill colors follow the same palette options as other overlays. See [Rendering → Segmentation Overlays](../rendering.md#segmentation-overlays).

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

---

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
| `SegmentationMask`  | `BoundingBox`         | `mask.to_bbox()`                  |
| `LabelImage`        | `list[SegmentationMask]` | `li.to_masks()`                |
| `LabelImage`        | `list[BoundingBox]`   | `li.to_bboxes()`                  |
| `list[SegmentationMask]` | `LabelImage`     | `UserLabelImage.from_masks(masks)` |

All conversions preserve metadata (track, instance, name,
category, source) when applicable. `to_bbox()` and `to_bboxes()` preserve
prediction semantics (`Predicted*` inputs produce `Predicted*` outputs with
scores). Other conversions return `User*` types.

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
        +centroid_xy
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
        +to_bbox()
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
        +to_bboxes()
        +from_masks()
        +from_numpy()
    }

    class UserLabelImage:::regions
    class PredictedLabelImage:::regions {
        +float score
        +ndarray score_map
    }

    class Centroid:::regions {
        <<abstract>>
        +float x
        +float y
        +xy
        +to_instance()
        +from_instance()
    }

    class UserCentroid:::regions
    class PredictedCentroid:::regions {
        +float score
    }

    class LabeledFrame:::labels {
        +centroids
        +bboxes
        +rois
        +masks
        +label_images
    }

    Centroid <|-- UserCentroid
    Centroid <|-- PredictedCentroid
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
    SegmentationMask --> BoundingBox : to_bbox()
    LabelImage --> SegmentationMask : to_masks()
    LabelImage --> BoundingBox : to_bboxes()
    SegmentationMask --> LabelImage : from_masks()
    LabeledFrame "1" *-- "0..*" Centroid
    LabeledFrame "1" *-- "0..*" BoundingBox
    LabeledFrame "1" *-- "0..*" ROI
    LabeledFrame "1" *-- "0..*" SegmentationMask
    LabeledFrame "1" *-- "0..*" LabelImage

    classDef regions fill:#d32f2f,stroke:#c62828,color:#fff
    classDef labels fill:#43a047,stroke:#2e7d32,color:#fff
```

!!! note "See also"

    - **[Labels & Frames](labels.md)**: Accessing annotations from a `Labels` dataset via `labels.centroids`, `labels.bboxes`, `labels.rois`, `labels.masks`, and `labels.label_images`, including filtered queries with `get_centroids()`, `get_bboxes()`, `get_rois()`, `get_masks()`, and `get_label_images()`.
    - **[Rendering](../rendering.md)**: Visualizing segmentation overlays and bounding boxes on video frames.
    - **[TIFF Format](../formats/tiff.md)**: Reading and writing label images as TIFF files with sidecar metadata.
    - **[SLP Format](../formats/slp.md#label-images)**: HDF5 storage layout for label images (blob and chunked formats).
    - **[Merging](../merging.md#merging-label-images)**: Combining label images from multiple SLP files.

---

## API reference

::: sleap_io.Centroid

::: sleap_io.UserCentroid

::: sleap_io.PredictedCentroid

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

::: sleap_io.normalize_label_ids
