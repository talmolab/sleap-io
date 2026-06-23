# COCO Format (.json)

[COCO](https://cocodataset.org/) (Common Objects in Context) format is widely used in computer vision and pose estimation. sleap-io provides full read and write support, making it compatible with tools like [mmpose](https://github.com/open-mmlab/mmpose), [CVAT](https://www.cvat.ai/), and other COCO-compatible frameworks.

sleap-io reads all three COCO flavors:

- **Pose** datasets (categories with `keypoints`) → `Instance` objects.
- **Detection** datasets (annotations with `bbox`) → `BoundingBox` objects.
- **Instance-segmentation** datasets (annotations with `segmentation`) →
  `SegmentationMask` (or `ROI`) objects.

These are not mutually exclusive: an annotation that carries both keypoints and a
segmentation/bbox preserves all of them, with the segmentation/bbox linked back
to the keypoint `Instance`.

!!! note "Unannotated images become empty frames"
    `load_coco` creates a `LabeledFrame` for every entry in the `images` array,
    including images with zero annotations — these become **empty**
    `LabeledFrame`s, so the frame count matches the input and unannotated images
    round-trip losslessly. One caveat: an image whose file path cannot be resolved
    is skipped, so a 0-annotation image with a missing file is dropped rather than
    preserved.

## Segmentation handling

COCO encodes segmentation either as a polygon (a list of `[x1, y1, x2, y2, ...]`
rings) or as RLE (a `{"counts": ..., "size": ...}` dict). RLE is always read as a
[`SegmentationMask`][sleap_io.SegmentationMask]. Polygon handling is controlled by
the `segmentation_format` argument of `load_coco` / `coco.read_labels`:

- `"mask"` (the **default**): each annotation's polygon(s) are rasterized into a
  single `SegmentationMask` at the image resolution. Multiple rings of one
  annotation collapse into one object mask. This is the representation that
  exercises the segmentation data model and round-trips through `.slp`.
- `"roi"`: keep the native vector geometry as [`ROI`][sleap_io.ROI] objects (one
  per ring), without rasterizing.

```python
import sleap_io as sio

# Polygon segmentation -> SegmentationMask (default).
labels = sio.load_file("annotations.coco.json")
masks = labels.labeled_frames[0].masks

# Keep polygons as vector ROIs instead.
labels = sio.load_coco("annotations.coco.json", segmentation_format="roi")
rois = labels.labeled_frames[0].rois
```

!!! note "Mask rasterization needs image dimensions"
    Rasterizing a polygon requires the image `height`/`width` from the `images`
    entry. In `"mask"` mode, a polygon whose image lacks those fields falls back
    to an `ROI` since there is no extent to rasterize into. When written back to
    COCO, `SegmentationMask` objects are exported as RLE.

!!! note "Predicted vs. user segmentation"
    A detection annotation carrying a `score` (i.e. a model prediction) is read
    as a `PredictedSegmentationMask` / `PredictedROI` with that score; annotations
    without a `score` become the `User*` variants. This mirrors how `bbox`
    annotations select `PredictedBoundingBox` vs. `UserBoundingBox`.

## Categories as identities

In a standard COCO dataset the `category` is an object *class* (e.g. `"person"`,
`"car"`). Some datasets instead use the category to encode a persistent
*identity* — for example a multi-animal segmentation dataset where each animal is
its own category. Set `category_as_track=True` to map each category to a shared
[`Track`][sleap_io.Track] (named after the category) and assign it to every
annotation of that category (masks, ROIs, bounding boxes, and keypoint instances
without an explicit track id):

```python
labels = sio.load_coco("annotations.coco.json", category_as_track=True)
[t.name for t in labels.tracks]      # one Track per category
labels.labeled_frames[0].masks[0].track.name  # == that mask's category
```

The identity tracks are persisted to `.slp` and survive a save/load round-trip.

::: sleap_io.io.main.load_coco

::: sleap_io.io.main.save_coco
