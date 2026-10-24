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
    preserved. `load_coco` warns with the number of skipped images when this
    happens.

!!! note "Image path resolution"
    Each `file_name` is looked up at `dataset_root / file_name`, then under the
    `images/`, `imgs/` and `data/images/` subdirectories, and finally by file name
    anywhere under `dataset_root`. The dataset tree is walked at most once per
    load for that last search, but pointing `dataset_root` at the directory that
    actually contains the images is still the fastest option for large datasets.

## Keypoint visibility

COCO stores each keypoint as `x, y, v`. sleap-io maps the visibility code `v` onto
the two [point flags](../model/poses.md#point-flags-and-states), `visible` and
`labeled`, so occluded estimates and unlabeled keypoints both survive a round
trip:

| `v` | Read as (`visibility_encoding="ternary"`) | `PointState` |
| --- | --- | --- |
| `2` | visible | `VISIBLE` |
| `1` | labeled, not visible, coordinates kept as an estimate | `OCCLUDED` |
| `0` | unlabeled, no location | `UNLABELED` |
| `< 0` | unlabeled, no location | `UNLABELED` |
| any other value (e.g. Halpe's float scores) | visible | `VISIBLE` |

Every labeled keypoint is marked `verified`, since COCO keypoints are human
annotations.

`load_coco` takes two options for files that don't follow COCO's encoding:

- `visibility_encoding`:
    - `"ternary"`: COCO's 0/1/2 encoding (the table above).
    - `"binary"`: `v > 0` is visible. This is what `save_coco(...,
      visibility_encoding="binary")` writes. Use it for CrowdPose too, whose codes
      are effectively reversed.
    - `"auto"` (the **default**): binary when some keypoint has `v=1` and none in
      the file has `v=2`, with a warning; ternary otherwise.
- `v0_as`: `"unlabeled"` (the **default**) reads `v=0` as unlabeled. `"absent"`
  reads it as labeled and not visible with no location, which is what sleap-io did
  before 0.10.

```python
import sleap_io as sio

labels = sio.load_coco("person_keypoints_val2017.json")  # v=0 -> unlabeled

# A binary file (no v=2 anywhere) is detected with a warning; be explicit instead:
labels = sio.load_coco("crowdpose_val.json", visibility_encoding="binary")

# Pre-0.10 behaviour: v=0 is a hidden keypoint, not an unlabeled one.
labels = sio.load_coco("annotations.json", v0_as="absent")
```

On write, `save_coco` encodes each point from its state:

| Point | `"ternary"` (default) | `"binary"` |
| --- | --- | --- |
| visible | `x, y, 2` | `x, y, 1` |
| hidden, with coordinates (occluded estimate) | `x, y, 1` | `0, 0, 0` |
| hidden, no location | `0, 0, 0` | `0, 0, 0` |
| unlabeled | `0, 0, 0` | `0, 0, 0` |

`num_keypoints` counts the keypoints written with `v > 0`, and the annotation
`bbox` covers them. COCO has no code for "hidden with no location", so those points
read back as unlabeled.

!!! warning "Changed in 0.10"
    Before 0.10, `v=0` read as a hidden keypoint and `v=1` as visible, so occluded
    estimates were lost. Now `v=0` is unlabeled and `v=1` is occluded with its
    coordinates. For **top-down** models trained with sleap-nn, COCO `v=0` joints
    get zero loss and `v=1` joints train at their coordinates, as in HRNet. For
    **bottom-up** models, frames that mix labeled and unlabeled instances still
    treat `v=0` joints as hidden until spatial ignore masks exist. Pass
    `v0_as="absent"` to keep the old reading of `v=0`.

## Annotation metadata

COCO fields that have no place in the pose model are kept as
[record metadata](../model/poses.md#metadata):

| Record | Metadata keys | From |
| --- | --- | --- |
| Pose `Instance` | `ann_id`, `category_id`, `iscrowd`, `area`, `num_keypoints` | the annotation's `id`, `category_id`, `iscrowd`, `area`, `num_keypoints` |
| `BoundingBox` (annotations read without keypoints) | `ann_id`, `category_id`, `iscrowd`, `area` | the same annotation fields |
| `LabeledFrame` | `image_id`, `file_name` | the image's `id` and `file_name` |

Only fields holding a scalar are kept. Crowd annotations that have keypoints are
read as pose instances, so their `iscrowd` lives on the instance. Annotations read
only as a mask or ROI keep no metadata, since those records have no `metadata`
field.

```python
labels = sio.load_coco("person_keypoints_val2017.json")
labels[0].metadata               # {"image_id": 139, "file_name": "000000000139.jpg"}
labels[0].instances[0].metadata  # {"ann_id": 230831, "iscrowd": 0, "area": ..., ...}
labels.save("coco_val.slp")      # metadata is stored in the SLP file
```

`save_coco` writes the metadata back:

- **Ids.** Stored `image_id` and `ann_id` values are reused when they are unique
  integers across the export; records without one get new ids above the largest
  stored id. If they collide, for example after merging labels loaded from two
  COCO files, every image (or annotation) is renumbered from 1 with a warning.
- **`area`.** The area of a mask or ROI linked to the instance is preferred, then
  the stored `area`, then the area of the keypoints' bounding box. For bounding
  boxes, the stored `area` is preferred over the box's own area.
- **`iscrowd`** is written from metadata (0 when missing).

`sio transform` (and [`transform_labels`][sleap_io.transform_labels])
drops `area` from the metadata of transformed instances and boxes, since it no
longer matches the transformed image.

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

!!! warning "Breaking change in 0.8.0"
    The default polygon segmentation handling changed in v0.8.0. In v0.7.1,
    polygon segmentation was read as vector [`ROI`][sleap_io.ROI] objects (one per
    ring). The new default (`segmentation_format="mask"`) rasterizes each
    annotation's polygon(s) into a single [`SegmentationMask`][sleap_io.SegmentationMask].
    **To restore the pre-0.8.0 vector-ROI behavior, pass
    `segmentation_format="roi"`.**

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
