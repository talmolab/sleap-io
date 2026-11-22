# Ultralytics YOLO Format

Support for [Ultralytics YOLO](https://docs.ultralytics.com/) pose format.

## Keypoint visibility

Ultralytics pose labels store each keypoint as `x y v`, with the same visibility
codes as COCO, and sleap-io maps them onto the
[point flags](../model/poses.md#point-flags-and-states) the same way (see
[COCO keypoint visibility](coco.md#keypoint-visibility)):

- `v=2` reads as visible, `v=1` as occluded with its coordinates kept as an
  estimate, and `v=0` (or `v<0`) as unlabeled. Labeled keypoints are marked
  `verified`.
- `visibility_encoding="binary"` reads `v>0` as visible. The default, `"auto"`,
  picks binary with a warning when some keypoint in the split has `v=1` and none has
  `v=2`.
- `v0_as` sets how `v=0` reads. By default it is unlabeled for the ternary
  encoding and hidden with no location (`ABSENT`) for the binary one, including
  when `"auto"` picks binary. Pass `v0_as="absent"` or `v0_as="unlabeled"` to
  choose explicitly; `"absent"` is how sleap-io read `v=0` before 0.10.

!!! warning "Changed in 0.10"
    Before 0.10, `v=0` read as a hidden keypoint and `v=1` as *visible*, so
    occluded estimates were indistinguishable from visible keypoints. Now `v=0` is
    unlabeled and `v=1` is occluded (hidden) with its coordinates kept.

    Because `v=1` points are no longer visible,
    [`Instance.numpy()`][sleap_io.Instance.numpy] and
    [`Labels.numpy()`][sleap_io.Labels.numpy] now return NaN for them where they
    used to return their coordinates. Use `instance.numpy(invisible_as_nan=False)`
    to get the estimated coordinates; `Labels.numpy(return_states=True)` tells
    these points apart by their `OCCLUDED` state. Datasets that use `v=1` to mean
    "visible" (binary visibility) read as before with
    `visibility_encoding="binary"`, which `"auto"` picks when no keypoint has
    `v=2`. See the [COCO notes](coco.md#keypoint-visibility) for how sleap-nn
    trains on `v=0` and `v=1` joints.

On write, `save_ultralytics` writes visible points as `2`, occluded estimates as
`1` and every other point as `0` with `0 0` coordinates
(`visibility_encoding="binary"` writes `1` and `0`). Instances are skipped only when
they have no labeled point with coordinates (`Instance.is_empty`), so instances
whose points are all occluded estimates are kept, and their box covers every
labeled point with coordinates.

A binary export writes every hidden point as `0 0 0`, so an instance whose points
are all occluded estimates is written with all-zero keypoints. Read back with the
binary encoding, those points are hidden (`ABSENT`), not unlabeled; their
estimated coordinates are lost, since binary has no code to carry them.

```python
import sleap_io as sio

labels = sio.load_ultralytics("dataset/", split="train")
labels = sio.load_ultralytics("dataset/", split="train", visibility_encoding="binary")
sio.save_ultralytics(labels, "export/", visibility_encoding="ternary")
```

::: sleap_io.io.main.load_ultralytics

::: sleap_io.io.main.save_ultralytics
