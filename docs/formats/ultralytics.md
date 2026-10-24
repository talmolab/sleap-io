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
- `v0_as="absent"` reads `v=0` as hidden with no location, as before 0.10.

On write, `save_ultralytics` writes visible points as `2`, occluded estimates as
`1` and every other point as `0` with `0 0` coordinates
(`visibility_encoding="binary"` writes `1` and `0`). Instances are skipped only when
they have no labeled point with coordinates (`Instance.is_empty`), so instances
whose points are all occluded estimates are kept, and their box covers every
labeled point with coordinates.

```python
import sleap_io as sio

labels = sio.load_ultralytics("dataset/", split="train")
labels = sio.load_ultralytics("dataset/", split="train", visibility_encoding="binary")
sio.save_ultralytics(labels, "export/", visibility_encoding="ternary")
```

::: sleap_io.io.main.load_ultralytics

::: sleap_io.io.main.save_ultralytics
