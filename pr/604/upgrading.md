# Upgrading to 0.10

**sleap-io 0.10 is backwards compatible by default for SLP files.** It reads every SLP file older releases wrote and the new format 3.0, and it writes the 2.x layout that sleap-io 0.9 and SLEAP 1.6 open. Anything that would change what an SLP file holds, or what a default produces in one, is opt-in. A few changes to the in-memory model and to exports to other formats happen by default; they are listed below.

This page lists what you can opt in to, the changes in behaviour that remain, and which versions work together.

## Nothing changes unless you opt in

| What | Default (as in 0.9) | Opt in with | What you get |
| --- | --- | --- | --- |
| The SLP layout a save writes | The 2.x layout, unless the labels were loaded from (or last saved as) format 3.0, or hold content only format 3.0 stores (then format 3.0, with a warning) | `slp_version="v3"` (`--slp-version v3`) | Always format 3.0. `slp_version="v2"` always writes the 2.x layout. See [Which layout a save writes](formats/slp.md#which-layout-a-save-writes). |
| Loading a file older than 3.0 | Points as stored: hidden points keep their coordinates, states unchanged | `load_slp(..., clean_legacy_points=True)`, or `labels.clean_legacy_points()` on labels already loaded | Hidden points lose their (stale) coordinates; visible points without coordinates become hidden. See [Files before 3.0](formats/slp.md#files-before-30). |
| COCO and Ultralytics keypoint visibility | `v=0` hidden with no location, any other code visible, nothing marked verified | `visibility_encoding="ternary"` or `"auto"`, `v0_as` | With the ternary reading, `v=0` unlabeled, `v=1` occluded with its coordinates, labeled points verified. `"binary"` (which `"auto"` picks for files with `v=1` but no `v=2`) reads `v>0` as visible; `v0_as="absent"` keeps `v=0` hidden. See [Keypoint visibility](formats/coco.md#keypoint-visibility). |
| COCO annotation fields | Not kept | `keep_metadata=True` | Ids, `iscrowd`, `area` and file names kept as record metadata, which `save_coco` reuses |
| Nodes added by a skeleton change or a merge | Hidden with no location | `unmapped="unlabeled"` | Unlabeled, so training ignores them |

**One switch for all of them.** `sio.set_v3_defaults(True)`, `with sio.v3_defaults(): ...`, or the environment variable `SLEAP_IO_V3_DEFAULTS=1` turn every newer default above on for a whole process; `sio.set_v3_defaults(False)` turns them off again, and `None` defers to the environment variable. Explicit options always win over it.

A `Labels` object remembers the layout it was last saved in: once it has been saved as format 3.0 (by choice, or inside `with sio.v3_defaults():`), its later saves with no option stay in format 3.0, so a file never changes layout without being asked to.

```python
import sleap_io as sio

labels = sio.load_slp("old_project.slp")          # points as stored
labels.save("copy.slp")                            # 2.x layout: SLEAP 1.6 opens it

labels.save("copy.v3.slp", slp_version="v3")       # format 3.0, by choice

with sio.v3_defaults():                            # all newer defaults at once
    coco = sio.load_coco("person_keypoints.json")  # v=0 unlabeled, v=1 occluded
    coco.save("coco.slp")                          # format 3.0
```

Points marked as not labeled and record metadata can only be saved in format 3.0, which is why the readings that create them are opt-in.

## Changes that remain

These hold whatever you opt in to. Only the last one can raise an error.

- **Hidden points with coordinates are occluded estimates.** The SLEAP GUI leaves the last position on a node it hides, and old files load with that position. sleap-io 0.10 treats a hidden point with coordinates as an occluded estimate, so an instance whose points are all hidden that way is not `is_empty`, `Labels.clean(empty_instances=True)` keeps it, and the COCO and Ultralytics exports write those coordinates as occluded keypoints. `Instance.numpy()` still returns NaN for hidden points, as before. Load with `clean_legacy_points=True`, or call `labels.clean_legacy_points()`, to get 0.9's results. In the 2.x layout an occluded estimate can't be told from a stale position, so save as format 3.0 to keep occluded estimates.
- **`is_empty` and `Labels.update_from_numpy`.** An instance is empty when no labeled point has a location. `Labels.update_from_numpy` marks finite coordinates visible and labeled (in 0.9 a finite value left a hidden point hidden), and NaN hides a point that was visible (in 0.9 NaN left the point unchanged).
- **A lazily loaded `numpy()` starts at frame 0,** as an eagerly loaded one does. In 0.9 a lazy `numpy()` started at the first labeled frame, so code that indexed it shifts.
- **The `complete` point field is now called `verified`.** `points["complete"]` (on arrays, single points and rows of an instance) still works, with a `DeprecationWarning`. The dictionary codec (`to_dict`, layout version 1.1.0) names the key `verified`, with no alias, and keeps hidden points' coordinates; the "points" dataframe has a new `labeled` column, and Analysis HDF5 files gain a `point_labeled` dataset.
- **The points dtype has a `labeled` field.** Code that builds structured point arrays by hand needs it. `Instance.from_numpy` with five columns takes `x, y, visible, verified, labeled`.
- **Exports and hidden points.** Label Studio skips hidden points. NWB predictions leaves a hidden point's frame out of that node's series (it reads back as NaN; 0.9 wrote the coordinates). NWB annotations writes occluded estimates with their coordinates and `node_visibility=False` (0.9 wrote NaN).
- **`save_coco` writes the box area** instead of 0, which changes COCO keypoint similarity (OKS) scores computed from exported files.
- **Saved files hold no URL credentials.** A video or labels file opened from a link with a password or token is saved with the secret masked, so a video whose link needs its token won't reopen from the saved file until it is given the link again (see [Saved files don't keep credentials](remote.md#saved-files-dont-keep-credentials)).
- **`LabelsSet.save` passes SLP options on** to `save_slp`, so an argument it doesn't accept now raises a `TypeError` instead of being ignored.

## Which versions work together

| | Reads | Writes by default | Accepted by |
| --- | --- | --- | --- |
| sleap-io 0.9.5 | SLP up to 2.8; refuses 3.0 with a clear message | 2.x | SLEAP 1.6.4 and later, sleap-nn 0.3.1 and later (0.3.x) |
| sleap-io 0.9.4 and older | SLP up to 2.x; fails on 3.0 with an h5py `TypeError` | 2.x | Check the release's requirements: SLEAP 1.6.4 and later and sleap-nn 0.3.1 and later accept 0.9.3 and 0.9.4 too |
| sleap-io 0.10 | SLP 1.x, 2.x and 3.0 | 2.x, unless 3.0 is needed or asked for | sleap-nn 0.4.x. The SLEAP 1.6 app can't install it, but opens the 2.x files it writes. |

If you see `TypeError: Accessing a group is done with bytes or str, not <class 'tuple'>` when opening a file, it is a format 3.0 file read by sleap-io 0.9.4 or older: upgrade sleap-io, or save the file again with `slp_version="v2"`.
