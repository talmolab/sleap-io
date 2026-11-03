# DeepLabCut Format (.csv)

Load annotations from [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), a popular markerless pose estimation tool. sleap-io reads DLC annotations from the project's CSV files (optionally augmented by a `config.yaml`); it does not read DLC `.h5` outputs.

`load_dlc` reads a single DLC annotation CSV. When a project `config.yaml` is
available — either passed explicitly via `config=` or auto-discovered by walking
up from the CSV — the following extra metadata is imported:

- **Skeleton edges** from the config `skeleton:` list. Edges that reference
  bodyparts not present in the labeled data are dropped with a warning.
- **Source videos**: each `labeled-data/<video>/` image folder is linked back to
  its original video file (from the config `video_sets`) via `Video.source_video`,
  matched by filename stem. The link is best-effort and left unset on a stem
  mismatch or missing file.

Pass `config=False` to disable config use entirely and reproduce the legacy,
config-free output.

!!! note "Cropping (`video_sets[...].crop`)"
    DeepLabCut's `video_sets[...].crop` is a *virtual* read-time crop (an ROI
    that DLC's video reader slices out of each full frame). The images under
    `labeled-data/<video>/` are the cropped region and the labels are stored in
    **cropped-frame coordinates**, while the linked `source_video` is the
    original, **uncropped** video. sleap-io now imports this crop:

    - The crop rect is parsed from `video_sets` (DLC stores it width-range-first
      as `x1, x2, y1, y2`; sleap-io reorders it to its `(x1, y1, x2, y2)`
      convention, `x2`/`y2` exclusive) and recorded under
      `labels.provenance["dlc_crops"]`, keyed by source-video path. This record
      **persists through an SLP round-trip**.
    - Labels are left **verbatim in cropped-frame coordinates** on the uncropped
      `labeled-data` `ImageVideo` — no offset is applied (and the already-cropped
      images are never cropped again). To map a label into the full source frame,
      use [`Video.to_source_coords`](../model/video.md#sleap_io.Video.to_source_coords)
      with the recorded rect (it adds the crop origin `(x1, y1)`).
    - When the source video file is available, `source_video` is set to a
      [`Video.from_crop`](../model/video.md#sleap_io.Video.from_crop) view of it,
      so `source_video.crop_rect` / `to_source_coords` work in memory (this view's
      crop is in-memory only; the persistent record is `provenance["dlc_crops"]`).
      When the source is absent, `source_video` is a closed `Video` as before.
    - Identity crops at the origin (`0, W, 0, H` — the DLC no-cropping default)
      record no crop and leave the link exact.

    See the [virtual cropping guide](../cropping.md) for the crop conventions.

```python
import sleap_io as sio

# Single CSV; auto-discovers config.yaml for edges + source-video links.
labels = sio.load_dlc("project/labeled-data/vid1/CollectedData_Scorer.csv")

# Strict legacy output (no edges/links), even inside a project.
labels = sio.load_dlc("project/labeled-data/vid1/CollectedData_Scorer.csv", config=False)
```

!!! note "Unannotated rows become empty frames"
    Every CSV row (image) is loaded as a `LabeledFrame`, including rows with no
    labeled bodyparts (all-NaN) — these become **empty** `LabeledFrame`s, so the
    frame count matches the input. Drop them with
    [`Labels.clean()`](../model/labels.md) if you don't want them.

::: sleap_io.io.main.load_dlc

## Loading a whole DLC project

`load_dlc_project` loads every `labeled-data/<video>/` folder in a project at
once and merges them into a single `Labels` that shares one `Skeleton` (with
edges) and one set of `Track`s, recording provenance under the `dlc_project`,
`dlc_scorer`, and `dlc_task` keys. A project directory or its `config.yaml` can
also be passed to `load_file`, which routes it here automatically.

```python
labels = sio.load_dlc_project("path/to/dlc_project")  # dir or config.yaml
labels = sio.load_file("path/to/dlc_project")          # auto-routed
```

::: sleap_io.io.main.load_dlc_project

## Importing train/test splits

`load_dlc_splits` recovers the train/test splits created by DLC's
`create_training_dataset`, reading the positional indices stored in the project's
`Documentation_data-*.pickle` and reconstructing DLC's globally, lexicographically
sorted merge of all per-video annotations. It returns a
[`LabelsSet`](../model/index.md) with `"train"` and `"test"` keys.

Splits require the labeled images to be present on disk. For non-zero-padded
image filenames a warning is emitted, since DLC's lexicographic ordering (e.g.
`img10` < `img2`) differs from numeric ordering and could otherwise cause silent
train/test mis-assignment. If a project has multiple training fractions or
shuffles, pass `train_fraction=` and/or `shuffle=` to disambiguate.

```python
splits = sio.load_dlc_splits("path/to/dlc_project", shuffle=1)
train, test = splits["train"], splits["test"]
```

::: sleap_io.io.main.load_dlc_splits
