# Diagnostics

Inference finishes, you are handed a `predictions.slp`, and the only real question is *"is this any good?"* [`Labels.diagnose()`](model/labels.md#sleap_io.Labels.diagnose) answers it by running a set of cheap consistency checks over the predictions and reporting what it finds in plain language.

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
print(labels.diagnose().summary())
```

```
1100 frames, 27 tracks, 24 nodes
confidence: mean=0.745 p05=0.459 below 0.3=2.0%
displacement px/frame: median=1.25 p99=7.95 max=60.19

[warning] 23 of 27 tracks occupy less than 1% of frames (4 above that).
    -> The tracker spawned identities it could not sustain. Run `Labels.clean()`
       to drop empty tracks, or re-track with a stricter matching threshold.
[warning] 35 track interruptions across 24 tracks.
    -> Identities are being lost and re-created mid-recording. Increase the
       tracker's gap-closing window, or proofread the breaks in the GUI.
[warning] 6 frames where an instance moved more than 10x the median displacement
          (median 1.25 px/frame, max 60.19).
    frames: 358, 359, 360, 476, 1098, ... (+1)
    -> Instances that teleport between frames are usually identity swaps. Render
       these frames to confirm, then fix the track assignments.
[info] 7 skeleton segments vary in length by more than 25%: forelegR1 --
       forelegR2 (cv=0.35), forelegL1 -- forelegL2 (cv=0.31), ...
    -> Rigid body segments should keep a constant length, so a high spread points
       at jittery keypoints. Articulated joints legitimately vary, so check these
       against the skeleton before acting.
```

!!! warning "This is a consistency check, not an accuracy check"

    Diagnostics compare the predictions **against themselves** — no ground truth
    is involved. They can tell you the output looks implausible (25 tracks for
    two animals, a fly that jumps 60 px in one frame), but not that a pose is
    *wrong*. Measuring accuracy needs labeled ground truth and lives in
    [sleap-nn](https://github.com/talmolab/sleap-nn), which computes OKS, PCK,
    and localization error against a labeled file.

---

## What gets checked

| Metric | What it means | Finding code |
|---|---|---|
| **Track occupancy** | Fraction of frames each track appears in. A real animal is present for most of a recording, so many near-empty tracks means the tracker invented identities. | `spurious_tracks` |
| **Track fragmentation** | How often a track stops after being present. Each interruption is a place the tracker lost an identity and started over. | `track_fragmentation` |
| **Node visibility** | How often each body part was detected, counted over *present* instances only. Names the parts the model never learned. | `low_node_visibility` |
| **Confidence** | Distribution of predicted point scores and how much of it falls below a usable threshold. | `low_confidence` |
| **Displacement outliers** | Frames where an instance moves far more than it usually does. A teleport between consecutive frames is the signature of an identity swap. | `displacement_outliers` |
| **Segment stability** | Coefficient of variation of each skeleton edge's length. A rigid body segment should keep a constant length, so spread is a units-free readout of keypoint jitter. | `unstable_segments` |
| **Empty file** | The labels contain no frames at all — usually a failed inference run. | `no_data` |

---

## Acting on findings

Each [`Finding`](#sleap_io.Finding) carries a stable `code`, a `severity` (`"error"`, `"warning"`, or `"info"`), the `message` with its supporting numbers, and a `suggestion`. Findings are ordered most severe first.

```python
report = labels.diagnose()

for finding in report.findings:
    print(finding.code, finding.severity)
    print(finding.suggestion)
```

Findings that localize to particular frames also carry `frames`, which pairs directly with [rendering](rendering.md) — the numeric outlier becomes something you can look at:

```python
report = labels.diagnose()
swaps = next(f for f in report.findings if f.code == "displacement_outliers")

for frame_idx in swaps.frames[:3]:
    sio.render_image(labels, f"swap_{frame_idx}.png", lf_ind=frame_idx, show_trails=True)
```

!!! tip "No video needed"

    Pass `background="black"` to `render_image` when the source video cannot be
    found. The skeleton overlay is often enough to spot a swapped pair or a
    flipped pose.

---

## Reading the raw metrics

`summary()` is the interpreted view. Every underlying array is also on the report, so you can do your own analysis:

```python
report = labels.diagnose()

report.track_occupancy   # (n_tracks,) fraction of frames each track appears in
report.node_visibility   # (n_nodes,) detection rate per body part
report.track_breaks      # (n_tracks,) interruptions per track
report.segment_cv        # (n_edges,) length variability per skeleton edge
report.segment_names     # ["head -- thorax", ...] matching segment_cv
report.confidence        # {"mean", "p05", "p50", "p95", "fraction_below"}
report.displacement      # {"median", "p99", "max"} in px/frame
report.outlier_frames    # frame indices with a teleport
```

Pair `node_visibility` with `node_names` to rank body parts:

```python
import numpy as np

order = np.argsort(report.node_visibility)
for i in order[:5]:
    print(f"{report.node_names[i]}: {report.node_visibility[i]:.1%}")
```

For a serializable form — writing to JSON, or handing the report to another tool — use `to_dict()`:

```python
import json

with open("diagnostics.json", "w") as f:
    json.dump(labels.diagnose().to_dict(), f, indent=2)
```

---

## Thresholds

The cutoffs that turn metrics into findings are module-level constants, documented so you know what a finding actually asserts:

| Constant | Default | Used for |
|---|---|---|
| `SPURIOUS_TRACK_OCCUPANCY` | `0.01` | Below this occupancy a track is called spurious. |
| `LOW_NODE_VISIBILITY` | `0.5` | Below this detection rate a node is flagged. |
| `LOW_CONFIDENCE` | `0.3` | Point scores below this are treated as unusable. |
| `LOW_CONFIDENCE_FRACTION` | `0.05` | Flag the score distribution when this much of it is low. |
| `TELEPORT_FACTOR` | `10.0` | Multiple of the median displacement that counts as a teleport. |
| `UNSTABLE_SEGMENT_CV` | `0.25` | Coefficient of variation above which a segment is unstable. |

---

## Notes on specific metrics

**Tracks.** When the labels have no tracks at all, `diagnose()` falls back to analyzing untracked instances and skips the track-based findings, since the track axis is then just an arbitrary per-frame ordering. Force either mode with the `untracked` argument.

**Confidence.** [`Labels.numpy()`](model/labels.md#sleap_io.Labels.numpy) reports a score of `1.0` for user-annotated points, which would read as a perfectly confident model. Diagnostics therefore report confidence as `nan` for labels with no predicted instances rather than a misleading `1.0`.

**Segment stability.** A high coefficient of variation means the distance between two connected nodes is not holding steady. For a genuinely rigid segment that is pure keypoint noise, but an articulated joint varies for real reasons — so this is reported at `info` severity and should be checked against your skeleton before you act on it.

---

## API reference

::: sleap_io.Diagnostics

::: sleap_io.Finding

::: sleap_io.diagnose
