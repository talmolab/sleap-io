# Merging annotations

Merging combines annotations from multiple sources into a single dataset.

## Quick start

```python
import sleap_io as sio

base = sio.load_file("manual_annotations.slp")
predictions = sio.load_file("predictions.slp")

base.merge(predictions)
base.save("merged.slp")
```

## How merging works

Merging proceeds in four steps:

1. **Match skeletons** — Find corresponding skeletons by node names
2. **Match videos** — Identify same videos across datasets (see [Video matching](#video-matching))
3. **Match tracks** — Map track identities by name
4. **Merge frames** — For each frame in `other`:
   - If frame doesn't exist in `base`: add it
   - If frame exists in both: apply the `frame` strategy

Two parameters control the key steps:

| Parameter | Stage | Controls | Default |
|-----------|-------|----------|---------|
| `video` | Step 2 | How videos are matched between labels | `"auto"` |
| `frame` | Step 4 | How overlapping frames are combined | `"smart"` |

**Video matching is upstream of frame strategy.** If video matching fails (adds video as new), frame strategy is never applied—there are no overlapping frames to merge.

```python
# Explicit control over both (simple string API)
base.merge(predictions, video="path", frame="smart")

# Or with Matcher objects for advanced configuration
from sleap_io.model.matching import VideoMatcher
base.merge(predictions, video=VideoMatcher(method="path", strict=True))
```

## Video matching

Videos must match for frames to merge. The default `AUTO` method uses a **safe matching cascade** designed to prevent data corruption.

### Design philosophy

The AUTO algorithm prioritizes **avoiding false positives** (matching wrong videos) over avoiding false negatives (failing to match correct videos):

- **False positive** → Annotations merged to wrong video → **Data corruption** (often unrecoverable)
- **False negative** → Video added as new → **Safe** (easily corrected manually)

When uncertain, AUTO adds the video as new rather than risk a wrong match.

### AUTO matching cascade

For each incoming video, AUTO runs these checks in order:

| Step | Check | Can yield | Description |
|------|-------|-----------|-------------|
| 1-2 | Shape rejection | NOT MATCH only | Different (frames, H, W) → reject |
| 3 | Provenance conflict | NOT MATCH only | Different original videos → reject |
| 4 | Physical file identity | MATCH | `os.path.samefile()` returns true |
| 5 | Exact path string | MATCH | Sanitized paths match exactly |
| 6 | Leaf uniqueness | MATCH or NOT MATCH | Minimal unique path suffixes match |
| 7 | Fallback | Add as new | No match found, add video to target |

**Shape is for rejection only.** Two videos with the same resolution are NOT automatically matched—they simply aren't rejected. This prevents matching unrelated videos that happen to have the same dimensions.

### Examples

**Example 1: Same file, different paths (symlink)**

```
Base:  /data/videos/fly.mp4
Other: /projects/exp1/fly.mp4  (symlink to /data/videos/fly.mp4)
```

Result: **MATCH** — Step 4 (physical file identity) detects same file via OS.

**Example 2: Cross-platform paths with unique basenames**

```
Base videos:
  - C:/Users/alice/data/exp1/fly.mp4

Other videos:
  - /home/bob/data/exp1/fly.mp4
```

Result: **MATCH** — Step 6 (leaf uniqueness). Both "fly.mp4" are unique in their sets, and the basenames match.

**Example 3: Ambiguous basenames, parent directory disambiguates**

```
Base videos:
  - /data/exp1/fly.mp4
  - /data/exp2/fly.mp4

Other videos:
  - /remote/exp2/fly.mp4
```

Result: **MATCH** to `/data/exp2/fly.mp4` — Step 6 builds unique leaves:
- Base: "exp1/fly.mp4" and "exp2/fly.mp4" (need parent to disambiguate)
- Other: "exp2/fly.mp4" (unique with parent)
- "exp2/fly.mp4" matches "exp2/fly.mp4" → MATCH

**Example 4: Ambiguous basenames, no distinguishing info**

```
Base videos:
  - /data/exp1/fly.mp4
  - /data/exp2/fly.mp4

Other videos:
  - /remote/fly.mp4  (no parent directory)
```

Result: **Add as new** — Cannot determine which base video to match. Step 6 can't find unique leaves, falls through to Step 7.

**Example 5: Same basename, different resolution**

```
Base:  fly.mp4 (1000 frames, 640×480)
Other: fly.mp4 (500 frames, 640×480)
```

Result: **NOT MATCH** — Step 1 (shape rejection). Different frame counts indicate different videos, despite same basename.

**Example 6: PKG.SLP predictions merged to external video**

```
Base: project.slp
  - video: /data/fly.mp4

Other: predictions.slp (from training on project.pkg.slp)
  - video: predictions.pkg.slp (embedded)
    - original_video: /data/fly.mp4
```

Result: **MATCH** — The embedded video's `original_video` attribute points to the same external file. AUTO traverses provenance chains and matches via physical file identity.

**Example 7: Two PKG.SLP files from different sources**

```
Base: project.pkg.slp
  - video embedded, original_video: /lab1/fly.mp4

Other: predictions.pkg.slp
  - video embedded, original_video: /lab2/fly.mp4
```

Result: **NOT MATCH** — Step 3 (provenance conflict). Both have `original_video` set but pointing to different files. Even with identical shapes, different provenance → reject.

### Shape comparison details

Shape rejection compares `(frames, height, width)` only—**channels are excluded**. Grayscale detection is noisy (affected by compression, user settings) and unreliable as a matching signal.

```python
# These videos are shape-compatible (not rejected):
video_a: (1000, 480, 640, 1)  # grayscale
video_b: (1000, 480, 640, 3)  # color

# These videos are shape-incompatible (rejected):
video_a: (1000, 480, 640, 3)
video_b: (500, 480, 640, 3)   # different frame count
```

### Other video matching methods

While AUTO is recommended, explicit matchers are available:

```python
# PATH: exact sanitized path match only
base.merge(other, video="path")

# BASENAME: filename match only (ignores directory)
# WARNING: Ambiguous with common filenames like "video.mp4"
base.merge(other, video="basename")

# CONTENT: shape + backend type match
# WARNING: Matches ANY videos with same resolution - use with caution
base.merge(other, video="content")

# IMAGE_DEDUP: for ImageVideo sequences with overlapping frames
base.merge(other, video="image_dedup")
```

For advanced configuration (e.g., strict path matching), use Matcher objects:

```python
from sleap_io.model.matching import VideoMatcher
base.merge(other, video=VideoMatcher(method="path", strict=True))
```

### Cross-platform paths

When merging labels created on different computers, use `replace_filenames` to normalize paths before merging:

```python
# Replace by prefix (useful for different mount points)
other.replace_filenames(prefix_map={
    "/home/bob/data": "C:/Users/alice/data"
})
base.merge(other)

# Or replace specific files
other.replace_filenames(filename_map={
    "/home/bob/data/video.mp4": "C:/Users/alice/data/video.mp4"
})
base.merge(other)
```

With the new AUTO algorithm, this is often unnecessary if basenames are unique—AUTO will match via leaf uniqueness (Step 6). However, `replace_filenames` is still useful when:

- You have multiple videos with the same basename in different directories
- You want deterministic matching without relying on the cascade
- You're using PATH matching mode explicitly

## Frame strategies

The `frame` parameter controls what happens when both datasets have the same frame (i.e., after video matching succeeds). Strategies fall into two categories:

**Frame-level strategies** operate on whole frames without comparing individual instances:

- `keep_original`, `keep_new`, `keep_both`, `replace_predictions`

**Instance-level strategies** use spatial matching to pair up instances, then decide per-pair:

- `smart`, `update_tracks`

### Spatial matching algorithm

Instance-level strategies use spatial matching to find corresponding instances between frames. The algorithm:

1. For each instance, extract all point coordinates as an array
2. For two instances to match:
   - Their NaN patterns must match exactly (same missing points)
   - All visible points must be within the threshold distance (default: 5 pixels)
3. Distance is computed as Euclidean distance per point: `sqrt((x1-x2)² + (y1-y2)²)`
4. All point distances must be ≤ threshold for instances to be considered "matched"

Unmatched instances (no corresponding instance within threshold) are handled according to the strategy.

| Strategy | Spatial matching | Use case |
|----------|------------------|----------|
| `smart` | Yes | HITL: add predictions, preserve labels |
| `update_tracks` | Yes | Apply tracking to existing poses |
| `keep_original` | No | Protect base from any changes |
| `keep_new` | No | Overwrite base with new data |
| `keep_both` | No | Combine multiple annotators |
| `replace_predictions` | No | Re-run inference, replace old predictions |

### Frame-level strategies

These strategies do NOT perform spatial matching. They operate on entire frames.

### `keep_original`

Keep base unchanged. Ignore everything from other.

| Frame location | Result |
|----------------|--------|
| Only in base | Keep |
| Only in other | Discard |
| In both | Keep base, ignore other |

```python
# Base frame 0: [User A, Pred B]
# Other frame 0: [Pred C, Pred D]
# Other frame 1: [Pred E]

base.merge(other, frame="keep_original")

# Result:
#   Frame 0: [User A, Pred B]  (unchanged)
#   Frame 1: does not exist    (discarded)
```

### `keep_new`

Replace base with other. Discard base.

| Frame location | Result |
|----------------|--------|
| Only in base | Discard |
| Only in other | Add |
| In both | Replace with other |

```python
# Base frame 0: [User A, Pred B]
# Base frame 1: [User C]
# Other frame 0: [Pred D]

base.merge(other, frame="keep_new")

# Result:
#   Frame 0: [Pred D]          (replaced)
#   Frame 1: does not exist    (discarded)
```

### `keep_both`

Concatenate all instances. No deduplication.

| Frame location | Result |
|----------------|--------|
| Only in base | Keep |
| Only in other | Add |
| In both | Concatenate (base + other) |

```python
# Base frame 0: [User A, Pred B]
# Other frame 0: [Pred C, Pred D]

base.merge(other, frame="keep_both")

# Result:
#   Frame 0: [User A, Pred B, Pred C, Pred D]
```

**Warning:** Creates duplicates at same locations. Use for multi-annotator review.

### `replace_predictions`

Keep user instances (`Instance`) from base. Remove predictions (`PredictedInstance`) from base. Add only predictions from other.

| Frame location | Result |
|----------------|--------|
| Only in base | Keep |
| Only in other | Add predictions only |
| In both | Base users + other predictions |

| Instance type | From base | From other |
|---------------|-----------|------------|
| User (`Instance`) | Keep | Ignore |
| Prediction (`PredictedInstance`) | Remove | Add |

```python
# Base frame 0: [User A, Pred B, Pred C]
# Other frame 0: [User X, Pred D, Pred E]

base.merge(other, frame="replace_predictions")

# Result:
#   Frame 0: [User A, Pred D, Pred E]
#   - User A kept (user from base)
#   - Pred B, C removed (old predictions)
#   - User X ignored (user from other)
#   - Pred D, E added (new predictions)
```

### Instance-level strategies

These strategies perform spatial matching to pair instances, then decide per-pair.

### `smart` (default)

Preserve user labels. Replace matched predictions. Add unmatched instances.

| Frame location | Result |
|----------------|--------|
| Only in base | Keep |
| Only in other | Add |
| In both | Spatial matching (see below) |

**When instances are matched:**

| Base | Other | Result |
|------|-------|--------|
| User | User | Keep base |
| User | Prediction | Keep base |
| Prediction | User | Replace with other |
| Prediction | Prediction | Replace with other |

**Unmatched:** Keep from base, add from other.

```python
# Base frame 0:
#   User A at (10, 10)
#   Pred B at (50, 50)
#   Pred C at (100, 100)  <- no match

# Other frame 0:
#   Pred D at (12, 12)    <- matches User A (within 5px)
#   Pred E at (52, 52)    <- matches Pred B
#   Pred F at (200, 200)  <- no match

base.merge(other, frame="smart")

# Result:
#   User A at (10, 10)    <- kept (user beats prediction)
#   Pred E at (52, 52)    <- replaced (newer prediction)
#   Pred C at (100, 100)  <- kept (unmatched)
#   Pred F at (200, 200)  <- added (unmatched)
```

### `update_tracks`

Update track assignments only. Do not modify poses. Do not add or remove instances.

| Frame location | Result |
|----------------|--------|
| Only in base | Keep |
| Only in other | Ignore |
| In both | Spatial matching (see below) |

**When matched:** Copy `.track` and `.tracking_score` from other to base.

**Unmatched:** Keep base unchanged, ignore other.

```python
# Base frame 0:
#   Pred A at (10, 10) with track=None
#   Pred B at (50, 50) with track=None

# Other frame 0:
#   Pred X at (12, 12) with track=Track("animal_1")
#   Pred Y at (52, 52) with track=Track("animal_2")

base.merge(other, frame="update_tracks")

# Result:
#   Pred A at (10, 10) with track=Track("animal_1")
#   Pred B at (50, 50) with track=Track("animal_2")
#   (poses unchanged, only tracks updated)
```

## Other matchers

### Instance matching

Configure how instances are paired for `smart` and `update_tracks` frame strategies:

```python
# Spatial (default): match by average point distance
base.merge(other, instance="spatial")

# Identity: match by track identity
base.merge(other, instance="identity")

# IoU: match by bounding box overlap
base.merge(other, instance="iou")
```

For advanced configuration (e.g., custom threshold), use Matcher objects:

```python
from sleap_io.model.matching import InstanceMatcher
base.merge(other, instance=InstanceMatcher(method="spatial", threshold=2.0))
base.merge(other, instance=InstanceMatcher(method="iou", threshold=0.5))
```

### Skeleton matching

```python
# Structure (default): same nodes (any order)
base.merge(other, skeleton="structure")

# Exact: same nodes in same order
base.merge(other, skeleton="exact")

# Overlap: partial match by Jaccard similarity
base.merge(other, skeleton="overlap")

# Subset: one skeleton's nodes are subset of other
base.merge(other, skeleton="subset")
```

For advanced configuration:

```python
from sleap_io.model.matching import SkeletonMatcher
base.merge(other, skeleton=SkeletonMatcher(method="overlap", threshold=0.7))
```

## Troubleshooting

### No frames merged (video added as new)

With AUTO matching, if videos aren't matched, they're added as new videos. This is the safe default—check if this was intentional:

```python
# Check what videos exist after merge
print("Videos after merge:", [v.filename for v in base.videos])
```

**Common causes and solutions:**

1. **Different shapes** — Videos have different frame counts or resolutions (correctly rejected)
   ```python
   # Check shapes
   for v in base.videos:
       print(f"{v.filename}: {v.shape}")
   ```

2. **Ambiguous basenames** — Multiple videos with same filename but different directories
   ```python
   # Use replace_filenames to make paths match
   other.replace_filenames(prefix_map={
       "/remote/data": "/local/data"
   })
   base.merge(other)
   ```

3. **Files don't exist** — Shape comparison requires accessible files
   - Ensure video files exist at the paths stored in the labels
   - Or use `replace_filenames` to update paths to accessible locations

**Don't blindly use BASENAME matching** — it ignores directory structure and can match wrong videos when you have multiple files with the same name.

### Video matched incorrectly

If frames were merged to the wrong video, the source data may have ambiguous paths. For future merges:

1. Use `replace_filenames` to ensure unique, correct paths before merging
2. Consider using PATH matching for strict control: `video="path"`

### Duplicate instances

Use `smart` instead of `keep_both`, or tighten match threshold:

```python
from sleap_io.model.matching import InstanceMatcher
base.merge(other, instance=InstanceMatcher(method="spatial", threshold=2.0))
```

## API reference

### Labels.merge

::: sleap_io.model.labels.Labels.merge
    options:
        heading_level: 4

### Labels.replace_filenames

::: sleap_io.model.labels.Labels.replace_filenames
    options:
        heading_level: 4

### Strategy enum

::: sleap_io.model.matching.FrameStrategy
    options:
        heading_level: 4

### Matcher classes

::: sleap_io.model.matching.VideoMatcher
    options:
        heading_level: 4

::: sleap_io.model.matching.InstanceMatcher
    options:
        heading_level: 4

::: sleap_io.model.matching.SkeletonMatcher
    options:
        heading_level: 4

### Pre-configured matchers

::: sleap_io.model.matching.BASENAME_VIDEO_MATCHER
    options:
        heading_level: 4

::: sleap_io.model.matching.AUTO_VIDEO_MATCHER
    options:
        heading_level: 4

### Result classes

::: sleap_io.model.matching.MergeResult
    options:
        heading_level: 4
