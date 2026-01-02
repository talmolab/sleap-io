# Merging annotations

Merging combines annotations from multiple sources into a single dataset.

## Quick start

```python
import sleap_io as sio

base = sio.load_file("manual_annotations.slp")
predictions = sio.load_file("predictions.slp")

base.merge(predictions)  # default: smart strategy
base.save("merged.slp")
```

## How merging works

Merging proceeds in four steps:

1. **Match skeletons** — Find corresponding skeletons by node names
2. **Match videos** — Identify same videos across datasets (see [Video matching](#video-matching))
3. **Match tracks** — Map track identities by name
4. **Merge frames** — For each frame in `other`:
   - If frame doesn't exist in `base`: add it
   - If frame exists in both: apply the `frame_strategy`

The `frame_strategy` parameter controls what happens when both datasets have the same frame. Strategies fall into two categories:

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

## Strategies

| Strategy | Spatial matching | Use case |
|----------|------------------|----------|
| `smart` | Yes | HITL: add predictions, preserve labels |
| `update_tracks` | Yes | Apply tracking to existing poses |
| `keep_original` | No | Protect base from any changes |
| `keep_new` | No | Overwrite base with new data |
| `keep_both` | No | Combine multiple annotators |
| `replace_predictions` | No | Re-run inference, replace old predictions |

## Frame-level strategies

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

base.merge(other, frame_strategy="keep_original")

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

base.merge(other, frame_strategy="keep_new")

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

base.merge(other, frame_strategy="keep_both")

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

base.merge(other, frame_strategy="replace_predictions")

# Result:
#   Frame 0: [User A, Pred D, Pred E]
#   - User A kept (user from base)
#   - Pred B, C removed (old predictions)
#   - User X ignored (user from other)
#   - Pred D, E added (new predictions)
```

## Instance-level strategies

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

base.merge(other, frame_strategy="smart")

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

base.merge(other, frame_strategy="update_tracks")

# Result:
#   Pred A at (10, 10) with track=Track("animal_1")
#   Pred B at (50, 50) with track=Track("animal_2")
#   (poses unchanged, only tracks updated)
```

## Matching configuration

### Video matching

Videos must match for frames to merge. The default `AUTO` method tries multiple strategies in order:

1. **Exact path** — Full resolved paths match exactly
2. **Basename** — Filename matches (ignoring directory)
3. **Content** — Same shape (frames, height, width, channels) and backend type

```python
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod, BASENAME_VIDEO_MATCHER

# AUTO (default): tries path → basename → content
base.merge(other)

# BASENAME: match by filename only
base.merge(other, video_matcher=BASENAME_VIDEO_MATCHER)

# CONTENT: match by video shape and backend type
matcher = VideoMatcher(method=VideoMatchMethod.CONTENT)
base.merge(other, video_matcher=matcher)
```

#### Cross-platform video paths

When merging labels created on different computers, video paths won't match:

```
Windows: C:\Users\alice\data\video.mp4
Linux:   /home/bob/data/video.mp4
```

**Option 1: Use basename matching** (if filenames are unique)

```python
from sleap_io.model.matching import BASENAME_VIDEO_MATCHER

base.merge(other, video_matcher=BASENAME_VIDEO_MATCHER)
```

**Option 2: Fix paths before merging** with `replace_filenames`

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

**Option 3: Use content matching** (if videos have same dimensions)

```python
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod

matcher = VideoMatcher(method=VideoMatchMethod.CONTENT)
base.merge(other, video_matcher=matcher)
```

Content matching compares video shape `(frames, height, width, channels)` and backend type. It does NOT compare actual pixel data.

### Instance matching

Configure how instances are paired for `smart` and `update_tracks`:

```python
from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

# Spatial (default): match by average point distance
matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)

# Identity: match by track identity
matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)

# IoU: match by bounding box overlap
matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.5)

base.merge(other, instance_matcher=matcher)
```

### Skeleton matching

```python
from sleap_io.model.matching import SkeletonMatcher, SkeletonMatchMethod

# Structure (default): same nodes (any order)
# Exact: same nodes in same order
# Overlap: partial match by Jaccard similarity
# Subset: one skeleton's nodes are subset of other

matcher = SkeletonMatcher(method=SkeletonMatchMethod.OVERLAP, threshold=0.7)
base.merge(other, skeleton_matcher=matcher)
```

## Troubleshooting

### No frames merged

Videos don't match. Check paths and try basename matching:

```python
print("Base videos:", [v.filename for v in base.videos])
print("Other videos:", [v.filename for v in other.videos])

from sleap_io.model.matching import BASENAME_VIDEO_MATCHER
base.merge(other, video_matcher=BASENAME_VIDEO_MATCHER)
```

### Duplicate instances

Use `smart` instead of `keep_both`, or tighten match threshold:

```python
from sleap_io.model.matching import InstanceMatcher
matcher = InstanceMatcher(method="spatial", threshold=2.0)
base.merge(other, instance_matcher=matcher)
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
