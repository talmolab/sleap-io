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
2. **Match videos** — Identify same videos across datasets
3. **Match tracks** — Map track identities by name
4. **Merge frames** — Combine frames based on the `frame` strategy

Two parameters control the key behavior:

| Parameter | Controls | Default |
|-----------|----------|---------|
| `video` | How videos are matched | `"auto"` |
| `frame` | How overlapping frames are combined | `"smart"` |

```python
base.merge(predictions, video="auto", frame="smart")
```

## Video matching

Videos must match for frames to merge. If video matching fails, the video is added as new—no frames merge because there's no overlap.

### Design philosophy

The default AUTO algorithm prioritizes **avoiding false positives** (matching wrong videos) over avoiding false negatives (failing to match correct videos):

| Error type | Consequence | Severity |
|------------|-------------|----------|
| **False positive** | Annotations merged to wrong video | **Data corruption** (often unrecoverable) |
| **False negative** | Video added as new | **Safe** (easily fixed, see below) |

**When uncertain, AUTO adds the video as new rather than risk a wrong match.**

### How AUTO matching works

For each incoming video, AUTO runs these checks in order:

| Step | Check | Result |
|------|-------|--------|
| 1 | Shape incompatible (frames, H, W differ) | Reject |
| 2 | Provenance conflict (different `original_video`) | Reject |
| 3 | Same physical file (`os.path.samefile`) | **Match** |
| 4 | Exact path string match | **Match** |
| 5 | Unique basename/parent suffix match | **Match** |
| 6 | No match found | Add as new |

**Shape is for rejection only.** Same resolution does NOT imply a match—it just means the videos aren't rejected. This prevents matching unrelated videos that happen to have the same dimensions.

### Examples

**Cross-platform paths with unique basenames:**
```
Base:  C:/Users/alice/data/fly.mp4
Other: /home/bob/data/fly.mp4
Result: MATCH — "fly.mp4" is unique in both sets
```

**Ambiguous basenames, parent disambiguates:**
```
Base:  /data/exp1/fly.mp4, /data/exp2/fly.mp4
Other: /remote/exp2/fly.mp4
Result: MATCH to exp2/fly.mp4 — parent directory disambiguates
```

**Same basename, different content:**
```
Base:  fly.mp4 (1000 frames)
Other: fly.mp4 (500 frames)
Result: NOT MATCH — shape rejection (different frame counts)
```

**PKG.SLP predictions to external video:**
```
Base: project.slp with /data/fly.mp4
Other: predictions.pkg.slp (embedded, original_video=/data/fly.mp4)
Result: MATCH — provenance chain links to same file
```

### Handling false negatives

If AUTO doesn't match videos that ARE the same, you have a **false negative**. This is safe—the video was added as new rather than corrupting data. Here's how to detect and fix it:

**Step 1: Detect** — Check video count after merge:
```python
base = sio.load_file("base.slp")
other = sio.load_file("other.slp")

print(f"Before: {len(base.videos)} videos")
result = base.merge(other)
print(f"After: {len(base.videos)} videos")

# If count increased unexpectedly, videos weren't matched
for v in base.videos:
    print(f"  {v.filename}")
```

**Step 2: Verify** — Confirm the videos ARE the same:
```python
# Check shapes match
video_a = base.videos[0]
video_b = base.videos[1]  # The one that should have matched

print(f"Video A: {video_a.shape}")  # e.g., (1000, 480, 640, 3)
print(f"Video B: {video_b.shape}")

# If files exist, check content
if video_a.exists() and video_b.exists():
    # Compare first frame visually or by hash
    frame_a = video_a[0]
    frame_b = video_b[0]
```

**Step 3: Fix** — Use `replace_filenames` before merging:
```python
# Reload and fix paths before merge
base = sio.load_file("base.slp")
other = sio.load_file("other.slp")

# Option A: Map the specific file
other.replace_filenames(filename_map={
    "/remote/path/fly.mp4": "/local/path/fly.mp4"
})

# Option B: Map by prefix (for multiple videos)
other.replace_filenames(prefix_map={
    "/remote/data": "/local/data"
})

# Now merge
result = base.merge(other)
print(f"Videos after fix: {len(base.videos)}")  # Should match original count
```

**Alternative: Force match with explicit matcher:**
```python
# Only use this if you're CERTAIN the videos are the same
base.merge(other, video="basename")  # Match by filename only
```

### Other video matching methods

| Method | Behavior | Use case |
|--------|----------|----------|
| `"auto"` | Safe cascade (default) | Most situations |
| `"path"` | Exact path match only | Strict control |
| `"basename"` | Filename only, ignores directory | Cross-platform (use with caution) |
| `"content"` | Shape + backend type | **Dangerous** — matches any same-resolution video |

## Frame strategies

The `frame` parameter controls what happens when both datasets have the same frame.

### `smart` (default)

The recommended strategy for human-in-the-loop workflows. Preserves user labels, updates predictions.

| Base instance | Other instance | Result |
|---------------|----------------|--------|
| User label | Prediction | Keep user label |
| User label | User label | Keep base (conflict) |
| Prediction | User label | **Replace** with user label |
| Prediction | Prediction | Replace with newer |

Unmatched instances from `other` are added.

```python
# Typical HITL workflow: merge predictions into labeled project
base.merge(predictions)  # Uses smart by default
```

### `replace_predictions`

Replace all predictions in base with predictions from other. User labels are preserved.

```python
# Re-ran inference, want to update predictions
base.merge(new_predictions, frame="replace_predictions")
```

| Instance type | From base | From other |
|---------------|-----------|------------|
| User label | **Keep** | Ignore |
| Prediction | Remove | **Add** |

### Other strategies

| Strategy | Behavior |
|----------|----------|
| `"keep_original"` | Ignore other entirely for overlapping frames |
| `"keep_new"` | Replace base with other for overlapping frames |
| `"keep_both"` | Concatenate all instances (creates duplicates) |
| `"update_tracks"` | Copy track assignments only, don't modify poses |

## Advanced configuration

### Instance matching

For `smart` and `update_tracks` strategies, instances are paired by spatial proximity (default 5px threshold):

```python
from sleap_io.model.matching import InstanceMatcher

# Tighter matching (2px)
base.merge(other, instance=InstanceMatcher(method="spatial", threshold=2.0))

# Match by track identity instead of position
base.merge(other, instance="identity")

# Match by bounding box IoU
base.merge(other, instance=InstanceMatcher(method="iou", threshold=0.5))
```

### Skeleton matching

```python
# Default: match if same nodes (any order)
base.merge(other, skeleton="structure")

# Partial overlap allowed
from sleap_io.model.matching import SkeletonMatcher
base.merge(other, skeleton=SkeletonMatcher(method="overlap", threshold=0.7))
```

### Video matcher objects

For advanced video matching configuration:

```python
from sleap_io.model.matching import VideoMatcher

# Strict path matching (paths must be identical)
base.merge(other, video=VideoMatcher(method="path", strict=True))
```

## Troubleshooting

### Videos weren't matched (false negative)

See [Handling false negatives](#handling-false-negatives) above.

### Videos matched incorrectly (false positive)

This shouldn't happen with AUTO matching. If it does:

1. Check if videos have identical shapes AND ambiguous paths
2. Use `video="path"` for strict matching
3. Report as a bug—AUTO should be conservative

### Duplicate instances after merge

Use `smart` instead of `keep_both`, or tighten the instance match threshold:

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

### FrameStrategy

::: sleap_io.model.matching.FrameStrategy
    options:
        heading_level: 4

### VideoMatcher

::: sleap_io.model.matching.VideoMatcher
    options:
        heading_level: 4

### InstanceMatcher

::: sleap_io.model.matching.InstanceMatcher
    options:
        heading_level: 4

### MergeResult

::: sleap_io.model.matching.MergeResult
    options:
        heading_level: 4
