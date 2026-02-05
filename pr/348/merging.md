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

Merging proceeds in five steps:

1. **[Match skeletons](#skeleton-matching)** — Find corresponding skeletons by node structure
2. **[Match videos](#video-matching)** — Identify same videos across datasets
3. **[Match tracks](#track-matching)** — Map track identities between datasets
4. **[Merge frames](#frame-strategies)** — Combine frames based on the `frame` strategy
5. **[Match instances](#instance-matching)** — Pair instances within overlapping frames

### Preset options {#preset-options}

These options are controlled via parameters to [`Labels.merge()`](#sleap_io.model.labels.Labels.merge):

| Parameter | Controls | Options |
|-----------|----------|---------|
| `skeleton` | How skeletons are matched | [`"structure"`](#skeleton-matching) (default), [`"subset"`](#skeleton-matching), [`"overlap"`](#skeleton-matching), [`"exact"`](#skeleton-matching) |
| `video` | How videos are matched | [`"auto"`](#how-auto-matching-works) (default), [`"path"`](#other-video-matching-methods), [`"basename"`](#other-video-matching-methods), [`"content"`](#other-video-matching-methods), [`"shape"`](#other-video-matching-methods), [`"image_dedup"`](#other-video-matching-methods) |
| `track` | How tracks are matched | [`"name"`](#track-matching) (default), [`"identity"`](#track-matching) |
| `frame` | How overlapping frames are combined | [`"auto"`](#auto-default) (default), [`"replace_predictions"`](#replace_predictions), [`"keep_original"`](#other-frame-strategies), [`"keep_new"`](#other-frame-strategies), [`"keep_both"`](#other-frame-strategies), [`"update_tracks"`](#other-frame-strategies) |
| `instance` | How instances are paired within frames | [`"spatial"`](#instance-matching) (default), [`"identity"`](#instance-matching), [`"iou"`](#instance-matching) |

```python
base.merge(predictions)  # All defaults
base.merge(predictions, video="auto", frame="auto")  # Explicit defaults
```

---

## Matching without merging {#matching-without-merging}

Sometimes you need to inspect matching results without actually merging datasets. This is useful for:

- **Evaluation workflows**: Aligning predictions with ground truth to compute metrics
- **Debugging**: Understanding why videos/skeletons aren't matching as expected
- **Validation**: Checking matches before committing to a merge

Use [`Labels.match()`](#sleap_io.model.labels.Labels.match) to build correspondence maps without modifying either dataset:

```python
import sleap_io as sio

gt_labels = sio.load_slp("ground_truth.slp")
pred_labels = sio.load_slp("predictions.slp")

# Match predictions to ground truth (doesn't modify either)
result = gt_labels.match(pred_labels)

# Inspect results
print(result.summary())
# Videos: 2/2 matched
# Skeletons: 1/1 matched
# Tracks: 0/0 matched

# Check if all videos matched
if not result.all_videos_matched:
    print("Unmatched videos:")
    for video in result.unmatched_videos:
        print(f"  - {video.filename}")

# Iterate through matched videos
for pred_video, gt_video in result.video_map.items():
    if gt_video is not None:
        print(f"{pred_video.filename} -> {gt_video.filename}")
```

### MatchResult properties

The [`MatchResult`](#sleap_io.model.matching.MatchResult) object contains:

| Property | Type | Description |
|----------|------|-------------|
| `video_map` | `dict[Video, Video \| None]` | Maps other's videos to self's videos |
| `skeleton_map` | `dict[Skeleton, Skeleton \| None]` | Maps other's skeletons to self's skeletons |
| `track_map` | `dict[Track, Track \| None]` | Maps other's tracks to self's tracks |
| `unmatched_videos` | `list[Video]` | Videos from other with no match |
| `unmatched_skeletons` | `list[Skeleton]` | Skeletons from other with no match |
| `unmatched_tracks` | `list[Track]` | Tracks from other with no match |
| `all_videos_matched` | `bool` | True if all videos matched |
| `n_videos_matched` | `int` | Count of matched videos |

### Customizing matching

`Labels.match()` accepts the same matching parameters as `merge()`:

```python
# Use specific video matching method
result = gt_labels.match(pred_labels, video="basename")

# Use custom matchers
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod

matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)
result = gt_labels.match(pred_labels, video=matcher)
```

---

## Step 1: Skeleton matching {#skeleton-matching}

Before merging can proceed, skeletons from both datasets must be matched. Each skeleton in the incoming dataset is compared against skeletons in the base dataset to find correspondence.

### Matching methods

| Method | Behavior | Use case |
|--------|----------|----------|
| `"structure"` | Match if same node names, regardless of order | **Default.** Most common case |
| `"subset"` | Match if incoming skeleton nodes are a subset of base | Merging partial annotations |
| `"overlap"` | Match if sufficient overlap between node sets | Flexible matching with threshold |
| `"exact"` | Match only if nodes and edges are identical | Strict validation |

If no match is found, the skeleton is added as new to the base dataset.

### String configuration

```python
# Default: match by structure (same nodes, any order)
base.merge(other, skeleton="structure")

# Allow partial matches (incoming can have fewer nodes)
base.merge(other, skeleton="subset")

# Exact match required (nodes and edges must be identical)
base.merge(other, skeleton="exact")
```

### Object configuration

For advanced control, use [`SkeletonMatcher`](#sleap_io.model.matching.SkeletonMatcher):

```python
from sleap_io.model.matching import SkeletonMatcher

# Overlap matching with custom threshold (70% of nodes must match)
matcher = SkeletonMatcher(method="overlap", threshold=0.7)
base.merge(other, skeleton=matcher)
```

---

## Step 2: Video matching {#video-matching}

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
| 2 | Provenance conflict (different `original_video`, verifiable) | Reject |
| 3 | Same physical file (`os.path.samefile`) | **Match** |
| 4 | Exact path string match | **Match** |
| 5 | Unique basename/parent suffix match | **Match** |
| 6 | Pose matching (identical annotations on common frames) | **Match** |
| 7 | Image matching (pixel similarity, if enabled) | **Match** |
| 8 | No match found | Add as new |

**Shape is for rejection only.** Same resolution does NOT imply a match—it just means the videos aren't rejected. This prevents matching unrelated videos that happen to have the same dimensions.

**Provenance conflict checking** only rejects when files can be verified on disk. If neither file exists (e.g., embedded videos in `.pkg.slp` files), the check is skipped to allow fall-through to content-based matching.

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

**Cross-platform embedded videos (pose matching):**
```
Base: valence.pkg.slp (Linux, original_video=/snlkt/.../CHR/fly.mp4)
Other: stress.pkg.slp (Windows, original_video=X:/.../CHR/fly.mp4)
Result: MATCH — poses on common frames are identical
```

### Content-based matching {#content-based-matching}

AUTO mode includes pose-based matching as a default step. When videos have overlapping labeled frames, the matcher compares pose coordinates to identify identical videos even when file paths differ.

#### How pose matching works

1. Find common frame indices between videos
2. For each common frame, check if ANY instance pair has identical poses
3. If enough frames match (default: 3), consider it a match

This is particularly useful for:
- Cross-platform merges (Linux ↔ Windows paths)
- Embedded videos in `.pkg.slp` files that can't be verified on disk
- Videos that have been moved or renamed

#### VideoMatcher parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `content_frames` | 3 | Minimum matching frames for confident match |
| `compare_predictions` | `"auto"` | Include predictions: `"auto"`, `True`, or `False` |
| `compare_images` | `False` | Enable image comparison (expensive) |
| `image_similarity_threshold` | 0.05 | Max pixel difference (0-1 scale) |

#### compare_predictions modes

- `"auto"` (default): Include predictions only if the video has NO user instances
- `True`: Always include predictions in comparison
- `False`: Only compare user-labeled instances

#### Image similarity threshold

When `compare_images=True`, frames are compared by mean pixel difference:

- `0.05` (default): ~13/255 pixel difference allowed
- `0.01`: Very strict (~3/255 pixels)
- `0.1`: Lenient (~26/255 pixels)

```python
# Enable image comparison with custom threshold
from sleap_io.model.matching import VideoMatcher

base.merge(other, video=VideoMatcher(
    method="auto",
    compare_images=True,
    image_similarity_threshold=0.1,  # More lenient
))
```

### String configuration

```python
# Default: safe AUTO cascade
base.merge(other, video="auto")

# Exact path match only
base.merge(other, video="path")

# Match by filename only (ignores directory)
base.merge(other, video="basename")
```

### Object configuration

For advanced control, use [`VideoMatcher`](#sleap_io.model.matching.VideoMatcher):

```python
from sleap_io.model.matching import VideoMatcher

# Strict path matching (paths must be identical, no normalization)
matcher = VideoMatcher(method="path", strict=True)
base.merge(other, video=matcher)
```

### Other video matching methods {#other-video-matching-methods}

| Method | Behavior | Use case |
|--------|----------|----------|
| `"auto"` | Safe cascade (default) | Most situations |
| `"path"` | Exact path match only | Strict control |
| `"basename"` | Filename only, ignores directory | Cross-platform (use with caution) |
| `"content"` | Shape + backend type | **Dangerous** — matches any same-resolution video |
| `"shape"` | Match and merge by shape | Image list merging |
| `"image_dedup"` | Deduplicate image lists | Remove duplicate images |

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

**Step 3: Fix** — Use [`replace_filenames`](#sleap_io.model.labels.Labels.replace_filenames) before merging:
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

---

## Step 3: Track matching {#track-matching}

Tracks represent identities (e.g., individual animals) that persist across frames. During merge, tracks from the incoming dataset are matched to tracks in the base dataset.

### Matching methods

| Method | Behavior | Use case |
|--------|----------|----------|
| `"name"` | Match tracks with identical names | **Default.** Named individuals |
| `"identity"` | Match by track object identity | Same `Track` object in memory |

If no match is found, the track is added as new to the base dataset.

### String configuration

```python
# Default: match by track name
base.merge(other, track="name")

# Match by object identity (same Track instance)
base.merge(other, track="identity")
```

### Object configuration

For advanced control, use [`TrackMatcher`](#sleap_io.model.matching.TrackMatcher):

```python
from sleap_io.model.matching import TrackMatcher

# Explicit name matching
matcher = TrackMatcher(method="name")
base.merge(other, track=matcher)
```

---

## Step 4: Frame strategies {#frame-strategies}

The `frame` parameter controls what happens when both datasets have the same frame (same video and frame index).

### `auto` (default) {#auto-default}

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
base.merge(predictions)  # Uses auto by default
```

### `replace_predictions` {#replace_predictions}

Replace all predictions in base with predictions from other. User labels are preserved.

```python
# Re-ran inference, want to update predictions
base.merge(new_predictions, frame="replace_predictions")
```

| Instance type | From base | From other |
|---------------|-----------|------------|
| User label | **Keep** | Ignore |
| Prediction | Remove | **Add** |

### Other frame strategies {#other-frame-strategies}

| Strategy | Behavior | Use case |
|----------|----------|----------|
| `"keep_original"` | Ignore other entirely for overlapping frames | Preserve base annotations |
| `"keep_new"` | Replace base with other for overlapping frames | Overwrite with new annotations |
| `"keep_both"` | Concatenate all instances (creates duplicates) | Manual deduplication later |
| `"update_tracks"` | Copy track assignments only, don't modify poses | Update identity labels |

```python
# Keep only the original annotations
base.merge(other, frame="keep_original")

# Replace with new annotations
base.merge(other, frame="keep_new")

# Keep everything (may create duplicates)
base.merge(other, frame="keep_both")

# Update track assignments without changing poses
base.merge(other, frame="update_tracks")
```

---

## Step 5: Instance matching {#instance-matching}

For frame strategies that need to pair instances (`auto`, `update_tracks`), instance matching determines how instances in the base frame correspond to instances in the incoming frame.

### Matching methods

| Method | Behavior | Use case |
|--------|----------|----------|
| `"spatial"` | Match by centroid distance | **Default.** Position-based matching |
| `"identity"` | Match by track identity | Same track assignment |
| `"iou"` | Match by bounding box IoU | Overlap-based matching |

### String configuration

```python
# Default: spatial matching with 5px threshold
base.merge(other, instance="spatial")

# Match by track identity
base.merge(other, instance="identity")

# Match by bounding box overlap
base.merge(other, instance="iou")
```

### Object configuration

For advanced control, use [`InstanceMatcher`](#sleap_io.model.matching.InstanceMatcher):

```python
from sleap_io.model.matching import InstanceMatcher

# Tighter spatial matching (2px threshold)
matcher = InstanceMatcher(method="spatial", threshold=2.0)
base.merge(other, instance=matcher)

# IoU matching with 50% overlap threshold
matcher = InstanceMatcher(method="iou", threshold=0.5)
base.merge(other, instance=matcher)
```

---

## Troubleshooting

### Videos weren't matched (false negative)

See [Handling false negatives](#handling-false-negatives) above.

### Videos matched incorrectly (false positive)

This shouldn't happen with AUTO matching. If it does:

1. Check if videos have identical shapes AND ambiguous paths
2. Use `video="path"` for strict matching
3. Report as a bug—AUTO should be conservative

### Duplicate instances after merge

Use `auto` instead of `keep_both`, or tighten the instance match threshold:

```python
from sleap_io.model.matching import InstanceMatcher
base.merge(other, instance=InstanceMatcher(method="spatial", threshold=2.0))
```

---

## Reference

### Labels.merge

::: sleap_io.model.labels.Labels.merge
    options:
        heading_level: 4

### Labels.add_video

::: sleap_io.model.labels.Labels.add_video
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

### SkeletonMatcher

::: sleap_io.model.matching.SkeletonMatcher
    options:
        heading_level: 4

### TrackMatcher

::: sleap_io.model.matching.TrackMatcher
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
