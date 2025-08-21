# Merging Annotations

The `sleap-io` library provides comprehensive functionality for merging multiple annotation files. This is particularly useful for:

- Combining predictions with manual labels (human-in-the-loop workflows)
- Merging annotations from multiple annotators
- Consolidating partial annotations
- Updating existing projects with new predictions

## Basic Usage

### Merging Two Labels Objects

```python
from sleap_io import load_file

# Load two annotation files
base_labels = load_file("base_annotations.slp")
new_labels = load_file("new_predictions.slp")

# Merge new labels into base
result = base_labels.merge(new_labels)

# Check the result
print(result.summary())
```

### Merge Strategies

The merge system supports different strategies for handling overlapping instances:

#### Smart Strategy (Default)
Preserves user labels over predictions when they overlap:

```python
result = base_labels.merge(new_labels, frame_strategy="smart")
```

#### Keep Original Strategy
Always keeps instances from the base labels:

```python
result = base_labels.merge(new_labels, frame_strategy="keep_original")
```

#### Keep New Strategy
Always keeps instances from the new labels:

```python
result = base_labels.merge(new_labels, frame_strategy="keep_new")
```

#### Keep Both Strategy
Keeps all instances from both sources:

```python
result = base_labels.merge(new_labels, frame_strategy="keep_both")
```

## Advanced Configuration

### Custom Instance Matching

You can customize how instances are matched between frames:

```python
from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

# Match instances by spatial proximity (default)
spatial_matcher = InstanceMatcher(
    method=InstanceMatchMethod.SPATIAL,
    threshold=5.0  # pixels
)

# Match instances by track identity
identity_matcher = InstanceMatcher(
    method=InstanceMatchMethod.IDENTITY
)

# Match instances by IoU overlap
iou_matcher = InstanceMatcher(
    method=InstanceMatchMethod.IOU,
    threshold=0.5
)

# Use custom matcher in merge
result = base_labels.merge(
    new_labels,
    instance_matcher=spatial_matcher
)
```

### Skeleton Matching

Control how skeletons are matched between projects:

```python
from sleap_io.model.matching import SkeletonMatcher, SkeletonMatchMethod

# Exact match (same nodes in same order)
exact_matcher = SkeletonMatcher(method=SkeletonMatchMethod.EXACT)

# Structure match (same nodes, any order)
structure_matcher = SkeletonMatcher(method=SkeletonMatchMethod.STRUCTURE)

# Overlap match (partial node overlap)
overlap_matcher = SkeletonMatcher(
    method=SkeletonMatchMethod.OVERLAP,
    min_overlap=0.7  # 70% node overlap required
)

result = base_labels.merge(
    new_labels,
    skeleton_matcher=overlap_matcher
)
```

### Error Handling

Configure how merge errors are handled:

```python
# Continue on errors (default)
result = base_labels.merge(new_labels, error_mode="continue")

# Stop on first error
result = base_labels.merge(new_labels, error_mode="strict")

# Warn about errors but continue
result = base_labels.merge(new_labels, error_mode="warn")
```

## Common Workflows

### Human-in-the-Loop (HITL) Workflow

Merge predictions back into a manually annotated project:

```python
from sleap_io import load_file, save_file

# Load manual annotations and predictions
manual_labels = load_file("manual_annotations.slp")
predictions = load_file("predictions.slp")

# Merge predictions, preserving manual labels
result = manual_labels.merge(
    predictions,
    frame_strategy="smart",  # Keep manual labels over predictions
    validate=True  # Validate skeleton compatibility
)

# Check what was merged
print(f"Frames merged: {result.frames_merged}")
print(f"Instances added: {result.instances_added}")
print(f"Instances updated: {result.instances_updated}")
print(f"Conflicts resolved: {len(result.conflicts)}")

# Save the merged project
save_file(manual_labels, "merged_project.slp")
```

### Combining Multiple Annotators

Merge annotations from different team members:

```python
# Load annotations from multiple annotators
annotator1 = load_file("annotator1.slp")
annotator2 = load_file("annotator2.slp")
annotator3 = load_file("annotator3.slp")

# Create base project
merged = annotator1

# Merge additional annotators
for labels in [annotator2, annotator3]:
    result = merged.merge(
        labels,
        frame_strategy="keep_both",  # Keep all annotations
        validate=True
    )
    print(f"Merged {result.frames_merged} frames from {labels.filename}")

# Review conflicts
for conflict in result.conflicts:
    print(f"Conflict at frame {conflict.frame.frame_idx}: {conflict.conflict_type}")
```

### Updating Predictions

Replace old predictions with new ones:

```python
# Load existing project and new predictions
project = load_file("project.slp")
new_predictions = load_file("new_predictions.slp")

# Replace predictions while keeping manual labels
result = project.merge(
    new_predictions,
    frame_strategy="smart",  # Smart replacement
    instance_matcher=InstanceMatcher(
        method=InstanceMatchMethod.SPATIAL,
        threshold=10.0  # More lenient matching
    )
)

print(result.summary())
```

## Merge Result Analysis

The merge operation returns a `MergeResult` object with detailed information:

```python
result = base_labels.merge(new_labels)

# Basic statistics
print(f"Successful: {result.successful}")
print(f"Frames merged: {result.frames_merged}")
print(f"Instances added: {result.instances_added}")
print(f"Instances updated: {result.instances_updated}")
print(f"Instances skipped: {result.instances_skipped}")

# Detailed information
print(f"Skeletons merged: {result.skeletons_merged}")
print(f"Videos merged: {result.videos_merged}")
print(f"Tracks merged: {result.tracks_merged}")

# Error information
if result.errors:
    print(f"Errors: {len(result.errors)}")
    for error in result.errors:
        print(f"  - {error.message}")

# Conflict information
if result.conflicts:
    print(f"Conflicts: {len(result.conflicts)}")
    for conflict in result.conflicts:
        print(f"  - Frame {conflict.frame.frame_idx}: {conflict.conflict_type}")

# Get full summary
print(result.summary())
```

## Provenance Tracking

Merge operations are automatically tracked in the Labels provenance:

```python
# After merging
result = base_labels.merge(new_labels)

# Check merge history
merge_history = base_labels.provenance.get("merge_history", [])
for merge in merge_history:
    print(f"Merged at: {merge['timestamp']}")
    print(f"Source frames: {merge['source_labels']['n_frames']}")
    print(f"Frames merged: {merge['result']['frames_merged']}")
```

## Frame-Level Merging

For fine-grained control, you can merge individual frames:

```python
from sleap_io.model.labeled_frame import LabeledFrame

# Get frames from different sources
frame1 = base_labels.labeled_frames[0]
frame2 = new_labels.labeled_frames[0]

# Merge frames
merged_instances, conflicts = frame1.merge(
    frame2,
    strategy="smart",
    instance_matcher=InstanceMatcher(method=InstanceMatchMethod.SPATIAL)
)

# Update frame with merged instances
frame1.instances = merged_instances
```

## Best Practices

1. **Always validate skeletons** when merging from different sources:
   ```python
   result = labels.merge(other, validate=True)
   ```

2. **Use appropriate matching thresholds** based on your data:
   - Tighter thresholds (2-5 pixels) for high-precision tracking
   - Looser thresholds (10-20 pixels) for noisy or low-resolution data

3. **Review conflicts** after merging:
   ```python
   for conflict in result.conflicts:
       # Handle or log conflicts
       pass
   ```

4. **Save provenance** for reproducibility:
   ```python
   # Provenance is automatically saved with the Labels
   save_file(merged_labels, "merged_with_provenance.slp")
   ```

5. **Test merge strategies** on a subset first:
   ```python
   # Test on first 100 frames
   test_base = base_labels[:100]
   test_new = new_labels[:100]
   test_result = test_base.merge(test_new)
   ```