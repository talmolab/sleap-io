# Merging Annotations

The `sleap-io` library provides comprehensive functionality for merging multiple annotation files. This is particularly useful for:

- Combining predictions with manual labels (human-in-the-loop workflows)
- Merging annotations from multiple annotators
- Consolidating partial annotations
- Updating existing projects with new predictions

## Basic Usage

### Merging Two Labels Objects

```python
import sleap_io as sio

# Load two annotation files
base_labels = sio.load_file("base_annotations.slp")
new_labels = sio.load_file("new_predictions.slp")

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

### Video Matching

Control how videos are matched between projects, especially useful for cross-platform workflows:

```python
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod

# AUTO mode (default) - tries multiple strategies
auto_matcher = VideoMatcher(method=VideoMatchMethod.AUTO)

# PATH mode - exact path matching
path_matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)

# BASENAME mode - match by filename only
basename_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

# CONTENT mode - match by video shape and backend type  
content_matcher = VideoMatcher(method=VideoMatchMethod.CONTENT)

# Use pre-configured matcher for filename-based matching
from sleap_io.model.matching import BASENAME_VIDEO_MATCHER

# Use in merge
result = base_labels.merge(
    new_labels,
    video_matcher=BASENAME_VIDEO_MATCHER
)
```

#### Cross-Platform Video Matching

Video paths often differ between systems, especially when collaborating or moving projects. Here's how to handle these scenarios:

##### Scenario 1: Same Files, Different Paths

**Problem**: You created annotations on Windows and received predictions generated on Linux:

```
# Original annotations (Windows)
├── C:\Users\researcher\Desktop\project\
│   ├── annotations.slp
│   └── videos\
│       ├── day1_trial1.mp4
│       ├── day1_trial2.mp4
│       └── day2_trial1.mp4

# New predictions (Linux) 
├── /home/lab/sleap_projects/mouse_behavior/
│   ├── predictions.slp
│   └── video_data/
│       ├── day1_trial1.mp4  # Same file!
│       ├── day1_trial2.mp4  # Same file!
│       └── day2_trial1.mp4  # Same file!
```

**Solution**: Use BASENAME matching to match by filename only:

```python
import sleap_io as sio
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod

# Load the files
annotations = sio.load_file(r"C:\Users\researcher\Desktop\project\annotations.slp")
predictions = sio.load_file("/home/lab/sleap_projects/mouse_behavior/predictions.slp")

# Create basename matcher - ignores directory paths
basename_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

# Merge - will match videos despite different directory structures
result = annotations.merge(predictions, video_matcher=basename_matcher)

print(f"Successfully matched {len(result.videos_matched)} videos")
# Output: Successfully matched 3 videos
```

##### Scenario 2: Reorganized Project Structure

**Problem**: You reorganized your project structure after creating annotations:

```
# Original structure when annotations were created
├── /data/experiments/
│   ├── session_001/
│   │   ├── behavior_recording.mp4
│   │   └── neural_recording.mp4
│   ├── session_002/
│   │   ├── behavior_recording.mp4
│   │   └── neural_recording.mp4
│   └── annotations.slp

# New organized structure  
├── /data/organized_experiments/
│   ├── behavior_videos/
│   │   ├── session_001_behavior_recording.mp4  # Renamed!
│   │   └── session_002_behavior_recording.mp4  # Renamed!
│   ├── neural_videos/
│   │   ├── session_001_neural_recording.mp4    # Renamed!
│   │   └── session_002_neural_recording.mp4    # Renamed!
│   ├── old_annotations.slp                     # Original annotations
│   └── new_predictions.slp                     # New predictions with new paths
```

**Problem**: Filenames changed, so BASENAME won't work. **Solution**: Use CONTENT matching:

```python
# Load files with mismatched names
old_annotations = sio.load_file("/data/organized_experiments/old_annotations.slp")
new_predictions = sio.load_file("/data/organized_experiments/new_predictions.slp")

# Use content matching - matches by video shape and backend type
content_matcher = VideoMatcher(method=VideoMatchMethod.CONTENT)

result = old_annotations.merge(new_predictions, video_matcher=content_matcher)
print(f"Matched {len(result.videos_matched)} videos by content analysis")
```

##### Scenario 3: Mixed Situations (Recommended)

**Problem**: Some videos have matching filenames, others have different names but same content:

```
# Collaboration scenario - mixed file organization
├── /lab_server/shared_data/
│   ├── student_annotations.slp
│   └── videos/
│       ├── mouse_01_day1.mp4
│       ├── mouse_01_day2.mp4
│       └── mouse_02_day1.mp4

# PI's predictions from different system
├── /home/pi/analysis_results/
│   ├── sleap_predictions.slp  
│   └── renamed_videos/
│       ├── mouse_01_day1.mp4        # Same name ✓
│       ├── subject01_session02.mp4  # Different name, same content as mouse_01_day2.mp4
│       └── mouse_02_day1.mp4        # Same name ✓
```

**Solution**: Use AUTO mode (tries BASENAME first, then CONTENT):

```python
# Load annotations and predictions
student_work = sio.load_file("/lab_server/shared_data/student_annotations.slp")
pi_predictions = sio.load_file("/home/pi/analysis_results/sleap_predictions.slp")

# AUTO mode: tries filename matching first, falls back to content matching
auto_matcher = VideoMatcher(method=VideoMatchMethod.AUTO)

result = student_work.merge(pi_predictions, video_matcher=auto_matcher)

# Check what happened
for i, (student_video, pred_video) in enumerate(zip(student_work.videos, pi_predictions.videos)):
    print(f"Video {i+1}:")
    print(f"  Student: {student_video.filename}")
    print(f"  Prediction: {pred_video.filename}")
    print(f"  Matched: {'✓' if auto_matcher.match(student_video, pred_video) else '✗'}")

# Output:
# Video 1:
#   Student: /lab_server/shared_data/videos/mouse_01_day1.mp4  
#   Prediction: /home/pi/analysis_results/renamed_videos/mouse_01_day1.mp4
#   Matched: ✓ (by filename)
# 
# Video 2:
#   Student: /lab_server/shared_data/videos/mouse_01_day2.mp4
#   Prediction: /home/pi/analysis_results/renamed_videos/subject01_session02.mp4  
#   Matched: ✓ (by content)
#
# Video 3:
#   Student: /lab_server/shared_data/videos/mouse_02_day1.mp4
#   Prediction: /home/pi/analysis_results/renamed_videos/mouse_02_day1.mp4
#   Matched: ✓ (by filename)
```

##### Scenario 4: Strict Path Matching

**Problem**: You need exact path matching (same system, no file movement):

```
# Development workflow - everything in same locations
├── /project/
│   ├── manual_labels.slp      # Created manually
│   ├── predictions.slp        # Generated by model
│   └── videos/
│       ├── trial_001.mp4
│       ├── trial_002.mp4
│       └── trial_003.mp4
```

**Solution**: Use PATH matching for strict validation:

```python
# Load files that should have identical paths
manual = sio.load_file("/project/manual_labels.slp")  
predictions = sio.load_file("/project/predictions.slp")

# Strict path matching - paths must be identical
strict_matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)

try:
    result = manual.merge(predictions, video_matcher=strict_matcher)
    print("✓ All video paths matched exactly")
except Exception as e:
    print(f"✗ Path mismatch detected: {e}")
    
    # Fall back to more lenient matching
    backup_matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
    result = manual.merge(predictions, video_matcher=backup_matcher)
    print("✓ Successful merge with AUTO matching")
```

##### Real-World Workflow Example

Here's a complete example of a typical collaboration workflow:

```python
import sleap_io as sio
from sleap_io.model.matching import VideoMatcher, VideoMatchMethod

def smart_merge_with_fallback(base_labels_path, new_labels_path):
    """Merge labels with intelligent video matching and fallback strategies."""
    
    # Load the label files
    print(f"Loading base labels: {base_labels_path}")
    base = sio.load_file(base_labels_path)
    print(f"  - {len(base.videos)} videos, {len(base)} frames")
    
    print(f"Loading new labels: {new_labels_path}")
    new = sio.load_file(new_labels_path)
    print(f"  - {len(new.videos)} videos, {len(new)} frames")
    
    # Try different matching strategies in order of preference
    strategies = [
        ("AUTO", VideoMatcher(method=VideoMatchMethod.AUTO)),
        ("BASENAME", VideoMatcher(method=VideoMatchMethod.BASENAME)),
        ("CONTENT", VideoMatcher(method=VideoMatchMethod.CONTENT)),
    ]
    
    for strategy_name, matcher in strategies:
        try:
            print(f"\nTrying {strategy_name} matching...")
            
            # Test matching before full merge
            matched_videos = 0
            for base_video in base.videos:
                for new_video in new.videos:
                    if matcher.match(base_video, new_video):
                        matched_videos += 1
                        break
            
            if matched_videos == len(base.videos):
                print(f"✓ All {matched_videos} videos would match with {strategy_name}")
                
                # Perform the actual merge
                result = base.merge(new, video_matcher=matcher)
                
                print(f"✓ Merge completed successfully!")
                print(f"  - Frames merged: {result.frames_merged}")
                print(f"  - Instances added: {result.instances_added}")
                
                return result
            else:
                print(f"✗ Only {matched_videos}/{len(base.videos)} videos match with {strategy_name}")
                
        except Exception as e:
            print(f"✗ {strategy_name} matching failed: {e}")
            continue
    
    raise ValueError("Could not match videos with any strategy")

# Usage example
try:
    result = smart_merge_with_fallback(
        "/lab/annotations/manual_session1.slp",
        "/home/student/predictions/session1_preds.slp"
    )
    print(f"\n🎉 Successfully merged! Summary:")
    print(result.summary())
    
except Exception as e:
    print(f"❌ Merge failed: {e}")
    print("💡 Try checking that video files exist and have matching content")
```

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
import sleap_io as sio

# Load manual annotations and predictions
manual_labels = sio.load_file("manual_annotations.slp")
predictions = sio.load_file("predictions.slp")

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
manual_labels.save("merged_project.slp")
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
   merged_labels.save("merged_with_provenance.slp")
   ```

5. **Test merge strategies** on a subset first:
   ```python
   # Test on first 100 frames
   test_base = base_labels[:100]
   test_new = new_labels[:100]
   test_result = test_base.merge(test_new)
   ```

## API

### Enums

::: sleap_io.model.matching.SkeletonMatchMethod

::: sleap_io.model.matching.InstanceMatchMethod

::: sleap_io.model.matching.TrackMatchMethod

::: sleap_io.model.matching.VideoMatchMethod

::: sleap_io.model.matching.FrameStrategy

::: sleap_io.model.matching.ErrorMode

### Matcher Classes

::: sleap_io.model.matching.SkeletonMatcher

::: sleap_io.model.matching.InstanceMatcher

::: sleap_io.model.matching.TrackMatcher

::: sleap_io.model.matching.VideoMatcher

::: sleap_io.model.matching.FrameMatcher

### Pre-configured Matchers

::: sleap_io.model.matching.STRUCTURE_SKELETON_MATCHER

::: sleap_io.model.matching.SUBSET_SKELETON_MATCHER

::: sleap_io.model.matching.OVERLAP_SKELETON_MATCHER

::: sleap_io.model.matching.DUPLICATE_MATCHER

::: sleap_io.model.matching.IOU_MATCHER

::: sleap_io.model.matching.IDENTITY_INSTANCE_MATCHER

::: sleap_io.model.matching.NAME_TRACK_MATCHER

::: sleap_io.model.matching.IDENTITY_TRACK_MATCHER

::: sleap_io.model.matching.AUTO_VIDEO_MATCHER

::: sleap_io.model.matching.SOURCE_VIDEO_MATCHER

::: sleap_io.model.matching.PATH_VIDEO_MATCHER

::: sleap_io.model.matching.BASENAME_VIDEO_MATCHER

### Result Classes

::: sleap_io.model.matching.MergeResult

::: sleap_io.model.matching.ConflictResolution

::: sleap_io.model.matching.MergeError

::: sleap_io.model.matching.SkeletonMismatchError

::: sleap_io.model.matching.VideoNotFoundError

### Progress Tracking

::: sleap_io.model.matching.MergeProgressBar

### Labels Merge Method

::: sleap_io.model.labels.Labels.merge

