## Implementation Plan: Merging with De-duplication

After thorough analysis of the requirements and existing codebase, here's the comprehensive plan for implementing merging functionality in sleap-io.

### Key Design Principles

1. **Clean API**: Semantic method names that clearly express intent (avoiding "complex merge" patterns)
2. **Identity Preservation**: Merged objects maintain base Labels' identities
3. **Resilient Operations**: Support partial merges where some objects fail but others succeed
4. **Scale-Ready**: Optimized for hundreds of thousands of frames
5. **Backwards Compatible**: Merged files readable by older SLEAP versions

### Core API Design

#### 1. Comparison Methods (Consistent Across All Classes)

All data classes will have standardized comparison APIs:

```python
# Instance class
instance.same_pose_as(other, tolerance=5.0)  # Spatial similarity
instance.same_identity_as(other)  # Track-based matching
instance.overlaps_with(other, iou_threshold=0.5)  # Bounding box overlap

# Skeleton class  
skeleton.matches(other, require_same_order=False)  # Structure comparison
skeleton.node_similarities(other)  # Detailed overlap metrics

# Track class
track.matches(other, method="name")  # Identity matching
track.similarity_to(other)  # Detailed metrics

# Video class
video.matches_path(other, strict=False)  # Path comparison
video.matches_content(other)  # Content verification

# LabeledFrame class
frame.matches(other, video_must_match=True)  # Frame identity
frame.similarity_to(other)  # Instance overlap metrics
```

#### 2. Unified Matcher System (`sleap_io/model/matching.py`)

Configurable matchers with string enum strategies for validation:

```python
# String enums for type safety and flexibility
class SkeletonMatchMethod(str, Enum):
    EXACT = "exact"
    STRUCTURE = "structure"
    OVERLAP = "overlap"
    SUBSET = "subset"

# Matcher classes using attrs
@attrs.define
class SkeletonMatcher:
    method: Union[SkeletonMatchMethod, str] = attrs.field(...)
    # Accepts both enum and string, validates automatically

# Pre-configured matchers for common use cases
STRUCTURE_SKELETON_MATCHER = SkeletonMatcher(method="structure")
DUPLICATE_MATCHER = InstanceMatcher(method="spatial", threshold=5.0)
```

#### 3. Labels-Level Merge API

```python
result = labels.merge(
    other,
    instance_matcher=None,  # Default: DUPLICATE_MATCHER
    skeleton_matcher=None,  # Default: STRUCTURE_SKELETON_MATCHER  
    video_matcher=None,     # Default: AUTO_VIDEO_MATCHER
    track_matcher=None,     # Default: NAME_TRACK_MATCHER
    frame_strategy="smart", # Options: smart, keep_original, keep_new, keep_both
    validate=True,          # Check for conflicts
    progress_callback=None, # For progress reporting
    error_mode="continue"   # Options: continue, strict, warn
) -> MergeResult
```

### Workflow Support

The API handles all identified workflows elegantly:

**1. HITL Merging** (Primary Use Case)
```python
# Simple one-liner with smart defaults
labels.merge(predictions)  # Preserves user labels, updates predictions
```

**2. Path Resolution** (Cross-Platform/Moved Files)
```python
video_matcher = VideoMatcher(
    method="resolve",
    base_path="/new/location",
    fallback_directories=["/backup", "/archive"]
)
labels.merge(other, video_matcher=video_matcher)
```

**3. Skeleton Harmonization** (Adding Nodes)
```python
# Accept old skeleton as subset of new
skeleton_matcher = SkeletonMatcher(method="subset")
new_labels.merge(old_labels, skeleton_matcher=skeleton_matcher)
```

**4. Package File Handling** (.pkg.slp)
```python
# Automatically restore source videos
labels.merge(pkg_predictions, 
             video_matcher=SOURCE_VIDEO_MATCHER)
```

### Error Handling & Recovery

```python
class MergeError(Exception): pass
class SkeletonMismatchError(MergeError): pass
class VideoNotFoundError(MergeError): pass

@attrs.define
class MergeResult:
    successful: bool
    frames_merged: int
    instances_added: int
    conflicts: list[ConflictResolution]
    errors: list[MergeError]
    
    def summary(self) -> str:
        """Human-readable summary"""
```

### Progress Reporting

Using existing `tqdm` dependency for progress feedback:

```python
from tqdm import tqdm

class MergeProgressBar:
    """Context manager for merge progress tracking."""
    def __init__(self, desc="Merging"):
        self.desc = desc
        self.pbar = None
        
    def callback(self, current, total, message):
        if self.pbar is None and total:
            self.pbar = tqdm(total=total, desc=self.desc)
        if self.pbar:
            self.pbar.set_description(message)
            self.pbar.n = current
            self.pbar.refresh()

# Usage
with MergeProgressBar("Merging predictions") as progress:
    result = labels.merge(predictions, progress_callback=progress)
```

### Provenance Tracking

Merge operations are tracked in Labels.provenance:

```python
{
    "merge_history": [{
        "timestamp": "2024-01-20T10:30:00",
        "source_labels": {...},
        "result": {
            "frames_merged": 1000,
            "instances_added": 5000,
            "conflicts": 23
        }
    }]
}
```

### Implementation Phases

**Phase 1**: Core comparison methods on Instance, Skeleton, Track, Video, LabeledFrame
**Phase 2**: Unified matcher system in `sleap_io/model/matching.py`
**Phase 3**: Frame-level merging with strategies
**Phase 4**: Labels-level merge with error handling
**Phase 5**: Integration tests for all workflows
**Phase 6**: Documentation and examples

### Key Advantages Over Core SLEAP

- **Cleaner API**: No "complex_merge" confusion
- **Type Safety**: String enums with validation
- **Better Errors**: Detailed error types and recovery
- **Progress Feedback**: Built-in progress reporting
- **Provenance**: Automatic merge history tracking
- **Flexibility**: Configurable strategies for different workflows

This design provides a robust, scalable solution that handles all identified use cases while maintaining a clean, intuitive API. The implementation can proceed incrementally with each phase providing immediate value.