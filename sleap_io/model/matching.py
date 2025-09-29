"""Unified matcher system for comparing and matching data structures during merging.

This module provides configurable matchers for comparing skeletons, instances, tracks,
videos, and frames during merge operations. The matchers use various strategies to
determine when data structures should be considered equivalent during merging.

Key features:
- Skeleton matching: exact, structure-based, overlap, and subset matching
- Instance matching: spatial proximity, track identity, and bounding box IoU
- Track matching: by name or object identity
- Video matching: path, basename, content, and auto matching
- Frame matching with configurable video matching requirements

Video matching supports path-based, filename-based, content-based, and
automatic strategies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Union

import attrs
import numpy as np

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


class SkeletonMatchMethod(str, Enum):
    """Methods for matching skeletons.

    Attributes:
        EXACT: Exact match requiring same nodes in the same order.
        STRUCTURE: Match requiring same nodes and edges, but order doesn't matter.
        OVERLAP: Partial match based on overlapping nodes (uses Jaccard similarity).
        SUBSET: Match if one skeleton's nodes are a subset of another's.
    """

    EXACT = "exact"
    STRUCTURE = "structure"
    OVERLAP = "overlap"
    SUBSET = "subset"


class InstanceMatchMethod(str, Enum):
    """Methods for matching instances.

    Attributes:
        SPATIAL: Match instances by spatial proximity using Euclidean distance.
        IDENTITY: Match instances by track identity (same track object).
        IOU: Match instances by bounding box Intersection over Union.
    """

    SPATIAL = "spatial"
    IDENTITY = "identity"
    IOU = "iou"


class TrackMatchMethod(str, Enum):
    """Methods for matching tracks.

    Attributes:
        NAME: Match tracks by their name attribute.
        IDENTITY: Match tracks by object identity (same Python object).
    """

    NAME = "name"
    IDENTITY = "identity"


class VideoMatchMethod(str, Enum):
    """Methods for matching videos.

    Attributes:
        PATH: Match by exact file path (strict or lenient based on
            VideoMatcher.strict setting).
        BASENAME: Match by filename only, ignoring directory paths.
        CONTENT: Match by video shape (frames, height, width, channels) and
            backend type.
        AUTO: Automatic matching - tries BASENAME first, then falls back to CONTENT.
        IMAGE_DEDUP: (ImageVideo only) Match ImageVideo instances with overlapping
            image files. Used to deduplicate individual images when merging datasets
            where videos are image sequences.
        SHAPE: Match videos by shape only (height, width, channels), ignoring
            filenames and frame count. Commonly used with ImageVideo to merge
            same-shaped image sequences.
    """

    PATH = "path"
    BASENAME = "basename"
    CONTENT = "content"
    AUTO = "auto"
    IMAGE_DEDUP = "image_dedup"
    SHAPE = "shape"


class FrameStrategy(str, Enum):
    """Strategies for handling frame merging.

    Attributes:
        SMART: Smart merging that preserves user labels over predictions when
            they overlap.
        KEEP_ORIGINAL: Always keep instances from the original (base) frame.
        KEEP_NEW: Always keep instances from the new (incoming) frame.
        KEEP_BOTH: Keep all instances from both frames without filtering.
    """

    SMART = "smart"
    KEEP_ORIGINAL = "keep_original"
    KEEP_NEW = "keep_new"
    KEEP_BOTH = "keep_both"
    UPDATE_TRACKS = "update_tracks"


class ErrorMode(str, Enum):
    """Error handling modes for merge operations.

    Attributes:
        CONTINUE: Continue merging on errors, collecting them in the result.
        STRICT: Raise an exception on the first error encountered.
        WARN: Issue warnings about errors but continue merging.
    """

    CONTINUE = "continue"
    STRICT = "strict"
    WARN = "warn"


@attrs.define
class SkeletonMatcher:
    """Matcher for comparing and matching skeletons.

    Attributes:
        method: The matching method to use. Can be a SkeletonMatchMethod enum value
            or a string that will be converted to the enum. Default is STRUCTURE.
        require_same_order: Whether to require nodes in the same order for STRUCTURE
            matching. Only used when method is STRUCTURE. Default is False.
        min_overlap: Minimum Jaccard similarity required for OVERLAP matching.
            Only used when method is OVERLAP. Default is 0.5.
    """

    method: Union[SkeletonMatchMethod, str] = attrs.field(
        default=SkeletonMatchMethod.STRUCTURE,
        converter=lambda x: SkeletonMatchMethod(x) if isinstance(x, str) else x,
    )
    require_same_order: bool = False
    min_overlap: float = 0.5

    def match(self, skeleton1: Skeleton, skeleton2: Skeleton) -> bool:
        """Check if two skeletons match according to the configured method."""
        if self.method == SkeletonMatchMethod.EXACT:
            return skeleton1.matches(skeleton2, require_same_order=True)
        elif self.method == SkeletonMatchMethod.STRUCTURE:
            return skeleton1.matches(
                skeleton2, require_same_order=self.require_same_order
            )
        elif self.method == SkeletonMatchMethod.OVERLAP:
            metrics = skeleton1.node_similarities(skeleton2)
            return metrics["jaccard"] >= self.min_overlap
        elif self.method == SkeletonMatchMethod.SUBSET:
            # Check if skeleton1 nodes are subset of skeleton2
            nodes1 = set(skeleton1.node_names)
            nodes2 = set(skeleton2.node_names)
            return nodes1.issubset(nodes2)
        else:
            raise ValueError(f"Unknown skeleton match method: {self.method}")


@attrs.define
class InstanceMatcher:
    """Matcher for comparing and matching instances.

    Attributes:
        method: The matching method to use. Can be an InstanceMatchMethod enum value
            or a string that will be converted to the enum. Default is SPATIAL.
        threshold: The threshold value used for matching. For SPATIAL method, this is
            the maximum pixel distance. For IOU method, this is the minimum IoU value.
            Not used for IDENTITY method. Default is 5.0.
    """

    method: Union[InstanceMatchMethod, str] = attrs.field(
        default=InstanceMatchMethod.SPATIAL,
        converter=lambda x: InstanceMatchMethod(x) if isinstance(x, str) else x,
    )
    threshold: float = 5.0

    def match(self, instance1: Instance, instance2: Instance) -> bool:
        """Check if two instances match according to the configured method."""
        if self.method == InstanceMatchMethod.SPATIAL:
            return instance1.same_pose_as(instance2, tolerance=self.threshold)
        elif self.method == InstanceMatchMethod.IDENTITY:
            return instance1.same_identity_as(instance2)
        elif self.method == InstanceMatchMethod.IOU:
            return instance1.overlaps_with(instance2, iou_threshold=self.threshold)
        else:
            raise ValueError(f"Unknown instance match method: {self.method}")

    def find_matches(
        self, instances1: list[Instance], instances2: list[Instance]
    ) -> list[tuple[int, int, float]]:
        """Find all matching instances between two lists.

        Returns:
            List of (idx1, idx2, score) tuples for matching instances.
        """
        matches = []

        for i, inst1 in enumerate(instances1):
            for j, inst2 in enumerate(instances2):
                if self.match(inst1, inst2):
                    # Calculate match score based on method
                    if self.method == InstanceMatchMethod.SPATIAL:
                        # Use inverse distance as score
                        pts1 = inst1.numpy()
                        pts2 = inst2.numpy()
                        valid = ~(np.isnan(pts1[:, 0]) | np.isnan(pts2[:, 0]))
                        if valid.any():
                            distances = np.linalg.norm(
                                pts1[valid] - pts2[valid], axis=1
                            )
                            score = 1.0 / (1.0 + np.mean(distances))
                        else:
                            score = 0.0
                    elif self.method == InstanceMatchMethod.IOU:
                        # Calculate actual IoU as score
                        bbox1 = inst1.bounding_box()
                        bbox2 = inst2.bounding_box()
                        if bbox1 is not None and bbox2 is not None:
                            # Calculate IoU
                            intersection_min = np.maximum(bbox1[0], bbox2[0])
                            intersection_max = np.minimum(bbox1[1], bbox2[1])
                            if np.all(intersection_min < intersection_max):
                                intersection_area = np.prod(
                                    intersection_max - intersection_min
                                )
                                area1 = np.prod(bbox1[1] - bbox1[0])
                                area2 = np.prod(bbox2[1] - bbox2[0])
                                union_area = area1 + area2 - intersection_area
                                score = (
                                    intersection_area / union_area
                                    if union_area > 0
                                    else 0
                                )
                            else:
                                score = 0.0
                        else:
                            score = 0.0
                    else:
                        score = 1.0  # Binary match for identity

                    matches.append((i, j, score))

        return matches


@attrs.define
class TrackMatcher:
    """Matcher for comparing and matching tracks.

    Attributes:
        method: The matching method to use. Can be a TrackMatchMethod enum value
            or a string that will be converted to the enum. Default is NAME.
    """

    method: Union[TrackMatchMethod, str] = attrs.field(
        default=TrackMatchMethod.NAME,
        converter=lambda x: TrackMatchMethod(x) if isinstance(x, str) else x,
    )

    def match(self, track1: Track, track2: Track) -> bool:
        """Check if two tracks match according to the configured method."""
        return track1.matches(track2, method=self.method.value)


@attrs.define
class VideoMatcher:
    """Matcher for comparing and matching videos.

    Attributes:
        method: The matching method to use. Can be a VideoMatchMethod enum value
            or a string that will be converted to the enum. Default is AUTO.
        strict: Whether to use strict path matching for the PATH method.
            When True, paths must be exactly identical. When False, paths
            are normalized before comparison. Only used when method is PATH.
            Default is False.
    """

    method: Union[VideoMatchMethod, str] = attrs.field(
        default=VideoMatchMethod.AUTO,
        converter=lambda x: VideoMatchMethod(x) if isinstance(x, str) else x,
    )
    strict: bool = False

    def match(self, video1: Video, video2: Video) -> bool:
        """Check if two videos match according to the configured method."""
        if self.method == VideoMatchMethod.AUTO:
            # Try different methods in order (identity check is redundant)
            if video1.matches_path(video2, strict=False):
                return True
            if video1.matches_content(video2):
                return True
            return False
        elif self.method == VideoMatchMethod.PATH:
            return video1.matches_path(video2, strict=self.strict)
        elif self.method == VideoMatchMethod.BASENAME:
            return video1.matches_path(video2, strict=False)
        elif self.method == VideoMatchMethod.CONTENT:
            return video1.matches_content(video2)
        elif self.method == VideoMatchMethod.IMAGE_DEDUP:
            # Match ImageVideo instances with overlapping images (ImageVideo only)
            return video1.has_overlapping_images(video2)
        elif self.method == VideoMatchMethod.SHAPE:
            # Match videos by shape only (height, width, channels)
            return video1.matches_shape(video2)
        else:
            raise ValueError(f"Unknown video match method: {self.method}")


@attrs.define
class FrameMatcher:
    """Matcher for comparing and matching labeled frames.

    Attributes:
        video_must_match: Whether frames must belong to the same video to be
            considered a match. Default is True.
    """

    video_must_match: bool = True

    def match(self, frame1: LabeledFrame, frame2: LabeledFrame) -> bool:
        """Check if two frames match."""
        return frame1.matches(frame2, video_must_match=self.video_must_match)


# Pre-configured matchers for common use cases
STRUCTURE_SKELETON_MATCHER = SkeletonMatcher(method=SkeletonMatchMethod.STRUCTURE)
SUBSET_SKELETON_MATCHER = SkeletonMatcher(method=SkeletonMatchMethod.SUBSET)
OVERLAP_SKELETON_MATCHER = SkeletonMatcher(
    method=SkeletonMatchMethod.OVERLAP, min_overlap=0.7
)

DUPLICATE_MATCHER = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
IOU_MATCHER = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.5)
IDENTITY_INSTANCE_MATCHER = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)

NAME_TRACK_MATCHER = TrackMatcher(method=TrackMatchMethod.NAME)
IDENTITY_TRACK_MATCHER = TrackMatcher(method=TrackMatchMethod.IDENTITY)

AUTO_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.AUTO)
SOURCE_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.BASENAME)
PATH_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
BASENAME_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.BASENAME)
IMAGE_DEDUP_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.IMAGE_DEDUP)
SHAPE_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.SHAPE)


@attrs.define
class ConflictResolution:
    """Information about a conflict that was resolved during merging.

    Attributes:
        frame: The labeled frame where the conflict occurred.
        conflict_type: Type of conflict (e.g., "duplicate_instance",
            "skeleton_mismatch").
        original_data: The original data before resolution.
        new_data: The new/incoming data that caused the conflict.
        resolution: Description of how the conflict was resolved.
    """

    frame: LabeledFrame
    conflict_type: str
    original_data: Any
    new_data: Any
    resolution: str


@attrs.define
class MergeError(Exception):
    """Base exception for merge errors.

    Attributes:
        message: Human-readable error message.
        details: Dictionary containing additional error details and context.
    """

    message: str
    details: dict = attrs.field(factory=dict)


class SkeletonMismatchError(MergeError):
    """Raised when skeletons don't match during merge."""

    pass


class VideoNotFoundError(MergeError):
    """Raised when a video file cannot be found during merge."""

    pass


@attrs.define
class MergeResult:
    """Result of a merge operation.

    Attributes:
        successful: Whether the merge completed successfully.
        frames_merged: Number of frames that were merged.
        instances_added: Number of new instances added.
        instances_updated: Number of existing instances that were updated.
        instances_skipped: Number of instances that were skipped.
        conflicts: List of conflicts that were resolved during merging.
        errors: List of errors encountered during merging.
    """

    successful: bool
    frames_merged: int = 0
    instances_added: int = 0
    instances_updated: int = 0
    instances_skipped: int = 0
    conflicts: list[ConflictResolution] = attrs.field(factory=list)
    errors: list[MergeError] = attrs.field(factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of the merge result."""
        lines = []

        if self.successful:
            lines.append("✓ Merge completed successfully")
        else:
            lines.append("✗ Merge completed with errors")

        lines.append(f"  Frames merged: {self.frames_merged}")
        lines.append(f"  Instances added: {self.instances_added}")

        if self.instances_updated:
            lines.append(f"  Instances updated: {self.instances_updated}")

        if self.instances_skipped:
            lines.append(f"  Instances skipped: {self.instances_skipped}")

        if self.conflicts:
            lines.append(f"  Conflicts resolved: {len(self.conflicts)}")

        if self.errors:
            lines.append(f"  Errors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                lines.append(f"    - {error.message}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")

        return "\n".join(lines)


class MergeProgressBar:
    """Context manager for merge progress tracking using tqdm.

    This provides a clean interface for tracking merge progress with visual feedback.

    Example:
        with MergeProgressBar("Merging predictions") as progress:
            result = labels.merge(predictions, progress_callback=progress.callback)
    """

    def __init__(self, desc: str = "Merging", leave: bool = True):
        """Initialize the progress bar.

        Args:
            desc: Description to show in the progress bar.
            leave: Whether to leave the progress bar on screen after completion.
        """
        self.desc = desc
        self.leave = leave
        self.pbar = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()

    def callback(self, current: int, total: int, message: str = ""):
        """Progress callback for merge operations.

        Args:
            current: Current progress value.
            total: Total items to process.
            message: Optional message to display.
        """
        from tqdm import tqdm

        if self.pbar is None and total:
            self.pbar = tqdm(total=total, desc=self.desc, leave=self.leave)

        if self.pbar:
            if message:
                self.pbar.set_description(f"{self.desc}: {message}")
            else:
                self.pbar.set_description(self.desc)
            self.pbar.n = current
            self.pbar.refresh()
