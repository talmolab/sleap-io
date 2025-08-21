"""Unified matcher system for comparing and matching data structures during merging.

This module provides configurable matchers for comparing skeletons, instances, tracks,
videos, and frames during merge operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

import attrs
import numpy as np

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


class SkeletonMatchMethod(str, Enum):
    """Methods for matching skeletons."""

    EXACT = "exact"  # Exact match including node order
    STRUCTURE = "structure"  # Same nodes and edges, order doesn't matter
    OVERLAP = "overlap"  # Partial overlap of nodes
    SUBSET = "subset"  # One skeleton is subset of another


class InstanceMatchMethod(str, Enum):
    """Methods for matching instances."""

    SPATIAL = "spatial"  # Match by spatial proximity
    IDENTITY = "identity"  # Match by track identity
    IOU = "iou"  # Match by bounding box IoU


class TrackMatchMethod(str, Enum):
    """Methods for matching tracks."""

    NAME = "name"  # Match by track name
    IDENTITY = "identity"  # Match by object identity


class VideoMatchMethod(str, Enum):
    """Methods for matching videos."""

    PATH = "path"  # Match by file path
    BASENAME = "basename"  # Match by filename only
    CONTENT = "content"  # Match by video content (shape, backend)
    RESOLVE = "resolve"  # Try to resolve paths
    AUTO = "auto"  # Automatically determine best method


class FrameStrategy(str, Enum):
    """Strategies for handling frame merging."""

    SMART = "smart"  # Smart merging based on instance types
    KEEP_ORIGINAL = "keep_original"  # Keep original frame instances
    KEEP_NEW = "keep_new"  # Keep new frame instances
    KEEP_BOTH = "keep_both"  # Keep all instances from both frames


class ErrorMode(str, Enum):
    """Error handling modes for merge operations."""

    CONTINUE = "continue"  # Continue on errors, log them
    STRICT = "strict"  # Raise exception on first error
    WARN = "warn"  # Warn about errors but continue


@attrs.define
class SkeletonMatcher:
    """Matcher for comparing and matching skeletons."""

    method: Union[SkeletonMatchMethod, str] = attrs.field(
        default=SkeletonMatchMethod.STRUCTURE,
        converter=lambda x: SkeletonMatchMethod(x) if isinstance(x, str) else x,
    )
    require_same_order: bool = False
    min_overlap: float = 0.5  # For OVERLAP method

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
    """Matcher for comparing and matching instances."""

    method: Union[InstanceMatchMethod, str] = attrs.field(
        default=InstanceMatchMethod.SPATIAL,
        converter=lambda x: InstanceMatchMethod(x) if isinstance(x, str) else x,
    )
    threshold: float = 5.0  # Distance threshold for SPATIAL, IoU threshold for IOU

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
    """Matcher for comparing and matching tracks."""

    method: Union[TrackMatchMethod, str] = attrs.field(
        default=TrackMatchMethod.NAME,
        converter=lambda x: TrackMatchMethod(x) if isinstance(x, str) else x,
    )

    def match(self, track1: Track, track2: Track) -> bool:
        """Check if two tracks match according to the configured method."""
        return track1.matches(track2, method=self.method.value)


@attrs.define
class VideoMatcher:
    """Matcher for comparing and matching videos."""

    method: Union[VideoMatchMethod, str] = attrs.field(
        default=VideoMatchMethod.AUTO,
        converter=lambda x: VideoMatchMethod(x) if isinstance(x, str) else x,
    )
    strict: bool = False  # For PATH method
    base_path: Optional[str] = None  # For RESOLVE method
    fallback_directories: list[str] = attrs.field(factory=list)  # For RESOLVE

    def match(self, video1: Video, video2: Video) -> bool:
        """Check if two videos match according to the configured method."""
        if self.method == VideoMatchMethod.AUTO:
            # Try different methods in order
            if video1 is video2:
                return True
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
        elif self.method == VideoMatchMethod.RESOLVE:
            # Try to resolve paths
            if video1.matches_path(video2, strict=False):
                return True
            # TODO: Implement path resolution logic
            return False
        else:
            raise ValueError(f"Unknown video match method: {self.method}")


@attrs.define
class FrameMatcher:
    """Matcher for comparing and matching labeled frames."""

    video_must_match: bool = True

    def match(self, frame1: LabeledFrame, frame2: LabeledFrame) -> bool:
        """Check if two frames match."""
        return frame1.matches(frame2, video_must_match=self.video_must_match)


# Pre-configured matchers for common use cases
STRUCTURE_SKELETON_MATCHER = SkeletonMatcher(method=SkeletonMatchMethod.STRUCTURE)
SUBSET_SKELETON_MATCHER = SkeletonMatcher(method=SkeletonMatchMethod.SUBSET)
DUPLICATE_MATCHER = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
IOU_MATCHER = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.5)
NAME_TRACK_MATCHER = TrackMatcher(method=TrackMatchMethod.NAME)
IDENTITY_TRACK_MATCHER = TrackMatcher(method=TrackMatchMethod.IDENTITY)
AUTO_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.AUTO)
SOURCE_VIDEO_MATCHER = VideoMatcher(method=VideoMatchMethod.RESOLVE)


@attrs.define
class ConflictResolution:
    """Information about a conflict that was resolved during merging."""

    frame: LabeledFrame
    conflict_type: str  # e.g., "duplicate_instance", "skeleton_mismatch"
    original_data: Any
    new_data: Any
    resolution: str  # How it was resolved


@attrs.define
class MergeError(Exception):
    """Base exception for merge errors."""

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
    """Result of a merge operation."""

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
