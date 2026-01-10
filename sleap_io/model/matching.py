"""Unified matcher system for comparing and matching data structures during merging.

This module provides configurable matchers for comparing skeletons, instances, tracks,
and videos during merge operations. The matchers use various strategies to determine
when data structures should be considered equivalent during merging.

Key features:
- Skeleton matching: exact, structure-based, overlap, and subset matching
- Instance matching: spatial proximity, track identity, and bounding box IoU
- Track matching: by name or object identity
- Video matching: path, basename, content, and auto matching

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
        AUTO: Automatic merging that preserves user labels over predictions when
            they overlap.
        KEEP_ORIGINAL: Always keep instances from the original (base) frame.
        KEEP_NEW: Always keep instances from the new (incoming) frame.
        KEEP_BOTH: Keep all instances from both frames without filtering.
        UPDATE_TRACKS: Update track assignments only without modifying poses.
        REPLACE_PREDICTIONS: Keep user instances from base, remove base predictions,
            add only predictions from incoming frame.
    """

    AUTO = "auto"
    KEEP_ORIGINAL = "keep_original"
    KEEP_NEW = "keep_new"
    KEEP_BOTH = "keep_both"
    UPDATE_TRACKS = "update_tracks"
    REPLACE_PREDICTIONS = "replace_predictions"


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


# =============================================================================
# Video Matching Helper Functions
# =============================================================================


def _get_root_video(video: Video) -> Video:
    """Traverse source_video/original_video chain to find the root video.

    Args:
        video: The video to find the root of.

    Returns:
        The root video in the provenance chain:
        - If original_video is set, returns it (original_video IS the root)
        - Otherwise, traverses source_video chain to find the root
    """
    # If original_video is explicitly set, it IS the root
    if video.original_video is not None:
        return video.original_video

    # Otherwise traverse source_video chain
    v = video
    while v.source_video is not None:
        v = v.source_video
    return v


def _is_same_file_direct(video1: Video, video2: Video) -> bool:
    """Check if two videos (without chain traversal) refer to the same file.

    This is the low-level comparison used by is_same_file() after resolving
    any provenance chains.

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        True if both videos refer to the same underlying file.
    """
    from pathlib import Path

    # Handle ImageVideo (list of filenames)
    if isinstance(video1.filename, list) and isinstance(video2.filename, list):
        # For ImageVideo, require exact same list (order matters for frame indices)
        return video1.filename == video2.filename
    elif isinstance(video1.filename, list) or isinstance(video2.filename, list):
        # One is ImageVideo, other is single file - can't be same
        return False

    # Both are single file paths
    path1 = Path(video1.filename)
    path2 = Path(video2.filename)

    # Try os.path.samefile first if both files exist (handles symlinks)
    try:
        if path1.exists() and path2.exists():
            import os

            return os.path.samefile(path1, path2)
    except (OSError, ValueError):
        # File access failed, fall through to path comparison
        pass

    # Compare resolved paths (handles relative vs absolute)
    try:
        if path1.resolve() == path2.resolve():
            return True
    except (OSError, ValueError):
        # Resolution failed, fall through to string comparison
        pass

    # Final check: exact path string match (normalized)
    return str(path1) == str(path2)


def is_same_file(video1: Video, video2: Video) -> bool:
    """Check if two videos refer to the same underlying file.

    This provides definitive file identity checking by:
    - Traversing source_video/original_video chains to find root videos
    - Using os.path.samefile() when files exist (handles symlinks)
    - Falling back to path resolution and string comparison

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        True if both videos refer to the same underlying file.

    Notes:
        This is stricter than matches_path(strict=False) - it only returns True
        when files are verifiably the same, not just same basename.
    """
    root1 = _get_root_video(video1)
    root2 = _get_root_video(video2)
    return _is_same_file_direct(root1, root2)


def _get_effective_shape(video: Video) -> tuple | None:
    """Get the effective shape for comparison purposes.

    For embedded videos with original_video, uses the original's shape
    if available. Falls back to backend_metadata for videos without
    loaded backends.

    Args:
        video: The video to get shape for.

    Returns:
        Shape tuple (frames, height, width, channels) or None if unavailable.
    """
    # For embedded videos, prefer original_video's shape
    if video.original_video is not None:
        original_shape = _get_effective_shape(video.original_video)
        if original_shape is not None:
            return original_shape

    # Try backend_metadata first (for videos with open_backend=False)
    if "shape" in video.backend_metadata:
        return video.backend_metadata["shape"]

    # Fall back to actual shape property
    return video.shape


def shapes_compatible(video1: Video, video2: Video) -> bool | None:
    """Check if two videos have compatible shapes.

    This is used for REJECTION only in video matching - incompatible shapes
    mean the videos definitely don't match. Compatible shapes (or unknown)
    do NOT imply the videos are the same.

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        False if shapes are definitely incompatible (different frames, H, or W).
        True if shapes are compatible.
        None if shape cannot be determined (missing metadata).

    Notes:
        Per algorithm design: Compare (frames, height, width) but NOT channels.
        Channels are excluded because grayscale detection is noisy (affected
        by compression) and user-configurable.
    """
    shape1 = _get_effective_shape(video1)
    shape2 = _get_effective_shape(video2)

    # If either shape is unknown, we can't determine compatibility
    if shape1 is None or shape2 is None:
        return None

    # Compare frames, height, width (indices 0, 1, 2) - NOT channels (index 3)
    return (
        shape1[0] == shape2[0]  # frames
        and shape1[1] == shape2[1]  # height
        and shape1[2] == shape2[2]  # width
    )


def original_videos_conflict(video1: Video, video2: Video) -> bool:
    """Check if two videos have conflicting original_video references.

    This is used for REJECTION in video matching. If both videos have
    provenance info but point to different root files, they definitely
    don't match (even if shapes are identical).

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        True if both have provenance AND their roots point to different files.
        False otherwise (including if either or both have no provenance).

    Example:
        Two videos with identical basenames and shapes but from different
        directories would conflict if both have original_video set to their
        respective source paths.
    """
    root1 = _get_root_video(video1)
    root2 = _get_root_video(video2)

    # Only conflict if BOTH have non-trivial chains (provenance info set)
    # AND the roots are different
    has_provenance1 = (
        video1.original_video is not None or video1.source_video is not None
    )
    has_provenance2 = (
        video2.original_video is not None or video2.source_video is not None
    )

    if not (has_provenance1 and has_provenance2):
        # At least one has no provenance - no conflict
        return False

    # Both have provenance - check if roots are the same file
    return not _is_same_file_direct(root1, root2)


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

    Notes:
        For AUTO method, use find_match() when matching against a list of
        candidates. The match() method for AUTO uses a simplified pairwise
        check that doesn't include the full leaf-uniqueness algorithm.
    """

    method: Union[VideoMatchMethod, str] = attrs.field(
        default=VideoMatchMethod.AUTO,
        converter=lambda x: VideoMatchMethod(x) if isinstance(x, str) else x,
    )
    strict: bool = False

    def match(self, video1: Video, video2: Video) -> bool:
        """Check if two videos match according to the configured method.

        For AUTO method, this performs pairwise checks (file identity, path match).
        For full AUTO matching with leaf-uniqueness, use find_match() instead.
        """
        if self.method == VideoMatchMethod.AUTO:
            # Pairwise AUTO: rejection checks + definitive identity + path match
            # (Leaf-uniqueness requires full candidate list - use find_match())

            # Rejection: incompatible shapes
            if shapes_compatible(video1, video2) is False:
                return False

            # Rejection: conflicting provenance
            if original_videos_conflict(video1, video2):
                return False

            # Definitive: same file identity
            if is_same_file(video1, video2):
                return True

            # String: strict path match
            if video1.matches_path(video2, strict=True):
                return True

            # String: basename match (for pairwise, this is the fallback)
            if video1.matches_path(video2, strict=False):
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

    def find_match(
        self,
        incoming: Video,
        candidates: list[Video],
    ) -> Video | None:
        """Find a matching video from candidates using the configured method.

        This is the preferred method for AUTO matching as it implements the
        full safe matching cascade including leaf-uniqueness disambiguation.

        Args:
            incoming: The video to find a match for.
            candidates: List of existing videos to search for matches.

        Returns:
            The matched video, or None if no match found.

        Notes:
            For AUTO method, implements the safe matching cascade:
            1. Shape rejection (filter candidates)
            2. original_video conflict rejection (filter candidates)
            3. Definitive file identity (is_same_file)
            4. Strict path match
            5. Leaf uniqueness matching at increasing depths

            Shape is for REJECTION only - compatible shapes don't imply a match.
        """
        from pathlib import Path

        from sleap_io.io.utils import sanitize_filename

        if self.method == VideoMatchMethod.AUTO:
            # Build list of viable candidates (not rejected by shape/provenance)
            viable = []
            for candidate in candidates:
                # REJECTION CHECK 1: Shape compatibility
                shape_compat = shapes_compatible(candidate, incoming)
                if shape_compat is False:
                    # Definitely incompatible shapes - skip
                    continue

                # REJECTION CHECK 2: original_video conflict
                if original_videos_conflict(candidate, incoming):
                    # Both have provenance pointing to different files - skip
                    continue

                viable.append(candidate)

            # DEFINITIVE CHECK: File identity (handles source_video chains)
            for candidate in viable:
                if is_same_file(candidate, incoming):
                    return candidate

            # STRING CHECK: Full path match
            for candidate in viable:
                if candidate.matches_path(incoming, strict=True):
                    return candidate

            # STRING CHECK: Leaf path uniqueness
            # Match paths by comparing suffixes at increasing depths
            if viable:

                def get_path_parts(video: Video) -> tuple[str, ...]:
                    """Get path parts for comparison."""
                    fn = video.filename
                    if isinstance(fn, list):
                        fn = fn[0]  # Use first for ImageVideo
                    return Path(sanitize_filename(fn)).parts

                incoming_parts = get_path_parts(incoming)
                candidate_parts = [(v, get_path_parts(v)) for v in viable]

                # Also need all existing videos for uniqueness check
                all_existing_parts = [(v, get_path_parts(v)) for v in candidates]

                # Compare at increasing depths until we find a unique match
                max_depth = max(
                    len(incoming_parts),
                    max((len(p) for _, p in all_existing_parts), default=0),
                )

                for depth in range(1, max_depth + 1):
                    if len(incoming_parts) < depth:
                        continue
                    incoming_leaf = "/".join(incoming_parts[-depth:])

                    # Find all viable candidates that match at this depth
                    matches_at_depth = []
                    for candidate, parts in candidate_parts:
                        if len(parts) < depth:
                            continue
                        candidate_leaf = "/".join(parts[-depth:])
                        if candidate_leaf == incoming_leaf:
                            matches_at_depth.append(candidate)

                    # If exactly one match at this depth, use it
                    if len(matches_at_depth) == 1:
                        return matches_at_depth[0]
                    # If no matches, try deeper
                    # If multiple matches, continue deeper to disambiguate

            # No match found
            return None

        else:
            # Non-AUTO methods: use pairwise match()
            for candidate in candidates:
                if self.match(candidate, incoming):
                    return candidate
            return None


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
