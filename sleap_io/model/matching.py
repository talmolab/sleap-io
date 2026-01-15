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
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels


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
    """Find the root video in the provenance chain.

    Args:
        video: The video to find the root of.

    Returns:
        The root video in the provenance chain. If the video has a source_video
        chain, returns the computed original_video (the root of the chain).
        Otherwise returns the video itself (it IS the root).

    Note:
        original_video is now a computed property that traverses the source_video
        chain to find the root, so this function simply uses that property.
    """
    # original_video is a computed property that traverses source_video chain
    if video.original_video is not None:
        return video.original_video

    # No chain - this video IS the root
    return video


def _file_exists(filename: str | list[str]) -> bool:
    """Check if file(s) exist on disk.

    Args:
        filename: Single file path or list of file paths.

    Returns:
        True if the file (or all files in the list) exist.
    """
    from pathlib import Path

    if isinstance(filename, list):
        return all(Path(f).exists() for f in filename)
    return Path(filename).exists()


def _get_frame_instances(
    labels: "Labels",
    video: "Video",
    include_predictions: bool,
) -> dict[int, list["Instance"]]:
    """Get frame_idx -> instances mapping for a video.

    Args:
        labels: Labels object containing the video's annotations.
        video: Video to get instances for.
        include_predictions: Whether to include PredictedInstance objects.

    Returns:
        Dictionary mapping frame indices to lists of instances.
    """
    from sleap_io.model.instance import PredictedInstance

    result: dict[int, list] = {}
    for lf in labels.labeled_frames:
        if lf.video is not video:
            continue

        instances = []
        for inst in lf.instances:
            if include_predictions or not isinstance(inst, PredictedInstance):
                instances.append(inst)

        if instances:
            result[lf.frame_idx] = instances

    return result


def _video_has_user_instances(labels: "Labels", video: "Video") -> bool:
    """Check if video has any user (non-predicted) instances.

    Args:
        labels: Labels object containing the video's annotations.
        video: Video to check for user instances.

    Returns:
        True if the video has at least one non-predicted instance.
    """
    from sleap_io.model.instance import PredictedInstance

    for lf in labels.labeled_frames:
        if lf.video is not video:
            continue
        for inst in lf.instances:
            if not isinstance(inst, PredictedInstance):
                return True
    return False


def _resolve_compare_predictions(
    compare_predictions: str | bool,
    labels: "Labels",
    video: "Video",
) -> bool:
    """Resolve 'auto' compare_predictions to boolean.

    "auto" means: include predictions only if video has NO user instances.

    Args:
        compare_predictions: The compare_predictions setting ("auto", True, or False).
        labels: Labels object containing the video's annotations.
        video: Video to check for user instances.

    Returns:
        True if predictions should be included, False otherwise.
    """
    if compare_predictions == "auto":
        return not _video_has_user_instances(labels, video)
    return bool(compare_predictions)


def _frame_has_matching_pose(
    instances_a: list["Instance"],
    instances_b: list["Instance"],
) -> bool:
    """Check if ANY pair of instances has identical poses.

    Args:
        instances_a: List of instances from the first frame.
        instances_b: List of instances from the second frame.

    Returns:
        True if at least one instance from A exactly matches at least one
        instance from B (0 coordinate difference).
    """
    for inst_a in instances_a:
        pts_a = inst_a.numpy()
        for inst_b in instances_b:
            pts_b = inst_b.numpy()
            if _poses_identical(pts_a, pts_b):
                return True
    return False


def _poses_identical(pts_a: "np.ndarray", pts_b: "np.ndarray") -> bool:
    """Check if two pose arrays are exactly identical.

    Handles NaN values by requiring same NaN pattern and exact match
    on all valid (non-NaN) coordinates.

    Args:
        pts_a: First pose array of shape (n_nodes, 2).
        pts_b: Second pose array of shape (n_nodes, 2).

    Returns:
        True if the poses are exactly identical (with NaN handling).
    """
    import numpy as np

    if pts_a.shape != pts_b.shape:
        return False

    valid_a = ~np.isnan(pts_a)
    valid_b = ~np.isnan(pts_b)

    # Must have same NaN pattern
    if not np.array_equal(valid_a, valid_b):
        return False

    # Must have at least some valid points
    if not np.any(valid_a):
        return False

    # Valid points must be exactly equal
    return np.array_equal(pts_a[valid_a], pts_b[valid_b])


def _sample_frame_indices(indices: set[int], max_samples: int) -> list[int]:
    """Sample frame indices evenly if too many, otherwise return all.

    Args:
        indices: Set of frame indices to sample from.
        max_samples: Maximum number of samples to return.

    Returns:
        List of sampled frame indices (sorted).
    """
    indices_list = sorted(indices)
    if len(indices_list) <= max_samples:
        return indices_list

    step = len(indices_list) / max_samples
    return [indices_list[int(i * step)] for i in range(max_samples)]


def _get_embedded_frame_indices(video: "Video") -> list[int] | None:
    """Get frame indices embedded in video, or None if not available.

    Args:
        video: Video to get embedded frame indices from.

    Returns:
        List of frame indices if available, None otherwise.
    """
    backend = video.backend
    if backend is None:
        return None
    if (
        hasattr(backend, "embedded_frame_inds")
        and backend.embedded_frame_inds is not None
    ):
        return list(backend.embedded_frame_inds)
    if hasattr(backend, "frame_map") and backend.frame_map:
        return list(backend.frame_map.keys())
    return None


def _get_common_embedded_indices(video1: "Video", video2: "Video") -> set[int]:
    """Get frame indices embedded in both videos.

    Args:
        video1: First video.
        video2: Second video.

    Returns:
        Set of frame indices embedded in both videos.
    """
    inds1 = _get_embedded_frame_indices(video1)
    inds2 = _get_embedded_frame_indices(video2)

    if inds1 is None or inds2 is None:
        return set()

    return set(inds1) & set(inds2)


def _frames_similar_by_image(
    video1: "Video",
    video2: "Video",
    frame_idx: int,
    threshold: float,
) -> bool:
    """Check if frames are similar by image content.

    Args:
        video1: First video.
        video2: Second video.
        frame_idx: Frame index to compare.
        threshold: Maximum mean pixel difference (0-1 scale, normalized by 255).

    Returns:
        True if images are within threshold similarity.
    """
    import numpy as np

    try:
        frame1 = video1[frame_idx]
        frame2 = video2[frame_idx]

        gray1 = _to_grayscale_float(frame1)
        gray2 = _to_grayscale_float(frame2)

        if gray1.shape != gray2.shape:
            return False

        diff = np.abs(gray1 - gray2).mean()
        return diff <= threshold

    except Exception:
        return False


def _to_grayscale_float(frame: "np.ndarray") -> "np.ndarray":
    """Convert frame to grayscale float in range [0, 1].

    Args:
        frame: Input frame array.

    Returns:
        Grayscale float array in range [0, 1].
    """
    import numpy as np

    if frame.ndim == 2:
        gray = frame.astype(np.float32)
    elif frame.ndim == 3:
        if frame.shape[2] == 1:
            gray = frame[:, :, 0].astype(np.float32)
        elif frame.shape[2] >= 3:
            gray = (
                0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            ).astype(np.float32)
        else:
            gray = frame[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    return gray / 255.0


def _is_same_file_direct(video1: Video, video2: Video) -> bool:
    """Check if two videos (without chain traversal) refer to the same file.

    This is the low-level comparison used by is_same_file() after resolving
    any provenance chains.

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        True if both videos refer to the same underlying file (and dataset for HDF5).
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

    # Check if files are the same
    files_match = False

    # Try os.path.samefile first if both files exist (handles symlinks)
    try:
        if path1.exists() and path2.exists():
            import os

            files_match = os.path.samefile(path1, path2)
    except (OSError, ValueError):
        # File access failed, fall through to path comparison
        pass

    if not files_match:
        # Compare resolved paths (handles relative vs absolute)
        try:
            if path1.resolve() == path2.resolve():
                files_match = True
        except (OSError, ValueError):
            # Resolution failed, fall through to string comparison
            pass

    if not files_match:
        # Final check: exact path string match (normalized)
        files_match = str(path1) == str(path2)

    if not files_match:
        return False

    # Files match - now check HDF5 datasets if applicable
    # For HDF5 videos within the same file, different datasets are different videos
    backend1 = video1.backend
    backend2 = video2.backend
    if backend1 is not None and backend2 is not None:
        dataset1 = getattr(backend1, "dataset", None)
        dataset2 = getattr(backend2, "dataset", None)
        if dataset1 is not None and dataset2 is not None:
            # Both have datasets - they must match
            return dataset1 == dataset2

    return True


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
    provenance info pointing to verifiably different files, they definitely
    don't match (even if shapes are identical).

    Returns True only if both videos have provenance pointing to verifiably
    different files. If files don't exist and can't be verified, returns False
    to allow fall-through to other matching strategies (pose, image).

    Args:
        video1: First video to compare.
        video2: Second video to compare.

    Returns:
        True if both have provenance AND their roots point to verifiably
        different files. False otherwise (including if either or both have
        no provenance, or if files don't exist and can't be verified).

    Example:
        Two videos with identical basenames and shapes but from different
        directories would conflict if both have original_video set to their
        respective source paths AND those files exist on disk.
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
    if _is_same_file_direct(root1, root2):
        return False  # Definitely same - no conflict

    # If neither file exists, we can't verify - don't reject
    if not _file_exists(root1.filename) and not _file_exists(root2.filename):
        return False  # Can't verify, allow fall-through to other matching

    # At least one file exists and they don't match - conflict
    return True


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

    method: SkeletonMatchMethod | str = attrs.field(
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

    method: InstanceMatchMethod | str = attrs.field(
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

    method: TrackMatchMethod | str = attrs.field(
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
        content_frames: Minimum number of matching frames required for pose/image
            matching to confirm a match. If fewer common frames exist, requires
            all of them to match. Default 3.
        compare_predictions: Whether to include predicted instances in pose matching.
            "auto" (default): Include only if video has 100% predictions (no user
            instances). True: Always include predictions. False: Never include
            predictions (user instances only).
        compare_images: Whether to compare frame images via pixel similarity.
            Expensive operation requiring frame decoding. Default False.
        image_similarity_threshold: Maximum mean pixel difference (0-1 scale,
            normalized by 255) for images to be considered matching.
            Only used when compare_images=True. Default 0.05 (~13/255 pixels).

    Notes:
        For AUTO method, use find_match() when matching against a list of
        candidates. The match() method for AUTO uses a simplified pairwise
        check that doesn't include the full leaf-uniqueness algorithm.
    """

    method: VideoMatchMethod | str = attrs.field(
        default=VideoMatchMethod.AUTO,
        converter=lambda x: VideoMatchMethod(x) if isinstance(x, str) else x,
    )
    strict: bool = False
    content_frames: int = 3
    compare_predictions: str | bool = "auto"
    compare_images: bool = False
    image_similarity_threshold: float = 0.05
    _frame_cache: dict = attrs.field(factory=dict, init=False, repr=False)

    def _get_cached_frame_instances(
        self,
        labels: "Labels",
        video: "Video",
        include_predictions: bool,
    ) -> dict[int, list["Instance"]]:
        """Get frame instances with caching for performance.

        Caches the result to avoid recomputing for the same video multiple times
        during merge operations.
        """
        cache_key = (id(labels), id(video), include_predictions)
        if cache_key not in self._frame_cache:
            self._frame_cache[cache_key] = _get_frame_instances(
                labels, video, include_predictions
            )
        return self._frame_cache[cache_key]

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
        labels_incoming: "Labels | None" = None,
        labels_base: "Labels | None" = None,
    ) -> Video | None:
        """Find a matching video from candidates using the configured method.

        This is the preferred method for AUTO matching as it implements the
        full safe matching cascade including leaf-uniqueness disambiguation.

        Args:
            incoming: The video to find a match for.
            candidates: List of existing videos to search for matches.
            labels_incoming: Labels object containing the incoming video's
                annotations. Used for pose-based matching in AUTO mode.
            labels_base: Labels object containing the candidates' annotations.
                Used for pose-based matching in AUTO mode.

        Returns:
            The matched video, or None if no match found.

        Notes:
            For AUTO method, implements the safe matching cascade:
            1. Shape rejection (filter candidates)
            2. original_video conflict rejection (filter candidates)
            3. Definitive file identity (is_same_file)
            4. Strict path match
            5. Leaf uniqueness matching at increasing depths
            6. Pose-based matching (if labels provided)
            7. Image-based matching (if compare_images=True)

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
                    """Get path parts for comparison, using root video for embedded."""
                    root = _get_root_video(video)
                    fn = root.filename
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

            # POSE MATCHING: Compare pose annotations (default in AUTO)
            if labels_incoming is not None and labels_base is not None:
                match = self._match_by_poses(
                    incoming, viable, labels_incoming, labels_base
                )
                if match is not None:
                    return match

            # IMAGE MATCHING: Compare frame images (opt-in)
            if self.compare_images:
                match = self._match_by_images(incoming, viable)
                if match is not None:
                    return match

            # No match found
            return None

        else:
            # Non-AUTO methods: use pairwise match()
            for candidate in candidates:
                if self.match(candidate, incoming):
                    return candidate
            return None

    def _match_by_poses(
        self,
        incoming: "Video",
        candidates: list["Video"],
        labels_incoming: "Labels",
        labels_base: "Labels",
    ) -> "Video | None":
        """Try to match video by comparing pose annotations.

        Returns matched video if poses match on enough common frames.
        """
        # Resolve whether to include predictions for incoming video
        include_preds = _resolve_compare_predictions(
            self.compare_predictions, labels_incoming, incoming
        )

        # Get incoming video's frame -> instances map (cached)
        incoming_frames = self._get_cached_frame_instances(
            labels_incoming, incoming, include_preds
        )
        if not incoming_frames:
            return None  # No annotations to compare

        for candidate in candidates:
            # Get candidate's frame -> instances map (cached for performance)
            # Use same prediction setting resolved for candidate
            include_preds_cand = _resolve_compare_predictions(
                self.compare_predictions, labels_base, candidate
            )
            candidate_frames = self._get_cached_frame_instances(
                labels_base, candidate, include_preds_cand
            )
            if not candidate_frames:
                continue

            # Find common frame indices
            common_indices = set(incoming_frames.keys()) & set(candidate_frames.keys())
            if not common_indices:
                continue

            # Determine required matches
            required_matches = min(self.content_frames, len(common_indices))

            # Sample frames if too many (performance)
            sample_indices = _sample_frame_indices(
                common_indices, max_samples=self.content_frames * 2
            )

            # Count matching frames
            matching_frames = 0
            for frame_idx in sample_indices:
                if _frame_has_matching_pose(
                    incoming_frames[frame_idx], candidate_frames[frame_idx]
                ):
                    matching_frames += 1
                    if matching_frames >= required_matches:
                        return candidate  # Found match!

        return None

    def _match_by_images(
        self,
        incoming: "Video",
        candidates: list["Video"],
    ) -> "Video | None":
        """Try to match video by comparing image content.

        Only used when compare_images=True. Expensive operation.
        Returns matched video if images match on enough common frames.
        """
        for candidate in candidates:
            # Get common embedded frame indices
            common_indices = _get_common_embedded_indices(incoming, candidate)
            if not common_indices:
                continue

            required_matches = min(self.content_frames, len(common_indices))

            # Sample frames
            sample_indices = _sample_frame_indices(
                common_indices, max_samples=self.content_frames * 2
            )

            # Count matching frames
            matching_frames = 0
            for frame_idx in sample_indices:
                if _frames_similar_by_image(
                    incoming, candidate, frame_idx, self.image_similarity_threshold
                ):
                    matching_frames += 1
                    if matching_frames >= required_matches:
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
