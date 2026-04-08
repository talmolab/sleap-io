"""Data structures for data contained within a single video frame.

The `LabeledFrame` class is a data structure that contains `Instance`s and
`PredictedInstance`s that are associated with a single frame within a video.
"""

from __future__ import annotations

import math
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from attrs import define, field

from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.centroid import Centroid
    from sleap_io.model.label_image import LabelImage
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.matching import InstanceMatcher
    from sleap_io.model.roi import ROI


def _annotation_centroid_xy(annotation: Any, attr: str) -> tuple[float, float] | None:
    """Extract centroid (x, y) from an annotation based on its modality.

    Args:
        annotation: An annotation object (Centroid, BoundingBox, etc.).
        attr: The attribute name indicating the modality.

    Returns:
        A tuple of (x, y) coordinates, or ``None`` if the centroid cannot be
        computed (e.g., empty mask or empty ROI geometry).
    """
    if attr == "centroids":
        return (annotation.x, annotation.y)
    elif attr == "bboxes":
        return annotation.centroid_xy
    elif attr == "rois":
        if annotation.geometry.is_empty:
            return None
        return annotation.centroid_xy
    elif attr == "masks":
        x, y, w, h = annotation.bbox
        if w == 0 and h == 0:
            return None
        return (x + w / 2, y + h / 2)
    elif attr == "label_images":
        sx, sy = annotation.scale
        ox, oy = annotation.offset
        return (
            (annotation.width / 2) / sx + ox,
            (annotation.height / 2) / sy + oy,
        )
    return None


def _find_annotation_matches(
    self_list: list,
    other_list: list,
    attr: str,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Find matching annotations between two lists by centroid distance.

    Args:
        self_list: Annotations from the self frame.
        other_list: Annotations from the other frame.
        attr: The attribute name indicating the modality.
        threshold: Maximum centroid distance for a match (pixels).

    Returns:
        List of ``(self_idx, other_idx, score)`` tuples where
        ``score = 1 / (1 + distance)``.
    """
    # NOTE: O(n*m) brute-force without bipartite assignment. Callers are
    # responsible for resolving many-to-one conflicts (e.g., greedy 1:1 in
    # _resolve_annotation_auto). Fine for typical annotation counts per frame.
    matches = []
    for i, a in enumerate(self_list):
        c1 = _annotation_centroid_xy(a, attr)
        if c1 is None:
            continue
        for j, b in enumerate(other_list):
            c2 = _annotation_centroid_xy(b, attr)
            if c2 is None:
                continue
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            if dist <= threshold:
                matches.append((i, j, 1.0 / (1.0 + dist)))
    return matches


def _resolve_annotation_auto(
    self_list: list,
    other_list: list,
    attr: str,
    threshold: float,
) -> list:
    """Apply auto merge resolution to a list of annotations.

    Mirrors the instance auto-merge cascade: keep user from self, spatially
    match, apply user-vs-predicted resolution rules, add unmatched from other,
    keep unmatched predictions from self.

    Args:
        self_list: Annotations from the self frame.
        other_list: Annotations from the other frame.
        attr: The attribute name indicating the modality.
        threshold: Maximum centroid distance for a match (pixels).

    Returns:
        Merged list of annotations.
    """
    merged = []
    used_self_indices: set[int] = set()

    # 1. Keep all user annotations from self
    for ann in self_list:
        if not ann.is_predicted:
            merged.append(ann)

    # 2. Find spatial matches
    matches = _find_annotation_matches(self_list, other_list, attr, threshold)

    # 3. Greedy one-to-one matching: sort by score descending, assign each
    # self/other index at most once so no annotation is silently dropped.
    matches.sort(key=lambda m: m[2], reverse=True)
    matched_self: set[int] = set()
    matched_other: set[int] = set()
    other_to_self: dict[int, int] = {}
    for self_idx, other_idx, _score in matches:
        if self_idx not in matched_self and other_idx not in matched_other:
            other_to_self[other_idx] = self_idx
            matched_self.add(self_idx)
            matched_other.add(other_idx)

    # 4. Process each annotation from other
    for other_idx, other_ann in enumerate(other_list):
        if other_idx in other_to_self:
            self_idx = other_to_self[other_idx]
            self_ann = self_list[self_idx]
            used_self_indices.add(self_idx)

            if not self_ann.is_predicted and not other_ann.is_predicted:
                # user + user → keep self (already in merged)
                pass
            elif self_ann.is_predicted and not other_ann.is_predicted:
                # predicted + user → replace with other's user
                merged.append(copy(other_ann))
            elif not self_ann.is_predicted and other_ann.is_predicted:
                # user + predicted → keep self (already in merged)
                pass
            else:
                # predicted + predicted → keep other's (newer)
                merged.append(copy(other_ann))
        else:
            # No match → add from other
            merged.append(copy(other_ann))

    # 5. Keep unmatched predictions from self
    for self_idx, self_ann in enumerate(self_list):
        if self_ann.is_predicted and self_idx not in used_self_indices:
            merged.append(self_ann)

    return merged


def _resolve_annotation_update_tracks(
    self_list: list,
    other_list: list,
    attr: str,
    threshold: float,
) -> None:
    """Update track assignments on self's annotations from spatially matched other's.

    Args:
        self_list: Annotations from the self frame (modified in place).
        other_list: Annotations from the other frame.
        attr: The attribute name indicating the modality.
        threshold: Maximum centroid distance for a match (pixels).
    """
    if attr == "label_images":
        # LabelImage tracks are per-object in .objects dict, not top-level.
        return

    matches = _find_annotation_matches(self_list, other_list, attr, threshold)

    # Best match per self_idx
    self_to_other: dict[int, tuple[int, float]] = {}
    for self_idx, other_idx, score in matches:
        if self_idx not in self_to_other or score > self_to_other[self_idx][1]:
            self_to_other[self_idx] = (other_idx, score)

    for self_idx, (other_idx, _score) in self_to_other.items():
        self_list[self_idx].track = other_list[other_idx].track
        self_list[self_idx].tracking_score = other_list[other_idx].tracking_score


@define(eq=False)
class LabeledFrame:
    """Labeled data for a single frame of a video.

    Attributes:
        video: The `Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the `Video`.
        instances: List of `Instance` objects associated with this `LabeledFrame`.
        is_negative: If True, this frame is explicitly marked as containing no
            instances (a "negative" or background frame for training). This is
            distinct from frames that are simply empty (e.g., instances were deleted).
        centroids: List of `Centroid` annotations for this frame.
        bboxes: List of `BoundingBox` annotations for this frame.
        masks: List of `SegmentationMask` annotations for this frame.
        label_images: List of `LabelImage` annotations for this frame.
        rois: List of `ROI` annotations for this frame.

    Notes:
        Instances of this class are hashed by identity, not by value. This means that
        two `LabeledFrame` instances with the same attributes will NOT be considered
        equal in a set or dict.
    """

    video: Video
    frame_idx: int = field(converter=int)
    instances: list[Instance | PredictedInstance] = field(factory=list)
    is_negative: bool = field(default=False)
    centroids: "list[Centroid]" = field(factory=list)
    bboxes: "list[BoundingBox]" = field(factory=list)
    masks: "list[SegmentationMask]" = field(factory=list)
    label_images: "list[LabelImage]" = field(factory=list)
    rois: "list[ROI]" = field(factory=list)

    def __len__(self) -> int:
        """Return the number of instances in the frame."""
        return len(self.instances)

    def __getitem__(self, key: int) -> Instance | PredictedInstance:
        """Return the `Instance` at `key` index in the `instances` list."""
        return self.instances[key]

    def __iter__(self):
        """Iterate over `Instance`s in `instances` list."""
        return iter(self.instances)

    @property
    def user_instances(self) -> list[Instance]:
        """Frame instances that are user-labeled (`Instance` objects)."""
        return [inst for inst in self.instances if type(inst) is Instance]

    @property
    def has_user_instances(self) -> bool:
        """Return True if the frame has any user-labeled instances."""
        for inst in self.instances:
            if type(inst) is Instance:
                return True
        return False

    @property
    def is_user_labeled(self) -> bool:
        """Return True if frame has user instances/annotations OR is negative.

        This property indicates whether the frame represents intentional user
        annotation, either through labeled instances, user annotations
        (centroids, bboxes, masks, label images), or explicit marking as a
        negative/background frame.
        """
        from sleap_io.model.label_image import PredictedLabelImage
        from sleap_io.model.mask import PredictedSegmentationMask

        return (
            self.has_user_instances
            or self.is_negative
            or any(not c.is_predicted for c in self.centroids)
            or any(not b.is_predicted for b in self.bboxes)
            or any(not isinstance(m, PredictedSegmentationMask) for m in self.masks)
            or any(not isinstance(li, PredictedLabelImage) for li in self.label_images)
        )

    @property
    def predicted_instances(self) -> list[Instance]:
        """Frame instances that are predicted by a model (`PredictedInstance`)."""
        return [inst for inst in self.instances if type(inst) is PredictedInstance]

    @property
    def has_predicted_instances(self) -> bool:
        """Return True if the frame has any predicted instances."""
        for inst in self.instances:
            if type(inst) is PredictedInstance:
                return True
        return False

    def numpy(self) -> np.ndarray:
        """Return all instances in the frame as a numpy array.

        Returns:
            Points as a numpy array of shape `(n_instances, n_nodes, 2)`.

            Note that the order of the instances is arbitrary.
        """
        n_instances = len(self.instances)
        n_nodes = len(self.instances[0]) if n_instances > 0 else 0
        pts = np.full((n_instances, n_nodes, 2), np.nan)
        for i, inst in enumerate(self.instances):
            pts[i] = inst.numpy()[:, 0:2]
        return pts

    @property
    def image(self) -> np.ndarray:
        """Return the image of the frame as a numpy array."""
        return self.video[self.frame_idx]

    @property
    def unused_predictions(self) -> list[Instance]:
        """Return a list of "unused" `PredictedInstance` objects in frame.

        This is all of the `PredictedInstance` objects which do not have a corresponding
        `Instance` in the same track in the same frame.
        """
        unused_predictions = []
        any_tracks = [inst.track for inst in self.instances if inst.track is not None]
        if len(any_tracks):
            # Use tracks to determine which predicted instances have been used
            used_tracks = [
                inst.track
                for inst in self.instances
                if type(inst) is Instance and inst.track is not None
            ]
            unused_predictions = [
                inst
                for inst in self.instances
                if inst.track not in used_tracks and type(inst) is PredictedInstance
            ]

        else:
            # Use from_predicted to determine which predicted instances have been used
            # TODO: should we always do this instead of using tracks?
            used_instances = [
                inst.from_predicted
                for inst in self.instances
                if inst.from_predicted is not None
            ]
            unused_predictions = [
                inst
                for inst in self.instances
                if type(inst) is PredictedInstance and inst not in used_instances
            ]

        return unused_predictions

    def remove_predictions(self):
        """Remove all predicted instances and annotations from the frame."""
        from sleap_io.model.bbox import PredictedBoundingBox
        from sleap_io.model.centroid import PredictedCentroid
        from sleap_io.model.label_image import PredictedLabelImage
        from sleap_io.model.mask import PredictedSegmentationMask
        from sleap_io.model.roi import PredictedROI

        self.instances = [inst for inst in self.instances if type(inst) is Instance]
        self.centroids = [
            c for c in self.centroids if not isinstance(c, PredictedCentroid)
        ]
        self.bboxes = [
            b for b in self.bboxes if not isinstance(b, PredictedBoundingBox)
        ]
        self.masks = [
            m for m in self.masks if not isinstance(m, PredictedSegmentationMask)
        ]
        self.label_images = [
            li for li in self.label_images if not isinstance(li, PredictedLabelImage)
        ]
        self.rois = [r for r in self.rois if not isinstance(r, PredictedROI)]

    def remove_empty_instances(self):
        """Remove all instances with no visible points."""
        self.instances = [inst for inst in self.instances if not inst.is_empty]

    def matches(self, other: "LabeledFrame", video_must_match: bool = True) -> bool:
        """Check if this frame matches another frame's identity.

        Args:
            other: Another LabeledFrame to compare with.
            video_must_match: If True, frames must be from the same video.
                If False, only frame index needs to match.

        Returns:
            True if the frames have the same identity, False otherwise.

        Notes:
            Frame identity is determined by video and frame index.
            This does not compare the instances within the frame.
        """
        if self.frame_idx != other.frame_idx:
            return False

        if video_must_match:
            # Check if videos are the same object
            if self.video is other.video:
                return True
            # Check if videos have matching paths
            return self.video.matches_path(other.video, strict=False)

        return True

    def similarity_to(self, other: "LabeledFrame") -> dict[str, any]:
        """Calculate instance overlap metrics with another frame.

        Args:
            other: Another LabeledFrame to compare with.

        Returns:
            A dictionary with similarity metrics:
            - 'n_user_self': Number of user instances in this frame
            - 'n_user_other': Number of user instances in the other frame
            - 'n_pred_self': Number of predicted instances in this frame
            - 'n_pred_other': Number of predicted instances in the other frame
            - 'n_overlapping': Number of instances that overlap (by IoU)
            - 'mean_pose_distance': Mean distance between matching poses
        """
        metrics = {
            "n_user_self": len(self.user_instances),
            "n_user_other": len(other.user_instances),
            "n_pred_self": len(self.predicted_instances),
            "n_pred_other": len(other.predicted_instances),
            "n_overlapping": 0,
            "mean_pose_distance": None,
        }

        # Count overlapping instances and compute pose distances
        pose_distances = []
        for inst1 in self.instances:
            for inst2 in other.instances:
                # Check if instances overlap
                if inst1.overlaps_with(inst2, iou_threshold=0.1):
                    metrics["n_overlapping"] += 1

                    # If they have the same skeleton, compute pose distance
                    if inst1.skeleton.matches(inst2.skeleton):
                        # Get visible points for both
                        pts1 = inst1.numpy()
                        pts2 = inst2.numpy()

                        # Compute distances for visible points in both
                        valid = ~(np.isnan(pts1[:, 0]) | np.isnan(pts2[:, 0]))
                        if valid.any():
                            distances = np.linalg.norm(
                                pts1[valid] - pts2[valid], axis=1
                            )
                            pose_distances.extend(distances.tolist())

        if pose_distances:
            metrics["mean_pose_distance"] = np.mean(pose_distances)

        return metrics

    def merge(
        self,
        other: "LabeledFrame",
        instance: "InstanceMatcher | None" = None,
        frame: str = "auto",
    ) -> tuple[list[Instance], list[tuple[Instance, Instance, str]]]:
        """Merge instances from another frame into this frame.

        Args:
            other: Another LabeledFrame to merge instances from.
            instance: Matcher to use for finding duplicate instances.
                If None, uses default spatial matching with 5px tolerance.
            frame: Merge strategy:
                - "auto": Keep user labels, update predictions only if no user label
                - "keep_original": Keep all original instances, ignore new ones
                - "keep_new": Replace with new instances
                - "keep_both": Keep all instances from both frames
                - "update_tracks": Update track and score of the original instances
                    from the new instances.
                - "replace_predictions": Keep all user instances from original frame,
                    remove all predictions from original frame, add only predictions
                    from the incoming frame. No spatial matching is performed.

        Returns:
            A tuple of (merged_instances, conflicts) where:
            - merged_instances: List of instances after merging
            - conflicts: List of (original, new, resolution) tuples for conflicts

        Notes:
            This method doesn't modify the frame in place. It returns the merged
            instance list which can be assigned back if desired.
        """
        from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

        if instance is None:
            instance_matcher = InstanceMatcher(
                method=InstanceMatchMethod.SPATIAL, threshold=5.0
            )
        else:
            instance_matcher = instance

        conflicts = []

        if frame == "keep_original":
            self._merge_annotations(other, strategy="keep_original")
            return self.instances.copy(), conflicts
        elif frame == "keep_new":
            self._merge_annotations(other, strategy="keep_new")
            return other.instances.copy(), conflicts
        elif frame == "keep_both":
            self._merge_annotations(other, strategy="keep_both")
            return self.instances + other.instances, conflicts
        elif frame == "update_tracks":
            # match instances and update .track and tracking score of the old instances
            matches = instance_matcher.find_matches(self.instances, other.instances)
            for self_idx, other_idx, score in matches:
                self.instances[self_idx].track = other.instances[other_idx].track
                self.instances[self_idx].tracking_score = other.instances[
                    other_idx
                ].tracking_score
            self._merge_annotations(
                other,
                strategy="update_tracks",
                threshold=instance_matcher.threshold,
            )
            return self.instances, conflicts
        elif frame == "replace_predictions":
            # Keep all user instances from original frame
            merged = [inst for inst in self.instances if type(inst) is Instance]
            # Add only predictions from incoming frame (not user instances)
            merged.extend(
                inst for inst in other.instances if type(inst) is PredictedInstance
            )
            self._merge_annotations(other, strategy="replace_predictions")
            # No conflicts to report - this is a clean replacement
            return merged, []

        # Auto merging strategy
        merged_instances = []
        used_indices = set()

        # First, keep all user instances from self
        for inst in self.instances:
            if type(inst) is Instance:
                merged_instances.append(inst)

        # Find matches between instances
        matches = instance_matcher.find_matches(self.instances, other.instances)

        # Group matches by instance in other frame
        other_to_self = {}
        for self_idx, other_idx, score in matches:
            if other_idx not in other_to_self or score > other_to_self[other_idx][1]:
                other_to_self[other_idx] = (self_idx, score)

        # Process instances from other frame
        for other_idx, other_inst in enumerate(other.instances):
            if other_idx in other_to_self:
                self_idx, score = other_to_self[other_idx]
                self_inst = self.instances[self_idx]

                # Check for conflicts
                if type(self_inst) is Instance and type(other_inst) is Instance:
                    # Both are user instances - conflict
                    conflicts.append((self_inst, other_inst, "kept_original"))
                    used_indices.add(self_idx)
                elif (
                    type(self_inst) is PredictedInstance
                    and type(other_inst) is Instance
                ):
                    # Replace prediction with user instance
                    if self_idx not in used_indices:
                        merged_instances.append(other_inst)
                        used_indices.add(self_idx)
                elif (
                    type(self_inst) is Instance
                    and type(other_inst) is PredictedInstance
                ):
                    # Keep user instance, ignore prediction
                    conflicts.append((self_inst, other_inst, "kept_user"))
                    used_indices.add(self_idx)
                else:
                    # Both are predictions - keep the new one
                    if self_idx not in used_indices:
                        merged_instances.append(other_inst)
                        used_indices.add(self_idx)
            else:
                # No match found, add new instance
                merged_instances.append(other_inst)

        # Add remaining instances from self that weren't matched
        for self_idx, self_inst in enumerate(self.instances):
            if type(self_inst) is PredictedInstance and self_idx not in used_indices:
                # Check if this prediction should be kept
                # NOTE: This defensive logic should be unreachable under normal
                # circumstances since all matched instances should have been added to
                # used_indices above. However, we keep this as a safety net for edge
                # cases or future changes.
                keep = True
                for other_idx, (matched_self_idx, _) in other_to_self.items():
                    if matched_self_idx == self_idx:
                        keep = False
                        break
                if keep:
                    merged_instances.append(self_inst)

        # Merge annotations from the other frame (spatial matching + resolution)
        self._merge_annotations(
            other, strategy="auto", threshold=instance_matcher.threshold
        )

        return merged_instances, conflicts

    def _merge_annotations(
        self,
        other: "LabeledFrame",
        strategy: str = "keep_both",
        threshold: float = 5.0,
    ):
        """Merge annotation lists from another frame into this frame.

        Shallow-copies annotations from the other frame to avoid mutating the
        source when references are later remapped. Video and track references
        are preserved so that ``_remap_frame_annotations`` can find them in
        the mapping dicts.

        Args:
            other: The frame to merge annotations from.
            strategy: The merge strategy, matching the ``frame`` parameter of
                ``merge()``. Controls which annotations are kept:

                - ``"keep_original"``: Keep self only.
                - ``"keep_new"``: Replace with other's annotations.
                - ``"keep_both"``: Keep self + add other's (default).
                - ``"replace_predictions"``: Keep user from self, replace
                  predicted with other's predicted.
                - ``"auto"``: Spatial matching + user-vs-predicted resolution
                  cascade (mirrors instance auto-merge logic).
                - ``"update_tracks"``: Spatial matching, then update track
                  assignments on matched self annotations.
            threshold: Maximum centroid distance (pixels) for spatial matching
                in ``"auto"`` and ``"update_tracks"`` strategies.
        """
        attrs = ("centroids", "bboxes", "masks", "label_images", "rois")

        if strategy == "keep_original":
            return

        if strategy == "keep_new":
            for attr in attrs:
                setattr(self, attr, [copy(item) for item in getattr(other, attr)])
            return

        if strategy == "replace_predictions":
            for attr in attrs:
                kept = [a for a in getattr(self, attr) if not a.is_predicted]
                for item in getattr(other, attr):
                    if item.is_predicted:
                        kept.append(copy(item))
                setattr(self, attr, kept)
            return

        if strategy == "auto":
            for attr in attrs:
                setattr(
                    self,
                    attr,
                    _resolve_annotation_auto(
                        getattr(self, attr), getattr(other, attr), attr, threshold
                    ),
                )
            return

        if strategy == "update_tracks":
            for attr in attrs:
                _resolve_annotation_update_tracks(
                    getattr(self, attr), getattr(other, attr), attr, threshold
                )
            return

        # "keep_both" (default)
        for attr in attrs:
            existing_ids = set(id(x) for x in getattr(self, attr))
            for item in getattr(other, attr):
                if id(item) not in existing_ids:
                    getattr(self, attr).append(copy(item))
