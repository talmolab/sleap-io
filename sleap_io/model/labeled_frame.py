"""Data structures for data contained within a single video frame.

The `LabeledFrame` class is a data structure that contains `Instance`s and
`PredictedInstance`s that are associated with a single frame within a video.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from attrs import define, field

from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.matching import InstanceMatcher


@define(eq=False)
class LabeledFrame:
    """Labeled data for a single frame of a video.

    Attributes:
        video: The `Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the `Video`.
        instances: List of `Instance` objects associated with this `LabeledFrame`.

    Notes:
        Instances of this class are hashed by identity, not by value. This means that
        two `LabeledFrame` instances with the same attributes will NOT be considered
        equal in a set or dict.
    """

    video: Video
    frame_idx: int = field(converter=int)
    instances: list[Union[Instance, PredictedInstance]] = field(factory=list)

    def __len__(self) -> int:
        """Return the number of instances in the frame."""
        return len(self.instances)

    def __getitem__(self, key: int) -> Union[Instance, PredictedInstance]:
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
        """Remove all `PredictedInstance` objects from the frame."""
        self.instances = [inst for inst in self.instances if type(inst) is Instance]

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
        instance: Optional["InstanceMatcher"] = None,
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
            return self.instances.copy(), conflicts
        elif frame == "keep_new":
            return other.instances.copy(), conflicts
        elif frame == "keep_both":
            return self.instances + other.instances, conflicts
        elif frame == "update_tracks":
            # match instances and update .track and tracking score of the old instances
            matches = instance_matcher.find_matches(self.instances, other.instances)
            for self_idx, other_idx, score in matches:
                self.instances[self_idx].track = other.instances[other_idx].track
                self.instances[self_idx].tracking_score = other.instances[
                    other_idx
                ].tracking_score
            return self.instances, conflicts
        elif frame == "replace_predictions":
            # Keep all user instances from original frame
            merged = [inst for inst in self.instances if type(inst) is Instance]
            # Add only predictions from incoming frame (not user instances)
            merged.extend(
                inst for inst in other.instances if type(inst) is PredictedInstance
            )
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

        return merged_instances, conflicts
