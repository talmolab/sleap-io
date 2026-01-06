"""Lazy loading support for SLP files.

This module provides LazyDataStore and LazyFrameList classes that enable
deferred materialization of LabeledFrame and Instance objects when loading
SLP files with lazy=True.

These classes are implementation details - users interact with Labels objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional, Union

import attrs
import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, PredictedInstance, Track
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Skeleton
    from sleap_io.model.video import Video

from sleap_io.io.slp import InstanceType


@attrs.define
class LazyDataStore:
    """Holds raw HDF5 data and provides lazy access methods.

    Attributes:
        frames_data: Structured array from /frames HDF5 dataset.
            Fields: frame_id, video_id, frame_idx, instance_id_start, instance_id_end.
        instances_data: Structured array from /instances HDF5 dataset.
            Fields vary by format_id but include: instance_id, instance_type, frame_id,
            skeleton_id, track_id, from_predicted, instance_score, point_id_start,
            point_id_end, and optionally tracking_score.
        pred_points_data: Structured array from /pred_points HDF5 dataset.
            Fields: x, y, visible, complete, score.
        points_data: Structured array from /points HDF5 dataset.
            Fields: x, y, visible, complete.
        videos: List of eagerly loaded Video objects.
        skeletons: List of eagerly loaded Skeleton objects.
        tracks: List of eagerly loaded Track objects.
        format_id: SLP format version.
        _source_path: Path to source SLP file (for debugging).
    """

    # Raw arrays
    frames_data: np.ndarray
    instances_data: np.ndarray
    pred_points_data: np.ndarray
    points_data: np.ndarray

    # References
    videos: list["Video"]
    skeletons: list["Skeleton"]
    tracks: list["Track"]

    # Metadata
    format_id: float
    _source_path: Optional[str] = attrs.field(default=None, alias="source_path")

    def __attrs_post_init__(self) -> None:
        """Validate index bounds on construction."""
        self.validate()

    def validate(self) -> None:
        """Check that all indices are within bounds.

        Raises:
            ValueError: If any index is out of bounds.
        """
        n_frames = len(self.frames_data)
        n_instances = len(self.instances_data)
        n_points = len(self.points_data)
        n_pred_points = len(self.pred_points_data)

        if n_frames == 0:
            return  # Empty data is valid

        # Validate frame -> instance references
        max_inst_end = self.frames_data["instance_id_end"].max() if n_frames > 0 else 0
        if max_inst_end > n_instances:
            raise ValueError(
                f"Frame references instance index {max_inst_end} but only "
                f"{n_instances} instances exist."
            )

        if n_instances == 0:
            return  # No instances means no points to validate

        # Validate instance -> point references
        # Separate user instances and predicted instances
        user_mask = self.instances_data["instance_type"] == InstanceType.USER
        pred_mask = self.instances_data["instance_type"] == InstanceType.PREDICTED

        if np.any(user_mask):
            user_max_end = self.instances_data[user_mask]["point_id_end"].max()
            if user_max_end > n_points:
                raise ValueError(
                    f"User instance references point index {user_max_end} but only "
                    f"{n_points} points exist."
                )

        if np.any(pred_mask):
            pred_max_end = self.instances_data[pred_mask]["point_id_end"].max()
            if pred_max_end > n_pred_points:
                raise ValueError(
                    f"Predicted instance references pred_point index {pred_max_end} "
                    f"but only {n_pred_points} pred_points exist."
                )

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frames_data)

    def copy(self) -> "LazyDataStore":
        """Create an independent copy with copied arrays.

        The returned copy has independent numpy arrays but shares references
        to Video, Skeleton, and Track objects. The metadata objects are shared
        because they are the canonical objects referenced by the Labels; copying
        them here would create inconsistency with Labels.videos/skeletons/tracks.

        Returns:
            A new LazyDataStore with copied arrays.
        """
        return LazyDataStore(
            frames_data=self.frames_data.copy(),
            instances_data=self.instances_data.copy(),
            pred_points_data=self.pred_points_data.copy(),
            points_data=self.points_data.copy(),
            videos=self.videos,  # Share references (canonical objects)
            skeletons=self.skeletons,  # Share references (canonical objects)
            tracks=self.tracks,  # Share references (canonical objects)
            format_id=self.format_id,
            source_path=self._source_path,
        )

    def materialize_frame(self, idx: int) -> "LabeledFrame":
        """Create a fully materialized LabeledFrame.

        Args:
            idx: Index into frames_data array.

        Returns:
            A real LabeledFrame with real Instance objects.
        """
        from sleap_io.model.labeled_frame import LabeledFrame

        frame_row = self.frames_data[idx]
        video_id = int(frame_row[1])  # video_id
        frame_idx = int(frame_row[2])  # frame_idx
        inst_start = int(frame_row[3])  # instance_id_start
        inst_end = int(frame_row[4])  # instance_id_end

        instances = []
        for inst_idx in range(inst_start, inst_end):
            inst = self._materialize_instance(inst_idx)
            instances.append(inst)

        return LabeledFrame(
            video=self.videos[video_id],
            frame_idx=frame_idx,
            instances=instances,
        )

    def _materialize_instance(self, idx: int) -> Union["Instance", "PredictedInstance"]:
        """Create a single Instance from raw data.

        Args:
            idx: Index into instances_data array.

        Returns:
            Instance or PredictedInstance with populated points.
        """
        from sleap_io.model.instance import Instance, PredictedInstance

        inst_row = self.instances_data[idx]

        # Parse instance data - handle format differences
        if self.format_id < 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
            ) = inst_row
            tracking_score = 0.0
        else:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
                tracking_score,
            ) = inst_row

        skeleton = self.skeletons[skeleton_id]
        track = self.tracks[track_id] if track_id >= 0 else None

        if instance_type == InstanceType.USER:
            pts_data = self.points_data[point_id_start:point_id_end]
            points_array = self._make_points_array(pts_data, skeleton)
            if self.format_id < 1.1:
                # Legacy coordinate system adjustment
                points_array["xy"] -= 0.5
            return Instance(
                points=points_array,
                skeleton=skeleton,
                track=track,
                tracking_score=float(tracking_score),
            )
        else:  # PREDICTED
            pts_data = self.pred_points_data[point_id_start:point_id_end]
            points_array = self._make_predicted_points_array(pts_data, skeleton)
            if self.format_id < 1.1:
                # Legacy coordinate system adjustment
                points_array["xy"] -= 0.5
            return PredictedInstance(
                points=points_array,
                skeleton=skeleton,
                track=track,
                score=float(instance_score),
                tracking_score=float(tracking_score),
            )

    def _make_points_array(
        self, pts_data: np.ndarray, skeleton: "Skeleton"
    ) -> np.ndarray:
        """Create PointsArray from raw point data.

        Args:
            pts_data: Structured array with x, y, visible, complete fields.
            skeleton: Skeleton defining node structure.

        Returns:
            Populated PointsArray.
        """
        from sleap_io.model.instance import PointsArray

        n = len(pts_data)
        points = PointsArray.empty(n)
        points["xy"][:, 0] = pts_data["x"]
        points["xy"][:, 1] = pts_data["y"]
        points["visible"] = pts_data["visible"]
        points["complete"] = pts_data["complete"]
        points["name"] = skeleton.node_names
        return points

    def _make_predicted_points_array(
        self, pts_data: np.ndarray, skeleton: "Skeleton"
    ) -> np.ndarray:
        """Create PredictedPointsArray from raw point data.

        Args:
            pts_data: Structured array with x, y, visible, complete, score fields.
            skeleton: Skeleton defining node structure.

        Returns:
            Populated PredictedPointsArray.
        """
        from sleap_io.model.instance import PredictedPointsArray

        n = len(pts_data)
        points = PredictedPointsArray.empty(n)
        points["xy"][:, 0] = pts_data["x"]
        points["xy"][:, 1] = pts_data["y"]
        points["visible"] = pts_data["visible"]
        points["complete"] = pts_data["complete"]
        points["score"] = pts_data["score"]
        points["name"] = skeleton.node_names
        return points

    def materialize_all(self) -> list["LabeledFrame"]:
        """Materialize all frames.

        Returns:
            List of all LabeledFrame objects.
        """
        return [self.materialize_frame(i) for i in range(len(self))]

    def get_user_frame_indices(self) -> list[int]:
        """Find indices of frames containing user (non-predicted) instances.

        Returns:
            List of frame indices (into frames_data) that have at least one user
            instance.
        """
        from sleap_io.io.slp import InstanceType

        # Find all user instances
        user_mask = self.instances_data["instance_type"] == InstanceType.USER
        if not np.any(user_mask):
            return []

        # Get frame boundaries for binary search
        frame_ends = self.frames_data["instance_id_end"]

        # Use binary search to find frame for each user instance - O(n log m)
        user_instance_indices = np.where(user_mask)[0]

        # searchsorted finds insertion point; instance i is in frame fi where
        # frame_ends[fi-1] <= i < frame_ends[fi] (with frame_ends[-1] = 0)
        frame_indices = np.searchsorted(frame_ends, user_instance_indices, side="right")

        # Get unique frame indices (already sorted by searchsorted)
        unique_frames = np.unique(frame_indices)

        # Filter out any out-of-bounds indices (shouldn't happen with valid data)
        valid_mask = unique_frames < len(self.frames_data)

        return unique_frames[valid_mask].tolist()

    def to_numpy(
        self,
        video: Optional["Video"] = None,
        untracked: bool = False,
        return_confidence: bool = False,
        user_instances: bool = True,
    ) -> np.ndarray:
        """Build numpy array directly from raw data (fast path).

        This method builds the output array directly from raw HDF5 data without
        creating any Instance or LabeledFrame objects, providing significant
        performance improvement for workflows that only need numpy output.

        Args:
            video: Video to filter by. If None, uses the first video.
            untracked: If True, index by instance order instead of tracks.
                If False (default), organize instances by their track assignment.
            return_confidence: If True, include confidence as third coordinate.
                For user instances, confidence is set to 1.0.
            user_instances: If True (default), prefer user instances over predicted
                instances. If False, only include predicted instances.

        Returns:
            Array of shape (n_frames, n_tracks, n_nodes, 2) or
            (n_frames, n_tracks, n_nodes, 3) if return_confidence is True.
            Missing data is filled with np.nan.
        """
        # Step 1: Determine video_id to filter
        if video is None:
            video_id = 0
        else:
            video_id = self.videos.index(video)

        # Step 2: Filter frames by video and get frame range
        frames_data = self.frames_data
        video_mask = frames_data["video"] == video_id
        video_frame_data = frames_data[video_mask]
        n_frames_data = len(video_frame_data)

        if n_frames_data == 0:
            # No frames for this video, return empty array
            skeleton = self.skeletons[0] if self.skeletons else None
            n_nodes = len(skeleton.nodes) if skeleton else 0
            n_coords = 3 if return_confidence else 2
            return np.full((0, 0, n_nodes, n_coords), np.nan, dtype="float32")

        # Get frame index range for this video
        frame_indices = video_frame_data["frame_idx"]
        first_frame = int(frame_indices.min())
        last_frame = int(frame_indices.max())
        n_frames = last_frame - first_frame + 1

        # Step 3: Determine output dimensions
        skeleton = self.skeletons[-1]  # Use last skeleton (consistent with eager)
        n_nodes = len(skeleton.nodes)

        # Count max instances across frames (matches eager behavior)
        n_instances = self._count_max_instances_per_frame(
            video_frame_data, user_instances
        )

        # Single instance case forces untracked mode (matches eager behavior)
        is_single_instance = n_instances == 1
        untracked = untracked or is_single_instance

        if untracked:
            n_tracks = n_instances
        else:
            # Use track count (can be 0 if no tracks)
            n_tracks = len(self.tracks)

        n_coords = 3 if return_confidence else 2

        # Step 4: Allocate output array
        output = np.full(
            (n_frames, n_tracks, n_nodes, n_coords), np.nan, dtype="float32"
        )

        # Step 5: Build frame_idx to data index mapping
        frame_idx_to_data = {}
        for data_idx, row in enumerate(video_frame_data):
            frame_idx_to_data[int(row["frame_idx"])] = (row, data_idx)

        # Step 6: Fill from raw data
        for frame_idx in range(first_frame, last_frame + 1):
            if frame_idx not in frame_idx_to_data:
                continue

            frame_row, _ = frame_idx_to_data[frame_idx]
            out_idx = frame_idx - first_frame
            self._fill_frame_numpy(
                output[out_idx],
                frame_row,
                untracked=untracked,
                return_confidence=return_confidence,
                user_instances=user_instances,
            )

        return output

    def _count_max_instances_per_frame(
        self,
        video_frame_data: np.ndarray,
        user_instances: bool,
    ) -> int:
        """Count maximum instances across frames for untracked mode.

        This matches the eager implementation: when user_instances=True,
        counts max(n_user, n_predicted); when user_instances=False, counts
        only predicted instances.

        Args:
            video_frame_data: Filtered frame data for a single video.
            user_instances: Whether to include user instances.

        Returns:
            Maximum number of instances in any frame.
        """
        max_count = 0
        for frame_row in video_frame_data:
            inst_start = int(frame_row["instance_id_start"])
            inst_end = int(frame_row["instance_id_end"])

            n_user = 0
            n_pred = 0
            for i in range(inst_start, inst_end):
                if self.instances_data[i]["instance_type"] == InstanceType.USER:
                    n_user += 1
                else:
                    n_pred += 1

            if user_instances:
                # When user_instances=True (and predicted_instances=True implicitly),
                # count max of either user or predicted (matches eager behavior)
                frame_count = max(n_user, n_pred)
            else:
                # When user_instances=False, only count predicted instances
                frame_count = n_pred

            max_count = max(max_count, frame_count)

        return max_count

    def _fill_frame_numpy(
        self,
        output: np.ndarray,
        frame_row: np.ndarray,
        untracked: bool,
        return_confidence: bool,
        user_instances: bool,
    ) -> None:
        """Fill a single frame's slice of the output array.

        Args:
            output: Output array slice of shape (n_tracks, n_nodes, n_coords).
            frame_row: Row from frames_data for this frame.
            untracked: Whether to use untracked (arbitrary order) indexing.
            return_confidence: Whether to include confidence scores.
            user_instances: Whether to prefer user instances.
        """
        inst_start = int(frame_row["instance_id_start"])
        inst_end = int(frame_row["instance_id_end"])

        if untracked:
            # Fill instances in arbitrary order
            # When user_instances=True: prefer user instances, then add predicted
            #   instances that don't overlap
            # When user_instances=False: only include predicted instances
            j = 0
            instances_to_process = []

            # Separate instances by type
            user_insts = []
            pred_insts = []
            for i in range(inst_start, inst_end):
                if self.instances_data[i]["instance_type"] == InstanceType.USER:
                    user_insts.append(i)
                else:
                    pred_insts.append(i)

            if user_instances:
                # First collect user instances
                if user_insts:
                    instances_to_process.extend(user_insts)

                    # Check if this is single-instance case (n_instances == 1)
                    # In that case, we don't add predicted instances
                    is_single_instance = output.shape[0] == 1
                    if not is_single_instance:
                        # Add predicted instances that don't overlap with user instances
                        for pred_idx in pred_insts:
                            pred_row = self.instances_data[pred_idx]
                            skip = False

                            for user_idx in user_insts:
                                user_row = self.instances_data[user_idx]
                                # Skip if user and predicted share same track
                                user_track = int(user_row["track"])
                                pred_track = int(pred_row["track"])
                                if (
                                    user_track >= 0
                                    and pred_track >= 0
                                    and user_track == pred_track
                                ):
                                    skip = True
                                    break
                                # Skip if linked via from_predicted
                                from_pred = int(user_row["from_predicted"])
                                inst_id = int(pred_row["instance_id"])
                                if from_pred == inst_id:
                                    skip = True
                                    break

                            if not skip:
                                instances_to_process.append(pred_idx)
                else:
                    # No user instances, use predicted
                    instances_to_process.extend(pred_insts)
            else:
                # Only include predicted instances
                instances_to_process.extend(pred_insts)

            # Fill output
            for inst_idx in instances_to_process:
                if j >= output.shape[0]:
                    break
                self._fill_instance_numpy(output[j], inst_idx, return_confidence)
                j += 1
        else:
            # Organize by track
            # Build track -> instance mapping, preferring user instances
            track_to_inst = {}

            # First pass: add predicted instances
            for i in range(inst_start, inst_end):
                inst_row = self.instances_data[i]
                if inst_row["instance_type"] == InstanceType.PREDICTED:
                    track_id = int(inst_row["track"])
                    if track_id >= 0:
                        track_to_inst[track_id] = i

            # Second pass: add user instances (overwriting predicted if same track)
            if user_instances:
                for i in range(inst_start, inst_end):
                    inst_row = self.instances_data[i]
                    if inst_row["instance_type"] == InstanceType.USER:
                        track_id = int(inst_row["track"])
                        if track_id >= 0:
                            track_to_inst[track_id] = i

            # Fill output by track
            for track_id, inst_idx in track_to_inst.items():
                if track_id < output.shape[0]:
                    self._fill_instance_numpy(
                        output[track_id], inst_idx, return_confidence
                    )

    def _fill_instance_numpy(
        self,
        output: np.ndarray,
        inst_idx: int,
        return_confidence: bool,
    ) -> None:
        """Fill instance points into output array.

        Args:
            output: Output array slice of shape (n_nodes, n_coords).
            inst_idx: Index into instances_data.
            return_confidence: Whether to include confidence scores.
        """
        inst_row = self.instances_data[inst_idx]

        # Parse instance data - handle format differences
        if self.format_id < 1.2:
            (
                _instance_id,
                instance_type,
                _frame_id,
                _skeleton_id,
                _track_id,
                _from_predicted,
                _instance_score,
                point_id_start,
                point_id_end,
            ) = inst_row
        else:
            (
                _instance_id,
                instance_type,
                _frame_id,
                _skeleton_id,
                _track_id,
                _from_predicted,
                _instance_score,
                point_id_start,
                point_id_end,
                _tracking_score,
            ) = inst_row

        point_id_start = int(point_id_start)
        point_id_end = int(point_id_end)

        if instance_type == InstanceType.USER:
            pts_data = self.points_data[point_id_start:point_id_end]
            x = pts_data["x"]
            y = pts_data["y"]
            # Apply legacy coordinate adjustment if needed
            if self.format_id < 1.1:
                x = x - 0.5
                y = y - 0.5
            output[:, 0] = x
            output[:, 1] = y
            if return_confidence:
                # User instances have confidence of 1.0
                output[:, 2] = 1.0
        else:  # PREDICTED
            pts_data = self.pred_points_data[point_id_start:point_id_end]
            x = pts_data["x"]
            y = pts_data["y"]
            # Apply legacy coordinate adjustment if needed
            if self.format_id < 1.1:
                x = x - 0.5
                y = y - 0.5
            output[:, 0] = x
            output[:, 1] = y
            if return_confidence:
                output[:, 2] = pts_data["score"]


class LazyFrameList:
    """List-like proxy that materializes LabeledFrame objects on access.

    This provides backward compatibility for code that accesses
    labels.labeled_frames directly. Frames are created on-demand when
    accessed via indexing or iteration.

    Mutations are blocked with helpful error messages suggesting to call
    labels.materialize() first.
    """

    def __init__(self, store: LazyDataStore) -> None:
        """Initialize with a LazyDataStore.

        Args:
            store: The LazyDataStore containing raw frame data.
        """
        self._store = store

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self._store)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union["LabeledFrame", list["LabeledFrame"]]:
        """Get frame(s) by index or slice.

        Args:
            idx: Integer index or slice object.

        Returns:
            A single LabeledFrame for integer indexing, or a list of
            LabeledFrames for slicing.

        Raises:
            IndexError: If index is out of range.
        """
        n = len(self)

        if isinstance(idx, slice):
            # Handle slice
            indices = range(*idx.indices(n))
            return [self._store.materialize_frame(i) for i in indices]

        # Handle negative indexing
        if idx < 0:
            idx = n + idx

        # Bounds check
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for {n} frames")

        return self._store.materialize_frame(idx)

    def __iter__(self) -> Iterator["LabeledFrame"]:
        """Iterate over frames, materializing each."""
        for i in range(len(self)):
            yield self._store.materialize_frame(i)

    def __repr__(self) -> str:
        """Return informative representation."""
        return f"LazyFrameList(n_frames={len(self)})"

    def _mutation_error(self, operation: str) -> RuntimeError:
        """Create a RuntimeError with helpful guidance for blocked mutations.

        Args:
            operation: Name of the blocked operation.

        Returns:
            RuntimeError with guidance message.
        """
        return RuntimeError(
            f"Cannot {operation} on LazyFrameList (lazy-loaded Labels).\n\n"
            f"To modify, first materialize the Labels:\n"
            f"    labels = labels.materialize()\n"
            f"    labels.labeled_frames.{operation}(...)"
        )

    def append(self, item: "LabeledFrame") -> None:
        """Block append with helpful error.

        Raises:
            RuntimeError: Always, with guidance to materialize first.
        """
        raise self._mutation_error("append")

    def extend(self, items: list["LabeledFrame"]) -> None:
        """Block extend with helpful error.

        Raises:
            RuntimeError: Always, with guidance to materialize first.
        """
        raise self._mutation_error("extend")

    def insert(self, idx: int, item: "LabeledFrame") -> None:
        """Block insert with helpful error.

        Raises:
            RuntimeError: Always, with guidance to materialize first.
        """
        raise self._mutation_error("insert")

    def __setitem__(self, idx: int, value: "LabeledFrame") -> None:
        """Block item assignment with helpful error.

        Raises:
            RuntimeError: Always, with guidance to materialize first.
        """
        raise self._mutation_error("__setitem__")

    def __delitem__(self, idx: int) -> None:
        """Block item deletion with helpful error.

        Raises:
            RuntimeError: Always, with guidance to materialize first.
        """
        raise self._mutation_error("__delitem__")
