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
        max_inst_end = (
            self.frames_data["instance_id_end"].max() if n_frames > 0 else 0
        )
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

    def _materialize_instance(
        self, idx: int
    ) -> Union["Instance", "PredictedInstance"]:
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
