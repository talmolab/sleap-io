"""Data structures for data associated with a single instance such as an animal.

The `Instance` class is a SLEAP data structure that contains a collection of points that
correspond to landmarks within a `Skeleton`.

`PredictedInstance` additionally contains metadata associated with how the instance was
estimated, such as confidence scores.
"""

from __future__ import annotations
import attrs
from typing import ClassVar, Optional, Union
from sleap_io import Skeleton, Node
from sleap_io.model.skeleton import NodeOrIndex
import numpy as np


@attrs.define(eq=False)
class Track:
    """An object that represents the same animal/object across multiple detections.

    This allows tracking of unique entities in the video over time and space.

    A `Track` may also be used to refer to unique identity classes that span multiple
    videos, such as `"female mouse"`.

    Attributes:
        name: A name given to this track for identification purposes.

    Notes:
        `Track`s are compared by identity. This means that unique track objects with the
        same name are considered to be different.
    """

    name: str = ""


@attrs.define(auto_attribs=True, slots=True, eq=True)
class Instance:
    """This class represents a ground truth instance such as an animal.

    An `Instance` has a set of landmarks (points) that correspond to a `Skeleton`. Each
    point is associated with a `Node` in the skeleton. The points are stored in a
    structured numpy array with columns for x, y, visible, complete and name.

    The `Instance` may also be associated with a `Track` which links multiple instances
    together across frames or videos.

    Attributes:
        points: A numpy structured array with columns for xy, visible and complete. The
            array should have shape `(n_nodes,)`. This representation is useful for
            performance efficiency when working with large datasets.
        skeleton: The `Skeleton` that describes the `Node`s and `Edge`s associated with
            this instance.
        track: An optional `Track` associated with a unique animal/object across frames
            or videos.
        tracking_score: The score associated with the `Track` assignment. This is
            typically the value from the score matrix used in an identity assignment.
            This is `None` if the instance is not associated with a track or if the
            track was assigned manually.
        from_predicted: The `PredictedInstance` (if any) that this instance was
            initialized from. This is used with human-in-the-loop workflows.
    """

    ARRAY_DTYPE: ClassVar[np.dtype] = np.dtype(
        [
            ("xy", "<f8", (2,)),  # 64-bit (8-byte) little-endian double, ndim=2
            ("visible", "bool"),
            ("complete", "bool"),
            ("name", "O"),  # object dtype to store pointers to python string objects
        ]
    )

    points: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    skeleton: Skeleton
    track: Optional[Track] = None
    tracking_score: Optional[float] = None
    from_predicted: Optional[PredictedInstance] = None

    @classmethod
    def empty(
        cls,
        skeleton: Skeleton,
        track: Optional[Track] = None,
        tracking_score: Optional[float] = None,
        from_predicted: Optional[PredictedInstance] = None,
    ) -> "Instance":
        """Create an empty instance with no points.

        Args:
            skeleton: The `Skeleton` that this `Instance` is associated with.
            track: An optional `Track` associated with a unique animal/object across
                frames or videos.
            tracking_score: The score associated with the `Track` assignment. This is
                typically the value from the score matrix used in an identity
                assignment. This is `None` if the instance is not associated with a
                track or if the track was assigned manually.
            from_predicted: The `PredictedInstance` (if any) that this instance was
                initialized from. This is used with human-in-the-loop workflows.

        Returns:
            An `Instance` with an empty numpy array of shape `(n_nodes,)`.
        """
        inst = cls(
            points=np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE),
            skeleton=skeleton,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )
        inst.points["name"] = skeleton.node_names
        return inst

    @classmethod
    def _convert_points(
        cls, points_data: np.ndarray | dict | list, skeleton: Skeleton
    ) -> np.ndarray:
        """Convert points to a structured numpy array if needed."""
        points = None

        if type(points_data) == list:
            # Hopefully this is a list of lists.
            points_data = np.array(points_data)

        if type(points_data) == np.ndarray:
            if len(points_data) != len(skeleton):
                raise ValueError(
                    f"points must have length {len(skeleton)}, got {len(points_data)}."
                )

            if (
                points_data.dtype.fields is not None
                and points_data.dtype == cls.ARRAY_DTYPE
            ):
                # We got the right dtype, just use it without copying.
                return points_data
            else:
                # Create a new structured array.
                points = np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE)

                # Fill in the fields.
                if points_data is not None:
                    if points_data.dtype.fields is not None:
                        # We got a structured array!
                        # Try to fill in with the fields available.
                        for field_name in cls.ARRAY_DTYPE.names:
                            if field_name in points_data.dtype.names:
                                points[field_name] = points_data[field_name]

                    else:
                        # We got a plain array! Assume it's x and y.
                        points["xy"] = points_data[:, 0:2]

                        if points_data.shape[1] >= 3:
                            # Assume we have visibility.
                            points["visible"] = points_data[:, 2]
                        else:
                            # Default to visibility based on x being NaN.
                            points["visible"] = ~np.isnan(points_data[:, 0])

                        if points_data.shape[1] >= 4:
                            # Assume we have completion.
                            points["complete"] = points_data[:, 3]

        elif type(points_data) == dict:
            points = np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE)
            for node, data in points_data.items():
                if type(node) == Node or type(node) == str:
                    node = skeleton.index(node)

                points[node]["xy"] = data[:2]

                if len(data) >= 3:
                    points[node]["visible"] = data[2]
                else:
                    points[node]["visible"] = ~np.isnan(data[0])

                if len(data) >= 4:
                    points[node]["complete"] = data[3]

        elif points is None:
            raise ValueError("points must be a numpy array or dictionary.")

        points["name"] = skeleton.node_names
        return points

    @classmethod
    def from_numpy(
        cls,
        points_data: np.ndarray,
        skeleton: Skeleton,
        track: Optional[Track] = None,
        tracking_score: Optional[float] = None,
        from_predicted: Optional[PredictedInstance] = None,
    ) -> "Instance":
        """Create an instance object from a numpy array.

        Args:
            points_data: A numpy array of shape `(n_nodes, D)` corresponding to the
                points of the skeleton. Values of `np.nan` indicate "missing" nodes and
                will be reflected in the "visible" field.

                If `D == 2`, the array should have columns for x and y.
                If `D == 3`, the array should have columns for x, y and visible.
                If `D == 4`, the array should have columns for x, y, visible and
                complete.

                If this is provided as a structured array, it will be used without copy
                if it has the correct dtype. Otherwise, a new structured array will be
                created reusing the provided data.
            skeleton: The `Skeleton` that this `Instance` is associated with. It should
                have `n_nodes` nodes.
            track: An optional `Track` associated with a unique animal/object across
                frames or videos.
            tracking_score: The score associated with the `Track` assignment. This is
                typically the value from the score matrix used in an identity
                assignment. This is `None` if the instance is not associated with a
                track or if the track was assigned manually.
            from_predicted: The `PredictedInstance` (if any) that this instance was
                initialized from. This is used with human-in-the-loop workflows.

        Returns:
            An `Instance` object with the specified points.
        """
        return cls(
            points=points_data,
            skeleton=skeleton,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )

    def __attrs_post_init__(self):
        """Convert the points array after initialization."""
        self.points = self._convert_points(self.points, self.skeleton)

    def numpy(
        self,
        invisible_as_nan: bool = True,
    ) -> np.ndarray:
        """Return the instance points as a `(n_nodes, 2)` numpy array.

        Args:
            invisible_as_nan: If `True` (the default), points that are not visible will
                be set to `np.nan`. If `False`, they will be whatever the stored value
                of `Instance.points["xy"]` is.

        Returns:
            A numpy array of shape `(n_nodes, 2)` corresponding to the points of the
            skeleton. Values of `np.nan` indicate "missing" nodes.

        Notes:
            This will always return a copy of the array.

            If you need to avoid making a copy, just access the `Instance.points["xy"]`
            attribute directly. This will not replace invisible points with `np.nan`.
        """
        if invisible_as_nan:
            return np.where(
                self.points["visible"].reshape(-1, 1), self.points["xy"], np.nan
            )
        else:
            return self.points["xy"].copy()

    def validate_points(self):
        """Validate the points array.

        The points array must be a structured numpy array with fields for x, y, score,
        visible, complete, and score. The fields must have the correct data types and
        the array must have the correct shape.

        Raises:
            TypeError: If the points array is not a structured numpy array.
            ValueError: If the points array does not have the correct fields or data
                types.
        """

        if not isinstance(self.points, np.ndarray):
            raise TypeError(f"points must be a numpy array, got {type(self.points)}")

        if self.points.dtype.fields is None:
            raise TypeError("points must be a structured array")

        expected_fields = set(self.DTYPE.names)
        actual_fields = set(self.points.dtype.names)

        if expected_fields != actual_fields:
            raise ValueError(
                f"points must have fields {expected_fields}, " f"got {actual_fields}"
            )

        # Validate field types
        for field_name in self.DTYPE.names:
            expected_type = self.DTYPE[field_name]
            actual_type = self.points.dtype[field_name]
            if expected_type != actual_type:
                raise TypeError(
                    f"Field {field_name} must have dtype {expected_type}, "
                    f"got {actual_type}"
                )

        # Validate length.
        if len(self.points) != len(self.skeleton):
            raise ValueError(
                f"points must have length {len(self.skeleton)}, "
                f"got {len(self.points)}"
            )

        # Validate alignment with the skeleton.
        if not all(self.points["name"] == self.skeleton.node_names):
            raise ValueError("points must have the same node names as the skeleton.")

    def __getitem__(self, node: Union[int, str, Node]) -> np.ndarray:
        """Return the point associated with a node."""
        if type(node) != int:
            node = self.skeleton.index(node)

        return self.points[node]

    def __len__(self) -> int:
        """Return the number of points in the instance."""
        return len(self.points)

    def __repr__(self) -> str:
        """Return a readable representation of the instance."""
        pts = self.numpy().tolist()
        track = f'"{self.track.name}"' if self.track is not None else self.track

        return f"Instance(points={pts}, track={track})"

    @property
    def n_visible(self) -> int:
        """Return the number of visible points in the instance."""
        return sum(self.points["visible"])

    @property
    def is_empty(self) -> bool:
        """Return `True` if no points are visible on the instance."""
        return ~(self.points["visible"].any())

    def update_skeleton(self, names_only: bool = False):
        """Update or replace the skeleton associated with the instance.

        Args:
            names_only: If `True`, only update the node names in the points array. If
                `False`, the points array will be updated to match the new skeleton.
        """

        if names_only:
            # Update the node names.
            self.points["name"] = self.skeleton.node_names
            return

        # Find correspondences.
        new_node_inds, old_node_inds = self.skeleton.match_nodes(self.points["name"])

        # Update the points.
        new_points = np.empty(len(self.skeleton), dtype=self.ARRAY_DTYPE)
        new_points[new_node_inds] = self.points[old_node_inds]
        new_points["name"] = self.skeleton.node_names
        self.points = new_points

    def replace_skeleton(
        self,
        new_skeleton: Skeleton,
        node_names_map: dict[str, str] | None = None,
    ):
        """Replace the skeleton associated with the instance.

        Args:
            new_skeleton: The new `Skeleton` to associate with the instance.
            node_names_map: Dictionary mapping nodes in the old skeleton to nodes in the
                new skeleton. Keys and values should be specified as lists of strings.
                If not provided, only nodes with identical names will be mapped. Points
                associated with unmapped nodes will be removed.

        Notes:
            This method will update the `Instance.skeleton` attribute and the
            `Instance.points` attribute in place (a copy is made of the points array).

            It is recommended to use `Labels.replace_skeleton` instead of this method if
            more flexible node mapping is required.
        """
        # Update skeleton object.
        # old_skeleton = self.skeleton
        self.skeleton = new_skeleton

        # Get node names with replacements from node map if possible.
        # old_node_names = old_skeleton.node_names
        old_node_names = self.points["name"].tolist()
        if node_names_map is not None:
            old_node_names = [node_names_map.get(node, node) for node in old_node_names]

        # Find correspondences.
        new_node_inds, old_node_inds = self.skeleton.match_nodes(old_node_names)
        # old_node_inds = np.array(old_node_inds).reshape(-1, 1)
        # new_node_inds = np.array(new_node_inds).reshape(-1, 1)

        # Update the points.
        new_points = np.empty(len(self.skeleton), dtype=self.ARRAY_DTYPE)
        new_points[new_node_inds] = self.points[old_node_inds]
        self.points = new_points
        self.points["name"] = self.skeleton.node_names


@attrs.define
class PredictedInstance(Instance):
    """A `PredictedInstance` is an `Instance` that was predicted using a model.

    Attributes:
        skeleton: The `Skeleton` that this `Instance` is associated with.
        points: A dictionary where keys are `Skeleton` nodes and values are `Point`s.
        track: An optional `Track` associated with a unique animal/object across frames
            or videos.
        from_predicted: Not applicable in `PredictedInstance`s (must be set to `None`).
        score: The instance detection or part grouping prediction score. This is a
            scalar that represents the confidence with which this entire instance was
            predicted. This may not always be applicable depending on the model type.
        tracking_score: The score associated with the `Track` assignment. This is
            typically the value from the score matrix used in an identity assignment.
    """

    ARRAY_DTYPE: ClassVar[np.dtype] = np.dtype(
        [
            ("xy", "<f8", (2,)),  # 64-bit (8-byte) little-endian double, ndim=2
            ("score", "<f8"),  # 64-bit (8-byte) little-endian double
            ("visible", "bool"),
            ("complete", "bool"),
            ("name", "O"),  # object dtype to store pointers to python string objects
        ]
    )

    points: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    skeleton: Skeleton
    score: float = 0.0
    track: Optional[Track] = None
    tracking_score: Optional[float] = 0
    from_predicted: Optional[PredictedInstance] = None

    def __repr__(self) -> str:
        """Return a readable representation of the instance."""
        pts = self.numpy().tolist()
        track = f'"{self.track.name}"' if self.track is not None else self.track

        score = str(self.score) if self.score is None else f"{self.score:.2f}"
        tracking_score = (
            str(self.tracking_score)
            if self.tracking_score is None
            else f"{self.tracking_score:.2f}"
        )
        return (
            f"PredictedInstance(points={pts}, track={track}, "
            f"score={score}, tracking_score={tracking_score})"
        )

    @classmethod
    def empty(
        cls,
        skeleton: Skeleton,
        score: float = 0.0,
        track: Optional[Track] = None,
        tracking_score: Optional[float] = None,
        from_predicted: Optional[PredictedInstance] = None,
    ) -> "PredictedInstance":
        """Create an empty instance with no points.

        Args:
            skeleton: The `Skeleton` that this `Instance` is associated with.
            score: The instance detection or part grouping prediction score. This is a
                scalar that represents the confidence with which this entire instance
                was predicted. This may not always be applicable depending on the model
                type.
            track: An optional `Track` associated with a unique animal/object across
                frames or videos.
            tracking_score: The score associated with the `Track` assignment. This is
                typically the value from the score matrix used in an identity
                assignment. This is `None` if the instance is not associated with a
                track or if the track was assigned manually.
            from_predicted: The `PredictedInstance` (if any) that this instance was
                initialized from. This is used with human-in-the-loop workflows.

        Returns:
            An `PredictedInstance` with an empty numpy array of shape `(n_nodes,)`.
        """
        inst = cls(
            points=np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE),
            skeleton=skeleton,
            score=score,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )
        inst.points["name"] = skeleton.node_names
        return inst

    @classmethod
    def _convert_points(
        cls, points_data: np.ndarray | dict | list, skeleton: Skeleton
    ) -> np.ndarray:
        """Convert points to a structured numpy array if needed."""
        points = None

        if type(points_data) == list:
            # Hopefully this is a list of lists.
            points_data = np.array(points_data)

        if type(points_data) == np.ndarray:
            if len(points_data) != len(skeleton):
                raise ValueError(
                    f"points must have length {len(skeleton)}, got {len(points_data)}."
                )

            if (
                points_data.dtype.fields is not None
                and points_data.dtype == cls.ARRAY_DTYPE
            ):
                # We got the right dtype, just use it without copying.
                return points_data
            else:
                # Create a new structured array.
                points = np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE)

                # Fill in the fields.
                if points_data is not None:
                    if points_data.dtype.fields is not None:
                        # We got a structured array!
                        # Try to fill in with the fields available.
                        for field_name in cls.ARRAY_DTYPE.names:
                            if field_name in points_data.dtype.names:
                                points[field_name] = points_data[field_name]

                    else:
                        # We got a plain array! Assume it's x and y.
                        points["xy"] = points_data[:, 0:2]

                        if points_data.shape[1] >= 3:
                            # Assume we have score.
                            points["score"] = points_data[:, 2]

                        if points_data.shape[1] >= 4:
                            # Assume we have visibility.
                            points["visible"] = points_data[:, 3]
                        else:
                            # Default to visibility based on x being NaN.
                            points["visible"] = ~np.isnan(points_data[:, 0])

                        if points_data.shape[1] >= 5:
                            # Assume we have completion.
                            points["complete"] = points_data[:, 4]

        elif type(points_data) == dict:
            points = np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE)
            for node, data in points_data.items():
                if type(node) == Node or type(node) == str:
                    node = skeleton.index(node)

                points[node]["xy"] = data[:2]

                if len(data) >= 3:
                    points[node]["score"] = data[2]

                if len(data) >= 4:
                    points[node]["visible"] = data[3]
                else:
                    points[node]["visible"] = ~np.isnan(data[0])

                if len(data) >= 5:
                    points[node]["complete"] = data[4]

        elif points is None:
            raise ValueError("points must be a numpy array or dictionary.")

        points["name"] = skeleton.node_names
        return points

    @classmethod
    def from_numpy(
        cls,
        points_data: np.ndarray,
        skeleton: Skeleton,
        point_scores: Optional[np.ndarray] = None,
        score: float = 0.0,
        track: Optional[Track] = None,
        tracking_score: Optional[float] = None,
        from_predicted: Optional[PredictedInstance] = None,
    ) -> "PredictedInstance":
        """Create a predicted instance object from a numpy array.

        Args:
            points_data: A numpy array of shape `(n_nodes, D)` corresponding to the
                points of the skeleton. Values of `np.nan` indicate "missing" nodes and
                will be reflected in the "visible" field.

                If `D == 2`, the array should have columns for x and y.
                If `D == 3`, the array should have columns for x, y and score.
                If `D == 4`, the array should have columns for x, y, score and visible.
                If `D == 5`, the array should have columns for x, y, score, visible and
                complete.

                If this is provided as a structured array, it will be used without copy
                if it has the correct dtype. Otherwise, a new structured array will be
                created reusing the provided data.
            skeleton: The `Skeleton` that this `Instance` is associated with. It should
                have `n_nodes` nodes.
            point_scores: An optional numpy array of shape `(n_nodes,)` with the score
                associated with each point. This is typically the confidence with which
                each point was predicted. This is `None` if the scores in the
                `points_data` array will be used.
            score: The instance detection or part grouping prediction score. This is a
                scalar that represents the confidence with which this entire instance
                was predicted. This may not always be applicable depending on the model
                type.
            track: An optional `Track` associated with a unique animal/object across
                frames or videos.
            tracking_score: The score associated with the `Track` assignment. This is
                typically the value from the score matrix used in an identity
                assignment. This is `None` if the instance is not associated with a
                track or if the track was assigned manually.
            from_predicted: The `PredictedInstance` (if any) that this instance was
                initialized from. This is used with human-in-the-loop workflows.

        Returns:
            An `Instance` object with the specified points.
        """
        # if (
        #     points_data.dtype.fields is not None
        #     and points_data.dtype == cls.ARRAY_DTYPE
        # ):
        #     # We got the right dtype, just use it without copying.
        #     points = points_data
        # else:
        #     # Create a new structured array.
        #     points = np.empty(len(skeleton), dtype=cls.ARRAY_DTYPE)

        #     # Fill in the fields.
        #     if points_data is not None:
        #         if points_data.dtype.fields is not None:
        #             # We got a structured array!
        #             # Try to fill in with the fields available.
        #             for field_name in cls.ARRAY_DTYPE.names:
        #                 if field_name in points_data.dtype.names:
        #                     points[field_name] = points_data[field_name]

        #         else:
        #             # We got a plain array! Assume it's x and y.
        #             points["xy"] = points_data[:, :2]

        #             if points_data.shape[1] >= 3:
        #                 # Assume we have score.
        #                 points["score"] = points_data[:, 2]

        #             if points_data.shape[1] >= 4:
        #                 # Assume we have visibility.
        #                 points["visible"] = points_data[:, 2]
        #             else:
        #                 # Default to visibility based on x being NaN.
        #                 points["visible"] = np.isnan(points_data[:, 0])

        #             if points_data.shape[1] >= 4:
        #                 # Assume we have completion.
        #                 points["complete"] = points_data[:, 3]

        # # Set the node names.
        # points["name"] = skeleton.node_names

        points_data = cls._convert_points(points_data, skeleton)
        if point_scores is not None:
            points_data["score"] = point_scores

        return cls(
            points=points_data,
            skeleton=skeleton,
            score=score,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )

    def numpy(
        self,
        invisible_as_nan: bool = True,
        scores: bool = False,
    ) -> np.ndarray:
        """Return the instance points as a `(n_nodes, 2)` numpy array.

        Args:
            invisible_as_nan: If `True` (the default), points that are not visible will
                be set to `np.nan`. If `False`, they will be whatever the stored value
                of `PredictedInstance.points["xy"]` is.
            scores: If `True`, the score associated with each point will be
                included in the output.

        Returns:
            A numpy array of shape `(n_nodes, 2)` corresponding to the points of the
            skeleton. Values of `np.nan` indicate "missing" nodes.

            If `scores` is `True`, the array will have shape `(n_nodes, 3)` with the
            third column containing the score associated with each point.

        Notes:
            This will always return a copy of the array.

            If you need to avoid making a copy, just access the
            `PredictedInstance.points["xy"]` attribute directly. This will not replace
            invisible points with `np.nan`.
        """
        if invisible_as_nan:
            pts = np.where(
                self.points["visible"].reshape(-1, 1), self.points["xy"], np.nan
            )
        else:
            pts = self.points["xy"].copy()

        if scores:
            return np.column_stack((pts, self.points["score"]))
        else:
            return pts
