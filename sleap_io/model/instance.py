"""Data structures for data associated with a single instance such as an animal.

The `Instance` class is a SLEAP data structure that contains a collection of points that
correspond to landmarks within a `Skeleton`.

`PredictedInstance` additionally contains metadata associated with how the instance was
estimated, such as confidence scores.
"""

from __future__ import annotations

from typing import Optional, Union

import attrs
import numpy as np

from sleap_io.model.skeleton import Node, Skeleton


class PointsArray(np.ndarray):
    """A specialized array for storing instance points data.

    This class ensures that the array always uses the correct dtype and provides
    convenience methods for working with point data.

    The structured dtype includes the following fields:
        - xy: A float64 array of shape (2,) containing the x, y coordinates
        - visible: A boolean indicating if the point is visible
        - complete: A boolean indicating if the point is complete
        - name: An object dtype containing the name of the node
    """

    @classmethod
    def _get_dtype(cls):
        """Get the dtype for points array.

        Returns:
            np.dtype: A structured numpy dtype with fields for xy coordinates,
                visible flag, complete flag, and node names.
        """
        # Cache the dtype at the class level for performance
        # Use cls.__dict__ to check if defined on this class (not inherited)
        if "_cached_dtype" not in cls.__dict__:
            cls._cached_dtype = np.dtype(
                [
                    ("xy", "<f8", (2,)),  # 64-bit (8-byte) little-endian double, ndim=2
                    ("visible", "bool"),
                    ("complete", "bool"),
                    (
                        "name",
                        "O",
                    ),  # object dtype to store pointers to python string objects
                ]
            )
        return cls._cached_dtype

    @classmethod
    def empty(cls, length: int) -> "PointsArray":
        """Create an empty points array with the appropriate dtype.

        Args:
            length: The number of points (nodes) to allocate in the array.

        Returns:
            PointsArray: An empty array of the specified length with the appropriate
                dtype.
        """
        dtype = cls._get_dtype()
        arr = np.empty(length, dtype=dtype).view(cls)
        return arr

    @classmethod
    def from_array(cls, array: np.ndarray) -> "PointsArray":
        """Convert an existing array to a PointsArray with the appropriate dtype.

        Args:
            array: A numpy array to convert. Can be a structured array or a regular
                array. If a regular array, it is assumed to have columns for x, y
                coordinates and optionally visible and complete flags.

        Returns:
            PointsArray: A structured array view of the input data with the appropriate
                dtype.

        Notes:
            If the input is a structured array with fields matching the target dtype,
            those fields will be copied. Otherwise, a best-effort conversion is made:

            - First two columns (or first 2D element) are interpreted as x, y coords
            - Third column (if present) is interpreted as visible flag
            - Fourth column (if present) is interpreted as complete flag

            If visibility is not provided, it is inferred from NaN values in the x
            coordinate.
        """
        dtype = cls._get_dtype()

        # If already the right type, just view as PointsArray
        if isinstance(array, np.ndarray) and array.dtype == dtype:
            return array.view(cls)

        # Otherwise, create a new array with the right dtype
        new_array = np.empty(len(array), dtype=dtype).view(cls)

        # Copy available fields
        if isinstance(array, np.ndarray) and array.dtype.fields is not None:
            # Structured array, copy matching fields
            for field_name in dtype.names:
                if field_name in array.dtype.names:
                    new_array[field_name] = array[field_name]
        elif isinstance(array, np.ndarray):
            # Regular array, assume x, y coordinates
            new_array["xy"] = array[:, 0:2]

            # Default visibility based on NaN
            new_array["visible"] = ~np.isnan(array[:, 0])

            # If there are more columns, assume they are visible and complete
            if array.shape[1] >= 3:
                new_array["visible"] = array[:, 2].astype(bool)

            if array.shape[1] >= 4:
                new_array["complete"] = array[:, 3].astype(bool)

        return new_array

    @classmethod
    def from_dict(cls, points_dict: dict, skeleton: Skeleton) -> "PointsArray":
        """Create a PointsArray from a dictionary of node points.

        Args:
            points_dict: A dictionary mapping nodes (as Node objects, indices, or
                strings) to point data. Each point should be an array-like with at least
                2 elements for x, y coordinates, and optionally visible and complete
                flags.
            skeleton: The Skeleton object that defines the nodes.

        Returns:
            PointsArray: A structured array with the appropriate dtype containing the
                point data from the dictionary.

        Notes:
            For each entry in the points_dict:
            - First two values are treated as x, y coordinates
            - Third value (if present) is treated as visible flag
            - Fourth value (if present) is treated as complete flag

            If visibility is not provided, it is inferred from NaN values in the x
            coordinate.
        """
        points = cls.empty(len(skeleton))

        for node, data in points_dict.items():
            if isinstance(node, (Node, str)):
                node = skeleton.index(node)

            points[node]["xy"] = data[:2]

            idx = 2
            if len(data) > idx:
                points[node]["visible"] = data[idx]
            else:
                points[node]["visible"] = ~np.isnan(data[0])

            idx += 1
            if len(data) > idx:
                points[node]["complete"] = data[idx]

        return points


class PredictedPointsArray(PointsArray):
    """A specialized array for storing predicted instance points data with scores.

    This extends the PointsArray class to include score information for each point.

    The structured dtype includes the following fields:
        - xy: A float64 array of shape (2,) containing the x, y coordinates
        - score: A float64 containing the confidence score for the point
        - visible: A boolean indicating if the point is visible
        - complete: A boolean indicating if the point is complete
        - name: An object dtype containing the name of the node
    """

    @classmethod
    def _get_dtype(cls):
        """Get the dtype for predicted points array with scores.

        Returns:
            np.dtype: A structured numpy dtype with fields for xy coordinates,
                score, visible flag, complete flag, and node names.
        """
        # Cache the dtype at the class level for performance
        # Use cls.__dict__ to check if defined on this class (not inherited)
        if "_cached_dtype" not in cls.__dict__:
            cls._cached_dtype = np.dtype(
                [
                    ("xy", "<f8", (2,)),  # 64-bit (8-byte) little-endian double, ndim=2
                    ("score", "<f8"),  # 64-bit (8-byte) little-endian double
                    ("visible", "bool"),
                    ("complete", "bool"),
                    (
                        "name",
                        "O",
                    ),  # object dtype to store pointers to python string objects
                ]
            )
        return cls._cached_dtype

    @classmethod
    def from_array(cls, array: np.ndarray) -> "PredictedPointsArray":
        """Convert an existing array to a PredictedPointsArray with appropriate dtype.

        Args:
            array: A numpy array to convert. Can be a structured array or a regular
                array. If a regular array, it is assumed to have columns for x, y
                coordinates, scores, and optionally visible and complete flags.

        Returns:
            PredictedPointsArray: A structured array view of the input data with the
                appropriate dtype.

        Notes:
            If the input is a structured array with fields matching the target dtype,
            those fields will be copied. Otherwise, a best-effort conversion is made:

            - First two columns (or first 2D element) are interpreted as x, y coords
            - Third column (if present) is interpreted as the score
            - Fourth column (if present) is interpreted as visible flag
            - Fifth column (if present) is interpreted as complete flag

            If visibility is not provided, it is inferred from NaN values in the x
            coordinate.
        """
        dtype = cls._get_dtype()

        # If already the right type, just view as PredictedPointsArray
        if isinstance(array, np.ndarray) and array.dtype == dtype:
            return array.view(cls)

        # Otherwise, create a new array with the right dtype
        new_array = np.empty(len(array), dtype=dtype).view(cls)

        # Copy available fields
        if isinstance(array, np.ndarray) and array.dtype.fields is not None:
            # Structured array, copy matching fields
            for field_name in dtype.names:
                if field_name in array.dtype.names:
                    new_array[field_name] = array[field_name]
        elif isinstance(array, np.ndarray):
            # Regular array, assume x, y coordinates
            new_array["xy"] = array[:, 0:2]

            # Default visibility based on NaN
            new_array["visible"] = ~np.isnan(array[:, 0])

            # If there's a third column, assume it's the score
            if array.shape[1] >= 3:
                new_array["score"] = array[:, 2]

            # If there are more columns, assume they are visible and complete
            if array.shape[1] >= 4:
                new_array["visible"] = array[:, 3].astype(bool)

            if array.shape[1] >= 5:
                new_array["complete"] = array[:, 4].astype(bool)

        return new_array

    @classmethod
    def from_dict(cls, points_dict: dict, skeleton: Skeleton) -> "PredictedPointsArray":
        """Create a PredictedPointsArray from a dictionary of node points.

        Args:
            points_dict: A dictionary mapping nodes (as Node objects, indices, or
                strings) to point data. Each point should be an array-like with at least
                2 elements for x, y coordinates, and optionally score, visible, and
                complete flags.
            skeleton: The Skeleton object that defines the nodes.

        Returns:
            PredictedPointsArray: A structured array with the appropriate dtype
                containing the point data from the dictionary.

        Notes:
            For each entry in the points_dict:
            - First two values are treated as x, y coordinates
            - Third value (if present) is treated as score
            - Fourth value (if present) is treated as visible flag
            - Fifth value (if present) is treated as complete flag

            If visibility is not provided, it is inferred from NaN values in the x
            coordinate.
        """
        points = cls.empty(len(skeleton))

        for node, data in points_dict.items():
            if isinstance(node, (Node, str)):
                node = skeleton.index(node)

            points[node]["xy"] = data[:2]

            # Score is the third element
            idx = 2
            if len(data) > idx:
                points[node]["score"] = data[idx]
                idx += 1

            # Visibility is the fourth element (or third if no score)
            if len(data) > idx:
                points[node]["visible"] = data[idx]
            else:
                points[node]["visible"] = ~np.isnan(data[0])

            idx += 1
            # Completeness is the fifth element (or fourth if no score)
            if len(data) > idx:
                points[node]["complete"] = data[idx]

        return points


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

    def matches(self, other: "Track", method: str = "name") -> bool:
        """Check if this track matches another track.

        Args:
            other: Another track to compare with.
            method: Matching method - "name" (match by name) or "identity"
                (match by object identity).

        Returns:
            True if the tracks match according to the specified method.
        """
        if method == "name":
            return self.name == other.name
        elif method == "identity":
            return self is other
        else:
            raise ValueError(f"Unknown matching method: {method}")

    def similarity_to(self, other: "Track") -> dict[str, any]:
        """Calculate similarity metrics with another track.

        Args:
            other: Another track to compare with.

        Returns:
            A dictionary with similarity metrics:
            - 'same_name': Whether the tracks have the same name
            - 'same_identity': Whether the tracks are the same object
            - 'name_similarity': Simple string similarity score (0-1)
        """
        # Calculate simple string similarity
        if self.name and other.name:
            # Simple character overlap similarity
            common_chars = set(self.name.lower()) & set(other.name.lower())
            all_chars = set(self.name.lower()) | set(other.name.lower())
            name_similarity = len(common_chars) / len(all_chars) if all_chars else 0
        else:
            name_similarity = 1.0 if self.name == other.name else 0.0

        return {
            "same_name": self.name == other.name,
            "same_identity": self is other,
            "name_similarity": name_similarity,
        }


@attrs.define(auto_attribs=True, slots=True, eq=False)
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

    points: PointsArray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
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
        points = PointsArray.empty(len(skeleton))
        points["name"] = skeleton.node_names

        return cls(
            points=points,
            skeleton=skeleton,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )

    @classmethod
    def _convert_points(
        cls, points_data: np.ndarray | dict | list, skeleton: Skeleton
    ) -> PointsArray:
        """Convert points to a structured numpy array if needed."""
        if isinstance(points_data, dict):
            return PointsArray.from_dict(points_data, skeleton)
        elif isinstance(points_data, (list, np.ndarray)):
            if isinstance(points_data, list):
                points_data = np.array(points_data)

            points = PointsArray.from_array(points_data)
            points["name"] = skeleton.node_names
            return points
        else:
            raise ValueError("points must be a numpy array or dictionary.")

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
        if not isinstance(self.points, PointsArray):
            self.points = self._convert_points(self.points, self.skeleton)

        # Ensure points have node names
        if "name" in self.points.dtype.names and not all(self.points["name"]):
            self.points["name"] = self.skeleton.node_names

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

    def __getitem__(self, node: Union[int, str, Node]) -> np.ndarray:
        """Return the point associated with a node."""
        if type(node) is not int:
            node = self.skeleton.index(node)

        return self.points[node]

    def __setitem__(self, node: Union[int, str, Node], value):
        """Set the point associated with a node.

        Args:
            node: The node to set the point for. Can be an integer index, string name,
                or Node object.
            value: A tuple or array-like of length 2 containing (x, y) coordinates.

        Notes:
            This sets the point coordinates and marks the point as visible.
        """
        if type(node) is not int:
            node = self.skeleton.index(node)

        if len(value) < 2:
            raise ValueError("Value must have at least 2 elements (x, y)")

        self.points[node]["xy"] = value[:2]
        self.points[node]["visible"] = True

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
        new_points = PointsArray.empty(len(self.skeleton))
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
        new_points = PointsArray.empty(len(self.skeleton))
        new_points[new_node_inds] = self.points[old_node_inds]
        self.points = new_points
        self.points["name"] = self.skeleton.node_names

    def same_pose_as(self, other: "Instance", tolerance: float = None) -> bool:
        """Check if this instance has the same pose as another instance.

        Args:
            other: Another instance to compare with.
            tolerance: Maximum distance (in pixels) between corresponding points
                for them to be considered the same. If None (default), uses exact
                comparison including proper NaN handling.

        Returns:
            True if the instances have the same pose within tolerance, False otherwise.

        Notes:
            Two instances are considered to have the same pose if:
            - They have the same skeleton structure
            - When tolerance is None: All coordinates match exactly (including NaN)
            - When tolerance is specified: All visible points are within tolerance
              distance and NaN patterns match exactly
        """
        # Check skeleton compatibility
        if not self.skeleton.matches(other.skeleton):
            return False

        if tolerance is None:
            # Exact comparison using numpy arrays with proper NaN handling
            return np.array_equal(self.numpy(), other.numpy(), equal_nan=True)
        else:
            # Tolerance-based comparison with proper NaN handling
            self_array = self.numpy()
            other_array = other.numpy()

            # First, check if NaN patterns match exactly
            self_nan_mask = np.isnan(self_array)
            other_nan_mask = np.isnan(other_array)
            if not np.array_equal(self_nan_mask, other_nan_mask):
                return False

            # Get mask for non-NaN values
            non_nan_mask = ~self_nan_mask

            # If all values are NaN, they're considered equal
            if not non_nan_mask.any():
                return True

            # Calculate distances only for non-NaN points
            self_pts = self_array[non_nan_mask]
            other_pts = other_array[non_nan_mask]

            # Reshape to handle the coordinate pairs properly
            self_pts = self_pts.reshape(-1, 2)
            other_pts = other_pts.reshape(-1, 2)

            distances = np.linalg.norm(self_pts - other_pts, axis=1)

            return np.all(distances <= tolerance)

    def same_identity_as(self, other: "Instance") -> bool:
        """Check if this instance has the same identity (track) as another instance.

        Args:
            other: Another instance to compare with.

        Returns:
            True if both instances have the same track identity, False otherwise.

        Notes:
            Instances have the same identity if they share the same Track object
            (by identity, not just by name).
        """
        if self.track is None or other.track is None:
            return False
        return self.track is other.track

    def overlaps_with(self, other: "Instance", iou_threshold: float = 0.5) -> bool:
        """Check if this instance overlaps with another based on bounding box IoU.

        Args:
            other: Another instance to compare with.
            iou_threshold: Minimum IoU (Intersection over Union) value to consider
                the instances as overlapping.

        Returns:
            True if the instances overlap above the threshold, False otherwise.

        Notes:
            Overlap is computed using the bounding boxes of visible points.
            If either instance has no visible points, they don't overlap.
        """
        # Get visible points for both instances
        self_visible = self.points["visible"]
        other_visible = other.points["visible"]

        if not self_visible.any() or not other_visible.any():
            return False

        # Calculate bounding boxes
        self_pts = self.points["xy"][self_visible]
        other_pts = other.points["xy"][other_visible]

        self_bbox = np.array(
            [
                [np.min(self_pts[:, 0]), np.min(self_pts[:, 1])],  # min x, y
                [np.max(self_pts[:, 0]), np.max(self_pts[:, 1])],  # max x, y
            ]
        )

        other_bbox = np.array(
            [
                [np.min(other_pts[:, 0]), np.min(other_pts[:, 1])],
                [np.max(other_pts[:, 0]), np.max(other_pts[:, 1])],
            ]
        )

        # Calculate intersection
        intersection_min = np.maximum(self_bbox[0], other_bbox[0])
        intersection_max = np.minimum(self_bbox[1], other_bbox[1])

        if np.any(intersection_min >= intersection_max):
            # No intersection
            return False

        intersection_area = np.prod(intersection_max - intersection_min)

        # Calculate union
        self_area = np.prod(self_bbox[1] - self_bbox[0])
        other_area = np.prod(other_bbox[1] - other_bbox[0])
        union_area = self_area + other_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou >= iou_threshold

    def bounding_box(self) -> Optional[np.ndarray]:
        """Get the bounding box of visible points.

        Returns:
            A numpy array of shape (2, 2) with [[min_x, min_y], [max_x, max_y]],
            or None if there are no visible points.
        """
        visible = self.points["visible"]
        if not visible.any():
            return None

        pts = self.points["xy"][visible]
        return np.array(
            [
                [np.min(pts[:, 0]), np.min(pts[:, 1])],
                [np.max(pts[:, 0]), np.max(pts[:, 1])],
            ]
        )


@attrs.define(eq=False)
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

    points: PredictedPointsArray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
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
        """Create an empty instance with no points."""
        points = PredictedPointsArray.empty(len(skeleton))
        points["name"] = skeleton.node_names

        return cls(
            points=points,
            skeleton=skeleton,
            score=score,
            track=track,
            tracking_score=tracking_score,
            from_predicted=from_predicted,
        )

    @classmethod
    def _convert_points(
        cls, points_data: np.ndarray | dict | list, skeleton: Skeleton
    ) -> PredictedPointsArray:
        """Convert points to a structured numpy array if needed."""
        if isinstance(points_data, dict):
            return PredictedPointsArray.from_dict(points_data, skeleton)
        elif isinstance(points_data, (list, np.ndarray)):
            if isinstance(points_data, list):
                points_data = np.array(points_data)

            points = PredictedPointsArray.from_array(points_data)
            points["name"] = skeleton.node_names
            return points
        else:
            raise ValueError("points must be a numpy array or dictionary.")

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
        """Create a predicted instance object from a numpy array."""
        points = cls._convert_points(points_data, skeleton)
        if point_scores is not None:
            points["score"] = point_scores

        return cls(
            points=points,
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
        new_points = PredictedPointsArray.empty(len(self.skeleton))
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
            This method will update the `PredictedInstance.skeleton` attribute and the
            `PredictedInstance.points` attribute in place (a copy is made of the points
            array).

            It is recommended to use `Labels.replace_skeleton` instead of this method if
            more flexible node mapping is required.
        """
        # Update skeleton object.
        self.skeleton = new_skeleton

        # Get node names with replacements from node map if possible.
        old_node_names = self.points["name"].tolist()
        if node_names_map is not None:
            old_node_names = [node_names_map.get(node, node) for node in old_node_names]

        # Find correspondences.
        new_node_inds, old_node_inds = self.skeleton.match_nodes(old_node_names)

        # Update the points.
        new_points = PredictedPointsArray.empty(len(self.skeleton))
        new_points[new_node_inds] = self.points[old_node_inds]
        self.points = new_points
        self.points["name"] = self.skeleton.node_names

    def __getitem__(self, node: Union[int, str, Node]) -> np.ndarray:
        """Return the point associated with a node."""
        # Inherit from Instance.__getitem__
        return super().__getitem__(node)

    def __setitem__(self, node: Union[int, str, Node], value):
        """Set the point associated with a node.

        Args:
            node: The node to set the point for. Can be an integer index, string name,
                or Node object.
            value: A tuple or array-like of length 2 or 3 containing (x, y) coordinates
                and optionally a confidence score. If the score is not provided, it
                defaults to 1.0.

        Notes:
            This sets the point coordinates, score, and marks the point as visible.
        """
        if type(node) is not int:
            node = self.skeleton.index(node)

        if len(value) < 2:
            raise ValueError("Value must have at least 2 elements (x, y)")

        self.points[node]["xy"] = value[:2]

        # Set score if provided, otherwise default to 1.0
        if len(value) >= 3:
            self.points[node]["score"] = value[2]
        else:
            self.points[node]["score"] = 1.0

        self.points[node]["visible"] = True
