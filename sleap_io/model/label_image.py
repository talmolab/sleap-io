"""Data structure for integer label image annotations.

Label images represent per-pixel object segmentation for a single video frame,
where each pixel value encodes which object occupies that pixel. This is the
standard output format of instance segmentation tools like Cellpose and StarDist.

Unlike binary ``SegmentationMask`` objects (one mask per object), a single
``LabelImage`` efficiently stores all objects for a frame in one dense integer
array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Self

import attrs
import numpy as np
from attrs import Factory

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.video import Video


@attrs.define(eq=False)
class LabelImage:
    """Per-pixel object segmentation for a single video frame.

    Each pixel is either background (0) or belongs to a tracked, categorized
    object. The integer values in ``data`` are an internal encoding. Use the
    track-centric API (``__getitem__``, ``tracks``, ``items``) to query objects
    without dealing with raw label IDs.

    Attributes:
        data: Integer array of shape ``(H, W)`` with dtype int32. ``0`` is
            background, positive values are object IDs.
        objects: Mapping from label ID to object metadata. Defines what each
            non-zero pixel value represents. Label IDs not in this dict are
            treated as having default (empty) metadata.
        video: Associated ``Video``.
        frame_idx: Frame index. ``None`` means static (applies to all frames).
        source: Source identifier string.
        scale: Resolution ratio ``(sx, sy)`` where ``sx = label_width / image_width``
            and ``sy = label_height / image_height``. ``(1.0, 1.0)`` means full
            resolution. ``(0.5, 0.5)`` means half resolution. Coordinate mapping:
            ``image_coord = label_coord / scale + offset``.
        offset: Origin ``(x, y)`` of the label image in image pixel coordinates.
    """

    @attrs.define
    class Info:
        """Metadata for one segmented object within a ``LabelImage``.

        Attributes:
            track: Track identity for cross-frame association. ``None`` if
                untracked.
            category: Semantic class label (e.g., ``"neuron"``, ``"glia"``).
            name: Human-readable name (e.g., ``"cell_042"``).
            instance: Linked pose ``Instance``, if any.
        """

        track: "Track | None" = None
        category: str = ""
        name: str = ""
        instance: "Instance | None" = None
        score: float | None = None

    data: np.ndarray = attrs.field()
    objects: dict[int, Info] = Factory(dict)
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    source: str = attrs.field(default="")
    scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    @property
    def is_predicted(self) -> bool:
        """Whether this label image is a model prediction."""
        return isinstance(self, PredictedLabelImage)

    def __attrs_post_init__(self):
        """Validate and normalize data array on construction."""
        if type(self) is LabelImage:
            raise TypeError(
                "LabelImage is abstract. Use UserLabelImage or PredictedLabelImage."
            )
        if self.data.ndim != 2:
            raise ValueError(f"LabelImage data must be 2D, got shape {self.data.shape}")
        if np.any(self.data < 0):
            raise ValueError("LabelImage data must not contain negative values.")
        if self.data.dtype != np.int32:
            self.data = self.data.astype(np.int32)
        if self.scale[0] <= 0 or self.scale[1] <= 0:
            raise ValueError(f"Scale values must be positive, got {self.scale}.")

    @property
    def height(self) -> int:
        """Height of the label image in pixels."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Width of the label image in pixels."""
        return self.data.shape[1]

    @property
    def has_spatial_transform(self) -> bool:
        """Whether this label image has non-default scale or offset."""
        return self.scale != (1.0, 1.0) or self.offset != (0.0, 0.0)

    @property
    def image_extent(self) -> tuple[int, int]:
        """Image-space ``(height, width)`` this label image covers (excluding offset).

        Computed as ``(int(height / scale_y), int(width / scale_x))``.
        """
        return (
            int(self.height / self.scale[1]),
            int(self.width / self.scale[0]),
        )

    def resampled(self, target_height: int, target_width: int) -> Self:
        """Return a new label image resampled to the target dimensions.

        The returned label image has ``scale=(1.0, 1.0)`` and
        ``offset=(0.0, 0.0)`` with the data resized using nearest-neighbor
        interpolation to preserve label IDs.

        Args:
            target_height: Target height in pixels.
            target_width: Target width in pixels.

        Returns:
            A new label image of the same concrete type with resampled data.
        """
        from sleap_io.model.mask import _resize_nearest

        resized = _resize_nearest(self.data, target_height, target_width)
        kwargs: dict = dict(
            data=resized,
            objects={lid: attrs.evolve(info) for lid, info in self.objects.items()},
            video=self.video,
            frame_idx=self.frame_idx,
            source=self.source,
            scale=(1.0, 1.0),
            offset=(0.0, 0.0),
        )
        if isinstance(self, PredictedLabelImage):
            kwargs["score"] = self.score
            if self.score_map is not None:
                kwargs["score_map"] = _resize_nearest(
                    self.score_map, target_height, target_width
                )
            kwargs["score_map_scale"] = (1.0, 1.0)
            kwargs["score_map_offset"] = (0.0, 0.0)
        return type(self)(**kwargs)

    @property
    def n_objects(self) -> int:
        """Number of unique non-zero labels present in data."""
        return len(self.label_ids)

    @property
    def label_ids(self) -> np.ndarray:
        """Sorted array of unique non-zero label values in data."""
        ids = np.unique(self.data)
        return ids[ids > 0]

    @property
    def tracks(self) -> list["Track"]:
        """Tracks present in this frame (from objects with non-None track)."""
        return [
            self.objects[lid].track
            for lid in sorted(self.objects)
            if self.objects[lid].track is not None
        ]

    @property
    def categories(self) -> set[str]:
        """Unique non-empty category strings present."""
        return {info.category for info in self.objects.values() if info.category != ""}

    def __getitem__(self, track: "Track") -> np.ndarray:
        """Get binary (H, W) mask for a tracked object.

        Args:
            track: The Track to look up.

        Returns:
            Boolean array of shape (H, W).

        Raises:
            KeyError: If the track is not present in this frame.
        """
        for label_id, info in self.objects.items():
            if info.track is track:
                return self.data == label_id
        raise KeyError(f"Track {track} not found in this LabelImage.")

    def __contains__(self, track: "Track") -> bool:
        """Whether this track has pixels in this frame."""
        return any(info.track is track for info in self.objects.values())

    def get_track_mask(self, track: "Track") -> np.ndarray:
        """Get binary (H, W) mask for a tracked object. Same as __getitem__."""
        return self[track]

    def get_category_mask(self, category: str) -> np.ndarray:
        """Union mask of all objects matching a category.

        Args:
            category: Semantic class label to filter by.

        Returns:
            Boolean array of shape (H, W). All-False if no objects match.
        """
        label_ids = [
            lid for lid, info in self.objects.items() if info.category == category
        ]
        if not label_ids:
            return np.zeros((self.height, self.width), dtype=bool)
        return np.isin(self.data, label_ids)

    def items(self) -> Iterator[tuple["Track | None", str, np.ndarray]]:
        """Iterate over objects as (track, category, mask) tuples.

        Yields one tuple per unique non-zero label ID, in sorted label order.
        """
        for label_id in np.sort(self.label_ids):
            lid = int(label_id)
            info = self.objects.get(lid, LabelImage.Info())
            yield info.track, info.category, self.data == lid

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        tracks: "dict[int, Track] | list[Track] | None" = None,
        categories: dict[int, str] | list[str] | None = None,
        **kwargs,
    ) -> "LabelImage":
        """Create from an integer array.

        Args:
            data: (H, W) integer array. Cast to int32.
            tracks: Maps label IDs to Tracks.

                - ``None``: auto-creates one Track per unique non-zero label.
                  Track.name is set to the string of the label ID.
                - ``list``: positional — ``tracks[i]`` maps to label ``i + 1``.
                - ``dict``: explicit ``{label_id: Track}`` mapping.
            categories: Same pattern as tracks, for category strings.

                - ``None``: no categories set.
                - ``list``: positional — ``categories[i]`` maps to label ``i + 1``.
                - ``dict``: explicit ``{label_id: category}`` mapping.
            **kwargs: Passed to the LabelImage constructor (video, frame_idx,
                source).

        Returns:
            A ``LabelImage`` with populated ``objects`` dict.
        """
        from sleap_io.model.instance import Track

        data = np.asarray(data, dtype=np.int32)
        unique_ids = np.unique(data)
        unique_ids = unique_ids[unique_ids > 0]

        # Build track mapping
        track_map: dict[int, Track] = {}
        if tracks is None:
            for lid in unique_ids:
                track_map[int(lid)] = Track(name=str(int(lid)))
        elif isinstance(tracks, list):
            for i, t in enumerate(tracks):
                track_map[i + 1] = t
        else:
            track_map = dict(tracks)

        # Build category mapping
        cat_map: dict[int, str] = {}
        if categories is None:
            pass  # No categories
        elif isinstance(categories, list):
            for i, c in enumerate(categories):
                cat_map[i + 1] = c
        else:
            cat_map = dict(categories)

        # Build objects dict
        objects: dict[int, LabelImage.Info] = {}
        all_ids = set(int(lid) for lid in unique_ids) | set(track_map) | set(cat_map)
        for lid in sorted(all_ids):
            objects[lid] = LabelImage.Info(
                track=track_map.get(lid),
                category=cat_map.get(lid, ""),
            )

        return cls(data=data, objects=objects, **kwargs)

    @classmethod
    def from_masks(
        cls,
        masks: list["SegmentationMask"],
        **kwargs,
    ) -> "LabelImage":
        """Compose from binary SegmentationMasks.

        Each mask becomes one object with a unique label ID. Track, category,
        and name are inherited from each mask's metadata. Overlapping pixels
        are assigned to the last mask in the list.

        All masks must share the same ``scale`` and ``offset``. The resulting
        ``LabelImage`` inherits the shared spatial metadata (unless overridden
        via ``**kwargs``).

        Args:
            masks: Binary masks. Must all have the same height, width, scale,
                and offset.
            **kwargs: Passed to the LabelImage constructor.

        Returns:
            A ``LabelImage`` composing all masks.

        Raises:
            ValueError: If masks have inconsistent shapes, scale/offset, or the
                list is empty.
        """
        if not masks:
            raise ValueError("Cannot create LabelImage from empty mask list.")

        height, width = masks[0].height, masks[0].width
        for m in masks[1:]:
            if m.height != height or m.width != width:
                raise ValueError(
                    f"All masks must have the same shape. "
                    f"Expected ({height}, {width}), got ({m.height}, {m.width})."
                )

        scales = {m.scale for m in masks}
        offsets = {m.offset for m in masks}
        if len(scales) > 1 or len(offsets) > 1:
            raise ValueError(
                "All masks must share the same scale and offset. "
                "Use mask.resampled() to align them first."
            )

        # Inherit spatial metadata from masks unless explicitly overridden.
        if "scale" not in kwargs:
            kwargs["scale"] = masks[0].scale
        if "offset" not in kwargs:
            kwargs["offset"] = masks[0].offset

        data = np.zeros((height, width), dtype=np.int32)
        objects: dict[int, LabelImage.Info] = {}

        for i, mask in enumerate(masks):
            label_id = i + 1
            data[mask.data] = label_id
            objects[label_id] = LabelImage.Info(
                track=mask.track,
                category=mask.category,
                name=mask.name,
                instance=mask.instance,
            )

        return cls(data=data, objects=objects, **kwargs)

    def to_masks(self) -> list["SegmentationMask"]:
        """Decompose into per-object binary SegmentationMasks.

        Returns one SegmentationMask per unique non-zero label. Each mask
        inherits track, category, name, instance, video, frame_idx, source,
        scale, and offset from the ``LabelImage``.

        Returns:
            A list of ``SegmentationMask`` objects, one per object.
        """
        from sleap_io.model.mask import UserSegmentationMask

        result = []
        for label_id in np.sort(self.label_ids):
            lid = int(label_id)
            info = self.objects.get(lid, LabelImage.Info())
            binary_mask = self.data == lid
            result.append(
                UserSegmentationMask.from_numpy(
                    binary_mask,
                    name=info.name,
                    category=info.category,
                    track=info.track,
                    instance=info.instance,
                    video=self.video,
                    frame_idx=self.frame_idx,
                    source=self.source,
                    scale=self.scale,
                    offset=self.offset,
                )
            )
        return result


@attrs.define(eq=False)
class UserLabelImage(LabelImage):
    """Human-annotated label image."""

    pass


@attrs.define(eq=False)
class PredictedLabelImage(LabelImage):
    """Model-predicted label image with confidence score.

    Attributes:
        score: Image-level confidence score (0-1).
        score_map: Optional dense pixel-level confidence map of shape (H, W)
            as float32. This can be large and is stored separately in the SLP
            format. If ``None``, only per-object scores in ``Info`` are available.
        score_map_scale: Resolution ratio ``(sx, sy)`` for the score map,
            independent of the label image's own ``scale``.
        score_map_offset: Origin ``(x, y)`` of the score map in image pixel
            coordinates.
    """

    score: float = attrs.field(default=0.0)
    score_map: np.ndarray | None = attrs.field(default=None)
    score_map_scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    score_map_offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))
