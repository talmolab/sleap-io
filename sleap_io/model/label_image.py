"""Data structure for integer label image annotations.

Label images represent per-pixel object segmentation for a single video frame,
where each pixel value encodes which object occupies that pixel. This is the
standard output format of instance segmentation tools like Cellpose and StarDist.

Unlike binary ``SegmentationMask`` objects (one mask per object), a single
``LabelImage`` efficiently stores all objects for a frame in one dense integer
array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

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

    data: np.ndarray = attrs.field()
    objects: dict[int, Info] = Factory(dict)
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    source: str = attrs.field(default="")

    def __attrs_post_init__(self):
        """Validate and normalize data array on construction."""
        if self.data.ndim != 2:
            raise ValueError(f"LabelImage data must be 2D, got shape {self.data.shape}")
        if np.any(self.data < 0):
            raise ValueError("LabelImage data must not contain negative values.")
        if self.data.dtype != np.int32:
            self.data = self.data.astype(np.int32)

    @property
    def height(self) -> int:
        """Height of the label image in pixels."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Width of the label image in pixels."""
        return self.data.shape[1]

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
            info.track
            for label_id in sorted(self.objects)
            for info in [self.objects[label_id]]
            if info.track is not None
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

        Args:
            masks: Binary masks. Must all have the same height and width.
            **kwargs: Passed to the LabelImage constructor.

        Returns:
            A ``LabelImage`` composing all masks.

        Raises:
            ValueError: If masks have inconsistent shapes or the list is empty.
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
        inherits track, category, name, instance, video, frame_idx, and source
        from the corresponding ``LabelImage.Info`` entry.

        Returns:
            A list of ``SegmentationMask`` objects, one per object.
        """
        from sleap_io.model.mask import SegmentationMask

        result = []
        for label_id in np.sort(self.label_ids):
            lid = int(label_id)
            info = self.objects.get(lid, LabelImage.Info())
            binary_mask = self.data == lid
            result.append(
                SegmentationMask.from_numpy(
                    binary_mask,
                    name=info.name,
                    category=info.category,
                    track=info.track,
                    instance=info.instance,
                    video=self.video,
                    frame_idx=self.frame_idx,
                    source=self.source,
                )
            )
        return result
