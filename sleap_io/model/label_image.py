"""Data structure for integer label image annotations.

Label images represent per-pixel object segmentation for a single video frame,
where each pixel value encodes which object occupies that pixel. This is the
standard output format of instance segmentation tools like Cellpose and StarDist.

Unlike binary ``SegmentationMask`` objects (one mask per object), a single
``LabelImage`` efficiently stores all objects for a frame in one dense integer
array.

**When to use LabelImage vs SegmentationMask:**

- Use ``LabelImage`` when you have a dense integer array from a segmentation
  tool (Cellpose, StarDist, COCO panoptic) where each pixel value identifies
  an object. One ``LabelImage`` per frame stores all objects at once.
- Use ``SegmentationMask`` when you have individual binary masks per object
  (e.g., from Mask R-CNN, manual annotation, or ROI-based workflows). Each
  mask is stored separately with RLE compression.
- Use ``LabelImage.from_binary_masks()`` to create a label image directly from
  per-object binary numpy arrays (e.g., from SAM or Mask R-CNN output).
- To convert between them, use ``LabelImage.to_masks()`` and
  ``LabelImage.from_masks()``.

For TIFF I/O of label images, see ``sleap_io.load_label_images()`` and
``sleap_io.save_label_images()``.

See Also:
    ``sleap_io.model.mask``: Binary segmentation masks (one per object).
"""

from __future__ import annotations

import copy
import sys
from typing import TYPE_CHECKING, Callable, Iterator, Literal

import attrs
import numpy as np
from attrs import Factory

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.mask import SegmentationMask


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
        source: Source identifier string.
        scale: Resolution ratio ``(sx, sy)`` where ``sx = label_width / image_width``
            and ``sy = label_height / image_height``. ``(1.0, 1.0)`` means full
            resolution. ``(0.5, 0.5)`` means half resolution. Coordinate mapping:
            ``image_coord = label_coord / scale + offset``.
        offset: Origin ``(x, y)`` of the label image in image pixel coordinates.

    See Also:
        ``SegmentationMask``: Per-object binary masks (one mask per object).
        ``sleap_io.load_label_images``: Load label images from TIFF files.
        ``sleap_io.save_label_images``: Save label images to TIFF files.
    """

    @attrs.define
    class Info:
        """Metadata for one segmented object within a ``LabelImage``.

        Attributes:
            track: Track identity for cross-frame association. ``None`` if
                untracked.
            tracking_score: Confidence of the track identity assignment.
                ``None`` if unassigned or manually assigned.
            category: Semantic class label (e.g., ``"neuron"``, ``"glia"``).
            name: Human-readable name (e.g., ``"cell_042"``).
            instance: Linked pose ``Instance``, if any.
        """

        track: "Track | None" = None
        tracking_score: float | None = None
        category: str = ""
        name: str = ""
        instance: "Instance | None" = None
        score: float | None = None

        # Private: deferred instance index for lazy loading. When label images
        # are read from a file without materialized instances (e.g., lazy mode),
        # this stores the raw instance_idx so it can be resolved later or
        # written back as-is.
        _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)

    _data: "np.ndarray | None" = attrs.field(default=None, alias="data")
    objects: dict[int, Info] = Factory(dict)
    source: str = attrs.field(default="")
    scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    # Private: lazy loading support. When set, data is decompressed on first
    # access via the .data property and cached. The loader is cleared after use.
    _lazy_loader: "Callable[[], np.ndarray] | None" = attrs.field(
        default=None, init=False, repr=False, eq=False
    )
    # Private: cached dimensions from metadata (avoids triggering lazy load for
    # height/width queries). Set by the I/O layer after construction.
    _height: int = attrs.field(default=0, init=False, repr=False, eq=False)
    _width: int = attrs.field(default=0, init=False, repr=False, eq=False)

    @property
    def data(self) -> np.ndarray:
        """Integer array of shape ``(H, W)`` with dtype int32.

        ``0`` is background, positive values are object IDs. When the label
        image was loaded lazily, the pixel data is decompressed on first access
        and cached for subsequent reads.
        """
        if self._data is None:
            if self._lazy_loader is not None:
                self._data = self._lazy_loader()
                self._lazy_loader = None
                self._validate_data()
            else:
                raise ValueError("LabelImage has no data and no lazy loader.")
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value
        self._lazy_loader = None
        if value is not None:
            self._height = value.shape[0]
            self._width = value.shape[1]

    @property
    def is_predicted(self) -> bool:
        """Whether this label image is a model prediction."""
        return isinstance(self, PredictedLabelImage)

    def _validate_data(self) -> None:
        """Validate and normalize the data array.

        Called eagerly from ``__attrs_post_init__`` when data is provided at
        construction time, or lazily on first ``.data`` access when a
        ``_lazy_loader`` is used.
        """
        if self._data.ndim != 2:
            raise ValueError(
                f"LabelImage data must be 2D, got shape {self._data.shape}"
            )
        if np.any(self._data < 0):
            raise ValueError("LabelImage data must not contain negative values.")
        if self._data.dtype != np.int32:
            self._data = self._data.astype(np.int32)
        self._height = self._data.shape[0]
        self._width = self._data.shape[1]

    def __attrs_post_init__(self):
        """Validate and normalize data array on construction."""
        if type(self) is LabelImage:
            raise TypeError(
                "LabelImage is abstract. Use UserLabelImage or PredictedLabelImage."
            )
        if self._data is not None:
            self._validate_data()
        # When _data is None, validation is deferred to first .data access.
        if self.scale[0] <= 0 or self.scale[1] <= 0:
            raise ValueError(f"Scale values must be positive, got {self.scale}.")

    def __deepcopy__(self, memo: dict) -> "LabelImage":
        """Deep copy that materializes lazy data before copying.

        This is necessary because lazy loaders capture h5py dataset references
        which cannot be pickled/deepcopied.
        """
        # Materialize lazy data before copying (h5py refs can't survive deepcopy).
        if self._data is not None:
            data = self._data.copy()
        elif self._lazy_loader is not None:
            data = self.data.copy()  # Triggers lazy load, then copy
        else:
            data = None
        objects = {lid: copy.deepcopy(info, memo) for lid, info in self.objects.items()}

        kwargs: dict = dict(
            data=data,
            objects=objects,
            source=self.source,
            scale=self.scale,
            offset=self.offset,
        )
        if isinstance(self, PredictedLabelImage):
            sm = self.score_map
            kwargs["score"] = self.score
            kwargs["score_map"] = sm.copy() if sm is not None else None
            kwargs["score_map_scale"] = self.score_map_scale
            kwargs["score_map_offset"] = self.score_map_offset
        result = type(self)(**kwargs)
        memo[id(self)] = result
        return result

    @property
    def height(self) -> int:
        """Height of the label image in pixels."""
        if self._height > 0:
            return self._height
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Width of the label image in pixels."""
        if self._width > 0:
            return self._width
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
        objects: dict[int, LabelImage.Info] = {}
        for lid, info in self.objects.items():
            new_info = attrs.evolve(info)
            # Carry the deferred instance index through (init=False, so it is not
            # reproduced by attrs.evolve and must be set after construction;
            # mirrors how __deepcopy__ preserves the lazy association).
            new_info._instance_idx = info._instance_idx
            objects[lid] = new_info
        kwargs: dict = dict(
            data=resized,
            objects=objects,
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
        create_tracks: bool = False,
        **kwargs,
    ) -> "LabelImage":
        """Create from an integer array.

        Args:
            data: (H, W) integer array. Cast to int32.
            tracks: Maps label IDs to Tracks.

                - ``None``: no tracks unless ``create_tracks=True``.
                - ``list``: positional — ``tracks[i]`` maps to label ``i + 1``.
                - ``dict``: explicit ``{label_id: Track}`` mapping. When
                  combined with ``create_tracks=True``, the dict is used as
                  a shared accumulator — existing entries are reused and new
                  entries are added for unseen label IDs (mutated in place).
            categories: Same pattern as tracks, for category strings.

                - ``None``: no categories set.
                - ``list``: positional — ``categories[i]`` maps to label ``i + 1``.
                - ``dict``: explicit ``{label_id: category}`` mapping.
            create_tracks: If ``True`` and ``tracks`` is ``None``, auto-create
                one Track per unique non-zero label with Track.name set to the
                string of the label ID. If ``True`` and ``tracks`` is a dict,
                create new Tracks for any label IDs not already in the dict
                (the dict is mutated in place to accumulate mappings across
                calls). Default is ``False``.
            **kwargs: Passed to the LabelImage constructor (
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
            if create_tracks:
                for lid in unique_ids:
                    track_map[int(lid)] = Track(name=str(int(lid)))
        elif isinstance(tracks, dict):
            track_map = dict(tracks)
            if create_tracks:
                # Accumulate: create new tracks for unseen IDs, mutate
                # the caller's dict in place so it stays in sync.
                for lid in unique_ids:
                    lid_int = int(lid)
                    if lid_int not in track_map:
                        new_track = Track(name=str(lid_int))
                        track_map[lid_int] = new_track
                        tracks[lid_int] = new_track
        elif isinstance(tracks, list):
            for i, t in enumerate(tracks):
                track_map[i + 1] = t

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

    @classmethod
    def from_binary_masks(
        cls,
        masks: "np.ndarray | list[np.ndarray]",
        label_ids: list[int] | None = None,
        tracks: "list[Track] | None" = None,
        categories: list[str] | None = None,
        names: list[str] | None = None,
        scores: list[float] | None = None,
        create_tracks: bool = False,
        **kwargs,
    ) -> "LabelImage":
        """Create a LabelImage from per-object binary mask arrays.

        This is a convenience constructor for workflows that produce per-object
        binary masks, such as SAM, Mask R-CNN, or other instance segmentation
        tools. Each binary mask becomes one object in the composited label image
        with a unique label ID (1, 2, ..., N, unless ``label_ids`` is provided).
        Overlapping pixels are assigned to the last mask in the list.

        Unlike ``from_masks()``, this takes raw numpy arrays instead of
        ``SegmentationMask`` objects, avoiding RLE encoding overhead.

        Args:
            masks: Per-object binary masks as an ``(N, H, W)`` array or a list
                of ``(H, W)`` arrays. Values are cast to bool (nonzero = True).
            label_ids: Optional list of positive integer label IDs, one per
                mask. ``label_ids[i]`` sets the pixel value for mask ``i``. If
                ``None`` (default), masks are numbered 1, 2, ..., N. All values
                must be positive (0 is background) and unique.
            tracks: List of ``Track`` objects, one per mask. ``tracks[i]`` is
                assigned to mask ``i`` (label ID ``label_ids[i]`` or ``i + 1``
                by default).
            categories: List of category strings, one per mask.
            names: List of human-readable name strings, one per mask.
            scores: List of per-object confidence scores, one per mask. Stored
                in ``Info.score`` for each object.
            create_tracks: If ``True`` and ``tracks`` is ``None``, auto-create
                a ``Track`` per mask with ``name=str(label_id)``.
            **kwargs: Passed to the ``LabelImage`` constructor (e.g.,
                ``source``, ``scale``, ``offset``). For
                ``PredictedLabelImage``, also accepts ``score``, ``score_map``.

        Returns:
            A ``LabelImage`` compositing all masks.

        Raises:
            ValueError: If ``masks`` is empty, shapes are inconsistent, or any
                parallel array has the wrong length.

        Example:
            Create a label image from SAM output::

                li = PredictedLabelImage.from_binary_masks(
                    sam_masks,          # (N, H, W) bool
                    tracks=[t1, t2],    # per-object tracks
                    scores=[0.95, 0.87],# per-object confidence
                    score=0.9,          # image-level confidence
                )

        See Also:
            :meth:`from_masks`: Create from ``SegmentationMask`` objects.
            :meth:`from_numpy`: Create from a pre-composited integer array.
        """
        from sleap_io.model.instance import Track

        # Normalize input to list of 2D arrays.
        if isinstance(masks, np.ndarray):
            if masks.ndim == 3:
                mask_list = [masks[i] for i in range(masks.shape[0])]
            elif masks.ndim == 2:
                mask_list = [masks]
            else:
                raise ValueError(
                    f"Expected 2D or 3D array, got {masks.ndim}D with shape "
                    f"{masks.shape}."
                )
        else:
            mask_list = list(masks)

        if not mask_list:
            raise ValueError("Cannot create LabelImage from empty mask list.")

        # Validate consistent shapes.
        height, width = mask_list[0].shape[0], mask_list[0].shape[1]
        for i, m in enumerate(mask_list[1:], 1):
            if m.shape[0] != height or m.shape[1] != width:
                raise ValueError(
                    f"All masks must have the same shape. "
                    f"Expected ({height}, {width}), got ({m.shape[0]}, {m.shape[1]}) "
                    f"at index {i}."
                )

        n = len(mask_list)

        # Validate parallel array lengths.
        for param_name, param in [
            ("label_ids", label_ids),
            ("tracks", tracks),
            ("categories", categories),
            ("names", names),
            ("scores", scores),
        ]:
            if param is not None and len(param) != n:
                raise ValueError(
                    f"{param_name} length ({len(param)}) must match number of "
                    f"masks ({n})."
                )

        # Validate label_ids semantics.
        if label_ids is not None:
            if any(lid <= 0 for lid in label_ids):
                raise ValueError(
                    "All label_ids must be positive (0 is reserved for background)."
                )
            if len(set(label_ids)) != len(label_ids):
                raise ValueError("label_ids must contain unique values.")

        # Build track list.
        if tracks is not None:
            track_list = tracks
        elif create_tracks:
            track_list = [
                Track(name=str(label_ids[i] if label_ids is not None else i + 1))
                for i in range(n)
            ]
        else:
            track_list = [None] * n

        # Composite masks and build objects dict.
        data = np.zeros((height, width), dtype=np.int32)
        objects: dict[int, LabelImage.Info] = {}

        for i, mask in enumerate(mask_list):
            label_id = label_ids[i] if label_ids is not None else i + 1
            data[np.asarray(mask, dtype=bool)] = label_id
            objects[label_id] = LabelImage.Info(
                track=track_list[i],
                category=categories[i] if categories is not None else "",
                name=names[i] if names is not None else "",
                score=scores[i] if scores is not None else None,
            )

        return cls(data=data, objects=objects, **kwargs)

    @classmethod
    def from_stack(
        cls,
        data: "np.ndarray | list[np.ndarray]",
        tracks: "dict[int, Track] | list[Track] | None" = None,
        categories: dict[int, str] | list[str] | None = None,
        create_tracks: bool = False,
        score: "float | list[float] | None" = None,
        score_map: np.ndarray | None = None,
        **kwargs,
    ) -> "list[LabelImage]":
        """Create label images from a stack of frames.

        This is the batch equivalent of ``from_numpy()``. It accepts a
        ``(T, H, W)`` array (or list of 2D arrays) and returns one
        ``LabelImage`` per frame with consistent ``Track`` objects shared
        across frames.

        Args:
            data: Integer label data as a 3D ``(T, H, W)`` array or a list
                of 2D ``(H, W)`` arrays. Cast to int32.
            tracks: Maps label IDs to Tracks (shared across all frames).

                - ``None``: no tracks unless ``create_tracks=True``.
                - ``list``: positional — ``tracks[i]`` maps to label
                  ``i + 1``.
                - ``dict``: explicit ``{label_id: Track}`` mapping.
            categories: Same pattern as tracks, for category strings.
            create_tracks: If ``True`` and ``tracks`` is ``None``,
                auto-create one ``Track`` per unique non-zero label ID
                found across all frames. The same ``Track`` object is
                shared across frames. Default is ``False``.
            score: Confidence score(s) for ``PredictedLabelImage``. A
                single float is broadcast to all frames; a list must have
                length ``T``. Defaults to ``0.0`` for all frames if
                ``None``. Ignored for ``UserLabelImage``.
            score_map: Optional ``(T, H, W)`` float32 array of per-pixel
                confidence maps. Sliced per frame. Ignored for
                ``UserLabelImage``.
            **kwargs: Passed to every frame's constructor (``source``,
                ``scale``, ``offset``).

        Returns:
            A list of ``LabelImage`` objects, one per frame.

        Raises:
            ValueError: If ``data`` is not 3D (or a list), or if
                ``score`` lengths don't match.

        Note:
            For loading label images from TIFF files (single, multi-page,
            or directory), use ``sleap_io.load_label_images()`` which
            handles file I/O and sidecar metadata. ``from_stack()`` is
            for converting in-memory numpy arrays (e.g., direct Cellpose
            output).

        Example::

            masks = np.stack(cellpose_masks)  # (T, H, W) int32
            label_images = sio.PredictedLabelImage.from_stack(
                masks,
                source="cellpose:nuclei",
                create_tracks=True,
                score=1.0,
            )
        """
        from sleap_io.model.instance import Track

        # Normalize input to list of 2D arrays
        if isinstance(data, np.ndarray):
            if data.ndim != 3:
                raise ValueError(
                    f"from_stack expects a (T, H, W) array, got shape "
                    f"{data.shape}. Use from_numpy() for a single frame."
                )
            frames = [data[t] for t in range(data.shape[0])]
        elif isinstance(data, list):
            frames = data
        else:
            raise ValueError(
                f"data must be a (T, H, W) numpy array or list of 2D "
                f"arrays, got {type(data).__name__}."
            )

        n_frames = len(frames)
        if n_frames == 0:
            return []

        # Collect unique non-zero IDs across all frames
        all_ids: set[int] = set()
        for frame in frames:
            ids = np.unique(frame)
            all_ids.update(int(i) for i in ids if i > 0)

        # Build global track map (shared across frames)
        track_map: dict[int, Track] = {}
        if tracks is None:
            if create_tracks:
                for lid in sorted(all_ids):
                    track_map[lid] = Track(name=str(lid))
        elif isinstance(tracks, list):
            for i, t in enumerate(tracks):
                track_map[i + 1] = t
        else:
            track_map = dict(tracks)

        # Build global category map
        cat_map: dict[int, str] = {}
        if categories is None:
            pass
        elif isinstance(categories, list):
            for i, c in enumerate(categories):
                cat_map[i + 1] = c
        else:
            cat_map = dict(categories)

        # Handle PredictedLabelImage-specific parameters
        is_predicted = issubclass(cls, PredictedLabelImage)
        scores: list[float] = []
        if is_predicted:
            if score is None:
                scores = [0.0] * n_frames
            elif isinstance(score, (int, float)):
                scores = [float(score)] * n_frames
            else:
                if len(score) != n_frames:
                    raise ValueError(
                        f"score list length ({len(score)}) must match "
                        f"number of frames ({n_frames})."
                    )
                scores = [float(s) for s in score]

        score_maps: list[np.ndarray | None] = [None] * n_frames
        if is_predicted and score_map is not None:
            if score_map.ndim == 3 and score_map.shape[0] == n_frames:
                score_maps = [score_map[t] for t in range(n_frames)]
            else:
                raise ValueError(
                    f"score_map must be (T, H, W) with T={n_frames}, "
                    f"got shape {score_map.shape}."
                )

        # Build per-frame LabelImages with shared Track objects
        result: list[LabelImage] = []
        for t, frame in enumerate(frames):
            frame_data = np.asarray(frame, dtype=np.int32)
            frame_ids = np.unique(frame_data)
            frame_ids = frame_ids[frame_ids > 0]

            objects: dict[int, LabelImage.Info] = {}
            for lid in frame_ids:
                lid_int = int(lid)
                objects[lid_int] = LabelImage.Info(
                    track=track_map.get(lid_int),
                    category=cat_map.get(lid_int, ""),
                )

            frame_kwargs = dict(kwargs)
            if is_predicted:
                frame_kwargs["score"] = scores[t]
                frame_kwargs["score_map"] = score_maps[t]

            result.append(cls(data=frame_data, objects=objects, **frame_kwargs))

        return result

    def to_masks(self) -> list["SegmentationMask"]:
        """Decompose into per-object binary SegmentationMasks.

        Returns one SegmentationMask per unique non-zero label. Each mask
        inherits track, category, name, instance, source, scale, and offset
        from the ``LabelImage``.

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
                    source=self.source,
                    scale=self.scale,
                    offset=self.offset,
                )
            )
        return result

    def to_bboxes(self) -> list["BoundingBox"]:
        """Extract tight bounding boxes for each object in the label image.

        Returns a list of ``BoundingBox`` objects (``UserBoundingBox`` or
        ``PredictedBoundingBox`` depending on whether this label image is
        predicted), one per non-zero label. Each bounding box inherits track,
        category, name, instance, and score from the corresponding
        ``self.objects`` entry.

        Bounding boxes are in image coordinates (respecting scale/offset).
        Label IDs present in ``objects`` but with no pixels in the data are
        skipped.

        Returns:
            A list of ``BoundingBox`` objects, one per object.
        """
        from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox

        data = self.data
        cls = PredictedBoundingBox if self.is_predicted else UserBoundingBox
        sx, sy = self.scale
        ox, oy = self.offset

        # Single-pass: find all foreground pixels at once.
        fg_rows, fg_cols = np.where(data > 0)
        if len(fg_rows) == 0:
            return []

        # Map sparse label IDs to dense indices and compute per-label bounds.
        fg_labels = data[fg_rows, fg_cols]
        unique_labels, inverse = np.unique(fg_labels, return_inverse=True)
        n = len(unique_labels)

        row_min = np.full(n, np.iinfo(np.intp).max, dtype=np.intp)
        row_max = np.full(n, np.iinfo(np.intp).min, dtype=np.intp)
        col_min = np.full(n, np.iinfo(np.intp).max, dtype=np.intp)
        col_max = np.full(n, np.iinfo(np.intp).min, dtype=np.intp)

        np.minimum.at(row_min, inverse, fg_rows)
        np.maximum.at(row_max, inverse, fg_rows)
        np.minimum.at(col_min, inverse, fg_cols)
        np.maximum.at(col_max, inverse, fg_cols)

        label_to_idx = {int(lid): i for i, lid in enumerate(unique_labels)}

        # Build BoundingBox objects using precomputed bounds.
        bboxes = []
        for lid, info in self.objects.items():
            idx = label_to_idx.get(lid)
            if idx is None:
                continue

            x1 = float(col_min[idx] / sx + ox)
            y1 = float(row_min[idx] / sy + oy)
            x2 = float((col_max[idx] + 1) / sx + ox)
            y2 = float((row_max[idx] + 1) / sy + oy)

            kwargs: dict = dict(
                track=info.track,
                instance=info.instance,
                category=info.category,
                name=info.name,
                source=self.source,
            )
            if self.is_predicted:
                kwargs["score"] = info.score if info.score is not None else self.score

            bboxes.append(cls.from_xyxy(x1, y1, x2, y2, **kwargs))

        return bboxes


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
            When loaded lazily, decompressed on first access and cached.
        score_map_scale: Resolution ratio ``(sx, sy)`` for the score map,
            independent of the label image's own ``scale``.
        score_map_offset: Origin ``(x, y)`` of the score map in image pixel
            coordinates.
    """

    score: float = attrs.field(default=0.0)
    _score_map: "np.ndarray | None" = attrs.field(default=None, alias="score_map")
    score_map_scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    score_map_offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    # Private: lazy loading support for score maps.
    _score_map_lazy_loader: "Callable[[], np.ndarray] | None" = attrs.field(
        default=None, init=False, repr=False, eq=False
    )

    @property
    def score_map(self) -> np.ndarray | None:
        """Optional dense pixel-level confidence map of shape ``(H, W)``."""
        if self._score_map is None and self._score_map_lazy_loader is not None:
            self._score_map = self._score_map_lazy_loader()
            self._score_map_lazy_loader = None
        return self._score_map

    @score_map.setter
    def score_map(self, value: np.ndarray | None) -> None:
        self._score_map = value
        self._score_map_lazy_loader = None


def normalize_label_ids(
    label_images: list[LabelImage],
    by: "Literal['track', 'category']" = "track",
) -> "dict[Track, int] | dict[str, int]":
    """Remap label IDs so each group gets a globally consistent ID.

    Rewrites ``.data`` arrays and ``.objects`` dicts in place so that the same
    Track (or category) always maps to the same pixel value across all frames.
    IDs are assigned 1, 2, 3, ... in order of first appearance.

    Args:
        label_images: Label images to normalize. Modified in place.
        by: Grouping key.

            - ``"track"``: Each unique ``Track`` object gets one ID.
              Identity is by Python object reference (``is``), not by
              name — ensure the same ``Track`` instance is shared across
              frames. Objects with ``track=None`` each get a unique ID.
            - ``"category"``: Each unique category string gets one ID.
              Within a frame, multiple objects with the same category
              merge into one pixel value (semantic segmentation).

    Returns:
        Mapping of group key to assigned label ID. Keys are ``Track`` objects
        when ``by="track"`` or category strings when ``by="category"``.
        Objects with ``track=None`` or empty category are not included.

    Raises:
        ValueError: If ``by`` is not ``"track"`` or ``"category"``.
    """
    if by not in ("track", "category"):
        raise ValueError(f"by must be 'track' or 'category', got {by!r}.")

    if not label_images:
        return {}

    if by == "track":
        return _normalize_by_track(label_images)
    else:
        return _normalize_by_category(label_images)


def _normalize_by_track(label_images: list[LabelImage]) -> dict:
    """Normalize label IDs so each Track gets a consistent ID."""
    # Phase 1: Assign IDs to tracked objects (first-appearance order).
    track_to_id: dict[int, int] = {}  # id(track) -> new_label_id
    track_by_identity: dict[int, Track] = {}  # id(track) -> Track
    next_id = 1
    for li in label_images:
        for lid in sorted(li.objects):
            info = li.objects[lid]
            if info.track is not None and id(info.track) not in track_to_id:
                track_to_id[id(info.track)] = next_id
                track_by_identity[id(info.track)] = info.track
                next_id += 1

    # Phase 2: Remap each label image.
    none_id = next_id  # Counter for untracked objects
    for li in label_images:
        if not li.objects:
            continue
        old_ids = sorted(li.objects.keys())
        max_old = max(old_ids)
        lut = np.zeros(max_old + 1, dtype=np.int32)
        new_objects: dict[int, LabelImage.Info] = {}

        for old_id in old_ids:
            info = li.objects[old_id]
            if info.track is not None:
                new_id = track_to_id[id(info.track)]
            else:
                new_id = none_id
                none_id += 1
            lut[old_id] = new_id
            new_objects[new_id] = info

        li.data = lut[li.data]
        li.objects = new_objects

    # Build return mapping: Track -> label_id
    return {track_by_identity[tid]: lid for tid, lid in track_to_id.items()}


def _normalize_by_category(label_images: list[LabelImage]) -> dict:
    """Normalize label IDs so each category string gets a consistent ID."""
    # Phase 1: Assign IDs to categories (first-appearance order).
    cat_to_id: dict[str, int] = {}
    next_id = 1
    for li in label_images:
        for lid in sorted(li.objects):
            cat = li.objects[lid].category
            if cat and cat not in cat_to_id:
                cat_to_id[cat] = next_id
                next_id += 1

    # Phase 2: Remap each label image.
    empty_id = next_id  # Counter for empty-category objects
    for li in label_images:
        if not li.objects:
            continue
        old_ids = sorted(li.objects.keys())
        max_old = max(old_ids)
        lut = np.zeros(max_old + 1, dtype=np.int32)
        new_objects: dict[int, LabelImage.Info] = {}

        for old_id in old_ids:
            info = li.objects[old_id]
            if info.category and info.category in cat_to_id:
                new_id = cat_to_id[info.category]
            else:
                new_id = empty_id
                empty_id += 1
            lut[old_id] = new_id
            # Multiple old objects may map to same new_id (semantic merge).
            if new_id not in new_objects:
                new_objects[new_id] = info

        li.data = lut[li.data]
        li.objects = new_objects

    return dict(cat_to_id)
