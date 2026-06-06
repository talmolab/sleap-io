"""Data structures for segmentation mask annotations.

Segmentation masks represent raster (per-pixel) annotations stored in
run-length encoded (RLE) format for compact storage. They can be converted
to and from numpy arrays and polygon representations.

Each ``SegmentationMask`` stores a single binary mask for one object. For
dense per-pixel segmentation where all objects are stored in one integer
array, see ``LabelImage`` in ``sleap_io.model.label_image``.

**When to use SegmentationMask vs LabelImage:**

- Use ``SegmentationMask`` when you have individual binary masks per object
  (e.g., from Mask R-CNN, manual annotation, or ROI-based workflows).
- Use ``LabelImage`` when you have a dense integer array from an instance
  segmentation tool (Cellpose, StarDist) where each pixel value identifies
  an object.
- To convert: ``LabelImage.to_masks()`` decomposes into per-object masks,
  and ``LabelImage.from_masks(masks)`` composes masks into a label image.

See Also:
    ``sleap_io.model.label_image``: Dense integer label images.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.roi import ROI


def _encode_rle(mask: np.ndarray) -> np.ndarray:
    """Encode a binary mask as run-length encoded counts.

    The encoding stores alternating run lengths of 0s and 1s, starting with 0s.
    The mask is read in row-major (C) order.

    Args:
        mask: A 2D boolean or uint8 numpy array.

    Returns:
        A 1D uint32 numpy array of run-length counts.
    """
    flat = mask.ravel().astype(np.uint8)
    if len(flat) == 0:
        return np.array([], dtype=np.uint32)

    # Find positions where value changes
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1

    # Build run lengths
    positions = np.concatenate([[0], change_indices, [len(flat)]])
    run_lengths = np.diff(positions).astype(np.uint32)

    # Ensure we start with a 0-run. If the first pixel is 1, prepend a 0-length run.
    if flat[0] == 1:
        run_lengths = np.concatenate([[np.uint32(0)], run_lengths])

    return run_lengths


def _decode_rle(rle_counts: np.ndarray, height: int, width: int) -> np.ndarray:
    """Decode run-length encoded counts to a binary mask.

    Args:
        rle_counts: A 1D uint32 array of alternating run lengths (starting with 0s).
        height: Height of the output mask.
        width: Width of the output mask.

    Returns:
        A 2D boolean numpy array of shape (height, width).
    """
    total = height * width
    if len(rle_counts) == 0:
        return np.zeros((height, width), dtype=bool)

    # Build alternating False/True values matching each run, then expand
    values = np.zeros(len(rle_counts), dtype=bool)
    values[1::2] = True  # Odd indices are 1-runs
    flat = np.repeat(values, rle_counts.astype(np.intp))[:total]

    return flat.reshape(height, width)


def _resize_nearest(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resize of a 2D array using numpy index mapping.

    Args:
        array: 2D array of shape (H, W).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized array of shape (target_h, target_w).
    """
    h, w = array.shape[:2]
    row_idx = (np.arange(target_h) * h / target_h).astype(int).clip(0, h - 1)
    col_idx = (np.arange(target_w) * w / target_w).astype(int).clip(0, w - 1)
    return array[np.ix_(row_idx, col_idx)]


@attrs.define(eq=False)
class SegmentationMask:
    """A segmentation mask stored as run-length encoded (RLE) data.

    Attributes:
        rle_counts: Run-length encoded counts as a uint32 array. Alternating runs
            of 0s and 1s, starting with 0s.
        height: Height of the mask in pixels.
        width: Width of the mask in pixels.
        name: Optional human-readable name for this mask.
        category: Optional category label (e.g., class name for detection).
        source: Optional string indicating the source of this annotation.
        track: Optional `Track` this mask is associated with.
        tracking_score: Confidence of the track identity assignment. ``None``
            if unassigned or manually assigned.
        instance: Optional `Instance` this mask is associated with.
        scale: Resolution ratio ``(sx, sy)`` where ``sx = mask_width / image_width``
            and ``sy = mask_height / image_height``. ``(1.0, 1.0)`` means full
            resolution. ``(0.5, 0.5)`` means half resolution (each mask pixel
            covers 2x2 image pixels). Coordinate mapping:
            ``image_coord = mask_coord / scale + offset``.
        offset: Origin ``(x, y)`` of the mask in image pixel coordinates.

    Notes:
        Masks use identity-based equality (two mask objects are only equal if they
        are the same object in memory).

    See Also:
        ``LabelImage``: Dense integer label images (all objects in one array).
    """

    rle_counts: np.ndarray = attrs.field()
    height: int = attrs.field()
    width: int = attrs.field()
    name: str = attrs.field(default="")
    category: str = attrs.field(default="")
    source: str = attrs.field(default="")
    track: "Track | None" = attrs.field(default=None)
    tracking_score: float | None = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)
    _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)
    scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    def __attrs_post_init__(self):
        """Validate that this class is not instantiated directly."""
        if type(self) is SegmentationMask:
            raise TypeError(
                "SegmentationMask is abstract. "
                "Use UserSegmentationMask or PredictedSegmentationMask."
            )
        if self.scale[0] <= 0 or self.scale[1] <= 0:
            raise ValueError(f"Scale values must be positive, got {self.scale}.")

    @property
    def is_predicted(self) -> bool:
        """Whether this mask is a model prediction."""
        return isinstance(self, PredictedSegmentationMask)

    @property
    def has_spatial_transform(self) -> bool:
        """Whether this mask has non-default scale or offset."""
        return self.scale != (1.0, 1.0) or self.offset != (0.0, 0.0)

    @property
    def image_extent(self) -> tuple[int, int]:
        """Image-space ``(height, width)`` this mask covers (excluding offset).

        Computed as ``(int(height / scale_y), int(width / scale_x))``.
        """
        return (
            int(self.height / self.scale[1]),
            int(self.width / self.scale[0]),
        )

    def resampled(self, target_height: int, target_width: int) -> Self:
        """Return a new mask resampled to the target dimensions.

        The returned mask has ``scale=(1.0, 1.0)`` and ``offset=(0.0, 0.0)``
        with the mask data resized using nearest-neighbor interpolation.

        Args:
            target_height: Target height in pixels.
            target_width: Target width in pixels.

        Returns:
            A new mask of the same concrete type with resampled data.
        """
        resized = _resize_nearest(self.data, target_height, target_width)
        kwargs: dict = dict(
            name=self.name,
            category=self.category,
            source=self.source,
            track=self.track,
            instance=self.instance,
            scale=(1.0, 1.0),
            offset=(0.0, 0.0),
        )
        if isinstance(self, UserSegmentationMask):
            # Preserve the provenance link so resampling a corrected mask keeps
            # its source prediction (mirrors track/instance preservation above).
            kwargs["from_predicted"] = self.from_predicted
        if isinstance(self, PredictedSegmentationMask):
            kwargs["score"] = self.score
            if self.score_map is not None:
                kwargs["score_map"] = _resize_nearest(
                    self.score_map, target_height, target_width
                )
            kwargs["score_map_scale"] = (1.0, 1.0)
            kwargs["score_map_offset"] = (0.0, 0.0)
        return type(self).from_numpy(resized, **kwargs)

    @classmethod
    def from_numpy(
        cls,
        mask: np.ndarray,
        stride: float | None = None,
        **kwargs,
    ) -> "SegmentationMask":
        """Create a SegmentationMask from a 2D numpy array.

        A ``SegmentationMask`` is binary by design (one object per mask). If a
        multi-class or multi-instance integer array is passed in, the internal
        RLE cast would silently drop all class/instance distinctions. This
        method rejects such inputs with a pointed error instead.

        Args:
            mask: A 2D boolean or ``{0, 1}`` integer array of shape
                ``(height, width)``. Inputs with more than one distinct
                non-zero value are rejected.
            stride: Convenience for setting isotropic scale. If provided, sets
                ``scale = (1/stride, 1/stride)``. Overrides ``scale`` in kwargs.
            **kwargs: Additional keyword arguments passed to the constructor
                (including ``scale``, ``offset``, ``name``, ``category``, etc.).

        Returns:
            A `SegmentationMask` with RLE-encoded data.

        Raises:
            ValueError: If ``mask`` contains more than one distinct non-zero
                value. Use ``LabelImage.from_numpy`` (to keep all classes in
                one dense array) or ``LabelImage.from_binary_masks`` (to
                split per-class binaries) for multi-class inputs.
        """
        arr = np.asarray(mask)
        if arr.dtype != bool:
            nonzero = arr[arr != 0]
            if nonzero.size > 0:
                uniques = np.unique(nonzero)
                if uniques.size > 1:
                    preview = sorted(uniques.tolist())[:5]
                    raise ValueError(
                        f"SegmentationMask is binary (one object per mask) but "
                        f"got an array with {uniques.size} distinct non-zero "
                        f"values (e.g. {preview}). Use "
                        f"sleap_io.UserLabelImage.from_numpy(array) to keep all "
                        f"classes in one dense array, or "
                        f"sleap_io.UserLabelImage.from_binary_masks([...]) to "
                        f"split per-class binaries. To opt in to binarization "
                        f"explicitly, pass array.astype(bool)."
                    )
        if stride is not None:
            kwargs["scale"] = (1.0 / stride, 1.0 / stride)
        height, width = arr.shape
        rle_counts = _encode_rle(arr)
        return cls(rle_counts=rle_counts, height=height, width=width, **kwargs)

    @property
    def data(self) -> np.ndarray:
        """Decode the mask to a 2D boolean numpy array.

        Returns:
            A boolean array of shape (height, width).
        """
        return _decode_rle(self.rle_counts, self.height, self.width)

    @property
    def area(self) -> int:
        """Number of foreground (True) pixels in the mask."""
        # Sum the odd-indexed runs (1-runs)
        return int(sum(self.rle_counts[1::2]))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box of the mask as (x, y, width, height) in image coordinates.

        When ``scale`` or ``offset`` are non-default, the bounding box is
        transformed from mask-pixel space to image-pixel space using
        ``image_coord = mask_coord / scale + offset``.

        Returns:
            A tuple of (x, y, width, height) for the tightest axis-aligned
            bounding box containing all foreground pixels. Returns (0, 0, 0, 0)
            if the mask is empty.
        """
        mask = self.data
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows):
            return (0.0, 0.0, 0.0, 0.0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        sx, sy = self.scale
        ox, oy = self.offset
        return (
            float(cmin / sx + ox),
            float(rmin / sy + oy),
            float((cmax - cmin + 1) / sx),
            float((rmax - rmin + 1) / sy),
        )

    def to_bbox(self) -> "BoundingBox":
        """Convert to a BoundingBox object.

        Returns a ``UserBoundingBox`` or ``PredictedBoundingBox`` with metadata
        (track, category, name, instance) inherited from this mask. Coordinates
        are in image space (respecting scale/offset).

        Returns:
            A ``BoundingBox`` matching this mask's tight bounding box.
        """
        from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox

        x, y, w, h = self.bbox
        cls = PredictedBoundingBox if self.is_predicted else UserBoundingBox
        kwargs: dict = dict(
            track=self.track,
            instance=self.instance,
            category=self.category,
            name=self.name,
            source=self.source,
        )
        if self.is_predicted:
            kwargs["score"] = self.score
        return cls.from_xywh(x, y, w, h, **kwargs)

    def to_polygon(self) -> "ROI":
        """Convert the mask to a polygon ROI via row-rectangle union.

        Builds pixel-aligned rectangles for each horizontal run of foreground
        pixels, then merges them with Shapely's ``unary_union`` to produce an
        exact polygon boundary. Handles non-convex shapes and holes correctly.

        When ``scale`` or ``offset`` are non-default, the polygon coordinates
        are transformed from mask-pixel space to image-pixel space.

        Returns:
            An `ROI` with polygon geometry derived from the mask. Returns an
            ROI with an empty polygon if the mask has no foreground pixels.
        """
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union

        from sleap_io.model.roi import UserROI

        sx, sy = self.scale
        ox, oy = self.offset

        mask = self.data
        rectangles = []
        for y in range(self.height):
            row = mask[y].astype(np.uint8)
            diff = np.diff(np.concatenate([[0], row, [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                rectangles.append(
                    box(s / sx + ox, y / sy + oy, e / sx + ox, (y + 1) / sy + oy)
                )

        if not rectangles:
            geometry = Polygon()
        else:
            geometry = unary_union(rectangles)

        return UserROI(
            geometry=geometry,
            name=self.name,
            category=self.category,
            source=self.source,
            track=self.track,
            instance=self.instance,
        )


@attrs.define(eq=False)
class UserSegmentationMask(SegmentationMask):
    """Human-annotated segmentation mask.

    Attributes:
        from_predicted: The `PredictedSegmentationMask` (if any) that this user
            mask was initialized from, recorded by
            `PredictedSegmentationMask.to_user()` for human-in-the-loop
            correction workflows. `None` if the mask was created directly. This
            provenance link is persisted to the SLP format as an index into the
            saved mask list (mirroring instance `from_predicted`), so it survives
            a save/load round-trip as long as the source prediction is also
            saved. Files written before this column existed load it as `None`.
    """

    from_predicted: "PredictedSegmentationMask | None" = attrs.field(
        default=None, repr=False
    )


@attrs.define(eq=False)
class PredictedSegmentationMask(SegmentationMask):
    """Model-predicted segmentation mask with confidence score.

    Attributes:
        score: Object-level confidence score (0-1).
        score_map: Optional dense pixel-level confidence map of shape (H, W)
            as float32. This can be large and is stored separately in the SLP
            format. If ``None``, only the object-level score is available.
        score_map_scale: Resolution ratio ``(sx, sy)`` for the score map,
            independent of the mask's own ``scale``. Defaults to ``(1.0, 1.0)``.
        score_map_offset: Origin ``(x, y)`` of the score map in image pixel
            coordinates. Defaults to ``(0.0, 0.0)``.
    """

    score: float = attrs.field(default=0.0)
    score_map: np.ndarray | None = attrs.field(default=None)
    score_map_scale: tuple[float, float] = attrs.field(default=(1.0, 1.0))
    score_map_offset: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    def to_user(self, link: bool = True) -> "UserSegmentationMask":
        """Convert this predicted mask to a user mask, recording provenance.

        Returns a new `UserSegmentationMask` carrying a copy of the RLE raster
        and all shared metadata (`name`, `category`, `source`, `track`,
        `tracking_score`, `instance`, `scale`, `offset`). The prediction-only
        fields (`score`, `score_map`, `score_map_scale`, `score_map_offset`)
        are dropped. This is the predicted -> user adoption path for the
        inference -> human-correct -> retrain loop, mirroring
        `Instance.from_predicted` for poses.

        Args:
            link: If `True` (the default), set `from_predicted` on the returned
                mask to this prediction, recording that the user annotation
                originated from it. Pass `False` for an unlinked copy.

        Returns:
            A new `UserSegmentationMask` with an independent RLE buffer and the
            shared metadata above. `from_predicted` points back at this mask
            when `link` is `True`, otherwise `None`.

        Notes:
            The `track` and `instance` references are shared (not copied), so
            mutating them affects both masks. The `from_predicted` link is
            persisted to the SLP format (as an index into the saved mask list);
            it survives a save/load round-trip as long as this source prediction
            is also saved.
        """
        user = UserSegmentationMask(
            rle_counts=self.rle_counts.copy(),
            height=self.height,
            width=self.width,
            name=self.name,
            category=self.category,
            source=self.source,
            track=self.track,
            tracking_score=self.tracking_score,
            instance=self.instance,
            scale=self.scale,
            offset=self.offset,
            from_predicted=self if link else None,
        )
        user._instance_idx = self._instance_idx
        return user
