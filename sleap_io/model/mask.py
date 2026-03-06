"""Data structures for segmentation mask annotations.

Segmentation masks represent raster (per-pixel) annotations stored in
run-length encoded (RLE) format for compact storage. They can be converted
to and from numpy arrays and polygon representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

from sleap_io.model.roi import AnnotationType

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.roi import ROI
    from sleap_io.model.video import Video


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


@attrs.define(eq=False)
class SegmentationMask:
    """A segmentation mask stored as run-length encoded (RLE) data.

    Attributes:
        rle_counts: Run-length encoded counts as a uint32 array. Alternating runs
            of 0s and 1s, starting with 0s.
        height: Height of the mask in pixels.
        width: Width of the mask in pixels.
        annotation_type: Semantic type of the annotation.
        name: Optional human-readable name for this mask.
        category: Optional category label (e.g., class name for detection).
        score: Optional confidence score (0-1). If set, the mask is considered
            a prediction.
        source: Optional string indicating the source of this annotation.
        video: Optional `Video` this mask is associated with.
        frame_idx: Optional frame index. If `None`, the mask is static.
        track: Optional `Track` this mask is associated with.
        instance: Optional `Instance` this mask is associated with.

    Notes:
        Masks use identity-based equality (two mask objects are only equal if they
        are the same object in memory).
    """

    rle_counts: np.ndarray = attrs.field()
    height: int = attrs.field()
    width: int = attrs.field()
    annotation_type: AnnotationType = attrs.field(
        default=AnnotationType.SEGMENTATION, converter=AnnotationType
    )
    name: str = attrs.field(default="")
    category: str = attrs.field(default="")
    score: float | None = attrs.field(default=None)
    source: str = attrs.field(default="")
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    track: "Track | None" = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)

    @classmethod
    def from_numpy(
        cls,
        mask: np.ndarray,
        **kwargs,
    ) -> "SegmentationMask":
        """Create a SegmentationMask from a 2D numpy array.

        Args:
            mask: A 2D boolean or uint8 numpy array of shape (height, width).
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A `SegmentationMask` with RLE-encoded data.
        """
        height, width = mask.shape
        rle_counts = _encode_rle(mask)
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
        """Bounding box of the mask as (x, y, width, height).

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

        return (
            float(cmin),
            float(rmin),
            float(cmax - cmin + 1),
            float(rmax - rmin + 1),
        )

    def to_polygon(self) -> "ROI":
        """Convert the mask to a polygon ROI via row-rectangle union.

        Builds pixel-aligned rectangles for each horizontal run of foreground
        pixels, then merges them with Shapely's ``unary_union`` to produce an
        exact polygon boundary. Handles non-convex shapes and holes correctly.

        Returns:
            An `ROI` with polygon geometry derived from the mask. Returns an
            ROI with an empty polygon if the mask has no foreground pixels.
        """
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union

        from sleap_io.model.roi import ROI

        mask = self.data
        rectangles = []
        for y in range(self.height):
            row = mask[y].astype(np.uint8)
            diff = np.diff(np.concatenate([[0], row, [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                rectangles.append(box(s, y, e, y + 1))

        if not rectangles:
            geometry = Polygon()
        else:
            geometry = unary_union(rectangles)

        return ROI(
            geometry=geometry,
            annotation_type=self.annotation_type,
            name=self.name,
            category=self.category,
            score=self.score,
            source=self.source,
            video=self.video,
            frame_idx=self.frame_idx,
            track=self.track,
            instance=self.instance,
        )
