"""TIFF I/O for LabelImage data.

Reads and writes integer label images as TIFF files (single, multi-page stacks,
or directories of per-frame TIFFs) with optional JSON sidecar metadata for track
names and categories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.instance import Track
    from sleap_io.model.label_image import LabelImage
    from sleap_io.model.video import Video


def _read_sidecar(path: Path) -> dict | None:
    """Read sidecar metadata JSON if it exists.

    Args:
        path: Path to the TIFF file or directory. The sidecar is expected at
            ``{path}.meta.json``.

    Returns:
        Parsed sidecar dict, or None if no sidecar exists.
    """
    sidecar_path = Path(str(path) + ".meta.json")
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            return json.load(f)
    return None


def _write_sidecar(path: Path, label_images: list[LabelImage]) -> None:
    """Write sidecar metadata JSON alongside a TIFF file or directory.

    Args:
        path: Path to the TIFF file or directory. The sidecar is written at
            ``{path}.meta.json``.
        label_images: LabelImage objects whose objects dicts are merged into the
            sidecar.
    """
    objects: dict[str, dict[str, str]] = {}
    for li in label_images:
        for label_id, info in li.objects.items():
            key = str(label_id)
            if key not in objects:
                entry: dict[str, str] = {}
                if info.track is not None:
                    entry["track"] = info.track.name
                if info.category:
                    entry["category"] = info.category
                objects[key] = entry

    # Check if any label image has non-default spatial metadata
    has_spatial = any(li.has_spatial_transform for li in label_images)

    sidecar: dict = {
        "format": "sleap-io-label-image-meta",
        "version": 2 if has_spatial else 1,
        "objects": objects,
    }
    if has_spatial and label_images:
        # Use spatial metadata from first label image (all should share it
        # for a consistent stack).
        sidecar["scale"] = list(label_images[0].scale)
        sidecar["offset"] = list(label_images[0].offset)

    sidecar_path = Path(str(path) + ".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def _build_track_map(
    unique_ids: np.ndarray,
    tracks: dict[int, "Track"] | None,
    sidecar: dict | None,
) -> dict[int, "Track"]:
    """Build a label_id -> Track mapping.

    Args:
        unique_ids: All unique non-zero label IDs.
        tracks: User-provided track mapping, or None.
        sidecar: Parsed sidecar dict, or None.

    Returns:
        Dict mapping label ID to Track.
    """
    from sleap_io.model.instance import Track

    if tracks is not None:
        return dict(tracks)

    track_map: dict[int, Track] = {}
    sidecar_objects = sidecar.get("objects", {}) if sidecar else {}

    for lid in unique_ids:
        lid_int = int(lid)
        key = str(lid_int)
        entry = sidecar_objects.get(key, {})
        track_name = entry.get("track", str(lid_int))
        track_map[lid_int] = Track(name=track_name)

    return track_map


def _build_category_map(
    unique_ids: np.ndarray,
    categories: dict[int, str] | None,
    sidecar: dict | None,
) -> dict[int, str]:
    """Build a label_id -> category mapping.

    Args:
        unique_ids: All unique non-zero label IDs.
        categories: User-provided category mapping, or None.
        sidecar: Parsed sidecar dict, or None.

    Returns:
        Dict mapping label ID to category string.
    """
    if categories is not None:
        return dict(categories)

    cat_map: dict[int, str] = {}
    sidecar_objects = sidecar.get("objects", {}) if sidecar else {}

    for lid in unique_ids:
        lid_int = int(lid)
        key = str(lid_int)
        entry = sidecar_objects.get(key, {})
        category = entry.get("category", "")
        if category:
            cat_map[lid_int] = category

    return cat_map


def _build_objects_dict(
    data: np.ndarray,
    track_map: dict[int, "Track"],
    cat_map: dict[int, str],
) -> dict[int, "LabelImage.Info"]:
    """Build objects dict for a single frame from shared track/category maps.

    Args:
        data: (H, W) integer array for this frame.
        track_map: Shared label_id -> Track mapping.
        cat_map: Shared label_id -> category mapping.

    Returns:
        Objects dict for this frame (only IDs present in data).
    """
    from sleap_io.model.label_image import LabelImage

    ids = np.unique(data)
    ids = ids[ids > 0]

    objects: dict[int, LabelImage.Info] = {}
    for lid in ids:
        lid_int = int(lid)
        objects[lid_int] = LabelImage.Info(
            track=track_map.get(lid_int),
            category=cat_map.get(lid_int, ""),
        )
    return objects


def read_label_images(
    path: str | Path,
    video: "Video | None" = None,
    tracks: "dict[int, Track] | None" = None,
    categories: dict[int, str] | None = None,
) -> list["LabelImage"]:
    """Read label images from TIFF file(s).

    Args:
        path: One of:
            - Multi-page TIFF stack: one page per frame.
            - Single TIFF: one frame.
            - Directory of TIFFs: sorted alphanumerically, one per frame.
        video: Video to associate with all frames.
        tracks: Global label_id -> Track mapping. If None, auto-creates
            one Track per unique ID found across all frames.
        categories: Global label_id -> category mapping.

    Returns:
        List of LabelImage, one per frame, sorted by frame_idx.
    """
    import tifffile

    from sleap_io.model.label_image import UserLabelImage

    path = Path(path)
    sidecar = _read_sidecar(path)

    # Read spatial metadata from sidecar (v2+)
    sidecar_scale = (1.0, 1.0)
    sidecar_offset = (0.0, 0.0)
    if sidecar is not None:
        if "scale" in sidecar:
            s = sidecar["scale"]
            sidecar_scale = (float(s[0]), float(s[1]))
        if "offset" in sidecar:
            o = sidecar["offset"]
            sidecar_offset = (float(o[0]), float(o[1]))

    if path.is_dir():
        tiff_files = sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))
        if not tiff_files:
            return []

        # Read all files in a single pass, collecting data and unique IDs
        frames_data: list[np.ndarray] = []
        all_ids: set[int] = set()
        for tiff_path in tiff_files:
            data = tifffile.imread(str(tiff_path)).astype(np.int32)
            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array from {tiff_path}, got shape {data.shape}"
                )
            frames_data.append(data)
            ids = np.unique(data)
            all_ids.update(int(i) for i in ids if i > 0)

        unique_ids = np.array(sorted(all_ids), dtype=np.int32)
        track_map = _build_track_map(unique_ids, tracks, sidecar)
        cat_map = _build_category_map(unique_ids, categories, sidecar)

        result = []
        for frame_idx, data in enumerate(frames_data):
            objects = _build_objects_dict(data, track_map, cat_map)
            result.append(
                UserLabelImage(
                    data=data,
                    objects=objects,
                    scale=sidecar_scale,
                    offset=sidecar_offset,
                )
            )
        return result

    # Single file (possibly multi-page)
    # Read all pages in a single pass, collecting data and unique IDs
    frames_data: list[np.ndarray] = []
    all_ids: set[int] = set()
    with tifffile.TiffFile(str(path)) as tif:
        for page in tif.pages:
            data = page.asarray().astype(np.int32)
            if data.ndim != 2:
                raise ValueError(f"Expected 2D page in {path}, got shape {data.shape}")
            frames_data.append(data)
            ids = np.unique(data)
            all_ids.update(int(i) for i in ids if i > 0)

    unique_ids = np.array(sorted(all_ids), dtype=np.int32)
    track_map = _build_track_map(unique_ids, tracks, sidecar)
    cat_map = _build_category_map(unique_ids, categories, sidecar)

    result = []
    for frame_idx, data in enumerate(frames_data):
        objects = _build_objects_dict(data, track_map, cat_map)
        result.append(
            UserLabelImage(
                data=data,
                objects=objects,
                scale=sidecar_scale,
                offset=sidecar_offset,
            )
        )

    return result


def write_label_images(
    path: str | Path,
    label_images: list["LabelImage"],
    stack: bool = True,
) -> None:
    """Write label images to TIFF.

    Args:
        path: Output path. If stack=True, writes a single multi-page TIFF.
            If stack=False, writes per-frame files to this directory (named
            by zero-padded frame index).
        label_images: LabelImage objects to write.
        stack: Write as multi-page TIFF stack (True) or per-frame files in a
            directory (False).
    """
    import tifffile

    path = Path(path)

    if not label_images:
        return

    if stack:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tifffile.TiffWriter(str(path)) as tw:
            for li in label_images:
                tw.write(li.data)
    else:
        path.mkdir(parents=True, exist_ok=True)
        n_digits = max(1, len(str(len(label_images) - 1)))
        for i, li in enumerate(label_images):
            frame_path = path / f"{str(i).zfill(n_digits)}.tif"
            tifffile.imwrite(str(frame_path), li.data)

    _write_sidecar(path, label_images)
