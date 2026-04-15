"""TIFF I/O for LabelImage data.

Reads and writes integer label images as TIFF files (single, multi-page stacks,
or directories of per-frame TIFFs) with optional JSON sidecar metadata for track
names and categories.

Pages in a TIFF can represent either time (one frame per page) or classes
(one class per page, single frame). The reader distinguishes these layouts
via, in priority order: an explicit ``pages_as`` argument, a sidecar
``"axes"`` hint, TIFF-level metadata (OME-XML, ImageJ hyperstack), and
finally a fall-back assumption of pages-as-time for plain multi-page files.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.instance import Track
    from sleap_io.model.label_image import LabelImage
    from sleap_io.model.video import Video


# Set of tifffile axis placeholders that mean "unknown iteration axis".
_AMBIGUOUS_AXIS_CHARS = {"I", "Q", "S"}


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

    # Write an axes hint so the reader can round-trip without heuristics.
    axes = "YX" if len(label_images) == 1 else "TYX"

    sidecar: dict = {
        "format": "sleap-io-label-image-meta",
        "version": 3,
        "axes": axes,
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


def _normalize_axes(axes: str | None) -> str:
    """Normalize a TIFF axes string to one of the layouts the reader handles.

    Args:
        axes: Raw axes string from tifffile or a sidecar hint. Case-insensitive.

    Returns:
        One of:
            - ``"YX"``: single 2D frame.
            - ``"TYX"``: pages are time (``ZYX`` z-stacks are also mapped here).
            - ``"CYX"``: pages are classes, single frame.
            - ``"TCYX"``: time + classes.
            - ``"unknown"``: no usable axis info (empty, None, or placeholder
              like ``"IYX"``/``"QYX"``).
    """
    if not axes:
        return "unknown"
    a = axes.upper()
    if a == "YX":
        return "YX"
    if a in ("TYX", "ZYX"):
        return "TYX"
    if a == "CYX":
        return "CYX"
    if "T" in a and "C" in a and set(a) <= set("TCZYX"):
        return "TCYX"
    # Placeholder / unknown axes like 'IYX', 'QYX', 'YXS'.
    if any(c in a for c in _AMBIGUOUS_AXIS_CHARS):
        return "unknown"
    return "unknown"


def _infer_tiff_axes(path: Path) -> tuple[str, int, bool]:
    """Inspect a TIFF file and infer what its pages represent.

    Only OME-XML and ImageJ hyperstack metadata are treated as authoritative.
    Plain multi-page TIFFs (written by a naive ``TiffWriter`` loop) have no
    axis semantics — tifffile reports them as multiple ``'YX'`` series — and
    are returned as ``"unknown"`` so the caller can decide.

    Args:
        path: Path to a single TIFF file.

    Returns:
        Tuple ``(axes, n_pages, has_metadata)`` where ``axes`` is one of
        ``"YX"``, ``"TYX"``, ``"CYX"``, ``"TCYX"``, or ``"unknown"`` (see
        :func:`_normalize_axes`) and ``has_metadata`` is ``True`` when the
        file declared its axes via OME-XML or an ImageJ hyperstack header.
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        n_pages = len(tif.pages)
        has_metadata = bool(tif.is_ome or tif.is_imagej)
        if has_metadata and tif.series:
            return _normalize_axes(tif.series[0].axes), n_pages, True
        if n_pages == 1 and tif.pages[0].ndim == 2:
            return "YX", 1, False
        return "unknown", n_pages, False


def _warn_ambiguous_pages(path: Path, n_pages: int, dtype_name: str) -> None:
    """Emit a warning when falling back to pages-as-time with no metadata.

    Only fires for multi-page files; single pages are unambiguous.
    """
    if n_pages <= 1:
        return
    warnings.warn(
        f"Loaded {n_pages} frames from multi-page TIFF {path!s} with no axis "
        f"metadata (dtype={dtype_name}). Assuming pages are time. If pages "
        f"represent classes for a single frame, pass pages_as='classes' "
        f"(or add a sidecar {{path}}.meta.json with 'axes': 'CYX') to route "
        f"through from_binary_masks with categories.",
        stacklevel=3,
    )


def _categories_list_from_sidecar(
    sidecar: dict | None, n_classes: int
) -> list[str] | None:
    """Extract a positional category list from a sidecar's ``objects`` dict.

    For class-stacked layouts the sidecar's ``objects`` dict should be keyed
    by the post-composite label IDs (``"1"``..``"N"``) in page order.

    Args:
        sidecar: Parsed sidecar dict or None.
        n_classes: Number of classes (pages).

    Returns:
        A list of category strings of length ``n_classes``, or None if the
        sidecar has no per-class categories.
    """
    if sidecar is None:
        return None
    objects = sidecar.get("objects", {}) or {}
    if not objects:
        return None
    out: list[str] = []
    any_set = False
    for i in range(n_classes):
        entry = objects.get(str(i + 1), {}) or {}
        cat = entry.get("category", "") or ""
        if cat:
            any_set = True
        out.append(cat)
    return out if any_set else None


def _read_pages_as_time(
    frames_data: list[np.ndarray],
    tracks: "dict[int, Track] | None",
    categories: "dict[int, str] | None",
    sidecar: dict | None,
    scale: tuple[float, float],
    offset: tuple[float, float],
) -> list["LabelImage"]:
    """Build LabelImages treating each 2D page as one frame.

    Args:
        frames_data: List of 2D int32 arrays (one per frame).
        tracks: Optional user-provided label_id -> Track mapping.
        categories: Optional user-provided label_id -> category mapping.
        sidecar: Parsed sidecar dict or None.
        scale: Spatial scale to attach to each LabelImage.
        offset: Spatial offset to attach to each LabelImage.

    Returns:
        List of ``UserLabelImage``, one per frame, in input order.
    """
    from sleap_io.model.label_image import UserLabelImage

    all_ids: set[int] = set()
    for data in frames_data:
        ids = np.unique(data)
        all_ids.update(int(i) for i in ids if i > 0)

    unique_ids = np.array(sorted(all_ids), dtype=np.int32)
    track_map = _build_track_map(unique_ids, tracks, sidecar)
    cat_map = _build_category_map(unique_ids, categories, sidecar)

    result = []
    for data in frames_data:
        objects = _build_objects_dict(data, track_map, cat_map)
        result.append(
            UserLabelImage(
                data=data,
                objects=objects,
                scale=scale,
                offset=offset,
            )
        )
    return result


def _infer_label_ids_from_pages(
    pages_data: "list[np.ndarray] | np.ndarray",
) -> list[int] | None:
    """Infer per-page label IDs from the pixel values of each page.

    When each page has exactly one distinct non-zero value and the values
    across pages are all unique, those values were almost certainly intended
    as label IDs (e.g., COCO-style class IDs like ``{5, 17, 99}``). Returning
    them lets the caller preserve the original IDs instead of renumbering to
    positional ``1..N``.

    Returns ``None`` for any ambiguous case — purely binary stacks (each
    page has nonzero value ``1``) collide and fall back to positional; pages
    with multiple nonzero values aren't per-class binaries at all.

    Args:
        pages_data: Iterable of 2D arrays, one per page/class.

    Returns:
        A list of positive integer IDs, one per page, if unambiguous; else
        ``None``.
    """
    ids: list[int] = []
    for page in pages_data:
        vals = np.unique(np.asarray(page))
        vals = vals[vals > 0]
        if vals.size != 1:
            return None
        ids.append(int(vals[0]))
    if len(set(ids)) != len(ids):
        return None
    return ids


def _categories_as_list(
    categories: "list[str] | dict[int, str] | None",
    n_classes: int,
) -> list[str] | None:
    """Coerce a user-provided ``categories`` value to a positional list.

    Args:
        categories: ``list[str]`` (used as-is), ``dict[int, str]`` (read in
            label-ID order), or ``None``.
        n_classes: Number of classes (pages).

    Returns:
        List of length ``n_classes`` or None.
    """
    if categories is None:
        return None
    if isinstance(categories, list):
        return list(categories)
    out = [categories.get(i + 1, "") for i in range(n_classes)]
    return out


def _read_single_class_stack(
    pages_data: list[np.ndarray],
    categories: "list[str] | dict[int, str] | None",
    sidecar: dict | None,
    scale: tuple[float, float],
    offset: tuple[float, float],
) -> list["LabelImage"]:
    """Build one LabelImage from a stack of per-class binary pages.

    When each page carries a single distinct non-zero value and the values
    are all unique across pages (e.g., ``{5}``, ``{17}``, ``{99}``), those
    values are preserved as label IDs. Purely binary stacks (every page has
    the same value ``1``) fall back to positional IDs ``1..N``.

    Args:
        pages_data: List of 2D arrays (one per class).
        categories: Per-class category names (list) or label-ID mapping (dict).
        sidecar: Parsed sidecar dict (fallback source of categories).
        scale: Spatial scale.
        offset: Spatial offset.

    Returns:
        A list with a single ``UserLabelImage``.
    """
    from sleap_io.model.label_image import UserLabelImage

    n_classes = len(pages_data)
    cat_list = _categories_as_list(categories, n_classes)
    if cat_list is None:
        cat_list = _categories_list_from_sidecar(sidecar, n_classes)

    label_ids = _infer_label_ids_from_pages(pages_data)

    stack = np.stack([p.astype(bool) for p in pages_data], axis=0)
    li = UserLabelImage.from_binary_masks(
        stack,
        label_ids=label_ids,
        categories=cat_list,
        scale=scale,
        offset=offset,
    )
    return [li]


def read_label_images(
    path: str | Path,
    video: "Video | None" = None,
    tracks: "dict[int, Track] | None" = None,
    categories: "list[str] | dict[int, str] | None" = None,
    pages_as: str = "auto",
) -> list["LabelImage"]:
    """Read label images from TIFF file(s).

    Args:
        path: One of:

            - Multi-page TIFF stack: one page per frame (or per class, see
              ``pages_as``).
            - Single TIFF: one frame.
            - Directory of TIFFs: sorted alphanumerically, one per frame.

        video: Video to associate with all frames.
        tracks: Global ``label_id -> Track`` mapping. If ``None``, auto-creates
            one ``Track`` per unique ID found across all frames. Ignored for
            class-stacked layouts.
        categories: Category strings.

            - ``dict[int, str]`` — keyed by label ID (time mode).
            - ``list[str]`` — positional, one per class (class mode).
            - ``None`` — read from sidecar if present.

        pages_as: How to interpret multi-page TIFFs.

            - ``"auto"`` (default): consult sidecar ``"axes"``, then TIFF
              metadata (OME-XML / ImageJ hyperstack). Falls back to
              ``"time"`` for plain multi-page TIFFs with a one-time warning.
            - ``"time"``: force each page to be one frame (N pages -> N
              ``LabelImage`` objects).
            - ``"classes"``: force pages to be per-class binary masks for a
              single frame (N pages -> 1 ``LabelImage`` with label IDs 1..N).

    Returns:
        List of ``LabelImage``, one per frame, sorted by frame index.

    Raises:
        ValueError: For unknown ``pages_as`` values or unreadable pages
            (non-2D, negative values, etc.).
    """
    import tifffile

    if pages_as not in ("auto", "time", "classes"):
        raise ValueError(
            f"pages_as must be 'auto', 'time', or 'classes'; got {pages_as!r}."
        )

    path = Path(path)
    sidecar = _read_sidecar(path)

    # Read spatial metadata from sidecar (v2+)
    sidecar_scale: tuple[float, float] = (1.0, 1.0)
    sidecar_offset: tuple[float, float] = (0.0, 0.0)
    if sidecar is not None:
        if "scale" in sidecar:
            s = sidecar["scale"]
            sidecar_scale = (float(s[0]), float(s[1]))
        if "offset" in sidecar:
            o = sidecar["offset"]
            sidecar_offset = (float(o[0]), float(o[1]))

    # --- Directory input ------------------------------------------------
    if path.is_dir():
        tiff_files = sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))
        if not tiff_files:
            return []

        frames_data: list[np.ndarray] = []
        for tiff_path in tiff_files:
            data = tifffile.imread(str(tiff_path)).astype(np.int32)
            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array from {tiff_path}, got shape {data.shape}"
                )
            frames_data.append(data)

        if pages_as == "classes":
            return _read_single_class_stack(
                frames_data, categories, sidecar, sidecar_scale, sidecar_offset
            )
        return _read_pages_as_time(
            frames_data,
            tracks,
            categories,
            sidecar,
            sidecar_scale,
            sidecar_offset,
        )

    # --- Single file (possibly multi-page) ------------------------------
    # Decide layout. Priority: explicit pages_as -> sidecar axes -> TIFF
    # metadata -> fallback ('time' with warning for plain multi-page).
    sidecar_axes = None
    if sidecar is not None and "axes" in sidecar:
        sidecar_axes = _normalize_axes(sidecar["axes"])

    tiff_axes, n_pages, has_metadata = _infer_tiff_axes(path)

    if pages_as == "time":
        layout = "TYX"
    elif pages_as == "classes":
        layout = "CYX"
    elif sidecar_axes and sidecar_axes != "unknown":
        layout = sidecar_axes
    elif tiff_axes != "unknown":
        layout = tiff_axes
    else:
        layout = "TYX"  # fallback

    # Read series data once for authoritative layouts (OME/ImageJ declare
    # the full shape via series rather than per-page iteration).
    def _iter_pages() -> list[np.ndarray]:
        with tifffile.TiffFile(str(path)) as tif:
            out = []
            for page in tif.pages:
                arr = page.asarray().astype(np.int32)
                if arr.ndim != 2:
                    raise ValueError(
                        f"Expected 2D page in {path}, got shape {arr.shape}"
                    )
                out.append(arr)
            return out

    def _read_series() -> np.ndarray:
        with tifffile.TiffFile(str(path)) as tif:
            return tif.series[0].asarray()

    # --- Dispatch on layout ---------------------------------------------
    if layout in ("YX", "TYX"):
        # Warn when we're falling back on an ambiguous plain multi-page.
        used_fallback = (
            pages_as == "auto" and sidecar_axes is None and tiff_axes == "unknown"
        )
        if used_fallback:
            dtype_name = "unknown"
            with tifffile.TiffFile(str(path)) as tif:
                if tif.pages:
                    dtype_name = str(tif.pages[0].dtype)
            _warn_ambiguous_pages(path, n_pages, dtype_name)

        frames_data = _iter_pages()
        return _read_pages_as_time(
            frames_data,
            tracks,
            categories,
            sidecar,
            sidecar_scale,
            sidecar_offset,
        )

    if layout == "CYX":
        pages_data = _iter_pages()
        return _read_single_class_stack(
            pages_data,
            categories,
            sidecar,
            sidecar_scale,
            sidecar_offset,
        )

    if layout == "TCYX":
        # OME/ImageJ declared both T and C. Use the series array which
        # reshapes pages into a coherent (T, C, H, W) block. tifffile drops
        # size-1 axes, so a degenerate T=1 surfaces as layout="CYX" above
        # and doesn't reach here.
        series = _read_series()
        if series.ndim != 4:
            raise ValueError(
                f"Expected 4D (T,C,H,W) series for TCYX, got shape {series.shape}"
            )
        from sleap_io.model.label_image import UserLabelImage

        t_dim, c_dim = series.shape[0], series.shape[1]
        cat_list = _categories_as_list(categories, c_dim)
        if cat_list is None:
            cat_list = _categories_list_from_sidecar(sidecar, c_dim)

        result = []
        for t in range(t_dim):
            pages_t = [series[t, c] for c in range(c_dim)]
            label_ids_t = _infer_label_ids_from_pages(pages_t)
            stack_t = series[t].astype(bool)
            result.append(
                UserLabelImage.from_binary_masks(
                    stack_t,
                    label_ids=label_ids_t,
                    categories=cat_list,
                    scale=sidecar_scale,
                    offset=sidecar_offset,
                )
            )
        return result

    raise ValueError(f"Unhandled TIFF axes layout: {layout!r}")


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
