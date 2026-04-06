"""Read TrackMate CSV exports into sleap-io data structures.

TrackMate (ImageJ/Fiji) exports tracking results as three CSV files per video:

- ``*_spots.csv`` — Individual spot detections (required).
- ``*_edges.csv`` — Frame-to-frame linkages with assignment cost (optional).
- ``*_tracks.csv`` — Track-level summary statistics (not used).

All CSVs have **4 header rows** (field names, descriptions, abbreviations,
units) followed by data rows.

See Also:
    https://imagej.net/plugins/trackmate/
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video

# Number of header rows before data in TrackMate CSV exports.
_HEADER_ROWS = 4

# Required columns in a spots CSV (used for format detection).
_SPOTS_SIGNATURE = ("LABEL", "ID", "TRACK_ID", "QUALITY", "POSITION_X", "POSITION_Y")


def is_trackmate_file(path: str | Path) -> bool:
    """Check if a CSV file is a TrackMate spots export.

    Reads the first line and checks for the TrackMate column signature.

    Args:
        path: Path to a CSV file.

    Returns:
        ``True`` if the file looks like a TrackMate spots CSV.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            cols = next(reader)
        return tuple(cols[: len(_SPOTS_SIGNATURE)]) == _SPOTS_SIGNATURE
    except Exception:
        return False


def _find_sibling(spots_path: Path, suffix: str) -> Path | None:
    """Find a sibling file by replacing ``_spots`` with *suffix* in the stem.

    Given ``foo_spots.csv``, returns ``foo_edges.csv`` if it exists (for
    ``suffix="_edges"``). Also checks for ``foo.tif`` / ``foo.tiff`` when
    ``suffix`` is ``".tif"``.

    Args:
        spots_path: Path to the spots CSV.
        suffix: The suffix to substitute for ``_spots``, e.g., ``"_edges"``
            or ``".tif"``.

    Returns:
        The resolved path if the sibling exists, otherwise ``None``.
    """
    stem = spots_path.stem
    if "_spots" not in stem:
        return None

    parts = stem.rsplit("_spots", 1)
    base = parts[0] + parts[1] if len(parts) == 2 else stem

    if suffix.startswith("."):
        # Looking for a non-CSV sibling (e.g., .tif video).
        for ext in (suffix, suffix + "f"):  # .tif and .tiff
            candidate = spots_path.parent / (base + ext)
            if candidate.exists():
                return candidate
    else:
        # Looking for another CSV sibling (e.g., _edges.csv).
        candidate = spots_path.parent / (base + suffix + ".csv")
        if candidate.exists():
            return candidate

    return None


def _parse_edges(edges_path: Path) -> dict[int, float]:
    """Parse an edges CSV and return a mapping of target spot ID → link cost.

    Args:
        edges_path: Path to the TrackMate edges CSV.

    Returns:
        Dict mapping ``SPOT_TARGET_ID`` (int) → ``LINK_COST`` (float).
    """
    target_to_cost: dict[int, float] = {}

    with open(edges_path, newline="") as f:
        reader = csv.reader(f)

        # Read header row to find column indices.
        header = next(reader)
        try:
            target_col = header.index("SPOT_TARGET_ID")
            cost_col = header.index("LINK_COST")
        except ValueError:
            return target_to_cost

        # Skip remaining header rows.
        for _ in range(_HEADER_ROWS - 1):
            next(reader, None)

        for row in reader:
            if not row or not row[target_col]:
                continue
            try:
                target_id = int(row[target_col])
                cost = float(row[cost_col])
                target_to_cost[target_id] = cost
            except (ValueError, IndexError):
                continue

    return target_to_cost


def read_trackmate_csv(
    spots_path: str | Path,
    edges_path: str | Path | None = None,
    video: "Video | str | Path | None" = None,
) -> "Labels":
    """Load TrackMate CSV exports into a ``Labels`` object.

    The spots CSV is required. The edges CSV is optional but provides
    per-link ``tracking_score`` (from TrackMate's ``LINK_COST``).

    Args:
        spots_path: Path to the ``*_spots.csv`` file.
        edges_path: Path to the ``*_edges.csv`` file. If ``None``, attempts
            to auto-detect a sibling ``_edges.csv`` alongside the spots file.
        video: Video to associate with centroids. Can be a ``Video`` object,
            a string/path to a video file, or ``None`` (auto-detects a
            sibling ``.tif`` file).

    Returns:
        A ``Labels`` object with ``centroids``, ``tracks``, and optionally
        ``videos`` populated.

    Raises:
        FileNotFoundError: If the spots CSV does not exist.
        ValueError: If the spots CSV does not have the expected TrackMate
            column signature.
    """
    from sleap_io.model.centroid import PredictedCentroid
    from sleap_io.model.instance import Track
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video as VideoClass

    spots_path = Path(spots_path)
    if not spots_path.exists():
        raise FileNotFoundError(f"Spots CSV not found: {spots_path}")

    # --- Auto-detect sibling files ---
    if edges_path is not None:
        edges_path = Path(edges_path)
    else:
        edges_path = _find_sibling(spots_path, "_edges")

    video_obj: VideoClass | None = None
    if video is not None:
        if isinstance(video, (str, Path)):
            video_obj = VideoClass(filename=str(video), open_backend=False)
        else:
            video_obj = video
    else:
        tif_path = _find_sibling(spots_path, ".tif")
        if tif_path is not None:
            video_obj = VideoClass(filename=str(tif_path), open_backend=False)

    # --- Parse edges (if available) ---
    target_to_cost: dict[int, float] = {}
    if edges_path is not None and edges_path.exists():
        target_to_cost = _parse_edges(edges_path)

    # --- Parse spots CSV ---
    with open(spots_path, newline="") as f:
        reader = csv.reader(f)

        # Read header row to find column indices.
        header = next(reader)
        if tuple(header[: len(_SPOTS_SIGNATURE)]) != _SPOTS_SIGNATURE:
            raise ValueError(
                f"Not a TrackMate spots CSV. Expected columns starting with "
                f"{_SPOTS_SIGNATURE}, got {tuple(header[:6])}."
            )

        col = {name: header.index(name) for name in header}

        # Skip remaining header rows.
        for _ in range(_HEADER_ROWS - 1):
            next(reader, None)

        # First pass: collect data rows and unique track IDs.
        rows: list[list[str]] = []
        track_ids: set[int] = set()
        for row in reader:
            if not row:
                continue
            rows.append(row)
            tid = row[col["TRACK_ID"]]
            if tid:
                track_ids.add(int(tid))

    # --- Build Track objects ---
    track_map: dict[int, Track] = {}
    for tid in sorted(track_ids):
        track_map[tid] = Track(name=f"Track_{tid}")

    tracks = list(track_map.values())

    # --- Build PredictedCentroid objects ---
    centroids: list[PredictedCentroid] = []
    for row in rows:
        spot_id = int(row[col["ID"]])
        tid_str = row[col["TRACK_ID"]]

        x = float(row[col["POSITION_X"]])
        y = float(row[col["POSITION_Y"]])

        z_val = float(row[col["POSITION_Z"]]) if "POSITION_Z" in col else 0.0
        z = z_val if z_val != 0.0 else None

        frame_idx = int(float(row[col["FRAME"]]))
        score = float(row[col["QUALITY"]])

        track = track_map.get(int(tid_str)) if tid_str else None

        tracking_score = target_to_cost.get(spot_id)

        label = row[col["LABEL"]]

        centroid = PredictedCentroid(
            x=x,
            y=y,
            z=z,
            video=video_obj,
            frame_idx=frame_idx,
            track=track,
            tracking_score=tracking_score,
            score=score,
            name=label,
            source="trackmate",
        )
        centroids.append(centroid)

    # --- Assemble Labels ---
    videos = [video_obj] if video_obj is not None else []
    labels = Labels(videos=videos, tracks=tracks, centroids=centroids)
    labels.provenance["filename"] = str(spots_path)
    return labels
