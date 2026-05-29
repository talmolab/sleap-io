"""This module handles direct I/O operations for working with DeepLabCut (DLC) files.

In addition to reading a single DLC annotation CSV (`load_dlc`), this module can
import an entire DLC *project* from its `config.yaml` (`load_dlc_project`) and
recover the train/test splits stored by `create_training_dataset`
(`load_dlc_splits`).

When a project `config.yaml` is available, the following extra metadata is
imported:

- **Skeleton edges** from the config ``skeleton:`` list (edges referencing
  bodyparts that were not labeled are dropped with a warning).
- **Source videos**: each ``labeled-data/<video>/`` image folder is linked back
  to its original video file (from the config ``video_sets``) via
  `Video.source_video`, matched by filename stem.

Cropping/ROI metadata (``video_sets[...].crop``) is not yet imported.
"""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.labels_set import LabelsSet
from sleap_io.model.skeleton import Node, Skeleton
from sleap_io.model.video import Video


def is_dlc_file(filename: str | Path) -> bool:
    """Check if file is a DLC CSV file.

    Args:
        filename: Path to file to check.

    Returns:
        True if file appears to be a DLC CSV file.
    """
    try:
        # Read first few lines as raw text to check for DLC structure
        with open(filename, "r") as f:
            lines = [f.readline().strip() for _ in range(4)]

        # Join all lines to search for DLC patterns
        content = "\n".join(lines).lower()

        # Check for DLC's characteristic patterns
        has_scorer = "scorer" in content
        has_coords = "coords" in content
        has_xy = "x" in content and "y" in content
        has_bodyparts = "bodyparts" in content or any(
            part in content for part in ["animal", "individual"]
        )

        return has_scorer and has_coords and has_xy and has_bodyparts

    except Exception:
        return False


# -----------------------------------------------------------------------------
# Config parsing and discovery
# -----------------------------------------------------------------------------

#: Keys that identify a mapping as a DLC project ``config.yaml``.
_DLC_CONFIG_KEYS = (
    "video_sets",
    "bodyparts",
    "scorer",
    "Task",
    "skeleton",
    "individuals",
)


def _read_dlc_config(path: str | Path) -> dict | None:
    """Read a DLC project ``config.yaml`` into a dictionary.

    Args:
        path: Path to the config YAML file.

    Returns:
        The parsed config as a dictionary, or `None` if the file is missing or
        cannot be parsed as a YAML mapping. A warning is emitted on failure so
        that a malformed or foreign config never breaks plain CSV loading.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(f"DLC config file not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(f"Failed to parse DLC config {path}: {e}")
        return None
    if not isinstance(cfg, dict):
        warnings.warn(f"DLC config {path} did not parse to a mapping.")
        return None
    return cfg


def _looks_like_dlc_config(cfg: dict) -> bool:
    """Return whether a parsed mapping looks like a DLC project config.

    Args:
        cfg: A parsed YAML mapping.

    Returns:
        True if at least two characteristic DLC config keys are present. This
        guards auto-discovery against picking up an unrelated ``config.yaml``.
    """
    if not isinstance(cfg, dict):
        return False
    return sum(key in cfg for key in _DLC_CONFIG_KEYS) >= 2


def _discover_config(csv_path: str | Path, max_levels: int = 3) -> Path | None:
    """Search upward from a CSV for a DLC project ``config.yaml``.

    DLC projects store annotations at ``<project>/labeled-data/<video>/`` with
    ``config.yaml`` at the project root, so the config typically sits two
    directories above the CSV.

    Args:
        csv_path: Path to a DLC annotation CSV.
        max_levels: Maximum number of parent directories to search.

    Returns:
        The path to a validated DLC ``config.yaml``, or `None` if none is found
        within `max_levels` levels.
    """
    start = Path(csv_path).resolve().parent
    for d in [start, *start.parents][: max_levels + 1]:
        candidate = d / "config.yaml"
        if candidate.is_file():
            cfg = _read_dlc_config(candidate)
            if cfg is not None and _looks_like_dlc_config(cfg):
                return candidate
    return None


def _resolve_config(
    csv_path: str | Path, config: str | Path | bool | None
) -> dict | None:
    """Resolve the ``config`` argument of `load_dlc` to a parsed config dict.

    Args:
        csv_path: Path to the DLC CSV being loaded (used for auto-discovery).
        config: Either `None` (auto-discover ``config.yaml`` by walking up from
            the CSV), `False` (disable config entirely for strict legacy
            output), or an explicit path to a ``config.yaml``.

    Returns:
        The parsed config dictionary, or `None` if disabled or not found.
    """
    if config is False:
        return None
    if config is None:
        discovered = _discover_config(csv_path)
        return _read_dlc_config(discovered) if discovered is not None else None
    return _read_dlc_config(config)


def _is_dlc_project_path(filename: str | Path) -> bool:
    """Return whether a path refers to a DLC project (for `load_file` routing).

    Args:
        filename: A directory or a ``config.yaml`` path.

    Returns:
        True if `filename` is a directory containing both ``config.yaml`` and
        ``labeled-data/``, or a ``config.yaml`` file that validates as a DLC
        project config.
    """
    p = Path(filename)
    if p.is_dir():
        return (p / "config.yaml").is_file() and (p / "labeled-data").is_dir()
    if p.name == "config.yaml" and p.is_file():
        cfg = _read_dlc_config(p)
        return cfg is not None and _looks_like_dlc_config(cfg)
    return False


def _attach_config_skeleton(skeleton: Skeleton, cfg: dict) -> None:
    """Attach skeleton edges (and name) from a DLC config to a `Skeleton`.

    Edges referencing bodyparts that are not present in the skeleton's nodes are
    dropped with a warning. Resolution is strictly name-based to avoid corrupting
    edges when the node order differs from the config.

    Args:
        skeleton: The `Skeleton` to mutate in place.
        cfg: A parsed DLC config dictionary.
    """
    task = cfg.get("Task")
    if task and skeleton.name is None:
        skeleton.name = str(task)

    raw_edges = cfg.get("skeleton") or []
    node_names = set(skeleton.node_names)
    valid: list[tuple[str, str]] = []
    dropped: list = []
    for entry in raw_edges:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            dropped.append(entry)
            continue
        src, dst = str(entry[0]), str(entry[1])
        if src in node_names and dst in node_names:
            valid.append((src, dst))
        else:
            dropped.append((src, dst))

    if valid:
        skeleton.add_edges(valid)
    if dropped:
        warnings.warn(
            f"Dropped {len(dropped)} DLC skeleton edge(s) referencing bodyparts "
            f"not present in the labeled data: {dropped}"
        )


def _video_sets_stem_map(cfg: dict) -> dict[str, str]:
    """Map video filename stems to original video paths from config ``video_sets``.

    Paths are normalized so Windows-style backslash separators are handled on any
    OS. Placeholder entries (e.g. DLC demo templates) are skipped.

    Args:
        cfg: A parsed DLC config dictionary.

    Returns:
        A dictionary mapping each video's filename stem to its original path
        string (preserving config order).
    """
    out: dict[str, str] = {}
    video_sets = cfg.get("video_sets") or {}
    for key in video_sets.keys():
        key_str = str(key)
        if "WILL BE AUTOMATICALLY UPDATED" in key_str:
            continue
        name = key_str.replace("\\", "/").rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem:
            out[stem] = key_str
    return out


def _set_source_video(
    video: Video,
    folder_name: str,
    stem_map: dict[str, str],
    video_search_paths: list[str | Path] | None = None,
) -> None:
    """Link an image-folder `Video` back to its original source video.

    Args:
        video: The image-sequence `Video` to annotate (mutated in place).
        folder_name: The ``labeled-data/<folder>`` name for this video.
        stem_map: Mapping of video stems to original paths (see
            `_video_sets_stem_map`).
        video_search_paths: Optional paths to search for the original video file
            by basename (best-effort path repair).

    Notes:
        The source `Video` is created with ``open_backend=False`` so a missing
        original file (the common case on import) never raises. On a stem
        mismatch, `video.source_video` is left as `None`.
    """
    original = stem_map.get(folder_name)
    if original is None:
        return

    path = original
    if video_search_paths:
        basename = original.replace("\\", "/").rsplit("/", 1)[-1]
        for search_path in video_search_paths:
            candidate = Path(search_path) / basename
            if candidate.exists():
                path = str(candidate)
                break

    video.source_video = Video(filename=path, open_backend=False)


# -----------------------------------------------------------------------------
# CSV reading
# -----------------------------------------------------------------------------


def _read_dlc_dataframe(filename: str | Path) -> tuple[pd.DataFrame, bool]:
    """Read a DLC annotation CSV into a DataFrame with a flattened string index.

    Args:
        filename: Path to a DLC annotation CSV.

    Returns:
        A tuple of `(df, is_multianimal)` where `df` has image paths as its
        (string) index and multi-level columns, and `is_multianimal` indicates
        whether the file uses the multi-animal individuals/bodyparts/coords
        layout.
    """
    filename = Path(filename)

    # Try reading first few rows to determine format.
    try:
        # Try multi-animal format first (header rows 1-3, skipping scorer row).
        peek = pd.read_csv(filename, header=[1, 2, 3], nrows=2)
        is_multianimal = peek.columns[0][0] == "individuals"
        is_multiindex = peek.iloc[0, 0] == "labeled-data"
    except Exception:
        # Fall back to single-animal format.
        is_multianimal = False
        is_multiindex = False

    # Older DLC versions store image paths with OS-specific path separators.
    # Newer versions store paths without separators as MultiIndex.
    index_col = [0, 1, 2] if is_multiindex else 0

    # Multi-animal format: skip scorer row, use individuals/bodyparts/coords.
    # Single-animal format: use scorer/bodyparts/coords.
    header = [1, 2, 3] if is_multianimal else [0, 1, 2]

    df = pd.read_csv(filename, header=header, index_col=index_col)
    # Flatten MultiIndex to match older DLC format.
    if is_multiindex:
        df.index = df.index.map("/".join)

    return df, is_multianimal


# -----------------------------------------------------------------------------
# Single-CSV loading
# -----------------------------------------------------------------------------


def load_dlc(
    filename: str | Path,
    video_search_paths: list[str | Path] | None = None,
    config: str | Path | bool | None = None,
    **kwargs,
) -> Labels:
    """Load DeepLabCut annotations from a CSV file.

    Args:
        filename: Path to DLC CSV file.
        video_search_paths: List of paths to search for video files.
        config: Path to a DLC project ``config.yaml``. When provided (or
            auto-discovered), skeleton edges and source-video links are imported
            in addition to the pose annotations. Pass `None` (the default) to
            auto-discover ``config.yaml`` by walking up from the CSV, an explicit
            path to force a specific config, or `False` to disable config use
            entirely (strict legacy output).
        **kwargs: Additional arguments (unused).

    Returns:
        Labels object with loaded data.

    Notes:
        When a config is found, the returned `Labels` gains skeleton edges and
        per-video `Video.source_video` links that were previously absent. Pass
        ``config=False`` to reproduce the legacy config-free output.
    """
    cfg = _resolve_config(filename, config)
    return _load_dlc_csv(filename, config=cfg, video_search_paths=video_search_paths)


def _load_dlc_csv(
    filename: str | Path,
    *,
    config: dict | None = None,
    video_search_paths: list[str | Path] | None = None,
    skeleton: Skeleton | None = None,
    tracks: list[Track] | None = None,
) -> Labels:
    """Load a single DLC annotation CSV into a `Labels` object.

    Args:
        filename: Path to a DLC annotation CSV.
        config: A parsed DLC config dictionary (not a path), or `None`. When
            provided, skeleton edges and source-video links are applied.
        video_search_paths: Optional paths to search for original video files.
        skeleton: An optional pre-built `Skeleton` to reuse (shared across a
            project load). When provided, structure parsing and edge attachment
            are skipped so all frames reference the same skeleton.
        tracks: An optional pre-built list of `Track`s to reuse alongside
            `skeleton`.

    Returns:
        A `Labels` object for this CSV.
    """
    filename = Path(filename)
    df, is_multianimal = _read_dlc_dataframe(filename)

    # Parse structure based on format (unless a shared skeleton was provided).
    if skeleton is None:
        if is_multianimal:
            skeleton, tracks = _parse_multi_animal_structure(df)
        else:
            skeleton = _parse_single_animal_structure(df)
            tracks = []
        if config is not None:
            _attach_config_skeleton(skeleton, config)
    elif tracks is None:
        tracks = []

    # First, group all image paths by their video directory.
    video_image_paths = {}
    frame_map = {}  # Maps image path to frame index

    for idx in df.index:
        img_path = str(idx)
        frame_idx = _extract_frame_index(img_path)
        frame_map[img_path] = frame_idx

        # Extract video name from path
        # e.g., "labeled-data/video/img000.png" -> "video"
        path_parts = Path(img_path).parts
        if len(path_parts) >= 2 and path_parts[0] == "labeled-data":
            video_name = path_parts[1]
        else:
            video_name = Path(img_path).parent.name or "default"

        if video_name not in video_image_paths:
            video_image_paths[video_name] = []
        video_image_paths[video_name].append(img_path)

    # Create one Video object per video directory.
    videos = {}
    for video_name, image_paths in video_image_paths.items():
        # Sort image paths to ensure consistent ordering.
        sorted_paths = sorted(image_paths, key=lambda p: frame_map[p])

        # Find the actual image files.
        actual_image_files = []
        for img_path in sorted_paths:
            # First try the full path from CSV.
            full_path = filename.parent / img_path
            if full_path.exists():
                actual_image_files.append(str(full_path))
            else:
                # Try just the filename in the same directory as the CSV.
                img_name = Path(img_path).name
                simple_path = filename.parent / img_name
                if simple_path.exists():
                    actual_image_files.append(str(simple_path))
                else:
                    # Try going up one directory from CSV location
                    # (CSV in subdir references parent/subdir/img.png).
                    parent_path = filename.parent.parent / img_path
                    if parent_path.exists():
                        actual_image_files.append(str(parent_path))

        # Only create video if we found actual images.
        if actual_image_files:
            videos[video_name] = Video.from_filename(actual_image_files)

    # Link image folders back to their original videos from config video_sets.
    if config is not None and videos:
        stem_map = _video_sets_stem_map(config)
        for video_name, video in videos.items():
            _set_source_video(video, video_name, stem_map, video_search_paths)

    # Parse the actual data rows and create labeled frames.
    labeled_frames = []
    for idx, row in df.iterrows():
        # Get image path from index.
        img_path = str(idx)

        # Determine which video this frame belongs to.
        path_parts = Path(img_path).parts
        if len(path_parts) >= 2 and path_parts[0] == "labeled-data":
            video_name = path_parts[1]
        else:
            video_name = Path(img_path).parent.name or "default"

        # Skip if we don't have a video for this frame.
        if video_name not in videos:
            continue

        # Parse instances for this frame.
        if is_multianimal:
            instances = _parse_multi_animal_row(row, skeleton, tracks)
        else:
            instances = _parse_single_animal_row(row, skeleton)

        # Get the index of this image within its video.
        sorted_video_paths = sorted(
            video_image_paths[video_name], key=lambda p: frame_map[p]
        )
        video_frame_idx = sorted_video_paths.index(img_path)
        labeled_frames.append(
            LabeledFrame(
                video=videos[video_name],
                frame_idx=video_frame_idx,
                instances=instances,
            )
        )

    unique_videos = list(videos.values())

    return Labels(
        labeled_frames=labeled_frames,
        videos=unique_videos,
        tracks=tracks,
        skeletons=[skeleton] if skeleton.nodes else [],
    )


# -----------------------------------------------------------------------------
# Project loading
# -----------------------------------------------------------------------------


def _resolve_project_config_path(config: str | Path) -> Path:
    """Resolve a project argument to a ``config.yaml`` path.

    Args:
        config: A path to a ``config.yaml`` file or to a project directory
            containing one.

    Returns:
        The path to the ``config.yaml`` file.

    Raises:
        FileNotFoundError: If a directory is given but contains no
            ``config.yaml``.
    """
    p = Path(config)
    if p.is_dir():
        candidate = p / "config.yaml"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No config.yaml found in DLC project directory: {p}")
    return p


def _find_project_csvs(project_dir: Path, scorer: str | None) -> list[tuple[str, Path]]:
    """Find per-video annotation CSVs under ``labeled-data/``.

    Args:
        project_dir: The DLC project root (the directory containing
            ``config.yaml``).
        scorer: The project scorer name (used to locate
            ``CollectedData_<scorer>.csv``).

    Returns:
        A list of ``(folder_name, csv_path)`` tuples, sorted by folder name.
    """
    labeled_dir = project_dir / "labeled-data"
    folders: list[tuple[str, Path]] = []
    if not labeled_dir.is_dir():
        return folders
    for sub in sorted(labeled_dir.iterdir()):
        if not sub.is_dir():
            continue
        csv = sub / f"CollectedData_{scorer}.csv"
        if not csv.is_file():
            # Fall back to any DLC-looking CSV in the folder.
            candidates = [c for c in sorted(sub.glob("*.csv")) if is_dlc_file(c)]
            if not candidates:
                continue
            csv = candidates[0]
        folders.append((sub.name, csv))
    return folders


def load_dlc_project(
    config: str | Path,
    video_search_paths: list[str | Path] | None = None,
) -> Labels:
    """Load an entire DeepLabCut project from its ``config.yaml``.

    All ``labeled-data/<video>/`` folders are loaded and merged into a single
    `Labels` object that shares one `Skeleton` (with edges from the config) and
    one set of `Track`s. Each per-video `Video` is linked back to its original
    video file via `Video.source_video` when available.

    Args:
        config: Path to a DLC project ``config.yaml``, or to a project directory
            containing one.
        video_search_paths: Optional paths to search for original video files.

    Returns:
        A `Labels` object with frames from every labeled video in the project.

    Raises:
        ValueError: If the config cannot be read or no annotation CSVs are found.

    Notes:
        The returned `Labels` records DLC provenance under the keys
        ``"dlc_project"``, ``"dlc_scorer"`` and ``"dlc_task"``.
    """
    config_path = _resolve_project_config_path(config)
    cfg = _read_dlc_config(config_path)
    if cfg is None:
        raise ValueError(f"Could not read DLC config: {config_path}")

    project_dir = config_path.parent
    scorer = cfg.get("scorer")
    folders = _find_project_csvs(project_dir, scorer)
    if not folders:
        raise ValueError(
            f"No DLC annotation CSVs found under {project_dir / 'labeled-data'}"
        )

    # Build a single shared skeleton and track list across all videos so that
    # every instance references the same objects (one skeleton, deduped tracks).
    node_names: list[str] = []
    track_names: list[str] = []
    for _, csv in folders:
        df, is_multianimal = _read_dlc_dataframe(csv)
        if is_multianimal:
            folder_skeleton, folder_tracks = _parse_multi_animal_structure(df)
            for track in folder_tracks:
                if track.name not in track_names:
                    track_names.append(track.name)
        else:
            folder_skeleton = _parse_single_animal_structure(df)
        for name in folder_skeleton.node_names:
            if name not in node_names:
                node_names.append(name)

    shared_skeleton = Skeleton(nodes=[Node(name) for name in sorted(set(node_names))])
    _attach_config_skeleton(shared_skeleton, cfg)
    shared_tracks = [Track(name=name) for name in track_names]

    # Load each folder using the shared skeleton/tracks and collect the frames.
    all_frames: list[LabeledFrame] = []
    videos: list[Video] = []
    for _, csv in folders:
        folder_labels = _load_dlc_csv(
            csv,
            config=cfg,
            video_search_paths=video_search_paths,
            skeleton=shared_skeleton,
            tracks=shared_tracks,
        )
        all_frames.extend(folder_labels.labeled_frames)
        videos.extend(folder_labels.videos)

    labels = Labels(
        labeled_frames=all_frames,
        videos=videos,
        tracks=shared_tracks,
        skeletons=[shared_skeleton] if shared_skeleton.nodes else [],
    )
    labels.provenance.update(
        {
            "dlc_project": str(config_path),
            "dlc_scorer": scorer,
            "dlc_task": cfg.get("Task"),
        }
    )
    return labels


# -----------------------------------------------------------------------------
# Training-set splits
# -----------------------------------------------------------------------------


def _get_training_set_folder(
    project_dir: Path, cfg: dict, iteration: int | None
) -> Path:
    """Return the ``UnaugmentedDataSet`` folder for a project iteration.

    Mirrors DLC's ``get_training_set_folder``: the path is
    ``training-datasets/iteration-<iteration>/UnaugmentedDataSet_<Task><date>``
    (note: ``Task`` and ``date`` are concatenated with no separator).

    Args:
        project_dir: The DLC project root.
        cfg: The parsed DLC config.
        iteration: The iteration index, or `None` to use ``cfg['iteration']``.

    Returns:
        The path to the unaugmented training-set folder.
    """
    it = cfg.get("iteration", 0) if iteration is None else iteration
    task = cfg.get("Task", "")
    date = cfg.get("date", "")
    return (
        project_dir
        / "training-datasets"
        / f"iteration-{it}"
        / f"UnaugmentedDataSet_{task}{date}"
    )


def _select_documentation_pickle(
    project_dir: Path,
    cfg: dict,
    train_fraction: float | None,
    shuffle: int | None,
    iteration: int | None,
) -> Path:
    """Locate the ``Documentation_data-*.pickle`` for the requested split.

    Args:
        project_dir: The DLC project root.
        cfg: The parsed DLC config.
        train_fraction: The training fraction to select (e.g. ``0.95``), or
            `None` to leave unconstrained.
        shuffle: The shuffle index to select, or `None` to leave unconstrained.
        iteration: The iteration index, or `None` to use ``cfg['iteration']``.

    Returns:
        The path to the selected Documentation pickle.

    Raises:
        FileNotFoundError: If the training-set folder or any pickle is missing.
        ValueError: If the selection is ambiguous (multiple pickles match).
    """
    trainset_dir = _get_training_set_folder(project_dir, cfg, iteration)
    pickles = (
        sorted(trainset_dir.glob("Documentation_data-*.pickle"))
        if trainset_dir.is_dir()
        else []
    )
    if not pickles:
        raise FileNotFoundError(
            f"No DLC Documentation_data-*.pickle found in {trainset_dir}. "
            "Run create_training_dataset in DLC to generate splits."
        )

    pattern = re.compile(
        r"Documentation_data-(?P<task>.+)_(?P<frac>\d+)shuffle(?P<shuffle>\d+)\.pickle$"
    )
    parsed: list[tuple[Path, int, int]] = []
    for p in pickles:
        m = pattern.match(p.name)
        if m is not None:
            parsed.append((p, int(m.group("frac")), int(m.group("shuffle"))))

    if not parsed:
        # Cannot parse selectors; only safe when unambiguous.
        if len(pickles) == 1:
            return pickles[0]
        raise ValueError(
            f"Could not parse train_fraction/shuffle from pickles in {trainset_dir}: "
            f"{[p.name for p in pickles]}"
        )

    candidates = parsed
    if train_fraction is not None:
        frac_int = int(round(train_fraction * 100))
        candidates = [c for c in candidates if c[1] == frac_int]
    if shuffle is not None:
        candidates = [c for c in candidates if c[2] == shuffle]

    if not candidates:
        available = [(p.name, f, s) for (p, f, s) in parsed]
        raise FileNotFoundError(
            f"No Documentation pickle matched train_fraction={train_fraction}, "
            f"shuffle={shuffle}. Available: {available}"
        )
    if len(candidates) > 1:
        available = [(p.name, f, s) for (p, f, s) in candidates]
        raise ValueError(
            "Multiple DLC splits found; specify train_fraction and/or shuffle. "
            f"Available (name, train%, shuffle): {available}"
        )
    return candidates[0][0]


def _read_dlc_split(pickle_path: str | Path) -> tuple[list[int], list[int]]:
    """Read train/test positional indices from a DLC Documentation pickle.

    The pickle is a 4-element list ``[data, trainIndices, testIndices,
    trainFraction]``. Only ``trainIndices`` (``meta[1]``) and ``testIndices``
    (``meta[2]``) are used; ``meta[0]`` is the lossy train-only record list and
    is intentionally ignored.

    Args:
        pickle_path: Path to the Documentation pickle.

    Returns:
        A tuple ``(train_indices, test_indices)`` of integer positional indices
        into the globally merged frame order.
    """
    with open(pickle_path, "rb") as f:
        meta = pickle.load(f)
    train = [int(i) for i in meta[1] if int(i) != -1]
    test = [int(i) for i in meta[2] if int(i) != -1]
    return train, test


def _dlc_merged_order(project_dir: Path, cfg: dict) -> list[tuple[str, str]]:
    """Reconstruct DLC's globally merged frame order as ``(folder, filename)``.

    DLC builds its training set by concatenating each video's
    ``CollectedData_<scorer>`` frame (in ``video_sets`` order, deduped by stem,
    skipping folders whose annotation file is missing or labeled by a different
    scorer) and then sorting the whole thing lexicographically with
    ``DataFrame.sort_index()``. The train/test indices in the Documentation
    pickle are positional indices into this merged order.

    Args:
        project_dir: The DLC project root.
        cfg: The parsed DLC config.

    Returns:
        A list of ``(folder, filename)`` tuples in DLC's merged positional
        order. Position ``i`` corresponds to split index ``i``.
    """
    scorer = cfg.get("scorer")
    stem_map = _video_sets_stem_map(cfg)

    # Determine the included folders, mirroring DLC's merge skip-rules.
    # ``stem_map`` is keyed by stem, so stems are already unique.
    included: list[tuple[str, Path]] = []
    for stem in stem_map.keys():
        csv = project_dir / "labeled-data" / stem / f"CollectedData_{scorer}.csv"
        if not csv.is_file():
            # Folder absent / not annotated -> skipped by DLC too.
            continue
        csv_scorer = _read_csv_scorer(csv)
        if scorer is not None and csv_scorer is not None and csv_scorer != scorer:
            warnings.warn(
                f"Skipping {csv} labeled by '{csv_scorer}' (project scorer is "
                f"'{scorer}'); this matches DLC's training-set merge behavior."
            )
            continue
        included.append((stem, csv))

    if not included:
        # Fallback: video_sets stems did not match any labeled-data folder.
        # Include every folder with a DLC CSV so splits remain usable.
        for folder, csv in _find_project_csvs(project_dir, scorer):
            included.append((folder, csv))

    merged: list[tuple[str, str]] = []
    for _, csv in included:
        df, _ = _read_dlc_dataframe(csv)
        for idx in df.index:
            p = Path(str(idx))
            merged.append((p.parent.name, p.name))

    # DLC applies a global lexicographic sort_index() across all merged frames.
    merged.sort()
    return merged


def _read_csv_scorer(csv: str | Path) -> str | None:
    """Read the scorer name from the first row of a DLC CSV.

    Args:
        csv: Path to a DLC annotation CSV.

    Returns:
        The scorer name (the second field of the first line), or `None` if it
        cannot be determined.
    """
    try:
        with open(csv, "r") as f:
            first = f.readline().strip()
    except Exception:  # pragma: no cover - defensive
        return None
    parts = first.split(",")
    return parts[1] if len(parts) > 1 else None


def _warn_if_nonlexicographic(merged: list[tuple[str, str]]) -> None:
    """Warn if numeric filename order differs from DLC's lexicographic order.

    Args:
        merged: The merged ``(folder, filename)`` order from `_dlc_merged_order`.
    """

    def numeric_key(key: tuple[str, str]):
        folder, fname = key
        nums = re.findall(r"\d+", fname)
        return (folder, int(nums[-1]) if nums else -1, fname)

    if sorted(merged) != sorted(merged, key=numeric_key):
        warnings.warn(
            "DLC split import: image filenames are not zero-padded, so DLC's "
            "lexicographic ordering differs from numeric order (e.g. 'img10' < "
            "'img2'). Train/test assignment follows DLC's lexicographic order; "
            "verify the result."
        )


def load_dlc_splits(
    config: str | Path,
    shuffle: int | None = None,
    train_fraction: float | None = None,
    iteration: int | None = None,
    video_search_paths: list[str | Path] | None = None,
) -> LabelsSet:
    """Load DeepLabCut train/test splits from a project's Documentation pickle.

    Args:
        config: Path to a DLC project ``config.yaml`` (or its project directory).
        shuffle: The shuffle index to load. Required if the project has more than
            one shuffle.
        train_fraction: The training fraction to load (e.g. ``0.95``). Required
            if the project has more than one training fraction.
        iteration: The project iteration. Defaults to ``cfg['iteration']``.
        video_search_paths: Optional paths to search for original video files.

    Returns:
        A `LabelsSet` with ``"train"`` and ``"test"`` keys, each a `Labels`
        containing the frames assigned to that split.

    Raises:
        FileNotFoundError: If no matching Documentation pickle exists.
        ValueError: If the config cannot be read or the split selection is
            ambiguous.

    Notes:
        Splits require the labeled images to be present on disk so that each
        merged frame maps to a loaded `LabeledFrame`. DLC stores split membership
        as positional indices into a globally, lexicographically sorted merge of
        all per-video annotations; this function reconstructs that order. For
        non-zero-padded filenames, a warning is emitted because lexicographic and
        numeric orderings diverge.
    """
    config_path = _resolve_project_config_path(config)
    cfg = _read_dlc_config(config_path)
    if cfg is None:
        raise ValueError(f"Could not read DLC config: {config_path}")
    project_dir = config_path.parent

    # Load the full project, then partition its frames into train/test.
    labels = load_dlc_project(config_path, video_search_paths=video_search_paths)

    merged = _dlc_merged_order(project_dir, cfg)
    _warn_if_nonlexicographic(merged)

    pickle_path = _select_documentation_pickle(
        project_dir, cfg, train_fraction, shuffle, iteration
    )
    train_indices, test_indices = _read_dlc_split(pickle_path)

    # Build a lookup from (folder, filename) -> global LabeledFrame index.
    lf_lookup: dict[tuple[str, str], int] = {}
    for global_idx, lf in enumerate(labels.labeled_frames):
        fname = lf.video.filename
        if isinstance(fname, list):
            fname = fname[lf.frame_idx]
        p = Path(str(fname))
        lf_lookup[(p.parent.name, p.name)] = global_idx

    def map_indices(indices: list[int]) -> list[int]:
        out: list[int] = []
        for i in indices:
            if 0 <= i < len(merged):
                global_idx = lf_lookup.get(merged[i])
                if global_idx is not None:
                    out.append(global_idx)
        return out

    train_global = map_indices(train_indices)
    test_global = map_indices(test_indices)

    train_labels = labels.extract(train_global, copy=True)
    test_labels = labels.extract(test_global, copy=True)

    return LabelsSet({"train": train_labels, "test": test_labels})


# -----------------------------------------------------------------------------
# Structure / row parsing (unchanged behavior)
# -----------------------------------------------------------------------------


def _parse_multi_animal_structure(df: pd.DataFrame) -> tuple[Skeleton, list[Track]]:
    """Parse multi-animal DLC structure to extract skeleton and tracks."""
    # Extract unique node names and track names from columns
    tracks_dict = {}
    node_names = []

    # Iterate through columns (skip coords columns)
    for col in df.columns:
        if len(col) >= 3:  # Multi-level column (individuals, bodyparts, coords)
            individual = col[0]
            bodypart = col[1]
            coord = col[2]

            if coord == "x":  # Only process x coordinates to avoid duplicates
                # Add track (skip the header row name)
                if individual not in tracks_dict and individual not in [
                    "",
                    None,
                    "individuals",
                ]:
                    tracks_dict[individual] = Track(name=individual)

                # Add node (skip the header row name)
                if bodypart not in node_names and bodypart not in [
                    "",
                    None,
                    "bodyparts",
                ]:
                    node_names.append(bodypart)

    # Create skeleton with all unique nodes
    nodes = [Node(name=name) for name in sorted(set(node_names))]
    skeleton = Skeleton(nodes=nodes)

    # Create track list
    tracks = list(tracks_dict.values())

    return skeleton, tracks


def _parse_single_animal_structure(df: pd.DataFrame) -> Skeleton:
    """Parse single-animal DLC structure to extract skeleton."""
    # Extract node names from bodyparts level
    node_names = []

    for col in df.columns:
        if len(col) >= 3:  # Multi-level column
            bodypart = col[1]
            coord = col[2]

            if (
                coord == "x"
                and bodypart not in node_names
                and bodypart not in ["", None]
            ):
                node_names.append(bodypart)

    # Create skeleton
    nodes = [Node(name=name) for name in sorted(set(node_names))]
    skeleton = Skeleton(nodes=nodes)

    return skeleton


def _parse_multi_animal_row(
    row: pd.Series, skeleton: Skeleton, tracks: list[Track]
) -> list[Instance]:
    """Parse a row of multi-animal DLC data."""
    instances_dict = {}

    # Group data by individual
    for col_tuple, value in row.items():
        if len(col_tuple) >= 3:
            individual = col_tuple[0]
            bodypart = col_tuple[1]
            coord = col_tuple[2]

            # Skip empty individuals or header names
            if not individual or individual == "" or individual == "individuals":
                continue

            # Initialize instance data if needed
            if individual not in instances_dict:
                instances_dict[individual] = {}

            # Store coordinate data
            if bodypart and bodypart != "":
                if bodypart not in instances_dict[individual]:
                    instances_dict[individual][bodypart] = {}
                instances_dict[individual][bodypart][coord] = value

    # Create instances
    instances = []
    for individual_name, bodyparts_data in instances_dict.items():
        # Find matching track
        track = next((t for t in tracks if t.name == individual_name), None)

        # Create instance
        points = np.full((len(skeleton.nodes), 2), np.nan)
        has_valid_points = False

        for node_idx, node in enumerate(skeleton.nodes):
            if node.name in bodyparts_data:
                coords = bodyparts_data[node.name]
                if "x" in coords and "y" in coords:
                    x_val = coords["x"]
                    y_val = coords["y"]
                    if pd.notna(x_val) and pd.notna(y_val):
                        points[node_idx] = [float(x_val), float(y_val)]
                        has_valid_points = True

        # Only create instance if it has at least one valid point
        if has_valid_points:
            instance = Instance.from_numpy(
                points_data=points, skeleton=skeleton, track=track
            )
            instances.append(instance)

    return instances


def _parse_single_animal_row(row: pd.Series, skeleton: Skeleton) -> list[Instance]:
    """Parse a row of single-animal DLC data."""
    # Create instance
    points = np.full((len(skeleton.nodes), 2), np.nan)
    has_valid_points = False

    # Collect coordinates for each bodypart
    bodyparts_data = {}
    for col_tuple, value in row.items():
        if len(col_tuple) >= 3:
            bodypart = col_tuple[1]
            coord = col_tuple[2]

            if bodypart and bodypart != "":
                if bodypart not in bodyparts_data:
                    bodyparts_data[bodypart] = {}
                bodyparts_data[bodypart][coord] = value

    # Fill in points
    for node_idx, node in enumerate(skeleton.nodes):
        if node.name in bodyparts_data:
            coords = bodyparts_data[node.name]
            if "x" in coords and "y" in coords:
                x_val = coords["x"]
                y_val = coords["y"]
                if pd.notna(x_val) and pd.notna(y_val):
                    points[node_idx] = [float(x_val), float(y_val)]
                    has_valid_points = True

    # Only return instance if it has at least one valid point
    if has_valid_points:
        instance = Instance.from_numpy(points_data=points, skeleton=skeleton)
        return [instance]

    return []


def _extract_frame_index(img_path: str) -> int:
    """Extract frame index from image filename."""
    # Look for numbers in filename
    matches = re.findall(r"(\d+)", Path(img_path).stem)
    if matches:
        return int(matches[-1])  # Use last number found
    return 0
