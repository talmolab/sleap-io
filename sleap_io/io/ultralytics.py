"""Handles direct I/O operations for working with Ultralytics YOLO formats.

Ultralytics YOLO format specification:
- Directory structure: dataset_root/split/images/ and dataset_root/split/labels/
- Configuration: data.yaml file defining dataset structure
- Supported tasks:
  - Pose: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ... xn yn vn
  - Detection: class_id x_center y_center width height [confidence]
  - Segmentation: class_id x1 y1 x2 y2 ... xn yn
- Coordinates: Normalized to [0,1] range, origin at top-left
- Visibility (pose only): 0=not labeled, 1=labeled but occluded (coordinates are an
  estimate), 2=visible, mapped onto the `visible` and `labeled` point flags the same
  way as COCO keypoints (see `sleap_io.io.coco.decode_visibility`)
"""

from __future__ import annotations

import multiprocessing
import warnings
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import yaml
from tqdm import tqdm

from sleap_io.io.coco import (
    _check_visibility_options,
    decode_visibility,
    encode_visibility,
    resolve_visibility_encoding,
)
from sleap_io.model.bbox import BoundingBox, PredictedBoundingBox, UserBoundingBox
from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.roi import ROI, UserROI
from sleap_io.model.skeleton import Edge, Node, Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.labels_set import LabelsSet


def read_labels(
    dataset_path: str,
    split: str = "train",
    skeleton: Skeleton | None = None,
    image_size: tuple[int, int] = (480, 640),
    visibility_encoding: str = "auto",
    v0_as: str | None = None,
) -> Labels:
    """Read Ultralytics YOLO dataset and return a `Labels` object.

    Automatically detects the annotation format (pose, detection, or segmentation)
    from the label file content.

    Keypoint visibility codes map onto the point flags as for COCO (see
    `sleap_io.io.coco.decode_visibility`): with the ternary encoding, `v=0` reads as
    unlabeled, `v=1` as occluded with its coordinates kept, and `v=2` as visible.
    Every labeled keypoint is marked `verified`.

    Args:
        dataset_path: Path to the Ultralytics dataset root directory containing
            data.yaml.
        split: Dataset split to read ('train', 'val', or 'test'). Defaults to 'train'.
        skeleton: Optional skeleton to use. If not provided, will be inferred from
            data.yaml. Only required for pose format.
        image_size: Image dimensions (height, width) for coordinate denormalization.
                   Defaults to (480, 640). Will attempt to infer from actual images if
                   available.
        visibility_encoding: How to read keypoint visibility codes: ``"ternary"``
            (0/1/2), ``"binary"`` (``v>0`` is visible), or ``"auto"`` (the default),
            which picks binary, with a warning, when some keypoint in the split has
            ``v=1`` and none has ``v=2``.
        v0_as: ``"unlabeled"`` reads ``v=0`` keypoints as unlabeled. ``"absent"``
            reads them as labeled and not visible with no location, as sleap-io did
            before 0.10. ``None`` (the default) picks ``"absent"`` when the encoding
            is (or ``"auto"`` resolves to) binary, since binary files write every
            hidden keypoint as ``v=0``, and ``"unlabeled"`` otherwise.

    Returns:
        Parsed labels as a `Labels` instance.

    Raises:
        ValueError: If ``visibility_encoding`` or ``v0_as`` is not an accepted value.
    """
    _check_visibility_options(visibility_encoding, v0_as)
    dataset_path = Path(dataset_path)

    # Parse data.yaml configuration
    if dataset_path.name == "data.yaml":
        # If path already points to data.yaml, use its parent as dataset path
        data_yaml_path = dataset_path
        dataset_path = dataset_path.parent
    else:
        data_yaml_path = dataset_path / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    config = parse_data_yaml(data_yaml_path)

    # Use provided skeleton or create from config (may be None for non-pose tasks)
    if skeleton is None and "kpt_shape" in config:
        skeleton = create_skeleton_from_config(config)

    # Build class names mapping
    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        class_names = {i: name for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        class_names = raw_names
    else:
        class_names = {}

    # Get paths for the specified split
    split_path = config.get(split, f"{split}/images")
    images_dir = dataset_path / split_path
    labels_dir = dataset_path / split_path.replace("/images", "/labels")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Pick one visibility encoding for the whole split.
    visibility_encoding = resolve_visibility_encoding(
        visibility_encoding,
        _iter_visibility_codes(sorted(labels_dir.glob("*.txt"))),
        source=str(labels_dir),
    )

    # Process all image/label pairs
    labeled_frames = []
    tracks = {}  # Track synthetic tracks by instance order

    for image_file in sorted(images_dir.glob("*")):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            label_file = labels_dir / f"{image_file.stem}.txt"

            # Create video object for this image
            video = Video.from_filename(str(image_file))

            # Parse label file if it exists
            instances: list[Instance] = []
            rois: list[ROI] = []
            bboxes: list[BoundingBox] = []
            if label_file.exists():
                # Get image dimensions - try from video shape first, fallback to
                # reading image
                if video.shape is not None:
                    img_shape = video.shape[1:3]
                else:
                    # Read first frame to get dimensions
                    img = video[0]
                    img_shape = img.shape[:2]

                # Create a default skeleton for parse_label_file if none exists
                parse_skeleton = skeleton if skeleton is not None else Skeleton()
                instances, rois, bboxes = parse_label_file(
                    label_file,
                    parse_skeleton,
                    img_shape,
                    class_names=class_names,
                    video=video,
                    frame_idx=0,
                    visibility_encoding=visibility_encoding,
                    v0_as=v0_as,
                )

                # Assign tracks to instances based on order
                for i, instance in enumerate(instances):
                    track_name = f"track_{i}"
                    if track_name not in tracks:
                        tracks[track_name] = Track(name=track_name)
                    instance.track = tracks[track_name]

            # Create labeled frame with annotations attached
            frame = LabeledFrame(video=video, frame_idx=0, instances=instances)
            frame.rois.extend(rois)
            frame.bboxes.extend(bboxes)
            labeled_frames.append(frame)

    skeletons = [skeleton] if skeleton is not None else []

    return Labels(
        labeled_frames=labeled_frames,
        skeletons=skeletons,
        tracks=list(tracks.values()),
        provenance={"source": str(dataset_path), "split": split},
    )


def read_labels_set(
    dataset_path: str,
    splits: list[str] | None = None,
    skeleton: Skeleton | None = None,
    image_size: tuple[int, int] = (480, 640),
    visibility_encoding: str = "auto",
    v0_as: str | None = None,
) -> LabelsSet:
    """Read multiple splits from an Ultralytics dataset as a LabelsSet.

    Args:
        dataset_path: Path to the root directory of the Ultralytics dataset.
        splits: List of split names to load (e.g., ["train", "val", "test"]).
            If None, will attempt to load all available splits.
        skeleton: Skeleton to use for the dataset. If None, will attempt to
            load from data.yaml file in the dataset root.
        image_size: Default image size (height, width) to use if unable to
            determine from the actual images. Default: (480, 640).
        visibility_encoding: ``"auto"`` (the default), ``"ternary"`` or
            ``"binary"``. ``"auto"`` is resolved per split. See `read_labels`.
        v0_as: ``"unlabeled"``, ``"absent"`` or ``None`` (the default, which
            depends on the encoding). See `read_labels`.

    Returns:
        A LabelsSet containing Labels objects for each split.

    Example:
        >>> labels_set = read_labels_set("path/to/yolo_dataset/")
        >>> train_labels = labels_set["train"]
        >>> val_labels = labels_set["val"]
    """
    from sleap_io.model.labels_set import LabelsSet

    dataset_path = Path(dataset_path)

    # If no splits specified, try to detect available splits
    if splits is None:
        splits = []
        for split_name in ["train", "val", "test", "valid"]:
            if (dataset_path / split_name).exists():
                splits.append(split_name)

        if not splits:
            raise ValueError(f"No splits found in dataset path: {dataset_path}")

    # Try to load skeleton from data.yaml if not provided
    if skeleton is None:
        data_yaml_path = dataset_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path, "r") as f:
                data_config = yaml.safe_load(f)

            # Try to create skeleton from our custom metadata first
            if "node_names" in data_config and "skeleton" in data_config:
                try:
                    node_names = data_config["node_names"]
                    skeleton_connections = data_config["skeleton"]

                    # Create nodes from names
                    nodes = [Node(name) for name in node_names]

                    # Create edges from skeleton connections
                    edges = []
                    for connection in skeleton_connections:
                        if len(connection) == 2:
                            src_idx, dst_idx = connection
                            if 0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes):
                                edges.append(Edge(nodes[src_idx], nodes[dst_idx]))

                    skeleton = Skeleton(nodes=nodes, edges=edges)
                except (KeyError, IndexError, TypeError):
                    # Fall back to basic skeleton creation
                    pass

            # Fall back to basic skeleton creation if metadata approach failed
            if skeleton is None and "kpt_shape" in data_config:
                kpt_shape = data_config["kpt_shape"]
                if isinstance(kpt_shape, list) and len(kpt_shape) >= 2:
                    n_keypoints = kpt_shape[0]
                    # Create a basic skeleton with numbered nodes
                    nodes = [Node(name=str(i)) for i in range(n_keypoints)]
                    skeleton = Skeleton(nodes=nodes)

    labels_dict = {}

    for split in splits:
        try:
            labels = read_labels(
                dataset_path=str(dataset_path),
                split=split,
                skeleton=skeleton,
                image_size=image_size,
                visibility_encoding=visibility_encoding,
                v0_as=v0_as,
            )
            labels_dict[split] = labels
        except Exception:
            continue

    if not labels_dict:
        raise ValueError(f"Could not load any splits from dataset: {dataset_path}")

    return LabelsSet(labels=labels_dict)


def _save_frame_image(args: tuple[dict, str, int | None]) -> str | None:
    """Worker function to save a single frame image.

    Args:
        args: Tuple containing:
            - frame_data: Dict with frame metadata (video_path, frame_idx, lf_idx)
            - image_format: Image format to save
            - image_quality: Optional image quality parameter

    Returns:
        Path to saved image file if successful, None if failed
    """
    frame_data, image_format, image_quality = args

    try:
        # Reopen video in worker process
        video = Video.from_filename(frame_data["video_path"])

        # Extract frame
        frame_img = video[frame_data["frame_idx"]]
        if frame_img is None:
            return None

        # Handle grayscale
        if frame_img.ndim == 3 and frame_img.shape[-1] == 1:
            frame_img = np.squeeze(frame_img, axis=-1)

        # Save image
        save_kwargs = {}
        if image_format.lower() in ["jpg", "jpeg"]:
            if image_quality is not None:
                save_kwargs["quality"] = image_quality
        elif image_format.lower() == "png":
            if image_quality is not None:
                save_kwargs["compress_level"] = min(9, max(0, image_quality))

        iio.imwrite(frame_data["output_path"], frame_img, **save_kwargs)
        return frame_data["output_path"]

    except Exception as e:
        warnings.warn(
            f"Error processing frame {frame_data['frame_idx']} from "
            f"{frame_data['video_path']}: {str(e)}"
        )
        return None


def write_labels(
    labels: Labels,
    dataset_path: str,
    split_ratios: dict[str, float] = {"train": 0.8, "val": 0.2},
    class_id: int = 0,
    image_format: str = "png",
    image_quality: int | None = None,
    verbose: bool = True,
    use_multiprocessing: bool = False,
    n_workers: int | None = None,
    task: str = "pose",
    visibility_encoding: str = "ternary",
    **kwargs,
) -> None:
    """Write Labels to Ultralytics YOLO format.

    Args:
        labels: SLEAP Labels object to export.
        dataset_path: Path to write the Ultralytics dataset.
        split_ratios: Dictionary mapping split names to ratios (must sum to 1.0).
        class_id: Class ID to use for all instances (default: 0).
        image_format: Image format to use for saving frames. Either "png" (default,
            lossless) or "jpg".
        image_quality: Image quality for JPEG format (1-100). For PNG, this is the
            compression level (0-9).
            If None, uses default quality settings.
        verbose: If True (default), show progress bars during export.
        use_multiprocessing: If True, use multiprocessing for parallel image saving.
            Default is False.
        n_workers: Number of worker processes. If None, uses CPU count - 1. Only used
            if use_multiprocessing=True.
        task: YOLO task type. One of ``"pose"`` (default), ``"detect"``, or
            ``"segment"``. For ``"detect"`` and ``"segment"``, ROIs from the Labels
            object are written instead of pose instances.
        visibility_encoding: Keypoint visibility codes to write for the pose task.
            ``"ternary"`` (the default) writes visible points as 2, occluded points
            with coordinates as 1 and every other point as 0. ``"binary"`` writes
            visible points as 1 and every other point as 0.
        **kwargs: Additional arguments (unused, for compatibility).

    Notes:
        Pose instances without any labeled point that has coordinates
        (`Instance.is_empty`) are skipped. Instances whose points are all occluded
        estimates are kept.
    """
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Validate split ratios
    total_ratio = sum(split_ratios.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    if task == "pose":
        if len(labels.skeletons) == 0:
            raise ValueError("Labels must have at least one skeleton for pose task")
        skeleton = labels.skeletons[0]
    else:
        skeleton = None

    # Build class names from ROI/bbox categories for non-pose tasks
    if task == "detect":
        class_names = _build_class_names_from_bboxes(labels.bboxes)
    elif task == "segment":
        class_names = _build_class_names_from_rois(labels.rois)
    else:
        class_names = {0: "animal"}

    # Create data.yaml configuration
    create_data_yaml(
        dataset_path / "data.yaml",
        skeleton,
        split_ratios,
        task=task,
        class_names=class_names,
    )

    # For detection, write bboxes; for segmentation, write ROIs
    if task == "detect":
        _write_bbox_labels(
            labels,
            dataset_path,
            split_ratios,
            class_names,
            image_format,
            image_quality,
            verbose,
        )
        return

    if task == "segment":
        _write_roi_labels(
            labels,
            dataset_path,
            split_ratios,
            class_names,
            image_format,
            image_quality,
            verbose,
        )
        return

    # Split the labels if multiple splits requested
    if len(split_ratios) == 1:
        split_name = list(split_ratios.keys())[0]
        split_labels = {split_name: labels}
    else:
        split_labels = create_splits_from_labels(labels, split_ratios)

    # Write each split
    for split_name, split_data in split_labels.items():
        images_dir = dataset_path / split_name / "images"
        labels_dir = dataset_path / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        if use_multiprocessing:
            # Prepare frame data for multiprocessing
            frame_data_list = []
            for lf_idx, frame in enumerate(split_data.labeled_frames):
                image_filename = f"{lf_idx:07d}.{image_format}"
                image_path = images_dir / image_filename

                frame_data = {
                    "video_path": str(frame.video.filename),
                    "frame_idx": frame.frame_idx,
                    "lf_idx": lf_idx,
                    "output_path": str(image_path),
                    "frame": frame,  # Keep reference for annotation writing
                }
                frame_data_list.append(frame_data)

            # Set up worker pool
            if n_workers is None:
                n_workers = max(1, multiprocessing.cpu_count() - 1)

            # Process frames in parallel
            with Pool(processes=n_workers) as pool:
                # Create args for each frame
                args_list = [
                    (fd, image_format, image_quality) for fd in frame_data_list
                ]

                # Use imap for progress tracking
                if verbose:
                    results = list(
                        tqdm(
                            pool.imap(_save_frame_image, args_list),
                            total=len(args_list),
                            desc=f"Writing {split_name} images (parallel)",
                        )
                    )
                else:
                    results = pool.map(_save_frame_image, args_list)

            # Write annotations for successfully saved images
            for frame_data, result in zip(frame_data_list, results):
                if result is not None:
                    frame = frame_data["frame"]
                    # Get image shape from saved file
                    img = iio.imread(result)
                    label_filename = f"{frame_data['lf_idx']:07d}.txt"
                    label_path = labels_dir / label_filename
                    write_label_file(
                        label_path,
                        frame,
                        skeleton,
                        img.shape[:2],
                        class_id,
                        visibility_encoding=visibility_encoding,
                    )
        else:
            # Sequential processing (original implementation)
            frame_iterator = (
                tqdm(
                    split_data.labeled_frames,
                    desc=f"Writing {split_name} images",
                    disable=not verbose,
                )
                if verbose
                else split_data.labeled_frames
            )

            for lf_idx, frame in enumerate(frame_iterator):
                # Extract frame image
                try:
                    frame_img = frame.image
                    if frame_img is None:
                        warnings.warn(
                            f"Could not load frame {frame.frame_idx} from video, "
                            f"skipping."
                        )
                        continue

                    # Use labeled frame index for filename
                    image_filename = f"{lf_idx:07d}.{image_format}"
                    image_path = images_dir / image_filename

                    # Handle grayscale conversion if needed
                    if frame_img.ndim == 3 and frame_img.shape[-1] == 1:
                        # Squeeze single channel dimension for grayscale
                        frame_img = np.squeeze(frame_img, axis=-1)

                    # Save image with appropriate quality settings
                    save_kwargs = {}
                    if image_format.lower() in ["jpg", "jpeg"]:
                        if image_quality is not None:
                            save_kwargs["quality"] = image_quality
                    elif image_format.lower() == "png":
                        if image_quality is not None:
                            # PNG uses compress_level (0-9)
                            save_kwargs["compress_level"] = min(
                                9, max(0, image_quality)
                            )

                    # Save the image
                    iio.imwrite(image_path, frame_img, **save_kwargs)

                    # Save annotations with same base filename
                    label_filename = f"{lf_idx:07d}.txt"
                    label_path = labels_dir / label_filename
                    write_label_file(
                        label_path,
                        frame,
                        skeleton,
                        frame_img.shape[:2],
                        class_id,
                        visibility_encoding=visibility_encoding,
                    )

                except Exception as e:
                    warnings.warn(
                        f"Error processing frame {frame.frame_idx}: {str(e)}, skipping."
                    )
                    continue


def _build_class_names_from_rois(rois: list[ROI]) -> dict[int, str]:
    """Build a class ID to name mapping from ROI categories.

    Args:
        rois: List of ROIs to extract categories from.

    Returns:
        A dictionary mapping integer class IDs to category name strings.
    """
    categories = sorted({roi.category.name for roi in rois if roi.category})
    if not categories:
        return {0: "object"}
    return {i: name for i, name in enumerate(categories)}


def _build_class_names_from_bboxes(bboxes: list[BoundingBox]) -> dict[int, str]:
    """Build a class ID to name mapping from BoundingBox categories.

    Args:
        bboxes: List of bounding boxes to extract categories from.

    Returns:
        A dictionary mapping integer class IDs to category name strings.
    """
    categories = sorted({bbox.category.name for bbox in bboxes if bbox.category})
    if not categories:
        return {0: "object"}
    return {i: name for i, name in enumerate(categories)}


def _write_bbox_labels(
    labels: Labels,
    dataset_path: Path,
    split_ratios: dict[str, float],
    class_names: dict[int, str],
    image_format: str,
    image_quality: int | None,
    verbose: bool,
) -> None:
    """Write BoundingBox-based detection labels to Ultralytics format.

    Args:
        labels: Labels object with bboxes.
        dataset_path: Output dataset path.
        split_ratios: Split ratio mapping.
        class_names: Class ID to name mapping.
        image_format: Image format string.
        image_quality: Image quality parameter.
        verbose: Whether to show progress.
    """
    # Reverse class_names to get name -> id mapping
    name_to_id = {v: k for k, v in class_names.items()}

    # Group bboxes by video (derive from parent LabeledFrame)
    bboxes_by_video: dict[str, list[BoundingBox]] = {}
    video_by_key: dict[str, Video] = {}
    for lf in labels.labeled_frames:
        if lf.bboxes:
            key = str(lf.video.filename)
            if key not in bboxes_by_video:
                bboxes_by_video[key] = []
                video_by_key[key] = lf.video
            bboxes_by_video[key].extend(lf.bboxes)

    video_keys = sorted(bboxes_by_video.keys())

    # Calculate split boundaries
    n_videos = len(video_keys)
    split_names = list(split_ratios.keys())
    split_boundaries = []
    cumulative = 0
    for name in split_names:
        cumulative += split_ratios[name]
        split_boundaries.append(int(round(cumulative * n_videos)))

    # Write each split
    start_idx = 0
    for split_idx, split_name in enumerate(split_names):
        end_idx = split_boundaries[split_idx]
        split_video_keys = video_keys[start_idx:end_idx]
        start_idx = end_idx

        if not split_video_keys:
            continue

        images_dir = dataset_path / split_name / "images"
        labels_dir = dataset_path / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        iterator = (
            tqdm(
                split_video_keys,
                desc=f"Writing {split_name}",
                disable=not verbose,
            )
            if verbose
            else split_video_keys
        )

        for lf_idx, video_path in enumerate(iterator):
            bbox_list = bboxes_by_video[video_path]
            video = video_by_key[video_path]

            # Read image to get shape and save it
            try:
                frame_img = video[0]
                if frame_img is None:
                    continue

                if frame_img.ndim == 3 and frame_img.shape[-1] == 1:
                    frame_img = np.squeeze(frame_img, axis=-1)

                image_filename = f"{lf_idx:07d}.{image_format}"
                image_path = images_dir / image_filename

                save_kwargs = {}
                if (
                    image_format.lower() in ["jpg", "jpeg"]
                    and image_quality is not None
                ):
                    save_kwargs["quality"] = image_quality
                elif image_format.lower() == "png" and image_quality is not None:
                    save_kwargs["compress_level"] = min(9, max(0, image_quality))

                iio.imwrite(image_path, frame_img, **save_kwargs)

                height_px, width_px = frame_img.shape[:2]
            except Exception as e:
                warnings.warn(f"Error processing {video_path}: {e}, skipping.")
                continue

            # Write label file
            label_filename = f"{lf_idx:07d}.txt"
            label_path = labels_dir / label_filename
            write_bbox_label_file(
                label_path, bbox_list, (height_px, width_px), name_to_id
            )


def _write_roi_labels(
    labels: Labels,
    dataset_path: Path,
    split_ratios: dict[str, float],
    class_names: dict[int, str],
    image_format: str,
    image_quality: int | None,
    verbose: bool,
) -> None:
    """Write ROI-based labels (detection/segmentation) to Ultralytics format.

    Args:
        labels: Labels object with ROIs.
        dataset_path: Output dataset path.
        split_ratios: Split ratio mapping.
        class_names: Class ID to name mapping.
        image_format: Image format string.
        image_quality: Image quality parameter.
        verbose: Whether to show progress.
    """
    # Reverse class_names to get name -> id mapping
    name_to_id = {v: k for k, v in class_names.items()}

    # Group ROIs by video (derive from parent LabeledFrame)
    rois_by_video: dict[str, list[ROI]] = {}
    video_by_key: dict[str, Video] = {}
    for lf in labels.labeled_frames:
        if lf.rois:
            key = str(lf.video.filename)
            if key not in rois_by_video:
                rois_by_video[key] = []
                video_by_key[key] = lf.video
            rois_by_video[key].extend(lf.rois)

    video_keys = sorted(rois_by_video.keys())

    # Calculate split boundaries
    n_videos = len(video_keys)
    split_names = list(split_ratios.keys())
    split_boundaries = []
    cumulative = 0
    for name in split_names:
        cumulative += split_ratios[name]
        split_boundaries.append(int(round(cumulative * n_videos)))

    # Write each split
    start_idx = 0
    for split_idx, split_name in enumerate(split_names):
        end_idx = split_boundaries[split_idx]
        split_video_keys = video_keys[start_idx:end_idx]
        start_idx = end_idx

        if not split_video_keys:
            continue

        images_dir = dataset_path / split_name / "images"
        labels_dir = dataset_path / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        iterator = (
            tqdm(
                split_video_keys,
                desc=f"Writing {split_name}",
                disable=not verbose,
            )
            if verbose
            else split_video_keys
        )

        for lf_idx, video_path in enumerate(iterator):
            rois = rois_by_video[video_path]
            video = video_by_key[video_path]

            # Read image to get shape and save it
            try:
                frame_img = video[0]
                if frame_img is None:
                    continue

                if frame_img.ndim == 3 and frame_img.shape[-1] == 1:
                    frame_img = np.squeeze(frame_img, axis=-1)

                image_filename = f"{lf_idx:07d}.{image_format}"
                image_path = images_dir / image_filename

                save_kwargs = {}
                if (
                    image_format.lower() in ["jpg", "jpeg"]
                    and image_quality is not None
                ):
                    save_kwargs["quality"] = image_quality
                elif image_format.lower() == "png" and image_quality is not None:
                    save_kwargs["compress_level"] = min(9, max(0, image_quality))

                iio.imwrite(image_path, frame_img, **save_kwargs)

                height_px, width_px = frame_img.shape[:2]
            except Exception as e:
                warnings.warn(f"Error processing {video_path}: {e}, skipping.")
                continue

            # Write label file
            label_filename = f"{lf_idx:07d}.txt"
            label_path = labels_dir / label_filename
            write_roi_label_file(label_path, rois, (height_px, width_px), name_to_id)


def parse_data_yaml(yaml_path: Path) -> dict:
    """Parse Ultralytics data.yaml configuration file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_skeleton_from_config(config: dict) -> Skeleton:
    """Create a Skeleton object from Ultralytics configuration."""
    kpt_shape = config.get("kpt_shape", [1, 3])
    num_keypoints = kpt_shape[0]

    # Create nodes - use generic names if not specified
    node_names = config.get("node_names", [f"point_{i}" for i in range(num_keypoints)])
    nodes = [Node(name) for name in node_names[:num_keypoints]]

    # Create edges from skeleton connections
    edges = []
    skeleton_connections = config.get("skeleton", [])
    for connection in skeleton_connections:
        if len(connection) == 2:
            src_idx, dst_idx = connection
            if 0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes):
                edges.append(Edge(nodes[src_idx], nodes[dst_idx]))

    return Skeleton(nodes=nodes, edges=edges, name="ultralytics_skeleton")


def detect_line_format(parts: list[str]) -> str:
    """Detect the YOLO annotation format from a single line's parsed values.

    Args:
        parts: List of string values from a single annotation line.

    Returns:
        One of ``"detection"``, ``"detection_conf"``, ``"segmentation"``, or
        ``"pose"``.
    """
    n = len(parts)
    if n == 5:
        return "detection"
    if n == 6:
        return "detection_conf"
    # Check for pose format: 5 + 3*k values
    remainder = n - 5
    if remainder > 0 and remainder % 3 == 0:
        return "pose"
    # Even number of values after class_id -> segmentation polygon
    if (n - 1) % 2 == 0 and n > 5:
        return "segmentation"
    return "pose"


def parse_label_file(
    label_path: Path,
    skeleton: Skeleton,
    image_shape: tuple[int, int],
    class_names: dict[int, str] | None = None,
    video: Video | None = None,
    frame_idx: int = 0,
    visibility_encoding: str = "ternary",
    v0_as: str | None = None,
) -> tuple[list[Instance], list[ROI], list[BoundingBox]]:
    """Parse a single Ultralytics label file and return instances, ROIs, and bboxes.

    The format is auto-detected per line based on the number of values:

    - **5 values**: Detection (``class_id x_center y_center width height``)
      -> ``UserBoundingBox``
    - **6 values**: Detection with confidence -> ``PredictedBoundingBox``
    - **5 + 3k values**: Pose (existing keypoint format)
    - **Even count > 5 with (n-1) even**: Segmentation polygon

    Args:
        label_path: Path to the ``.txt`` label file.
        skeleton: Skeleton to use for pose instances.
        image_shape: Image dimensions ``(height, width)`` for denormalization.
        class_names: Optional mapping from class ID to category name.
        video: Optional ``Video`` to associate with ROIs and bounding boxes.
        frame_idx: Frame index for ROIs and bounding boxes. Defaults to 0.
        visibility_encoding: ``"ternary"`` (the default), ``"binary"`` or
            ``"auto"`` (resolved per line). See
            `sleap_io.io.coco.decode_visibility`.
        v0_as: ``"unlabeled"``, ``"absent"`` or ``None`` (the default, which
            depends on the encoding). See `sleap_io.io.coco.decode_visibility`.

    Returns:
        A tuple of ``(instances, rois, bboxes)`` parsed from the file.

    .. versionchanged:: 0.x
        Returns a 3-tuple ``(instances, rois, bboxes)`` instead of the previous
        2-tuple ``(instances, rois)``. Detection format lines now produce
        ``BoundingBox`` objects in the third element instead of ``ROI`` objects.
    """
    instances: list[Instance] = []
    rois: list[ROI] = []
    bboxes: list[BoundingBox] = []

    with open(label_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                parts = line.split()
                if len(parts) < 5:
                    warnings.warn(
                        f"Invalid line {line_num} in {label_path}: insufficient data"
                    )
                    continue

                fmt = detect_line_format(parts)
                class_id = int(parts[0])
                category = (
                    class_names.get(class_id, "") if class_names is not None else ""
                )
                height_px, width_px = image_shape

                if fmt in ("detection", "detection_conf"):
                    x_center_norm, y_center_norm, w_norm, h_norm = map(
                        float, parts[1:5]
                    )

                    # Denormalize to pixel coordinates
                    x_center_px = x_center_norm * width_px
                    y_center_px = y_center_norm * height_px
                    w_px = w_norm * width_px
                    h_px = h_norm * height_px

                    if fmt == "detection_conf":
                        score = float(parts[5])
                        bbox = PredictedBoundingBox.from_xywh(
                            x_center_px - w_px / 2,
                            y_center_px - h_px / 2,
                            w_px,
                            h_px,
                            category=category,
                            score=score,
                        )
                    else:
                        bbox = UserBoundingBox.from_xywh(
                            x_center_px - w_px / 2,
                            y_center_px - h_px / 2,
                            w_px,
                            h_px,
                            category=category,
                        )
                    bboxes.append(bbox)

                elif fmt == "segmentation":
                    # class_id x1 y1 x2 y2 ... xn yn
                    coord_values = list(map(float, parts[1:]))
                    coords = []
                    for i in range(0, len(coord_values), 2):
                        x_px = coord_values[i] * width_px
                        y_px = coord_values[i + 1] * height_px
                        coords.append((x_px, y_px))

                    roi = UserROI.from_polygon(
                        coords,
                        category=category,
                    )
                    rois.append(roi)

                else:
                    # Pose format (existing behavior)
                    x_center, y_center, width, height = map(float, parts[1:5])
                    keypoint_data = parts[5:]
                    if len(keypoint_data) % 3 != 0:
                        warnings.warn(
                            f"Invalid keypoint data in {label_path} line {line_num}"
                        )
                        continue

                    num_keypoints = len(keypoint_data) // 3
                    if num_keypoints != len(skeleton.nodes):
                        warnings.warn(
                            f"Keypoint count mismatch: expected "
                            f"{len(skeleton.nodes)}, got {num_keypoints} in "
                            f"{label_path} line {line_num}"
                        )
                        continue

                    values = np.array(keypoint_data, dtype="float64").reshape(-1, 3)
                    xy = values[:, :2] * [width_px, height_px]
                    points_array = decode_visibility(
                        xy,
                        values[:, 2],
                        visibility_encoding=visibility_encoding,
                        v0_as=v0_as,
                    )
                    instance = Instance.from_numpy(
                        points_data=points_array, skeleton=skeleton
                    )
                    instances.append(instance)

            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing line {line_num} in {label_path}: {e}")
                continue

    return instances, rois, bboxes


def write_label_file(
    label_path: Path,
    frame: LabeledFrame,
    skeleton: Skeleton,
    image_shape: tuple[int, int],
    class_id: int = 0,
    visibility_encoding: str = "ternary",
) -> None:
    """Write a single Ultralytics pose label file for a frame.

    Args:
        label_path: Path of the ``.txt`` label file to write.
        frame: The frame whose instances to write.
        skeleton: The skeleton the instances must match.
        image_shape: Image dimensions ``(height, width)`` for normalization.
        class_id: Class ID to write for every instance.
        visibility_encoding: ``"ternary"`` (the default) or ``"binary"``. See
            `sleap_io.io.coco.encode_visibility`.

    Notes:
        Instances without any labeled point that has coordinates are skipped. The
        bounding box covers every labeled point with coordinates, occluded
        estimates included.
    """
    height_px, width_px = image_shape

    with open(label_path, "w") as f:
        for instance in frame.instances:
            if len(instance.points) != len(skeleton.nodes):
                warnings.warn(
                    f"Instance has {len(instance.points)} points, "
                    f"skeleton has {len(skeleton.nodes)} nodes. Skipping."
                )
                continue

            if instance.is_empty:
                continue  # Skip instances with no labeled point that has a location

            # Calculate bounding box from labeled points with a location
            located_xy = instance.numpy(invisible_as_nan=False)
            located_xy = located_xy[np.isfinite(located_xy).all(axis=1)]
            x_coords = located_xy[:, 0]
            y_coords = located_xy[:, 1]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding to bounding box
            padding = 10  # pixels
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width_px, x_max + padding)
            y_max = min(height_px, y_max + padding)

            # Convert to normalized YOLO format
            x_center_norm = ((x_min + x_max) / 2) / width_px
            y_center_norm = ((y_min + y_max) / 2) / height_px
            width_norm = (x_max - x_min) / width_px
            height_norm = (y_max - y_min) / height_px

            # Build line: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
            line_parts = [
                str(class_id),
                f"{x_center_norm:.6f}",
                f"{y_center_norm:.6f}",
                f"{width_norm:.6f}",
                f"{height_norm:.6f}",
            ]

            # Add keypoints
            for x_norm, y_norm, visibility in normalize_coordinates(
                instance, image_shape, visibility_encoding=visibility_encoding
            ):
                line_parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", str(visibility)])

            f.write(" ".join(line_parts) + "\n")


def write_roi_label_file(
    label_path: Path,
    rois: list[ROI],
    image_shape: tuple[int, int],
    name_to_id: dict[str, int],
) -> None:
    """Write a single Ultralytics label file for detection/segmentation ROIs.

    Args:
        label_path: Path to write the label file.
        rois: List of ROIs for this image.
        image_shape: Image dimensions ``(height, width)``.
        name_to_id: Mapping from category name to class ID.
    """
    height_px, width_px = image_shape

    # Explode multi-geometries so each polygon gets its own line
    exploded_rois: list[ROI] = []
    for roi in rois:
        exploded_rois.extend(roi.explode())

    with open(label_path, "w") as f:
        for roi in exploded_rois:
            class_id = name_to_id.get(roi.category.name, 0) if roi.category else 0

            if not roi.is_bbox:
                # Write segmentation polygon
                if (
                    hasattr(roi.geometry, "interiors")
                    and len(list(roi.geometry.interiors)) > 0
                ):
                    warnings.warn(
                        f"ROI polygon has "
                        f"{len(list(roi.geometry.interiors))} interior "
                        f"ring(s) (holes) that will be dropped. YOLO "
                        f"segmentation format does not support polygon "
                        f"holes.",
                        stacklevel=2,
                    )
                coords = list(roi.geometry.exterior.coords)[:-1]  # Remove closing pt
                line_parts = [str(class_id)]
                for x, y in coords:
                    line_parts.append(f"{x / width_px:.6f}")
                    line_parts.append(f"{y / height_px:.6f}")
                f.write(" ".join(line_parts) + "\n")
            else:
                # Write detection bounding box
                minx, miny, maxx, maxy = roi.bounds
                x_center = ((minx + maxx) / 2) / width_px
                y_center = ((miny + maxy) / 2) / height_px
                w = (maxx - minx) / width_px
                h = (maxy - miny) / height_px

                line_parts = [
                    str(class_id),
                    f"{x_center:.6f}",
                    f"{y_center:.6f}",
                    f"{w:.6f}",
                    f"{h:.6f}",
                ]
                f.write(" ".join(line_parts) + "\n")


def write_bbox_label_file(
    label_path: Path,
    bboxes: list[BoundingBox],
    image_shape: tuple[int, int],
    name_to_id: dict[str, int],
) -> None:
    """Write a single Ultralytics label file for detection bounding boxes.

    Args:
        label_path: Path to write the label file.
        bboxes: List of ``BoundingBox`` objects for this image.
        image_shape: Image dimensions ``(height, width)``.
        name_to_id: Mapping from category name to class ID.
    """
    height_px, width_px = image_shape

    with open(label_path, "w") as f:
        for bbox in bboxes:
            class_id = name_to_id.get(bbox.category.name, 0) if bbox.category else 0

            # Normalize center coordinates and dimensions
            x_center_norm = bbox.x_center / width_px
            y_center_norm = bbox.y_center / height_px
            w_norm = bbox.width / width_px
            h_norm = bbox.height / height_px

            line_parts = [
                str(class_id),
                f"{x_center_norm:.6f}",
                f"{y_center_norm:.6f}",
                f"{w_norm:.6f}",
                f"{h_norm:.6f}",
            ]
            if isinstance(bbox, PredictedBoundingBox):
                line_parts.append(f"{bbox.score:.6f}")
            f.write(" ".join(line_parts) + "\n")


def create_data_yaml(
    yaml_path: Path,
    skeleton: Skeleton | None,
    split_ratios: dict[str, float],
    task: str = "pose",
    class_names: dict[int, str] | None = None,
) -> None:
    """Create Ultralytics data.yaml configuration file.

    Args:
        yaml_path: Path to write the YAML file.
        skeleton: Skeleton for pose tasks. Can be ``None`` for non-pose tasks.
        split_ratios: Mapping of split names to ratios.
        task: YOLO task type (``"pose"``, ``"detect"``, or ``"segment"``).
        class_names: Optional class name mapping. Defaults to ``{0: "animal"}``.
    """
    if class_names is None:
        class_names = {0: "animal"}

    config: dict = {
        "path": ".",
        "names": class_names,
    }

    if task != "pose":
        config["task"] = task
    else:
        # Pose-specific fields
        if skeleton is not None:
            # Build skeleton connections
            skeleton_connections = []
            for edge in skeleton.edges:
                src_idx = skeleton.nodes.index(edge.source)
                dst_idx = skeleton.nodes.index(edge.destination)
                skeleton_connections.append([src_idx, dst_idx])

            # Create flip indices (identity mapping by default)
            flip_idx = list(range(len(skeleton.nodes)))

            config["kpt_shape"] = [len(skeleton.nodes), 3]
            config["flip_idx"] = flip_idx
            config["skeleton"] = skeleton_connections
            config["node_names"] = [node.name for node in skeleton.nodes]

    # Add split paths
    for split_name in split_ratios.keys():
        config[split_name] = f"{split_name}/images"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_splits_from_labels(
    labels: Labels, split_ratios: dict[str, float]
) -> dict[str, Labels]:
    """Create dataset splits from Labels using the built-in splitting functionality."""
    split_names = list(split_ratios.keys())

    if len(split_names) == 2:
        # Two-way split
        ratio = split_ratios[split_names[0]]
        split1, split2 = labels.split(ratio)
        return {split_names[0]: split1, split_names[1]: split2}

    elif len(split_names) == 3:
        # Three-way split using make_training_splits
        train_ratio = split_ratios.get("train", 0.6)
        val_ratio = split_ratios.get("val", 0.2)
        test_ratio = split_ratios.get("test", 0.2)

        try:
            train_split, val_split, test_split = labels.make_training_splits(
                n_train=train_ratio, n_val=val_ratio, n_test=test_ratio
            )
            return {"train": train_split, "val": val_split, "test": test_split}
        except Exception:
            # Fallback to manual splitting
            first_split = train_ratio + val_ratio
            temp_split, test_split = labels.split(first_split)
            train_split, val_split = temp_split.split(train_ratio / first_split)
            return {"train": train_split, "val": val_split, "test": test_split}

    else:
        # Single split or custom splits
        return {split_names[0]: labels}


def normalize_coordinates(
    instance: Instance,
    image_shape: tuple[int, int],
    visibility_encoding: str = "ternary",
) -> list[tuple[float, float, int]]:
    """Normalize instance point coordinates to [0,1] range.

    Args:
        instance: The instance to normalize.
        image_shape: Image dimensions ``(height, width)``.
        visibility_encoding: ``"ternary"`` (the default) or ``"binary"``. See
            `sleap_io.io.coco.encode_visibility`.

    Returns:
        One ``(x_norm, y_norm, visibility)`` tuple per point. Points written with
        visibility 0 have ``(0.0, 0.0)`` coordinates.
    """
    height, width = image_shape
    xy, visibility = encode_visibility(instance.points, visibility_encoding)
    xy = xy / [width, height]
    return [(x, y, v) for (x, y), v in zip(xy.tolist(), visibility.tolist())]


def denormalize_coordinates(
    normalized_points: list[tuple[float, float, int]],
    image_shape: tuple[int, int],
    visibility_encoding: str = "ternary",
    v0_as: str | None = None,
) -> np.ndarray:
    """Denormalize coordinates from [0,1] range to pixel coordinates.

    Args:
        normalized_points: One ``(x_norm, y_norm, visibility)`` tuple per point.
        image_shape: Image dimensions ``(height, width)``.
        visibility_encoding: ``"ternary"`` (the default), ``"binary"`` or
            ``"auto"``. See `sleap_io.io.coco.decode_visibility`.
        v0_as: ``"unlabeled"``, ``"absent"`` or ``None`` (the default, which
            depends on the encoding). See `sleap_io.io.coco.decode_visibility`.

    Returns:
        A numpy array of shape (n_points, 5) with columns x, y, visible, verified
        and labeled, ready for `Instance.from_numpy`.
    """
    height, width = image_shape
    values = np.array(normalized_points, dtype="float64").reshape(-1, 3)
    return decode_visibility(
        values[:, :2] * [width, height],
        values[:, 2],
        visibility_encoding=visibility_encoding,
        v0_as=v0_as,
    )


def _iter_visibility_codes(label_files: list[Path]) -> Iterator[list[float]]:
    """Yield the keypoint visibility codes of every pose line in some label files.

    Args:
        label_files: Ultralytics ``.txt`` label files.

    Yields:
        The visibility codes of one pose line. Comments, lines that aren't pose
        lines, and lines with values that aren't numbers are skipped (the parser
        warns about the latter).
    """
    for label_file in label_files:
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if detect_line_format(parts) == "pose":
                try:
                    yield [float(v) for v in parts[7::3]]
                except ValueError:
                    continue
