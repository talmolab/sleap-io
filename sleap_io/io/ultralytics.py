"""This module handles direct I/O operations for working with Ultralytics YOLO pose format.

Ultralytics YOLO pose format specification:
- Directory structure: dataset_root/split/images/ and dataset_root/split/labels/
- Configuration: data.yaml file defining skeleton and dataset structure
- Annotation format: Each .txt file contains lines with format:
  class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ... xn yn vn
- Coordinates: Normalized to [0,1] range, origin at top-left
- Visibility: 0=not visible, 1=visible but occluded, 2=visible and not occluded
"""

from __future__ import annotations
import numpy as np
import yaml
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings
from sleap_io.model.video import Video
from sleap_io.model.skeleton import Skeleton, Edge, Node
from sleap_io.model.instance import Track, Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
import imageio.v3 as iio
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial


def read_labels(
    dataset_path: str,
    split: str = "train",
    skeleton: Optional[Skeleton] = None,
    image_size: Tuple[int, int] = (480, 640),
) -> Labels:
    """Read Ultralytics YOLO pose dataset and return a `Labels` object.

    Args:
        dataset_path: Path to the Ultralytics dataset root directory containing data.yaml.
        split: Dataset split to read ('train', 'val', or 'test'). Defaults to 'train'.
        skeleton: Optional skeleton to use. If not provided, will be inferred from data.yaml.
        image_size: Image dimensions (height, width) for coordinate denormalization.
                   Defaults to (480, 640). Will attempt to infer from actual images if available.

    Returns:
        Parsed labels as a `Labels` instance.
    """
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

    # Use provided skeleton or create from config
    if skeleton is None:
        skeleton = create_skeleton_from_config(config)

    # Get paths for the specified split
    split_path = config.get(split, f"{split}/images")
    images_dir = dataset_path / split_path
    labels_dir = dataset_path / split_path.replace("/images", "/labels")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Process all image/label pairs
    labeled_frames = []
    tracks = {}  # Track synthetic tracks by instance order

    for image_file in sorted(images_dir.glob("*")):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            label_file = labels_dir / f"{image_file.stem}.txt"

            # Create video object for this image
            video = Video.from_filename(str(image_file))

            # Parse label file if it exists
            instances = []
            if label_file.exists():
                # Get image dimensions - try from video shape first, fallback to reading image
                if video.shape is not None:
                    img_shape = video.shape[:2]
                else:
                    # Read first frame to get dimensions
                    img = video[0]
                    img_shape = img.shape[:2]

                instances = parse_label_file(label_file, skeleton, img_shape)

                # Assign tracks to instances based on order
                for i, instance in enumerate(instances):
                    track_name = f"track_{i}"
                    if track_name not in tracks:
                        tracks[track_name] = Track(name=track_name)
                    instance.track = tracks[track_name]

            # Create labeled frame
            frame = LabeledFrame(video=video, frame_idx=0, instances=instances)
            labeled_frames.append(frame)

    return Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        tracks=list(tracks.values()),
        provenance={"source": str(dataset_path), "split": split},
    )


def _save_frame_image(args: Tuple[dict, str, Optional[int]]) -> Optional[str]:
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
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.2},
    class_id: int = 0,
    image_format: str = "png",
    image_quality: Optional[int] = None,
    verbose: bool = True,
    use_multiprocessing: bool = False,
    n_workers: Optional[int] = None,
    **kwargs,
) -> None:
    """Write Labels to Ultralytics YOLO pose format.

    Args:
        labels: SLEAP Labels object to export.
        dataset_path: Path to write the Ultralytics dataset.
        split_ratios: Dictionary mapping split names to ratios (must sum to 1.0).
        class_id: Class ID to use for all instances (default: 0).
        image_format: Image format to use for saving frames. Either "png" (default, lossless) or "jpg".
        image_quality: Image quality for JPEG format (1-100). For PNG, this is the compression level (0-9).
            If None, uses default quality settings.
        verbose: If True (default), show progress bars during export.
        use_multiprocessing: If True, use multiprocessing for parallel image saving. Default is False.
        n_workers: Number of worker processes. If None, uses CPU count - 1. Only used if use_multiprocessing=True.
        **kwargs: Additional arguments (unused, for compatibility).
    """
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Validate split ratios
    total_ratio = sum(split_ratios.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    if len(labels.skeletons) == 0:
        raise ValueError("Labels must have at least one skeleton")

    skeleton = labels.skeletons[0]

    # Create data.yaml configuration
    create_data_yaml(dataset_path / "data.yaml", skeleton, split_ratios)

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
                        label_path, frame, skeleton, img.shape[:2], class_id
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
                            f"Could not load frame {frame.frame_idx} from video, skipping."
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
                        label_path, frame, skeleton, frame_img.shape[:2], class_id
                    )

                except Exception as e:
                    warnings.warn(
                        f"Error processing frame {frame.frame_idx}: {str(e)}, skipping."
                    )
                    continue


def parse_data_yaml(yaml_path: Path) -> Dict:
    """Parse Ultralytics data.yaml configuration file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_skeleton_from_config(config: Dict) -> Skeleton:
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


def parse_label_file(
    label_path: Path, skeleton: Skeleton, image_shape: Tuple[int, int]
) -> List[Instance]:
    """Parse a single Ultralytics label file and return instances."""
    instances = []

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

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Parse keypoints
                keypoint_data = parts[5:]
                if len(keypoint_data) % 3 != 0:
                    warnings.warn(
                        f"Invalid keypoint data in {label_path} line {line_num}"
                    )
                    continue

                num_keypoints = len(keypoint_data) // 3
                if num_keypoints != len(skeleton.nodes):
                    warnings.warn(
                        f"Keypoint count mismatch: expected {len(skeleton.nodes)}, "
                        f"got {num_keypoints} in {label_path} line {line_num}"
                    )
                    continue

                # Convert normalized coordinates to pixel coordinates
                height_px, width_px = image_shape
                points = []

                for i in range(num_keypoints):
                    x_norm = float(keypoint_data[i * 3])
                    y_norm = float(keypoint_data[i * 3 + 1])
                    visibility = int(keypoint_data[i * 3 + 2])

                    # Denormalize coordinates
                    x_px = x_norm * width_px
                    y_px = y_norm * height_px

                    # Convert visibility: 0=not visible, 1=occluded, 2=visible
                    is_visible = visibility > 0

                    if visibility == 0:
                        # Not visible - use NaN coordinates
                        points.append([np.nan, np.nan, False])
                    else:
                        points.append([x_px, y_px, is_visible])

                # Create instance from numpy array
                points_array = np.array(points, dtype=np.float32)
                instance = Instance.from_numpy(
                    points_data=points_array, skeleton=skeleton
                )
                instances.append(instance)

            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing line {line_num} in {label_path}: {e}")
                continue

    return instances


def write_label_file(
    label_path: Path,
    frame: LabeledFrame,
    skeleton: Skeleton,
    image_shape: Tuple[int, int],
    class_id: int = 0,
) -> None:
    """Write a single Ultralytics label file for a frame."""
    height_px, width_px = image_shape

    with open(label_path, "w") as f:
        for instance in frame.instances:
            if len(instance.points) != len(skeleton.nodes):
                warnings.warn(
                    f"Instance has {len(instance.points)} points, "
                    f"skeleton has {len(skeleton.nodes)} nodes. Skipping."
                )
                continue

            # Calculate bounding box from visible keypoints
            visible_mask = instance.points["visible"] & ~np.isnan(
                instance.points["xy"][:, 0]
            )
            if not visible_mask.any():
                continue  # Skip instances with no visible points

            visible_xy = instance.points["xy"][visible_mask]
            x_coords = visible_xy[:, 0]
            y_coords = visible_xy[:, 1]

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
            for point in instance.points:
                x, y = point["xy"]
                if point["visible"] and not np.isnan(x):
                    x_norm = x / width_px
                    y_norm = y / height_px
                    visibility = 2  # visible and not occluded
                else:
                    x_norm = 0.0
                    y_norm = 0.0
                    visibility = 0  # not visible

                line_parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", str(visibility)])

            f.write(" ".join(line_parts) + "\n")


def create_data_yaml(
    yaml_path: Path, skeleton: Skeleton, split_ratios: Dict[str, float]
) -> None:
    """Create Ultralytics data.yaml configuration file."""
    # Build skeleton connections
    skeleton_connections = []
    for edge in skeleton.edges:
        src_idx = skeleton.nodes.index(edge.source)
        dst_idx = skeleton.nodes.index(edge.destination)
        skeleton_connections.append([src_idx, dst_idx])

    # Create flip indices (identity mapping by default)
    flip_idx = list(range(len(skeleton.nodes)))

    config = {
        "path": ".",
        "names": {0: "animal"},
        "kpt_shape": [len(skeleton.nodes), 3],
        "flip_idx": flip_idx,
        "skeleton": skeleton_connections,
        "node_names": [node.name for node in skeleton.nodes],
    }

    # Add split paths
    for split_name in split_ratios.keys():
        config[split_name] = f"{split_name}/images"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_splits_from_labels(
    labels: Labels, split_ratios: Dict[str, float]
) -> Dict[str, Labels]:
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
        except:
            # Fallback to manual splitting
            first_split = train_ratio + val_ratio
            temp_split, test_split = labels.split(first_split)
            train_split, val_split = temp_split.split(train_ratio / first_split)
            return {"train": train_split, "val": val_split, "test": test_split}

    else:
        # Single split or custom splits
        return {split_names[0]: labels}


def normalize_coordinates(
    instance: Instance, image_shape: Tuple[int, int]
) -> List[Tuple[float, float, int]]:
    """Normalize instance point coordinates to [0,1] range."""
    height, width = image_shape
    normalized = []

    for point in instance.points:
        x, y = point["xy"]
        if point["visible"] and not np.isnan(x):
            x_norm = x / width
            y_norm = y / height
            visibility = 2  # visible
        else:
            x_norm = 0.0
            y_norm = 0.0
            visibility = 0  # not visible

        normalized.append((x_norm, y_norm, visibility))

    return normalized


def denormalize_coordinates(
    normalized_points: List[Tuple[float, float, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """Denormalize coordinates from [0,1] range to pixel coordinates.

    Returns:
        A numpy array of shape (n_points, 3) with columns [x, y, visible].
    """
    height, width = image_shape
    points = []

    for x_norm, y_norm, visibility in normalized_points:
        if visibility > 0:
            x_px = x_norm * width
            y_px = y_norm * height
            is_visible = True
        else:
            x_px = np.nan
            y_px = np.nan
            is_visible = False

        points.append([x_px, y_px, is_visible])

    return np.array(points, dtype=np.float32)
