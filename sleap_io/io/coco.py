"""Handles direct I/O operations for working with COCO-style pose datasets.

COCO-style pose format specification:
- JSON annotation files containing images, annotations, and categories
- Image directory structure can vary (flat, categorized, nested, multi-source)
- Keypoint annotations with coordinates and visibility flags
- Support for multiple animal categories with different skeletons
- Visibility encoding: binary (0/1) or ternary (0/1/2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Edge, Node, Skeleton
from sleap_io.model.video import Video


def parse_coco_json(json_path: Union[str, Path]) -> Dict:
    """Parse COCO annotation JSON file and validate structure.

    Args:
        json_path: Path to the COCO annotation JSON file.

    Returns:
        Parsed COCO annotation dictionary.

    Raises:
        FileNotFoundError: If JSON file doesn't exist.
        ValueError: If JSON structure is invalid.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Validate required COCO fields
    required_fields = ["images", "annotations", "categories"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required COCO field: {field}")

    # Validate that we have pose data (keypoints in categories)
    has_keypoints = any("keypoints" in cat for cat in data["categories"])
    if not has_keypoints:
        raise ValueError(
            "No keypoint definitions found in categories. "
            "This appears to be a detection-only COCO dataset."
        )

    return data


def create_skeleton_from_category(category: Dict) -> Skeleton:
    """Create a Skeleton object from a COCO category definition.

    Args:
        category: COCO category dictionary with keypoints and skeleton.

    Returns:
        Skeleton object corresponding to the category.
    """
    if "keypoints" not in category:
        raise ValueError(f"Category '{category['name']}' has no keypoint definitions")

    # Create nodes from keypoint names
    keypoint_names = category["keypoints"]
    nodes = [Node(name) for name in keypoint_names]

    # Create edges from skeleton connections
    edges = []
    if "skeleton" in category:
        for connection in category["skeleton"]:
            if len(connection) == 2:
                # COCO skeleton uses 1-based indexing
                src_idx, dst_idx = connection[0] - 1, connection[1] - 1
                if 0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes):
                    edges.append(Edge(nodes[src_idx], nodes[dst_idx]))

    skeleton_name = category.get("name", "unknown")
    return Skeleton(nodes, edges, name=skeleton_name)


def resolve_image_path(image_filename: str, dataset_root: Path) -> Path:
    """Resolve image file path handling various directory structures.

    Args:
        image_filename: Image filename from COCO annotation.
        dataset_root: Root directory of the dataset.

    Returns:
        Resolved absolute path to the image file.

    Raises:
        FileNotFoundError: If image file cannot be found.
    """
    # Try direct path first
    image_path = dataset_root / image_filename
    if image_path.exists():
        return image_path

    # Try common variations
    common_prefixes = ["images", "imgs", "data/images", ""]

    for prefix in common_prefixes:
        if prefix:
            test_path = dataset_root / prefix / image_filename
        else:
            # Try finding the file anywhere in the dataset
            test_path = None
            for found_path in dataset_root.rglob(Path(image_filename).name):
                if found_path.is_file():
                    test_path = found_path
                    break

        if test_path and test_path.exists():
            return test_path

    raise FileNotFoundError(
        f"Image file not found: {image_filename} (searched in {dataset_root})"
    )


def decode_keypoints(
    keypoints: List[float], num_keypoints: int, skeleton: Skeleton
) -> np.ndarray:
    """Decode COCO keypoint format to numpy array for Instance creation.

    Args:
        keypoints: Flat list of [x1, y1, v1, x2, y2, v2, ...] values.
        num_keypoints: Number of keypoints (for validation).
        skeleton: Skeleton object defining the keypoint structure.

    Returns:
        Numpy array of shape (num_keypoints, 3) with [x, y, visibility] values.
    """
    if len(keypoints) != num_keypoints * 3:
        raise ValueError(
            f"Keypoints length {len(keypoints)} doesn't match expected "
            f"{num_keypoints * 3}"
        )

    if len(skeleton.nodes) != num_keypoints:
        raise ValueError(
            f"Skeleton has {len(skeleton.nodes)} nodes but annotation has "
            f"{num_keypoints} keypoints"
        )

    points = []
    for i in range(num_keypoints):
        x = keypoints[i * 3]
        y = keypoints[i * 3 + 1]
        visibility = keypoints[i * 3 + 2]

        # Handle different visibility encodings
        # 0 = not labeled/not visible, 1 = labeled but not visible,
        # 2 = labeled and visible
        # For binary encoding: 0 = not visible, 1 = visible
        if visibility == 0:
            # Not labeled or not visible - use NaN coordinates
            points.append([np.nan, np.nan, False])
        elif visibility == 1:
            # Labeled but not visible (occluded) OR visible (in binary encoding)
            # For now, treat as visible since we can't distinguish binary vs ternary
            points.append([x, y, True])
        elif visibility == 2:
            # Labeled and visible
            points.append([x, y, True])
        else:
            # Unknown visibility value, default to visible
            points.append([x, y, True])

    return np.array(points, dtype=np.float32)


def read_labels(
    json_path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    grayscale: bool = False,
) -> Labels:
    """Read COCO-style pose dataset and return a Labels object.

    Args:
        json_path: Path to the COCO annotation JSON file.
        dataset_root: Root directory of the dataset. If None, uses parent directory
                     of json_path.
        grayscale: If True, load images as grayscale (1 channel). If False, load as
                   RGB (3 channels). Default is False.

    Returns:
        Parsed labels as a Labels instance.
    """
    json_path = Path(json_path)

    if dataset_root is None:
        dataset_root = json_path.parent
    else:
        dataset_root = Path(dataset_root)

    # Parse COCO annotation file
    coco_data = parse_coco_json(json_path)

    # Create skeletons from categories
    skeletons = {}
    for category in coco_data["categories"]:
        if "keypoints" in category:
            skeleton = create_skeleton_from_category(category)
            skeletons[category["id"]] = skeleton

    # Track management: maps track_id -> Track object
    track_dict = {}

    # Create image id to annotation mapping
    image_annotations = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(annotation)

    # Group images by shape (height, width) for shared Video objects
    shape_to_images = {}
    image_id_to_path = {}
    image_id_to_shape = {}

    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        height = image_info.get("height", 0)
        width = image_info.get("width", 0)

        # Resolve image path
        try:
            image_path = resolve_image_path(image_filename, dataset_root)
            image_id_to_path[image_id] = image_path

            # Group by shape
            shape_key = (height, width)
            image_id_to_shape[image_id] = shape_key
            if shape_key not in shape_to_images:
                shape_to_images[shape_key] = []
            shape_to_images[shape_key].append(str(image_path))
        except FileNotFoundError:
            # Skip missing images
            continue

    # Create Video objects for each unique shape
    shape_to_video = {}
    for shape_key, image_paths in shape_to_images.items():
        height, width = shape_key
        # Create Video from the list of images with this shape
        video = Video.from_filename(
            image_paths,
            grayscale=grayscale,
        )
        # Store shape metadata from JSON (useful when images can't be read)
        channels = 1 if grayscale else 3
        video.backend_metadata["shape"] = (len(image_paths), height, width, channels)
        shape_to_video[shape_key] = video

    # Process images and annotations
    labeled_frames = []
    image_id_to_frame_idx = {}

    # Build frame index mapping for each image
    for shape_key, image_paths in shape_to_images.items():
        for frame_idx, image_path in enumerate(image_paths):
            # Find the image_id for this path
            for img_id, path in image_id_to_path.items():
                if str(path) == image_path:
                    image_id_to_frame_idx[img_id] = frame_idx
                    break

    for image_info in coco_data["images"]:
        image_id = image_info["id"]

        # Skip if image was not found
        if image_id not in image_id_to_path:
            continue

        # Get the video and frame index for this image
        shape_key = image_id_to_shape[image_id]
        video = shape_to_video[shape_key]
        frame_idx = image_id_to_frame_idx[image_id]

        # Create instances from annotations
        instances = []
        if image_id in image_annotations:
            for annotation in image_annotations[image_id]:
                category_id = annotation["category_id"]

                if category_id not in skeletons:
                    continue  # Skip non-pose annotations

                skeleton = skeletons[category_id]

                # Extract track ID from various possible sources
                track = None
                track_id = (
                    annotation.get("attributes", {}).get("object_id")
                    or annotation.get("track_id")
                    or annotation.get("instance_id")
                )

                if track_id is not None:
                    # Create or reuse Track object
                    if track_id not in track_dict:
                        track_dict[track_id] = Track(name=f"track_{track_id}")
                    track = track_dict[track_id]

                # Decode keypoints
                keypoints = annotation.get("keypoints", [])
                # Always use the skeleton length, not num_keypoints which may count
                # only visible points
                expected_keypoints = len(skeleton.nodes)

                if keypoints:
                    points_array = decode_keypoints(
                        keypoints, expected_keypoints, skeleton
                    )
                    instance = Instance.from_numpy(
                        points_data=points_array, skeleton=skeleton, track=track
                    )
                    instances.append(instance)

        # Create labeled frame
        if (
            instances or image_id in image_annotations
        ):  # Include frames even without instances
            labeled_frame = LabeledFrame(
                video=video, frame_idx=frame_idx, instances=instances
            )
            labeled_frames.append(labeled_frame)

    # Create Labels object (skeletons will be auto-added from instances)
    return Labels(labeled_frames=labeled_frames)


def read_labels_set(
    dataset_path: Union[str, Path],
    json_files: Optional[List[str]] = None,
    grayscale: bool = False,
) -> Dict[str, Labels]:
    """Read multiple COCO annotation files and return a dictionary of Labels.

    This function is designed to handle datasets with multiple splits (train/val/test)
    or multiple annotation files.

    Args:
        dataset_path: Root directory containing COCO annotation files.
        json_files: List of specific JSON filenames to load. If None, automatically
                   discovers all .json files in the dataset directory.
        grayscale: If True, load images as grayscale (1 channel). If False, load as
                   RGB (3 channels). Default is False.

    Returns:
        Dictionary mapping split names to Labels objects.
    """
    dataset_path = Path(dataset_path)

    if json_files is None:
        # Auto-discover JSON files
        json_files = [f.name for f in dataset_path.glob("*.json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON annotation files found in {dataset_path}")

    labels_dict = {}

    for json_file in json_files:
        json_path = dataset_path / json_file

        # Use filename (without extension) as split name
        split_name = json_path.stem

        # Load labels for this split
        labels = read_labels(json_path, dataset_root=dataset_path, grayscale=grayscale)
        labels_dict[split_name] = labels

    return labels_dict


def encode_keypoints(
    points_array: np.ndarray, visibility_encoding: str = "ternary"
) -> List[float]:
    """Encode numpy array of points into COCO keypoint format.

    Args:
        points_array: Numpy array of shape (num_keypoints, 2) or (num_keypoints, 3)
                     with [x, y] or [x, y, visibility] values.
        visibility_encoding: Visibility encoding to use. Either "binary" (0/1) or
                           "ternary" (0/1/2). Default is "ternary".

    Returns:
        Flat list of [x1, y1, v1, x2, y2, v2, ...] values.
    """
    keypoints = []
    for i in range(len(points_array)):
        if points_array.shape[1] == 2:
            x, y = points_array[i]
            visible = not (np.isnan(x) or np.isnan(y))
        else:
            x, y = points_array[i, :2]
            visible = points_array[i, 2] if points_array.shape[1] > 2 else True

        # Handle NaN coordinates
        if np.isnan(x) or np.isnan(y):
            keypoints.extend([0.0, 0.0, 0])  # Not labeled
        else:
            # Encode visibility
            if visibility_encoding == "binary":
                # Binary: 0 = not visible, 1 = visible
                visibility_value = 1 if visible else 0
            else:
                # Ternary: 0 = not labeled, 1 = labeled but occluded,
                # 2 = labeled and visible
                visibility_value = 2 if visible else 1
            keypoints.extend([float(x), float(y), visibility_value])

    return keypoints


def convert_labels(
    labels: Labels,
    image_filenames: Optional[Union[str, List[str]]] = None,
    visibility_encoding: str = "ternary",
) -> Dict:
    """Convert a Labels object into COCO-formatted annotations.

    Args:
        labels: SLEAP `Labels` object to be converted to COCO format.
        image_filenames: Optional image filenames to use. If provided, must be a single
                        string (for single-frame videos) or a list of strings matching
                        the number of labeled frames. If None, generates filenames from
                        video filenames and frame indices.
        visibility_encoding: Visibility encoding to use. Either "binary" (0/1) or
                           "ternary" (0/1/2). Default is "ternary".

    Returns:
        COCO annotation dictionary with "images", "annotations", and "categories"
        fields.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Build skeleton/category mapping
    skeleton_to_category = {}
    category_id_counter = 1
    for skeleton in labels.skeletons:
        if skeleton not in skeleton_to_category:
            category = {
                "id": category_id_counter,
                "name": skeleton.name
                if skeleton.name
                else f"skeleton_{category_id_counter}",
                "keypoints": [node.name for node in skeleton.nodes],
                "skeleton": [
                    [i + 1, j + 1]
                    for i, j in [
                        (
                            skeleton.nodes.index(edge.source),
                            skeleton.nodes.index(edge.destination),
                        )
                        for edge in skeleton.edges
                    ]
                ],  # Convert to 1-based indexing
            }
            coco_data["categories"].append(category)
            skeleton_to_category[skeleton] = category_id_counter
            category_id_counter += 1

    # Build track mapping
    track_to_id = {}
    track_id_counter = 1
    for track in labels.tracks:
        if track not in track_to_id:
            track_to_id[track] = track_id_counter
            track_id_counter += 1

    # Process image filenames
    if image_filenames is not None:
        if isinstance(image_filenames, str):
            image_filenames = [image_filenames]
        if len(image_filenames) != len(labels.labeled_frames):
            raise ValueError(
                f"Number of image filenames ({len(image_filenames)}) must match "
                f"number of labeled frames ({len(labels.labeled_frames)})"
            )

    # Process labeled frames
    image_id_counter = 1
    annotation_id_counter = 1

    for frame_idx, labeled_frame in enumerate(labels.labeled_frames):
        # Determine image filename
        if image_filenames is not None:
            image_filename = image_filenames[frame_idx]
        else:
            # Generate from video filename and frame index
            video = labeled_frame.video
            if isinstance(video.filename, list):
                # Image sequence - use the specific frame
                if labeled_frame.frame_idx < len(video.filename):
                    image_filename = Path(video.filename[labeled_frame.frame_idx]).name
                else:
                    image_filename = f"frame_{labeled_frame.frame_idx:06d}.png"
            else:
                # Video file - generate image name
                video_name = Path(video.filename).stem
                image_filename = f"{video_name}_frame_{labeled_frame.frame_idx:06d}.png"

        # Get image dimensions
        if labeled_frame.video.shape is not None:
            height = labeled_frame.video.shape[1]
            width = labeled_frame.video.shape[2]
        else:
            # Default dimensions if shape unavailable
            height = 0
            width = 0

        # Add image entry
        image_info = {
            "id": image_id_counter,
            "file_name": image_filename,
            "width": width,
            "height": height,
        }
        coco_data["images"].append(image_info)

        # Process instances
        for instance in labeled_frame.instances:
            # Get category ID
            category_id = skeleton_to_category[instance.skeleton]

            # Encode keypoints
            points_array = instance.numpy()
            keypoints = encode_keypoints(points_array, visibility_encoding)

            # Count visible keypoints
            num_keypoints = sum(
                1 for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0
            )

            # Compute bounding box from visible keypoints
            # mmpose requires bbox field for all annotations
            visible_points = []
            for i in range(0, len(keypoints), 3):
                if keypoints[i + 2] > 0:  # visible
                    visible_points.append([keypoints[i], keypoints[i + 1]])

            if visible_points:
                visible_points_array = np.array(visible_points)
                x_min = float(np.min(visible_points_array[:, 0]))
                y_min = float(np.min(visible_points_array[:, 1]))
                x_max = float(np.max(visible_points_array[:, 0]))
                y_max = float(np.max(visible_points_array[:, 1]))

                # Bbox in COCO format: [x, y, width, height]
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)
            else:
                # No visible keypoints - use zero bbox
                bbox = [0.0, 0.0, 0.0, 0.0]
                area = 0.0

            # Create annotation
            annotation = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_id,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }

            # Add track ID if present
            if instance.track is not None:
                annotation["attributes"] = {"object_id": track_to_id[instance.track]}

            coco_data["annotations"].append(annotation)
            annotation_id_counter += 1

        image_id_counter += 1

    return coco_data


def write_labels(
    labels: Labels,
    json_path: Union[str, Path],
    image_filenames: Optional[Union[str, List[str]]] = None,
    visibility_encoding: str = "ternary",
) -> None:
    """Write Labels to COCO-style JSON annotation file.

    Args:
        labels: SLEAP `Labels` object to save.
        json_path: Path to save the COCO annotation JSON file.
        image_filenames: Optional image filenames to use in the COCO JSON. If
                        provided, must be a single string (for single-frame videos) or
                        a list of strings matching the number of labeled frames. If
                        None, generates filenames from video filenames and frame
                        indices.
        visibility_encoding: Visibility encoding to use. Either "binary" (0/1) or
                           "ternary" (0/1/2). Default is "ternary".

    Notes:
        - This function only writes the JSON annotation file. It does not save images.
        - The generated JSON can be used with mmpose and other COCO-compatible tools.
        - For complete datasets with images, consider using save_dataset() instead.
    """
    json_path = Path(json_path)

    # Convert labels to COCO format
    coco_data = convert_labels(labels, image_filenames, visibility_encoding)

    # Write to JSON file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=2)
