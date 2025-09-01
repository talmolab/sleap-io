"""This module handles direct I/O operations for working with AlphaTracker files.

AlphaTracker is a multi-animal pose tracking system that exports annotations in
JSON format. Each JSON file contains an array of frame objects, where each frame
includes:
- filename: Reference to the image file
- class: Always "image" for frame objects
- annotations: Array containing bounding boxes (class: "Face") and keypoints
  (class: "point")

The format groups annotations by animal, with each animal having a Face bounding
box followed by its keypoint annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Node, Skeleton
from sleap_io.model.video import Video


def read_labels(labels_path: str) -> Labels:
    """Read AlphaTracker style annotations from a file and return a `Labels` object.

    Args:
        labels_path: Path to the AlphaTracker annotation file in JSON format.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    labels_path = Path(labels_path)

    # Load the JSON data
    with open(labels_path, "r") as f:
        data = json.load(f)

    # Create skeleton with 3 nodes named "1", "2", "3"
    skeleton = Skeleton(nodes=[Node("1"), Node("2"), Node("3")])

    # Collect all image filenames to create video
    image_files = []
    for frame_data in data:
        filename = frame_data["filename"]
        # Build full path to image file (assuming in same directory as JSON)
        img_path = labels_path.parent / filename
        image_files.append(str(img_path))

    # Create video from image files
    video = Video.from_filename(image_files)

    # Parse frames
    labeled_frames = []
    for frame_idx, frame_data in enumerate(data):
        instances = []

        # Get all annotations for this frame
        annotations = frame_data.get("annotations", [])

        # Group annotations into instances (Face + 3 points per instance)
        # Skip Face annotations, only use point annotations
        point_annotations = [a for a in annotations if a["class"] == "point"]

        # Split points into instances (3 points per instance)
        points_per_instance = 3
        num_instances = len(point_annotations) // points_per_instance

        for inst_idx in range(num_instances):
            # Get the 3 points for this instance
            start_idx = inst_idx * points_per_instance
            inst_points = point_annotations[start_idx : start_idx + points_per_instance]

            # Create points array for this instance
            points = np.full((len(skeleton.nodes), 2), np.nan)
            for point_idx, point_data in enumerate(inst_points):
                points[point_idx] = [point_data["x"], point_data["y"]]

            # Create instance
            instance = Instance(points=points, skeleton=skeleton)
            instances.append(instance)

        # Create labeled frame
        labeled_frame = LabeledFrame(
            video=video, frame_idx=frame_idx, instances=instances
        )
        labeled_frames.append(labeled_frame)

    # Create and return Labels object
    labels = Labels(labeled_frames=labeled_frames)
    labels.provenance["filename"] = str(labels_path)

    return labels
