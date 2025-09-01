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

    # First pass: determine the number of nodes by scanning all frames
    max_points = 0
    for frame_data in data:
        annotations = frame_data.get("annotations", [])

        # Count points per instance by finding Face followed by points
        current_points = 0
        for i, ann in enumerate(annotations):
            if ann.get("class") == "Face":
                # Count subsequent point annotations
                current_points = 0
                for j in range(i + 1, len(annotations)):
                    if annotations[j].get("class") == "point":
                        current_points += 1
                    else:
                        break  # Stop at next Face or other annotation
                max_points = max(max_points, current_points)

    # Create skeleton with dynamically determined nodes
    nodes = [Node(str(i + 1)) for i in range(max_points)]
    skeleton = Skeleton(nodes=nodes)

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

        # Group annotations into instances (Face + subsequent points per instance)
        i = 0
        while i < len(annotations):
            if annotations[i].get("class") == "Face":
                # Found a new instance
                # Collect subsequent point annotations
                inst_points = []
                j = i + 1
                while j < len(annotations) and annotations[j].get("class") == "point":
                    inst_points.append(annotations[j])
                    j += 1

                # Create points array for this instance
                points = np.full((len(skeleton.nodes), 2), np.nan)
                for point_idx, point_data in enumerate(inst_points):
                    if point_idx < len(skeleton.nodes):
                        points[point_idx] = [point_data["x"], point_data["y"]]

                # Create instance
                instance = Instance(points=points, skeleton=skeleton)
                instances.append(instance)

                # Move to next annotation after the points
                i = j
            else:
                # Skip non-Face annotations that aren't part of an instance
                i += 1

        # Create labeled frame
        labeled_frame = LabeledFrame(
            video=video, frame_idx=frame_idx, instances=instances
        )
        labeled_frames.append(labeled_frame)

    # Create and return Labels object
    labels = Labels(labeled_frames=labeled_frames)
    labels.provenance["filename"] = str(labels_path)

    return labels
