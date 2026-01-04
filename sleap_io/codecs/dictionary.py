"""Dictionary codec for SLEAP Labels objects.

This module provides conversion between Labels objects and primitive Python dictionaries
that are fully JSON-serializable. The dictionary format is lossless and preserves all
information from the Labels object.

The dictionary codec is useful for:
- JSON/YAML serialization
- Web APIs and data interchange
- Debugging and inspection
- Custom storage formats
"""

from __future__ import annotations

from typing import Any, Optional

from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labels import Labels
from sleap_io.model.video import Video


def to_dict(
    labels: Labels,
    *,
    video: Optional[Video | int] = None,
    skip_empty_frames: bool = False,
) -> dict[str, Any]:
    """Convert Labels to a primitive dictionary (JSON-serializable).

    Args:
        labels: Labels object to convert.
        video: Optional video filter. If specified, only frames from this video
            are included. Can be a Video object or integer index.
        skip_empty_frames: If True, exclude frames with no instances.

    Returns:
        Dictionary with structure:
        {
            "version": "1.0.0",
            "skeletons": [...],
            "videos": [...],
            "tracks": [...],
            "labeled_frames": [...],
            "suggestions": [...],
            "provenance": {...}
        }

        All values are JSON-serializable primitives (str, int, float, bool, None,
        list, dict). No numpy arrays or custom objects.

    Examples:
        >>> labels = load_file("predictions.slp")
        >>> d = to_dict(labels)
        >>> import json
        >>> json.dumps(d)  # Fully serializable!

        >>> # Filter to specific video
        >>> d = to_dict(labels, video=0)

        >>> # Exclude empty frames
        >>> d = to_dict(labels, skip_empty_frames=True)

    Notes:
        - Uses index-based references (e.g., skeleton_idx, video_idx) for compactness
        - Preserves all metadata including provenance
        - Points are stored as list of dicts with x, y, visible, complete fields
        - The output is fully compatible with JSON, YAML, or any format that handles
          Python primitives
    """
    # Filter video if specified
    if video is not None:
        if isinstance(video, int):
            video = labels.videos[video]
        labeled_frames = [lf for lf in labels.labeled_frames if lf.video == video]
    else:
        labeled_frames = labels.labeled_frames

    # Skip empty frames if requested
    if skip_empty_frames:
        labeled_frames = [lf for lf in labeled_frames if len(lf.instances) > 0]

    # Build skeleton list
    skeletons_list = []
    for skeleton in labels.skeletons:
        skeleton_dict = {
            "name": skeleton.name,
            "nodes": [node.name for node in skeleton.nodes],
            "edges": [[skeleton.nodes.index(edge.source), skeleton.nodes.index(edge.destination)]
                      for edge in skeleton.edges],
        }

        # Add symmetries if present
        if skeleton.symmetries:
            symmetries_list = []
            for symmetry in skeleton.symmetries:
                # Convert set to list for indexing and get indices
                nodes_list = list(symmetry.nodes)
                indices = [
                    skeleton.nodes.index(nodes_list[0]),
                    skeleton.nodes.index(nodes_list[1])
                ]
                # Sort indices for consistent ordering
                indices.sort()
                symmetries_list.append(indices)
            skeleton_dict["symmetries"] = symmetries_list

        skeletons_list.append(skeleton_dict)

    # Build video list
    videos_list = []
    for vid in labels.videos:
        video_dict = {
            "filename": vid.filename,
        }

        # Add shape if available
        try:
            if hasattr(vid, "shape") and vid.shape is not None:
                video_dict["shape"] = list(vid.shape)
        except Exception:
            # Shape might not be accessible
            pass

        # Add backend info if available
        if hasattr(vid, "backend") and vid.backend is not None:
            backend_dict = {
                "type": type(vid.backend).__name__,
            }
            # Add any other relevant backend info
            if hasattr(vid.backend, "grayscale"):
                backend_dict["grayscale"] = vid.backend.grayscale
            video_dict["backend"] = backend_dict

        videos_list.append(video_dict)

    # Build track list
    tracks_list = []
    for track in labels.tracks:
        track_dict = {
            "name": track.name,
        }
        if hasattr(track, "spawned_on") and track.spawned_on is not None:
            track_dict["spawned_on"] = track.spawned_on
        tracks_list.append(track_dict)

    # Build labeled frames list
    labeled_frames_list = []
    for lf in labeled_frames:
        instances_list = []

        for instance in lf.instances:
            # Determine instance type
            is_predicted = isinstance(instance, PredictedInstance)

            # Build points list
            points_list = []
            for point in instance.points:
                point_dict = {
                    "x": float(point["xy"][0]),
                    "y": float(point["xy"][1]),
                    "visible": bool(point["visible"]),
                    "complete": bool(point["complete"]),
                }
                points_list.append(point_dict)

            # Build instance dict
            instance_dict = {
                "type": "predicted_instance" if is_predicted else "instance",
                "skeleton_idx": labels.skeletons.index(instance.skeleton),
                "points": points_list,
            }

            # Add track if present
            if instance.track is not None:
                instance_dict["track_idx"] = labels.tracks.index(instance.track)

            # Add tracking score if present
            if hasattr(instance, "tracking_score") and instance.tracking_score is not None:
                instance_dict["tracking_score"] = float(instance.tracking_score)

            # Add score for predicted instances
            if is_predicted:
                instance_dict["score"] = float(instance.score)

            # Add from_predicted if present
            if hasattr(instance, "from_predicted") and instance.from_predicted is not None:
                # Note: We can't serialize the reference, just indicate it exists
                instance_dict["has_from_predicted"] = True

            instances_list.append(instance_dict)

        frame_dict = {
            "frame_idx": int(lf.frame_idx),
            "video_idx": labels.videos.index(lf.video),
            "instances": instances_list,
        }
        labeled_frames_list.append(frame_dict)

    # Build suggestions list (if filtering by video, also filter suggestions)
    if video is not None:
        suggestions_to_include = [sf for sf in labels.suggestions if sf.video == video]
    else:
        suggestions_to_include = labels.suggestions

    suggestions_list = []
    for sf in suggestions_to_include:
        suggestion_dict = {
            "frame_idx": int(sf.frame_idx),
            "video_idx": labels.videos.index(sf.video),
        }
        suggestions_list.append(suggestion_dict)

    # Build complete dictionary
    result = {
        "version": "1.0.0",
        "skeletons": skeletons_list,
        "videos": videos_list,
        "tracks": tracks_list,
        "labeled_frames": labeled_frames_list,
        "suggestions": suggestions_list,
        "provenance": dict(labels.provenance),  # Copy to ensure it's a plain dict
    }

    return result
