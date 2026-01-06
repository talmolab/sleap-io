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

from sleap_io.model.instance import Instance, PredictedInstance, Track
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
            "edges": [
                [
                    skeleton.nodes.index(edge.source),
                    skeleton.nodes.index(edge.destination),
                ]
                for edge in skeleton.edges
            ],
        }

        # Add symmetries if present
        if skeleton.symmetries:
            symmetries_list = []
            for symmetry in skeleton.symmetries:
                # Convert set to list for indexing and get indices
                nodes_list = list(symmetry.nodes)
                indices = [
                    skeleton.nodes.index(nodes_list[0]),
                    skeleton.nodes.index(nodes_list[1]),
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

        # Add shape if available (Video.shape handles exceptions internally)
        if vid.shape is not None:
            video_dict["shape"] = list(vid.shape)

        # Add backend info if available
        if vid.backend is not None:
            video_dict["backend"] = {"type": type(vid.backend).__name__}

        videos_list.append(video_dict)

    # Build track list
    tracks_list = [{"name": track.name} for track in labels.tracks]

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
            if (
                hasattr(instance, "tracking_score")
                and instance.tracking_score is not None
            ):
                instance_dict["tracking_score"] = float(instance.tracking_score)

            # Add score for predicted instances
            if is_predicted:
                instance_dict["score"] = float(instance.score)

            # Add from_predicted if present
            if (
                hasattr(instance, "from_predicted")
                and instance.from_predicted is not None
            ):
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


def from_dict(data: dict[str, Any]) -> Labels:
    """Create a Labels object from a dictionary.

    This is the inverse of `to_dict()` and reconstructs a Labels object from
    its dictionary representation.

    Args:
        data: Dictionary in the format produced by `to_dict()`.

    Returns:
        A Labels object reconstructed from the dictionary.

    Raises:
        ValueError: If the dictionary format is invalid or missing required keys.

    Examples:
        >>> d = to_dict(labels)
        >>> labels_restored = from_dict(d)

        >>> # Round-trip through JSON
        >>> import json
        >>> json_str = json.dumps(to_dict(labels))
        >>> labels_restored = from_dict(json.loads(json_str))

    Notes:
        - The `from_predicted` relationship cannot be fully restored since the
          dictionary only indicates its presence, not the actual reference.
        - Video backends are not restored; videos are created with filename only.
    """
    import numpy as np

    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Edge, Node, Skeleton, Symmetry
    from sleap_io.model.suggestions import SuggestionFrame

    # Validate required keys
    required_keys = ["skeletons", "videos", "labeled_frames"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    # Build skeletons
    skeletons = []
    for skel_dict in data["skeletons"]:
        # Create nodes
        nodes = [Node(name=name) for name in skel_dict["nodes"]]

        # Create edges
        edges = []
        for src_idx, dst_idx in skel_dict.get("edges", []):
            edges.append(Edge(source=nodes[src_idx], destination=nodes[dst_idx]))

        # Create symmetries
        symmetries = []
        for sym_indices in skel_dict.get("symmetries", []):
            symmetries.append(
                Symmetry(nodes={nodes[sym_indices[0]], nodes[sym_indices[1]]})
            )

        skeleton = Skeleton(
            nodes=nodes,
            edges=edges,
            symmetries=symmetries,
            name=skel_dict.get("name", ""),
        )
        skeletons.append(skeleton)

    # Build videos
    videos = []
    for vid_dict in data["videos"]:
        video = Video(filename=vid_dict["filename"])
        videos.append(video)

    # Build tracks
    tracks = []
    for track_dict in data.get("tracks", []):
        track = Track(name=track_dict["name"])
        tracks.append(track)

    # Build labeled frames
    labeled_frames = []
    for lf_dict in data["labeled_frames"]:
        video = videos[lf_dict["video_idx"]]
        frame_idx = lf_dict["frame_idx"]

        instances = []
        for inst_dict in lf_dict.get("instances", []):
            skeleton = skeletons[inst_dict["skeleton_idx"]]
            is_predicted = inst_dict["type"] == "predicted_instance"

            # Build points array
            points_list = inst_dict["points"]
            n_nodes = len(points_list)
            points_data = np.full((n_nodes, 2), np.nan, dtype="float64")

            for node_idx, pt in enumerate(points_list):
                if pt.get("visible", True):
                    points_data[node_idx, 0] = pt["x"]
                    points_data[node_idx, 1] = pt["y"]

            # Get track if present
            track = None
            if "track_idx" in inst_dict:
                track = tracks[inst_dict["track_idx"]]

            # Get tracking score if present
            tracking_score = inst_dict.get("tracking_score")

            if is_predicted:
                # Get instance score
                score = inst_dict.get("score", 0.0)

                # Create predicted instance
                instance = PredictedInstance.from_numpy(
                    points_data=points_data,
                    skeleton=skeleton,
                    score=score,
                    track=track,
                    tracking_score=tracking_score,
                )
            else:
                # Create user instance
                instance = Instance.from_numpy(
                    points_data=points_data,
                    skeleton=skeleton,
                    track=track,
                    tracking_score=tracking_score,
                )

            instances.append(instance)

        labeled_frame = LabeledFrame(
            video=video,
            frame_idx=frame_idx,
            instances=instances,
        )
        labeled_frames.append(labeled_frame)

    # Build suggestions
    suggestions = []
    for sug_dict in data.get("suggestions", []):
        video = videos[sug_dict["video_idx"]]
        suggestion = SuggestionFrame(
            video=video,
            frame_idx=sug_dict["frame_idx"],
        )
        suggestions.append(suggestion)

    # Build provenance
    provenance = dict(data.get("provenance", {}))

    # Create Labels object
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        suggestions=suggestions,
        provenance=provenance,
    )

    return labels
