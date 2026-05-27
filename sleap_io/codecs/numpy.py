"""NumPy array codec for SLEAP Labels objects.

This module provides conversion between Labels objects and NumPy arrays with enhanced
flexibility compared to the original Labels.numpy() method. The codec supports various
array shapes, instance selection, and metadata handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    pass


def _max_instances_per_frame(
    lfs: list[LabeledFrame],
    *,
    user_instances: bool,
    predicted_instances: bool,
) -> int:
    """Return the maximum number of instances to export from any single frame.

    Args:
        lfs: Labeled frames to scan.
        user_instances: If True, count user instances.
        predicted_instances: If True, count predicted instances.

    Returns:
        The largest per-frame instance count. When both instance types are
        included, user and predicted instances are assumed to overlap, so the
        per-frame count is `max(n_user, n_predicted)` rather than their sum.
    """
    n_instances = 0
    for lf in lfs:
        n_user = len(lf.user_instances) if user_instances else 0
        n_predicted = len(lf.predicted_instances) if predicted_instances else 0

        if user_instances and predicted_instances:
            # Count max of either user or predicted instances per frame (not sum).
            n_frame_instances = max(n_user, n_predicted)
        else:
            n_frame_instances = n_user + n_predicted

        n_instances = max(n_instances, n_frame_instances)
    return n_instances


def _untracked_frame_instances(
    lf: LabeledFrame,
    *,
    is_single_instance: bool,
    user_instances: bool,
    predicted_instances: bool,
) -> list[Instance]:
    """Select instances to export from a frame without using track identity.

    User instances are preferred. Predicted instances that duplicate a user
    instance (linked via `from_predicted` or sharing the same track) are
    dropped. For single-instance projects, any user instance fully suppresses
    predicted instances in the same frame.

    Args:
        lf: The labeled frame to select instances from.
        is_single_instance: True if the project has at most one instance per
            frame.
        user_instances: If True, include user instances, preferring them over
            predicted instances.
        predicted_instances: If True, include predicted instances.

    Returns:
        A new list of instances to export, in frame order.
    """
    instances_to_include: list[Instance] = []

    if user_instances and lf.has_user_instances:
        # Collect all user instances first.
        for inst in lf.user_instances:
            instances_to_include.append(inst)

        # For the trivial case (single instance per frame), if we found user
        # instances, we shouldn't include any predicted instances.
        if is_single_instance and len(instances_to_include) > 0:
            return instances_to_include

        # Add predicted instances that don't have a corresponding user instance.
        if predicted_instances:
            for inst in lf.predicted_instances:
                skip = False
                for user_inst in lf.user_instances:
                    # Skip if this predicted instance is linked to a user
                    # instance via from_predicted.
                    if (
                        hasattr(user_inst, "from_predicted")
                        and user_inst.from_predicted == inst
                    ):
                        skip = True
                        break
                    # Skip if user and predicted instances share same track.
                    if (
                        user_inst.track is not None
                        and inst.track is not None
                        and user_inst.track == inst.track
                    ):
                        skip = True
                        break
                if not skip:
                    instances_to_include.append(inst)
    else:
        # If user_instances=False or there are no user instances, only include
        # predicted instances (or fall back to user instances).
        if predicted_instances:
            instances_to_include = list(lf.predicted_instances)
        elif user_instances:
            instances_to_include = list(lf.user_instances)

    return instances_to_include


def _tracked_frame_instances(
    lf: LabeledFrame,
    *,
    user_instances: bool,
    predicted_instances: bool,
) -> dict[Track, Instance]:
    """Select one instance per track from a frame.

    Predicted instances are added first, then user instances override them for
    the same track. Instances without a track assignment are ignored.

    Args:
        lf: The labeled frame to select instances from.
        user_instances: If True, include user instances, preferring them over
            predicted instances with the same track.
        predicted_instances: If True, include predicted instances.

    Returns:
        A mapping from `Track` to the instance to export for that track.
    """
    track_to_instance: dict[Track, Instance] = {}

    # First, add predicted instances to the mapping.
    if predicted_instances:
        for inst in lf.predicted_instances:
            if inst.track is not None:
                track_to_instance[inst.track] = inst

    # Then, add user instances to the mapping (they override predicted).
    if user_instances:
        for inst in lf.user_instances:
            if inst.track is not None:
                track_to_instance[inst.track] = inst

    return track_to_instance


def to_numpy(
    labels: Labels,
    *,
    video: Video | int | None = None,
    untracked: bool = False,
    return_confidence: bool = False,
    user_instances: bool = True,
    predicted_instances: bool = True,
) -> np.ndarray:
    """Convert Labels to a numpy array.

    This is a more flexible version of Labels.numpy() with enhanced options.

    Args:
        labels: Labels object to convert.
        video: Video or video index to convert to numpy arrays. If None (the default),
            uses the first video.
        untracked: If False (the default), include only instances that have a track
            assignment. If True, includes all instances in each frame in arbitrary
            order.
        return_confidence: If False (the default), only return points of nodes. If
            True, return the points and scores of nodes.
        user_instances: If True (the default), include user instances when available,
            preferring them over predicted instances with the same track. If False,
            only include predicted instances.
        predicted_instances: If True (the default), include predicted instances.
            If False, only include user instances.

    Returns:
        An array of tracks of shape `(n_frames, n_tracks, n_nodes, 2)` if
        `return_confidence` is False. Otherwise returned shape is
        `(n_frames, n_tracks, n_nodes, 3)` if `return_confidence` is True.

        Missing data will be replaced with `np.nan`.

        If this is a single instance project, a track does not need to be assigned.

        When `user_instances=False`, only predicted instances will be returned.
        When `user_instances=True`, user instances will be preferred over predicted
        instances with the same track or if linked via `from_predicted`.

    Notes:
        This method assumes that instances have tracks assigned and is intended to
        function primarily for single-video prediction results.

        This function contains the core logic for numpy conversion. The
        Labels.numpy() method delegates to this function.

    Examples:
        >>> arr = to_numpy(labels, video=0, return_confidence=True)
        >>> arr.shape
        (100, 2, 5, 3)  # 100 frames, 2 tracks, 5 nodes, (x, y, score)

        >>> # Get only user instances
        >>> arr = to_numpy(labels, user_instances=True, predicted_instances=False)

        >>> # Include untracked instances
        >>> arr = to_numpy(labels, untracked=True)
    """
    # Convert video parameter to Video object
    if video is None:
        video = labels.videos[0] if labels.videos else None
    elif type(video) is int:
        video = labels.videos[video]

    # Use lazy fast path when available
    if labels.is_lazy:
        store = labels.labeled_frames._store
        return store.to_numpy(
            video=video,
            untracked=untracked,
            return_confidence=return_confidence,
            user_instances=user_instances,
        )

    # Eager path: filter labeled frames by video
    lfs = [lf for lf in labels.labeled_frames if lf.video == video]

    # Figure out frame index range.
    first_frame, last_frame = 0, 0
    for lf in lfs:
        first_frame = min(first_frame, lf.frame_idx)
        last_frame = max(last_frame, lf.frame_idx)

    # Use video length when available so output spans the full video duration.
    video_length = len(video) if video is not None else 0
    if video_length > 0:
        last_frame = max(last_frame, video_length - 1)

    # Figure out the number of tracks based on number of instances in each frame.
    n_instances = _max_instances_per_frame(
        lfs, user_instances=user_instances, predicted_instances=predicted_instances
    )

    # Case 1: We don't care about order because there's only 1 instance per frame,
    # or we're considering untracked instances.
    is_single_instance = n_instances == 1
    untracked = untracked or is_single_instance
    if untracked:
        n_tracks = n_instances
    else:
        # Case 2: We're considering only tracked instances.
        n_tracks = len(labels.tracks)

    n_frames = int(last_frame - first_frame + 1)
    skeleton = labels.skeletons[-1]  # Assume project only uses last skeleton
    n_nodes = len(skeleton.nodes)

    if return_confidence:
        tracks = np.full((n_frames, n_tracks, n_nodes, 3), np.nan, dtype="float32")
    else:
        tracks = np.full((n_frames, n_tracks, n_nodes, 2), np.nan, dtype="float32")

    for lf in lfs:
        i = int(lf.frame_idx - first_frame)

        if untracked:
            # For untracked instances, fill them in arbitrary order.
            frame_instances = _untracked_frame_instances(
                lf,
                is_single_instance=is_single_instance,
                user_instances=user_instances,
                predicted_instances=predicted_instances,
            )
            for j, inst in enumerate(frame_instances):
                if j < n_tracks:
                    if return_confidence:
                        if isinstance(inst, PredictedInstance):
                            tracks[i, j] = inst.numpy(scores=True)
                        else:
                            # For user instances, set confidence to 1.0
                            points_data = inst.numpy()
                            confidence = np.ones(
                                (points_data.shape[0], 1), dtype="float32"
                            )
                            tracks[i, j] = np.hstack((points_data, confidence))
                    else:
                        tracks[i, j] = inst.numpy()
        else:  # untracked is False
            # For tracked instances, organize by track ID.
            track_to_instance = _tracked_frame_instances(
                lf,
                user_instances=user_instances,
                predicted_instances=predicted_instances,
            )
            for track, inst in track_to_instance.items():
                j = labels.tracks.index(track)

                if type(inst) is PredictedInstance:
                    tracks[i, j] = inst.numpy(scores=return_confidence)
                elif type(inst) is Instance:
                    tracks[i, j, :, :2] = inst.numpy()

                    # If return_confidence is True, add dummy confidence scores
                    if return_confidence:
                        tracks[i, j, :, 2] = 1.0

    return tracks


def to_analysis_arrays(
    labels: Labels,
    *,
    video: Video | int | None = None,
    all_frames: bool = True,
    min_occupancy: float = 0.0,
    user_instances: bool = True,
    predicted_instances: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
]:
    """Build occupancy and point-data matrices for an analysis HDF5 export.

    All returned arrays are in canonical shape (frame-first ordering). This is
    the shared builder behind `save_analysis_h5`; it reuses the same
    instance-slotting logic as `to_numpy` so that projects without track
    assignments keep every instance instead of collapsing to one per frame.

    This is an internal building block — most users should call
    `sleap_io.save_analysis_h5` instead, which writes these arrays to disk in
    the SLEAP analysis HDF5 layout.

    Args:
        labels: The `Labels` from which to get data.
        video: Video (or video index) to export. If None, uses the first video.
        all_frames: If True, include all frames from 0 to the end of the video.
            If False, only include frames from the first labeled frame onward.
        min_occupancy: Minimum occupancy ratio (0-1) for a track to be kept.
            0 keeps all non-empty tracks (default SLEAP behavior).
        user_instances: If True, include user instances, preferring them over
            predicted instances.
        predicted_instances: If True, include predicted instances.

    Returns:
        A tuple of (all arrays in canonical shape):

        - occupancy: shape `(n_frames, n_tracks)`, binary `uint8` matrix
          indicating track presence per frame.
        - locations: shape `(n_frames, n_tracks, n_nodes, 2)`, point
          coordinates `(x, y)`.
        - point_scores: shape `(n_frames, n_tracks, n_nodes)`, per-point
          confidence.
        - instance_scores: shape `(n_frames, n_tracks)`, instance confidence.
        - tracking_scores: shape `(n_frames, n_tracks)`, tracking confidence.
        - track_names: list of track name strings, length `n_tracks`.
        - first_frame: first frame index (int).

    Raises:
        ValueError: If there are no labeled frames for the selected video.

    Notes:
        When the project has no `Track` assignments, instances are slotted in
        arbitrary per-frame order and `n_tracks` is sized to the largest number
        of instances in any frame (matching `to_numpy(untracked=True)`).
        Synthetic track names `track_0 ... track_{n-1}` are generated in that
        case. Per-slot animal identity is arbitrary across frames without real
        track information, but no data is dropped.
    """
    # Resolve video.
    if video is None:
        video = labels.videos[0]
    elif type(video) is int:
        video = labels.videos[video]

    # Get labeled frames for this video.
    lfs = labels.find(video)
    if not lfs:
        raise ValueError(f"No labeled frames in video: {video.filename}")

    frame_idxs = sorted(lf.frame_idx for lf in lfs)
    first_frame = 0 if all_frames else frame_idxs[0]

    # Use video length when available so output spans the full video duration.
    last_frame = frame_idxs[-1]
    video_length = len(video)
    if video_length > 0:
        last_frame = max(last_frame, video_length - 1)

    n_frames = last_frame - first_frame + 1

    skeleton = labels.skeletons[0]
    node_count = len(skeleton.nodes)

    # Size the track axis. With no track assignments, fall back to untracked
    # slotting so multi-animal projects keep every instance.
    untracked = len(labels.tracks) == 0
    if untracked:
        n_instances = _max_instances_per_frame(
            lfs,
            user_instances=user_instances,
            predicted_instances=predicted_instances,
        )
        is_single_instance = n_instances == 1
        n_tracks = n_instances
    else:
        n_tracks = len(labels.tracks)
        track_to_slot = {track: i for i, track in enumerate(labels.tracks)}

    # Initialize matrices in canonical shape: (frame, track, ...).
    occupancy = np.zeros((n_frames, n_tracks), dtype=np.uint8)
    locations = np.full((n_frames, n_tracks, node_count, 2), np.nan, dtype=np.float64)
    point_scores = np.full((n_frames, n_tracks, node_count), np.nan, dtype=np.float64)
    instance_scores = np.full((n_frames, n_tracks), np.nan, dtype=np.float64)
    tracking_scores = np.full((n_frames, n_tracks), np.nan, dtype=np.float64)

    # Build lookup for frame index -> LabeledFrame.
    lf_map = {lf.frame_idx: lf for lf in lfs}

    # Fill matrices.
    for frame_idx in range(first_frame, last_frame + 1):
        frame_i = frame_idx - first_frame
        lf = lf_map.get(frame_idx)
        if lf is None:
            continue

        # Determine which instances go into which track slot.
        if untracked:
            slotted = enumerate(
                _untracked_frame_instances(
                    lf,
                    is_single_instance=is_single_instance,
                    user_instances=user_instances,
                    predicted_instances=predicted_instances,
                )
            )
        else:
            slotted = (
                (track_to_slot[track], inst)
                for track, inst in _tracked_frame_instances(
                    lf,
                    user_instances=user_instances,
                    predicted_instances=predicted_instances,
                ).items()
                if track in track_to_slot
            )

        for track_i, inst in slotted:
            # Defensive: per-frame dedup can occasionally leave more instances
            # than the global maximum; extra instances are dropped to stay
            # consistent with to_numpy's `j < n_tracks` guard.
            if track_i >= n_tracks:
                continue

            occupancy[frame_i, track_i] = 1
            locations[frame_i, track_i, :, :] = inst.numpy()

            if hasattr(inst, "tracking_score") and inst.tracking_score is not None:
                tracking_scores[frame_i, track_i] = inst.tracking_score

            if isinstance(inst, PredictedInstance):
                if "score" in inst.points.dtype.names:
                    point_scores[frame_i, track_i, :] = inst.points["score"]
                if inst.score is not None:
                    instance_scores[frame_i, track_i] = inst.score

    # Filter empty/low-occupancy tracks.
    occupied_frames = np.sum(occupancy, axis=0)
    occupancy_ratio = occupied_frames / n_frames
    keep_mask = (occupied_frames > 0) & (occupancy_ratio >= min_occupancy)

    if not np.all(keep_mask):
        occupancy = occupancy[:, keep_mask]
        locations = locations[:, keep_mask, :, :]
        point_scores = point_scores[:, keep_mask, :]
        instance_scores = instance_scores[:, keep_mask]
        tracking_scores = tracking_scores[:, keep_mask]

    # Build track names sized to the surviving tracks.
    if untracked:
        # Synthesize positional names, renumbered after filtering (no gaps).
        track_names = [f"track_{i}" for i in range(occupancy.shape[1])]
    else:
        track_names = [
            track.name for i, track in enumerate(labels.tracks) if keep_mask[i]
        ]

    return (
        occupancy,
        locations,
        point_scores,
        instance_scores,
        tracking_scores,
        track_names,
        first_frame,
    )


def from_numpy(
    tracks_array: np.ndarray,
    *,
    videos: list[Video] | None = None,
    video: Video | None = None,
    skeletons: list[Skeleton] | Skeleton | None = None,
    skeleton: Skeleton | None = None,
    tracks: list[Track] | None = None,
    track_names: list[str] | None = None,
    first_frame: int = 0,
    return_confidence: bool = False,
) -> Labels:
    """Create a new Labels object from a numpy array of tracks.

    This factory method creates a new Labels object with instances constructed from
    the provided numpy array. It is a more flexible version of Labels.from_numpy().

    Args:
        tracks_array: A numpy array of tracks, with shape
            `(n_frames, n_tracks, n_nodes, 2)` or `(n_frames, n_tracks, n_nodes, 3)`,
            where the last dimension contains the x,y coordinates (and optionally
            confidence scores).
        videos: List of Video objects to associate with the labels. At least one
            video is required. Mutually exclusive with `video`.
        video: Single Video object to associate with the labels. Mutually exclusive
            with `videos`.
        skeletons: Skeleton or list of Skeleton objects to use for the instances.
            At least one skeleton is required. Mutually exclusive with `skeleton`.
        skeleton: Single Skeleton object to use. Mutually exclusive with `skeletons`.
        tracks: List of Track objects corresponding to the second dimension of the
            array. If not specified, new tracks will be created automatically using
            `track_names` if provided, or default names.
        track_names: List of track names to use when auto-creating tracks. Only used
            if `tracks` is None.
        first_frame: Frame index to start the labeled frames from. Default is 0.
        return_confidence: Whether the tracks array contains confidence scores in the
            last dimension. If True, tracks.shape[-1] should be 3. If False or None,
            will be inferred from array shape.

    Returns:
        A new Labels object with instances constructed from the numpy array.

    Raises:
        ValueError: If the array dimensions are invalid, or if no videos or
            skeletons are provided, or if both `videos` and `video` are provided.

    Examples:
        >>> import numpy as np
        >>> from sleap_io import Video, Skeleton
        >>> from sleap_io.codecs import from_numpy
        >>> # Create a simple tracking array for 2 frames, 1 track, 2 nodes
        >>> arr = np.zeros((2, 1, 2, 2))
        >>> arr[0, 0] = [[10, 20], [30, 40]]  # Frame 0
        >>> arr[1, 0] = [[15, 25], [35, 45]]  # Frame 1
        >>> # Create labels from the array
        >>> video = Video(filename="example.mp4")
        >>> skeleton = Skeleton(["head", "tail"])
        >>> labels = from_numpy(arr, video=video, skeleton=skeleton)

        >>> # With custom track names
        >>> labels = from_numpy(arr, video=video, skeleton=skeleton,
        ...                     track_names=["mouse1"])

        >>> # With confidence scores
        >>> arr_with_conf = np.zeros((2, 1, 2, 3))
        >>> arr_with_conf[0, 0] = [[10, 20, 0.95], [30, 40, 0.98]]
        >>> labels = from_numpy(arr_with_conf, video=video, skeleton=skeleton,
        ...                     return_confidence=True)
    """
    # Check dimensions
    if len(tracks_array.shape) != 4:
        raise ValueError(
            f"Array must have 4 dimensions (n_frames, n_tracks, n_nodes, 2 or 3), "
            f"but got {tracks_array.shape}"
        )

    # Handle video/videos parameter
    if video is not None and videos is not None:
        raise ValueError("Cannot specify both 'video' and 'videos' parameters")

    if video is not None:
        videos = [video]
    elif videos is None:
        raise ValueError("At least one video must be provided via 'video' or 'videos'")

    if not videos:
        raise ValueError("At least one video must be provided")

    video = videos[0]  # Use the first video for creating labeled frames

    # Handle skeleton/skeletons parameter
    if skeleton is not None and skeletons is not None:
        raise ValueError("Cannot specify both 'skeleton' and 'skeletons' parameters")

    if skeleton is not None:
        skeletons = [skeleton]
    elif skeletons is None:
        raise ValueError(
            "At least one skeleton must be provided via 'skeleton' or 'skeletons'"
        )
    elif isinstance(skeletons, Skeleton):
        skeletons = [skeletons]
    elif not skeletons:  # Check for empty list
        raise ValueError("At least one skeleton must be provided")

    skeleton = skeletons[0]  # Use the first skeleton for creating instances
    n_nodes = len(skeleton.nodes)

    # Check if tracks_array contains confidence scores
    has_confidence = tracks_array.shape[-1] == 3 or return_confidence

    # Get dimensions
    n_frames, n_tracks_arr, _ = tracks_array.shape[:3]

    # Create or validate tracks
    if tracks is None:
        # Auto-create tracks
        if track_names is not None:
            if len(track_names) < n_tracks_arr:
                # Extend with default names if needed
                track_names = list(track_names) + [
                    f"track_{i}" for i in range(len(track_names), n_tracks_arr)
                ]
            tracks = [Track(name=name) for name in track_names[:n_tracks_arr]]
        else:
            tracks = [Track(f"track_{i}") for i in range(n_tracks_arr)]
    elif len(tracks) < n_tracks_arr:
        # Add missing tracks if needed
        original_len = len(tracks)
        for i in range(n_tracks_arr - original_len):
            tracks.append(Track(f"track_{i}"))

    # Create a new empty Labels object
    labels = Labels()
    labels.videos = list(videos)
    labels.skeletons = list(skeletons)
    labels.tracks = list(tracks)

    # Create labeled frames and instances from the array data
    for i in range(n_frames):
        frame_idx = i + first_frame

        # Check if this frame has any valid data across all tracks
        frame_has_valid_data = False
        for j in range(n_tracks_arr):
            track_data = tracks_array[i, j]
            # Check if at least one node in this track has valid xy coordinates
            if np.any(~np.isnan(track_data[:, 0])):
                frame_has_valid_data = True
                break

        # Skip creating a frame if there's no valid data
        if not frame_has_valid_data:
            continue

        # Create a new labeled frame
        labeled_frame = LabeledFrame(video=video, frame_idx=frame_idx)
        frame_has_valid_instances = False

        # Process each track in this frame
        for j in range(n_tracks_arr):
            track = tracks[j]
            track_data = tracks_array[i, j]

            # Check if there's any valid data for this track at this frame
            valid_points = ~np.isnan(track_data[:, 0])
            if not np.any(valid_points):
                continue

            # Create points from numpy data
            points = track_data[:, :2].copy()

            # Create new instance
            if has_confidence:
                # Get confidence scores
                if tracks_array.shape[-1] == 3:
                    scores = track_data[:, 2].copy()
                else:
                    scores = np.ones(n_nodes)

                # Fix NaN scores
                scores = np.where(np.isnan(scores), 1.0, scores)

                # Create instance with confidence scores
                new_instance = PredictedInstance.from_numpy(
                    points_data=points,
                    skeleton=skeleton,
                    point_scores=scores,
                    score=1.0,
                    track=track,
                )
            else:
                # Create instance with default scores
                new_instance = PredictedInstance.from_numpy(
                    points_data=points,
                    skeleton=skeleton,
                    point_scores=np.ones(n_nodes),
                    score=1.0,
                    track=track,
                )

            # Add to frame
            labeled_frame.instances.append(new_instance)
            frame_has_valid_instances = True

        # Only add frames that have instances
        if frame_has_valid_instances:
            labels.append(labeled_frame, update=False)

    # Update internal references
    labels.update()

    return labels
