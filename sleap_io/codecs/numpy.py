"""NumPy array codec for SLEAP Labels objects.

This module provides conversion between Labels objects and NumPy arrays with enhanced
flexibility compared to the original Labels.numpy() method. The codec supports various
array shapes, instance selection, and metadata handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    pass


def to_numpy(
    labels: Labels,
    *,
    video: Optional[Union[Video, int]] = None,
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

    # Figure out the number of tracks based on number of instances in each frame.
    n_instances = 0
    for lf in lfs:
        # Count instances based on what we're including
        n_user = len(lf.user_instances) if user_instances else 0
        n_predicted = len(lf.predicted_instances) if predicted_instances else 0

        if user_instances and predicted_instances:
            # Count max of either user or predicted instances per frame (not sum)
            n_frame_instances = max(n_user, n_predicted)
        else:
            n_frame_instances = n_user + n_predicted

        n_instances = max(n_instances, n_frame_instances)

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
            # For untracked instances, fill them in arbitrary order
            j = 0
            instances_to_include = []

            # If user instances are preferred, add them first
            if user_instances and lf.has_user_instances:
                # First collect all user instances
                for inst in lf.user_instances:
                    instances_to_include.append(inst)

                # For the trivial case (single instance per frame), if we found
                # user instances, we shouldn't include any predicted instances
                if is_single_instance and len(instances_to_include) > 0:
                    pass  # Skip adding predicted instances
                else:
                    # Add predicted instances that don't have a corresponding
                    # user instance
                    if predicted_instances:
                        for inst in lf.predicted_instances:
                            skip = False
                            for user_inst in lf.user_instances:
                                # Skip if this predicted instance is linked to a user
                                # instance via from_predicted
                                if (
                                    hasattr(user_inst, "from_predicted")
                                    and user_inst.from_predicted == inst
                                ):
                                    skip = True
                                    break
                                # Skip if user and predicted instances share same track
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
                # If user_instances=False, only include predicted instances
                if predicted_instances:
                    instances_to_include = lf.predicted_instances
                elif user_instances:
                    instances_to_include = lf.user_instances

            # Now process all the instances we want to include
            for inst in instances_to_include:
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
                    j += 1
        else:  # untracked is False
            # For tracked instances, organize by track ID

            # Create mapping from track to best instance for this frame
            track_to_instance = {}

            # First, add predicted instances to the mapping
            if predicted_instances:
                for inst in lf.predicted_instances:
                    if inst.track is not None:
                        track_to_instance[inst.track] = inst

            # Then, add user instances to the mapping (if user_instances=True)
            if user_instances:
                for inst in lf.user_instances:
                    if inst.track is not None:
                        track_to_instance[inst.track] = inst

            # Process the preferred instances for each track
            for track in track_to_instance:
                inst = track_to_instance[track]
                j = labels.tracks.index(track)

                if type(inst) is PredictedInstance:
                    tracks[i, j] = inst.numpy(scores=return_confidence)
                elif type(inst) is Instance:
                    tracks[i, j, :, :2] = inst.numpy()

                    # If return_confidence is True, add dummy confidence scores
                    if return_confidence:
                        tracks[i, j, :, 2] = 1.0

    return tracks


def from_numpy(
    tracks_array: np.ndarray,
    *,
    videos: Optional[list[Video]] = None,
    video: Optional[Video] = None,
    skeletons: Optional[list[Skeleton] | Skeleton] = None,
    skeleton: Optional[Skeleton] = None,
    tracks: Optional[list[Track]] = None,
    track_names: Optional[list[str]] = None,
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
