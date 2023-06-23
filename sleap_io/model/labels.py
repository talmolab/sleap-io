"""Data structure for the labels, a top-level container for pose data.

`Label`s contain `LabeledFrame`s, which in turn contain `Instance`s, which contain
`Point`s.

This structure also maintains metadata that is common across all child objects such as
`Track`s, `Video`s, `Skeleton`s and others.

It is intended to be the entrypoint for deserialization and main container that should
be used for serialization. It is designed to support both labeled data (used for
training models) and predictions (inference results).
"""

from __future__ import annotations
from sleap_io import LabeledFrame, Instance, PredictedInstance, Video, Track
from attrs import define, field
from typing import Union, Optional, Any
import numpy as np

from sleap_io.model.skeleton import Skeleton


@define
class Labels:
    """Pose data for a set of videos that have user labels and/or predictions.

    Attributes:
        labeled_frames: A list of `LabeledFrame`s that are associated with this dataset.
        videos: A list of `Video`s that are associated with this dataset. Videos do not
            need to have corresponding `LabeledFrame`s if they do not have any
            labels or predictions yet.
        skeletons: A list of `Skeleton`s that are associated with this dataset. This
            should generally only contain a single skeleton.
        tracks: A list of `Track`s that are associated with this dataset.
        provenance: Dictionary of arbitrary metadata providing additional information
            about where the dataset came from.

    Notes:
        `Video`s in contain `LabeledFrame`s, and `Skeleton`s and `Track`s in contained
        `Instance`s are added to the respective lists automatically.
    """

    labeled_frames: list[LabeledFrame] = field(factory=list)
    videos: list[Video] = field(factory=list)
    skeletons: list[Skeleton] = field(factory=list)
    tracks: list[Track] = field(factory=list)
    provenance: dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self):
        """Append videos, skeletons, and tracks seen in `labeled_frames` to `Labels`."""
        for lf in self.labeled_frames:
            if lf.video not in self.videos:
                self.videos.append(lf.video)

            for inst in lf:
                if inst.skeleton not in self.skeletons:
                    self.skeletons.append(inst.skeleton)

                if inst.track is not None and inst.track not in self.tracks:
                    self.tracks.append(inst.track)

    def __getitem__(self, key: int) -> Union[list[LabeledFrame], LabeledFrame]:
        """Return one or more labeled frames based on indexing criteria."""
        if type(key) == int:
            return self.labeled_frames[key]
        else:
            raise IndexError(f"Invalid indexing argument for labels: {key}")

    def __iter__(self):
        """Iterate over `labeled_frames` list when calling iter method on `Labels`."""
        return iter(self.labeled_frames)

    def __len__(self) -> int:
        """Return number of labeled frames."""
        return len(self.labeled_frames)

    def __repr__(self) -> str:
        """Return a readable representation of the labels."""
        return (
            "Labels("
            f"labeled_frames={len(self.labeled_frames)}, "
            f"videos={len(self.videos)}, "
            f"skeletons={len(self.skeletons)}, "
            f"tracks={len(self.tracks)}"
            ")"
        )

    def __str__(self) -> str:
        """Return a readable representation of the labels."""
        return self.__repr__()

    def numpy(
        self,
        video: Optional[Union[Video, int]] = None,
        all_frames: bool = True,
        untracked: bool = False,
        return_confidence: bool = False,
    ) -> np.ndarray:
        """Construct a numpy array from instance points.

        Args:
            video: Video or video index to convert to numpy arrays. If `None` (the
                default), uses the first video.
            untracked: If `False` (the default), include only instances that have a
                track assignment. If `True`, includes all instances in each frame in
                arbitrary order.
            return_confidence: If `False` (the default), only return points of nodes. If
                `True`, return the points and scores of nodes.

        Returns:
            An array of tracks of shape `(n_frames, n_tracks, n_nodes, 2)` if
            `return_confidence` is `False`. Otherwise returned shape is
            `(n_frames, n_tracks, n_nodes, 3)` if `return_confidence` is `True`.

            Missing data will be replaced with `np.nan`.

            If this is a single instance project, a track does not need to be assigned.

            Only predicted instances (NOT user instances) will be returned.

        Notes:
            This method assumes that instances have tracks assigned and is intended to
            function primarily for single-video prediction results.
        """
        # Get labeled frames for specified video.
        if video is None:
            video = 0
        if type(video) == int:
            video = self.videos[video]
        lfs = [lf for lf in self.labeled_frames if lf.video == video]

        # Figure out frame index range.
        first_frame, last_frame = 0, 0
        for lf in lfs:
            first_frame = min(first_frame, lf.frame_idx)
            last_frame = max(last_frame, lf.frame_idx)

        # Figure out the number of tracks based on number of instances in each frame.
        # First, let's check the max number of predicted instances (regardless of
        # whether they're tracked.
        n_preds = 0
        for lf in lfs:
            n_pred_instances = len(lf.predicted_instances)
            n_preds = max(n_preds, n_pred_instances)

        # Case 1: We don't care about order because there's only 1 instance per frame,
        # or we're considering untracked instances.
        untracked = untracked or n_preds == 1
        if untracked:
            n_tracks = n_preds
        else:
            # Case 2: We're considering only tracked instances.
            n_tracks = len(self.tracks)

        n_frames = int(last_frame - first_frame + 1)
        skeleton = self.skeletons[-1]  # Assume project only uses last skeleton
        n_nodes = len(skeleton.nodes)

        if return_confidence:
            tracks = np.full((n_frames, n_tracks, n_nodes, 3), np.nan, dtype="float32")
        else:
            tracks = np.full((n_frames, n_tracks, n_nodes, 2), np.nan, dtype="float32")
        for lf in lfs:
            i = int(lf.frame_idx - first_frame)
            if untracked:
                for j, inst in enumerate(lf.predicted_instances):
                    tracks[i, j] = inst.numpy(scores=return_confidence)
            else:
                tracked_instances = [
                    inst
                    for inst in lf.instances
                    if type(inst) == PredictedInstance and inst.track is not None
                ]
                for inst in tracked_instances:
                    j = self.tracks.index(inst.track)  # type: ignore[arg-type]
                    tracks[i, j] = inst.numpy(scores=return_confidence)

        return tracks
