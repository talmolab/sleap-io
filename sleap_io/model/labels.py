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
from sleap_io import (
    LabeledFrame,
    Instance,
    PredictedInstance,
    Video,
    Track,
    SuggestionFrame,
)
from attrs import define, field
from typing import Union, Optional, Any
import numpy as np
from pathlib import Path
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
        suggestions: A list of `SuggestionFrame`s that are associated with this dataset.
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
    suggestions: list[SuggestionFrame] = field(factory=list)
    provenance: dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self):
        """Append videos, skeletons, and tracks seen in `labeled_frames` to `Labels`."""
        self.update()

    def update(self):
        """Update data structures based on contents.

        This function will update the list of skeletons, videos and tracks from the
        labeled frames, instances and suggestions.
        """
        for lf in self.labeled_frames:
            if lf.video not in self.videos:
                self.videos.append(lf.video)

            for inst in lf:
                if inst.skeleton not in self.skeletons:
                    self.skeletons.append(inst.skeleton)

                if inst.track is not None and inst.track not in self.tracks:
                    self.tracks.append(inst.track)

        for sf in self.suggestions:
            if sf.video not in self.videos:
                self.videos.append(sf.video)

    def __getitem__(
        self, key: int | slice | list[int] | np.ndarray | tuple[Video, int]
    ) -> list[LabeledFrame] | LabeledFrame:
        """Return one or more labeled frames based on indexing criteria."""
        if type(key) == int:
            return self.labeled_frames[key]
        elif type(key) == slice:
            return [self.labeled_frames[i] for i in range(*key.indices(len(self)))]
        elif type(key) == list:
            return [self.labeled_frames[i] for i in key]
        elif isinstance(key, np.ndarray):
            return [self.labeled_frames[i] for i in key.tolist()]
        elif type(key) == tuple and len(key) == 2:
            video, frame_idx = key
            res = self.find(video, frame_idx)
            if len(res) == 1:
                return res[0]
            elif len(res) == 0:
                raise IndexError(
                    f"No labeled frames found for video {video} and "
                    f"frame index {frame_idx}."
                )
        elif type(key) == Video:
            res = self.find(key)
            if len(res) == 0:
                raise IndexError(f"No labeled frames found for video {key}.")
            return res
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
            f"tracks={len(self.tracks)}, "
            f"suggestions={len(self.suggestions)}"
            ")"
        )

    def __str__(self) -> str:
        """Return a readable representation of the labels."""
        return self.__repr__()

    def append(self, lf: LabeledFrame, update: bool = True):
        """Append a labeled frame to the labels.

        Args:
            lf: A labeled frame to add to the labels.
            update: If `True` (the default), update list of videos, tracks and
                skeletons from the contents.
        """
        self.labeled_frames.append(lf)

        if update:
            if lf.video not in self.videos:
                self.videos.append(lf.video)

            for inst in lf:
                if inst.skeleton not in self.skeletons:
                    self.skeletons.append(inst.skeleton)

                if inst.track is not None and inst.track not in self.tracks:
                    self.tracks.append(inst.track)

    def extend(self, lfs: list[LabeledFrame], update: bool = True):
        """Append a labeled frame to the labels.

        Args:
            lfs: A list of labeled frames to add to the labels.
            update: If `True` (the default), update list of videos, tracks and
                skeletons from the contents.
        """
        self.labeled_frames.extend(lfs)

        if update:
            for lf in lfs:
                if lf.video not in self.videos:
                    self.videos.append(lf.video)

                for inst in lf:
                    if inst.skeleton not in self.skeletons:
                        self.skeletons.append(inst.skeleton)

                    if inst.track is not None and inst.track not in self.tracks:
                        self.tracks.append(inst.track)

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

    @property
    def video(self) -> Video:
        """Return the video if there is only a single video in the labels."""
        if len(self.videos) == 0:
            raise ValueError("There are no videos in the labels.")
        elif len(self.videos) == 1:
            return self.videos[0]
        else:
            raise ValueError(
                "Labels.video can only be used when there is only a single video saved "
                "in the labels. Use Labels.videos instead."
            )

    @property
    def skeleton(self) -> Skeleton:
        """Return the skeleton if there is only a single skeleton in the labels."""
        if len(self.skeletons) == 0:
            raise ValueError("There are no skeletons in the labels.")
        elif len(self.skeletons) == 1:
            return self.skeletons[0]
        else:
            raise ValueError(
                "Labels.skeleton can only be used when there is only a single skeleton "
                "saved in the labels. Use Labels.skeletons instead."
            )

    def find(
        self,
        video: Video,
        frame_idx: int | list[int] | None = None,
        return_new: bool = False,
    ) -> list[LabeledFrame]:
        """Search for labeled frames given video and/or frame index.

        Args:
            video: A `Video` that is associated with the project.
            frame_idx: The frame index (or indices) which we want to find in the video.
                If a range is specified, we'll return all frames with indices in that
                range. If not specific, then we'll return all labeled frames for video.
            return_new: Whether to return singleton of new and empty `LabeledFrame` if
                none are found in project.

        Returns:
            List of `LabeledFrame` objects that match the criteria.

            The list will be empty if no matches found, unless return_new is True, in
            which case it contains new (empty) `LabeledFrame` objects with `video` and
            `frame_index` set.
        """
        results = []

        if frame_idx is None:
            for lf in self.labeled_frames:
                if lf.video == video:
                    results.append(lf)
            return results

        if np.isscalar(frame_idx):
            frame_idx = np.array(frame_idx).reshape(-1)

        for frame_ind in frame_idx:
            result = None
            for lf in self.labeled_frames:
                if lf.video == video and lf.frame_idx == frame_ind:
                    result = lf
                    results.append(result)
                    break
            if result is None and return_new:
                results.append(LabeledFrame(video=video, frame_idx=frame_ind))

        return results

    def save(
        self,
        filename: str,
        format: Optional[str] = None,
        embed: str | list[tuple[Video, int]] | None = None,
        **kwargs,
    ):
        """Save labels to file in specified format.

        Args:
            filename: Path to save labels to.
            format: The format to save the labels in. If `None`, the format will be
                inferred from the file extension. Available formats are "slp", "nwb",
                "labelstudio", and "jabs".
            embed: One of `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or
                list of tuples of `(video, frame_idx)` specifying the frames to embed.
                If `"source"` is specified, no images will be embedded and the source
                video will be restored if available. This argument is only valid for the
                SLP backend.
        """
        from sleap_io import save_file

        save_file(self, filename, format=format, embed=embed, **kwargs)

    def clean(
        self,
        frames: bool = True,
        empty_instances: bool = False,
        skeletons: bool = True,
        tracks: bool = True,
        videos: bool = False,
    ):
        """Remove empty frames, unused skeletons, tracks and videos.

        Args:
            frames: If `True` (the default), remove empty frames.
            empty_instances: If `True` (NOT default), remove instances that have no
                visible points.
            skeletons: If `True` (the default), remove unused skeletons.
            tracks: If `True` (the default), remove unused tracks.
            videos: If `True` (NOT default), remove videos that have no labeled frames.
        """
        used_skeletons = []
        used_tracks = []
        used_videos = []
        kept_frames = []
        for lf in self.labeled_frames:

            if empty_instances:
                lf.remove_empty_instances()

            if frames and len(lf) == 0:
                continue

            if videos and lf.video not in used_videos:
                used_videos.append(lf.video)

            if skeletons or tracks:
                for inst in lf:
                    if skeletons and inst.skeleton not in used_skeletons:
                        used_skeletons.append(inst.skeleton)
                    if (
                        tracks
                        and inst.track is not None
                        and inst.track not in used_tracks
                    ):
                        used_tracks.append(inst.track)

            if frames:
                kept_frames.append(lf)

        if videos:
            self.videos = [video for video in self.videos if video in used_videos]

        if skeletons:
            self.skeletons = [
                skeleton for skeleton in self.skeletons if skeleton in used_skeletons
            ]

        if tracks:
            self.tracks = [track for track in self.tracks if track in used_tracks]

        if frames:
            self.labeled_frames = kept_frames

    def remove_predictions(self, clean: bool = True):
        """Remove all predicted instances from the labels.

        Args:
            clean: If `True` (the default), also remove any empty frames and unused
                tracks and skeletons. It does NOT remove videos that have no labeled
                frames or instances with no visible points.

        See also: `Labels.clean`
        """
        for lf in self.labeled_frames:
            lf.remove_predictions()

        if clean:
            self.clean(
                frames=True,
                empty_instances=False,
                skeletons=True,
                tracks=True,
                videos=False,
            )

    @property
    def user_labeled_frames(self) -> list[LabeledFrame]:
        """Return all labeled frames with user (non-predicted) instances."""
        return [lf for lf in self.labeled_frames if lf.has_user_instances]

    def replace_videos(
        self,
        old_videos: list[Video] | None = None,
        new_videos: list[Video] | None = None,
        video_map: dict[Video, Video] | None = None,
    ):
        """Replace videos and update all references.

        Args:
            old_videos: List of videos to be replaced.
            new_videos: List of videos to replace with.
            video_map: Alternative input of dictionary where keys are the old videos and
                values are the new videos.
        """
        if video_map is None:
            video_map = {o: n for o, n in zip(old_videos, new_videos)}

        # Update the labeled frames with the new videos.
        for lf in self.labeled_frames:
            if lf.video in video_map:
                lf.video = video_map[lf.video]

        # Update suggestions with the new videos.
        for sf in self.suggestions:
            if sf.video in video_map:
                sf.video = video_map[sf.video]

    def replace_filenames(
        self,
        new_filenames: list[str | Path] | None = None,
        filename_map: dict[str | Path, str | Path] | None = None,
        prefix_map: dict[str | Path, str | Path] | None = None,
    ):
        """Replace video filenames.

        Args:
            new_filenames: List of new filenames. Must have the same length as the
                number of videos in the labels.
            filename_map: Dictionary mapping old filenames (keys) to new filenames
                (values).
            prefix_map: Dictonary mapping old prefixes (keys) to new prefixes (values).

        Notes:
            Only one of the argument types can be provided.
        """
        n = 0
        if new_filenames is not None:
            n += 1
        if filename_map is not None:
            n += 1
        if prefix_map is not None:
            n += 1
        if n != 1:
            raise ValueError(
                "Exactly one input method must be provided to replace filenames."
            )

        if new_filenames is not None:
            if len(self.videos) != len(new_filenames):
                raise ValueError(
                    f"Number of new filenames ({len(new_filenames)}) does not match "
                    f"the number of videos ({len(self.videos)})."
                )

            for video, new_filename in zip(self.videos, new_filenames):
                video.replace_filename(new_filename)

        elif filename_map is not None:
            for video in self.videos:
                for old_fn, new_fn in filename_map.items():
                    if type(video.filename) == list:
                        new_fns = []
                        for fn in video.filename:
                            if Path(fn) == Path(old_fn):
                                new_fns.append(new_fn)
                            else:
                                new_fns.append(fn)
                        video.replace_filename(new_fns)
                    else:
                        if Path(video.filename) == Path(old_fn):
                            video.replace_filename(new_fn)

        elif prefix_map is not None:
            for video in self.videos:
                for old_prefix, new_prefix in prefix_map.items():
                    old_prefix, new_prefix = Path(old_prefix), Path(new_prefix)

                    if type(video.filename) == list:
                        new_fns = []
                        for fn in video.filename:
                            fn = Path(fn)
                            if fn.as_posix().startswith(old_prefix.as_posix()):
                                new_fns.append(new_prefix / fn.relative_to(old_prefix))
                            else:
                                new_fns.append(fn)
                        video.replace_filename(new_fns)
                    else:
                        fn = Path(video.filename)
                        if fn.as_posix().startswith(old_prefix.as_posix()):
                            video.replace_filename(
                                new_prefix / fn.relative_to(old_prefix)
                            )
