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
    Skeleton,
    LabeledFrame,
    Instance,
    PredictedInstance,
    Video,
    Track,
    SuggestionFrame,
)
from sleap_io.model.skeleton import NodeOrIndex
from attrs import define, field
from typing import Iterator, Union, Optional, Any
import numpy as np
from pathlib import Path
from copy import deepcopy


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
        embed: bool | str | list[tuple[Video, int]] | None = None,
        **kwargs,
    ):
        """Save labels to file in specified format.

        Args:
            filename: Path to save labels to.
            format: The format to save the labels in. If `None`, the format will be
                inferred from the file extension. Available formats are `"slp"`,
                `"nwb"`, `"labelstudio"`, and `"jabs"`.
            embed: Frames to embed in the saved labels file. One of `None`, `True`,
                `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or
                list of tuples of `(video, frame_idx)`.

                If `None` is specified (the default) and the labels contains embedded
                frames, those embedded frames will be re-saved to the new file.

                If `True` or `"all"`, all labeled frames and suggested frames will be
                embedded.

                If `"source"` is specified, no images will be embedded and the source
                video will be restored if available.

                This argument is only valid for the SLP backend.
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

    @property
    def instances(self) -> Iterator[Instance]:
        """Return an iterator over all instances within all labeled frames."""
        return (instance for lf in self.labeled_frames for instance in lf.instances)

    def rename_nodes(
        self,
        name_map: dict[NodeOrIndex, str] | list[str],
        skeleton: Skeleton | None = None,
    ):
        """Rename nodes in the skeleton.

        Args:
            name_map: A dictionary mapping old node names to new node names. Keys can be
                specified as `Node` objects, integer indices, or string names. Values
                must be specified as string names.

                If a list of strings is provided of the same length as the current
                nodes, the nodes will be renamed to the names in the list in order.
            skeleton: `Skeleton` to update. If `None` (the default), assumes there is
                only one skeleton in the labels and raises `ValueError` otherwise.

        Raises:
            ValueError: If the new node names exist in the skeleton, if the old node
                names are not found in the skeleton, or if there is more than one
                skeleton in the `Labels` but it is not specified.

        Notes:
            This method is recommended over `Skeleton.rename_nodes` as it will update
            all instances in the labels to reflect the new node names.

        Example:
            >>> labels = Labels(skeletons=[Skeleton(["A", "B", "C"])])
            >>> labels.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
            >>> labels.skeleton.node_names
            ["X", "Y", "Z"]
            >>> labels.rename_nodes(["a", "b", "c"])
            >>> labels.skeleton.node_names
            ["a", "b", "c"]
        """
        if skeleton is None:
            if len(self.skeletons) != 1:
                raise ValueError(
                    "Skeleton must be specified when there is more than one skeleton in "
                    "the labels."
                )
            skeleton = self.skeleton

        skeleton.rename_nodes(name_map)

    def remove_nodes(self, nodes: list[NodeOrIndex], skeleton: Skeleton | None = None):
        """Remove nodes from the skeleton.

        Args:
            nodes: A list of node names, indices, or `Node` objects to remove.
            skeleton: `Skeleton` to update. If `None` (the default), assumes there is
                only one skeleton in the labels and raises `ValueError` otherwise.

        Raises:
            ValueError: If the nodes are not found in the skeleton, or if there is more
                than one skeleton in the `Labels` but it is not specified.

        Notes:
            This method should always be used when removing nodes from the skeleton as
            it handles updating the lookup caches necessary for indexing nodes by name,
            and updating instances to reflect the changes made to the skeleton.

            Any edges and symmetries that are connected to the removed nodes will also
            be removed.
        """
        if skeleton is None:
            if len(self.skeletons) != 1:
                raise ValueError(
                    "Skeleton must be specified when there is more than one skeleton "
                    "in the labels."
                )
            skeleton = self.skeleton

        skeleton.remove_nodes(nodes)

        for inst in self.instances:
            if inst.skeleton == skeleton:
                inst.update_skeleton()

    def reorder_nodes(
        self, new_order: list[NodeOrIndex], skeleton: Skeleton | None = None
    ):
        """Reorder nodes in the skeleton.

        Args:
            new_order: A list of node names, indices, or `Node` objects specifying the
                new order of the nodes.
            skeleton: `Skeleton` to update. If `None` (the default), assumes there is
                only one skeleton in the labels and raises `ValueError` otherwise.

        Raises:
            ValueError: If the new order of nodes is not the same length as the current
                nodes, or if there is more than one skeleton in the `Labels` but it is
                not specified.

        Notes:
            This method handles updating the lookup caches necessary for indexing nodes
            by name, as well as updating instances to reflect the changes made to the
            skeleton.
        """
        if skeleton is None:
            if len(self.skeletons) != 1:
                raise ValueError(
                    "Skeleton must be specified when there is more than one skeleton "
                    "in the labels."
                )
            skeleton = self.skeleton

        skeleton.reorder_nodes(new_order)

        for inst in self.instances:
            if inst.skeleton == skeleton:
                inst.update_skeleton()

    def replace_skeleton(
        self,
        new_skeleton: Skeleton,
        old_skeleton: Skeleton | None = None,
        node_map: dict[NodeOrIndex, NodeOrIndex] | None = None,
    ):
        """Replace the skeleton in the labels.

        Args:
            new_skeleton: The new `Skeleton` to replace the old skeleton with.
            old_skeleton: The old `Skeleton` to replace. If `None` (the default),
                assumes there is only one skeleton in the labels and raises `ValueError`
                otherwise.
            node_map: Dictionary mapping nodes in the old skeleton to nodes in the new
                skeleton. Keys and values can be specified as `Node` objects, integer
                indices, or string names. If not provided, only nodes with identical
                names will be mapped. Points associated with unmapped nodes will be
                removed.

        Raises:
            ValueError: If there is more than one skeleton in the `Labels` but it is not
                specified.

        Warning:
            This method will replace the skeleton in all instances in the labels that
            have the old skeleton. **All point data associated with nodes not in the
            `node_map` will be lost.**
        """
        if old_skeleton is None:
            if len(self.skeletons) != 1:
                raise ValueError(
                    "Old skeleton must be specified when there is more than one "
                    "skeleton in the labels."
                )
            old_skeleton = self.skeleton

        if node_map is None:
            node_map = {}
            for old_node in old_skeleton.nodes:
                for new_node in new_skeleton.nodes:
                    if old_node.name == new_node.name:
                        node_map[old_node] = new_node
                        break
        else:
            node_map = {
                old_skeleton.require_node(
                    old, add_missing=False
                ): new_skeleton.require_node(new, add_missing=False)
                for old, new in node_map.items()
            }

        # Make new -> old mapping for nodes for efficiency.
        rev_node_map = {new: old for old, new in node_map.items()}

        # Replace the skeleton in the instances.
        for inst in self.instances:
            if inst.skeleton == old_skeleton:
                inst.replace_skeleton(new_skeleton, rev_node_map=rev_node_map)

        # Replace the skeleton in the labels.
        self.skeletons[self.skeletons.index(old_skeleton)] = new_skeleton

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
        if (
            old_videos is None
            and new_videos is not None
            and len(new_videos) == len(self.videos)
        ):
            old_videos = self.videos

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

        # Update the list of videos.
        self.videos = [video_map.get(video, video) for video in self.videos]

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

    def extract(
        self, inds: list[int] | list[tuple[Video, int]] | np.ndarray, copy: bool = True
    ) -> Labels:
        """Extract a set of frames into a new Labels object.

        Args:
            inds: Indices of labeled frames. Can be specified as a list of array of
                integer indices of labeled frames or tuples of Video and frame indices.
            copy: If `True` (the default), return a copy of the frames and containing
                objects. Otherwise, return a reference to the data.

        Returns:
            A new `Labels` object containing the selected labels.

        Notes:
            This copies the labeled frames and their associated data, including
            skeletons and tracks, and tries to maintain the relative ordering.

            This also copies the provenance and inserts an extra key: `"source_labels"`
            with the path to the current labels, if available.

            It does NOT copy suggested frames.
        """
        lfs = self[inds]

        if copy:
            lfs = deepcopy(lfs)
        labels = Labels(lfs)

        # Try to keep the lists in the same order.
        track_to_ind = {track.name: ind for ind, track in enumerate(self.tracks)}
        labels.tracks = sorted(labels.tracks, key=lambda x: track_to_ind[x.name])

        skel_to_ind = {skel.name: ind for ind, skel in enumerate(self.skeletons)}
        labels.skeletons = sorted(labels.skeletons, key=lambda x: skel_to_ind[x.name])

        labels.provenance = deepcopy(labels.provenance)
        labels.provenance["source_labels"] = self.provenance.get("filename", None)

        return labels

    def split(self, n: int | float, seed: int | None = None) -> tuple[Labels, Labels]:
        """Separate the labels into random splits.

        Args:
            n: Size of the first split. If integer >= 1, assumes that this is the number
                of labeled frames in the first split. If < 1.0, this will be treated as
                a fraction of the total labeled frames.
            seed: Optional integer seed to use for reproducibility.

        Returns:
            A tuple of `split1, split2`.

            If an integer was specified, `len(split1) == n`.

            If a fraction was specified, `len(split1) == int(n * len(labels))`.

            The second split contains the remainder, i.e.,
            `len(split2) == len(labels) - len(split1)`.

            If there are too few frames, a minimum of 1 frame will be kept in the second
            split.

            If there is exactly 1 labeled frame in the labels, the same frame will be
            assigned to both splits.
        """
        n0 = len(self)
        if n0 == 0:
            return self, self
        n1 = n
        if n < 1.0:
            n1 = max(int(n0 * float(n)), 1)
        n2 = max(n0 - n1, 1)
        n1, n2 = int(n1), int(n2)

        rng = np.random.default_rng(seed=seed)
        inds1 = rng.choice(n0, size=(n1,), replace=False)

        if n0 == 1:
            inds2 = np.array([0])
        else:
            inds2 = np.setdiff1d(np.arange(n0), inds1)

        split1 = self.extract(inds1, copy=True)
        split2 = self.extract(inds2, copy=True)

        return split1, split2

    def make_training_splits(
        self,
        n_train: int | float,
        n_val: int | float | None = None,
        n_test: int | float | None = None,
        save_dir: str | Path | None = None,
        seed: int | None = None,
        embed: bool = True,
    ) -> tuple[Labels, Labels] | tuple[Labels, Labels, Labels]:
        """Make splits for training with embedded images.

        Args:
            n_train: Size of the training split as integer or fraction.
            n_val: Size of the validation split as integer or fraction. If `None`,
                this will be inferred based on the values of `n_train` and `n_test`. If
                `n_test` is `None`, this will be the remainder of the data after the
                training split.
            n_test: Size of the testing split as integer or fraction. If `None`, the
                test split will not be saved.
            save_dir: If specified, save splits to SLP files with embedded images.
            seed: Optional integer seed to use for reproducibility.
            embed: If `True` (the default), embed user labeled frame images in the saved
                files, which is useful for portability but can be slow for large
                projects. If `False`, labels are saved with references to the source
                videos files.

        Returns:
            A tuple of `labels_train, labels_val` or
            `labels_train, labels_val, labels_test` if `n_test` was specified.

        Notes:
            Predictions and suggestions will be removed before saving, leaving only
            frames with user labeled data (the source labels are not affected).

            Frames with user labeled data will be embedded in the resulting files.

            If `save_dir` is specified, this will save the randomly sampled splits to:

            - `{save_dir}/train.pkg.slp`
            - `{save_dir}/val.pkg.slp`
            - `{save_dir}/test.pkg.slp` (if `n_test` is specified)

            If `embed` is `False`, the files will be saved without embedded images to:

            - `{save_dir}/train.slp`
            - `{save_dir}/val.slp`
            - `{save_dir}/test.slp` (if `n_test` is specified)

        See also: `Labels.split`
        """
        # Clean up labels.
        labels = deepcopy(self)
        labels.remove_predictions()
        labels.suggestions = []
        labels.clean()

        # Make train split.
        labels_train, labels_rest = labels.split(n_train, seed=seed)

        # Make test split.
        if n_test is not None:
            if n_test < 1:
                n_test = (n_test * len(labels)) / len(labels_rest)
            labels_test, labels_rest = labels_rest.split(n=n_test, seed=seed)

        # Make val split.
        if n_val is not None:
            if n_val < 1:
                n_val = (n_val * len(labels)) / len(labels_rest)
            if isinstance(n_val, float) and n_val == 1.0:
                labels_val = labels_rest
            else:
                labels_val, _ = labels_rest.split(n=n_val, seed=seed)
        else:
            labels_val = labels_rest

        # Update provenance.
        source_labels = self.provenance.get("filename", None)
        labels_train.provenance["source_labels"] = source_labels
        if n_val is not None:
            labels_val.provenance["source_labels"] = source_labels
        if n_test is not None:
            labels_test.provenance["source_labels"] = source_labels

        # Save.
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

            if embed:
                labels_train.save(save_dir / "train.pkg.slp", embed="user")
                labels_val.save(save_dir / "val.pkg.slp", embed="user")
                labels_test.save(save_dir / "test.pkg.slp", embed="user")
            else:
                labels_train.save(save_dir / "train.slp", embed=False)
                labels_val.save(save_dir / "val.slp", embed=False)
                labels_test.save(save_dir / "test.slp", embed=False)

        if n_test is None:
            return labels_train, labels_val
        else:
            return labels_train, labels_val, labels_test

    def trim(
        self,
        save_path: str | Path,
        frame_inds: list[int] | np.ndarray,
        video: Video | int | None = None,
        video_kwargs: dict[str, Any] | None = None,
    ) -> Labels:
        """Trim the labels to a subset of frames and videos accordingly.

        Args:
            save_path: Path to the trimmed labels SLP file. Video will be saved with the
                same base name but with .mp4 extension.
            frame_inds: Frame indices to save. Can be specified as a list or array of
                frame integers.
            video: Video or integer index of the video to trim. Does not need to be
                specified for single-video projects.
            video_kwargs: A dictionary of keyword arguments to provide to
                `sio.save_video` for video compression.

        Returns:
            The resulting labels object referencing the trimmed data.

        Notes:
            This will remove any data outside of the trimmed frames, save new videos,
            and adjust the frame indices to match the newly trimmed videos.
        """
        if video is None:
            if len(self.videos) == 1:
                video = self.video
            else:
                raise ValueError(
                    "Video needs to be specified when trimming multi-video projects."
                )
        if type(video) == int:
            video = self.videos[video]

        # Write trimmed clip.
        save_path = Path(save_path)
        video_path = save_path.with_suffix(".mp4")
        fidx0, fidx1 = np.min(frame_inds), np.max(frame_inds)
        new_video = video.save(
            video_path,
            frame_inds=np.arange(fidx0, fidx1 + 1),
            video_kwargs=video_kwargs,
        )

        # Get frames in range.
        # TODO: Create an optimized search function for this access pattern.
        inds = []
        for ind, lf in enumerate(self):
            if lf.video == video and lf.frame_idx >= fidx0 and lf.frame_idx <= fidx1:
                inds.append(ind)
        trimmed_labels = self.extract(inds, copy=True)

        # Adjust video and frame indices.
        trimmed_labels.videos = [new_video]
        for lf in trimmed_labels:
            lf.video = new_video
            lf.frame_idx = lf.frame_idx - fidx0

        # Save.
        trimmed_labels.save(save_path)

        return trimmed_labels
