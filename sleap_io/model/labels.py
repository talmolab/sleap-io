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
