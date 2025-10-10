"""Data module for suggestions."""

from __future__ import annotations

import attrs

from sleap_io.model.video import Video


@attrs.define(auto_attribs=True)
class SuggestionFrame:
    """Data structure for a single frame of suggestions.

    Attributes:
        video: The video associated with the frame.
        frame_idx: The index of the frame in the video.
        metadata: Dictionary containing additional metadata that is not explicitly
            represented in the data model. This is used to store arbitrary metadata
            such as the "group" key when reading/writing SLP files.
    """

    video: Video
    frame_idx: int
    metadata: dict[str, any] = attrs.field(factory=dict)
