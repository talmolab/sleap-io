"""Data structures for frame-spanning event annotations.

Unlike every other annotation in sleap-io (`Instance`, `Centroid`, `BoundingBox`,
`SegmentationMask`, `LabelImage`, `ROI`), which are strictly per-frame and live on a
single `LabeledFrame`, an `Event` is the first annotation with a *temporal extent*: a
``(video, start_frame, end_frame, type)`` interval. Events model anything with a
duration -- behavior bouts, stimulus epochs, physiological events, feeding bouts,
review flags -- and are stored on ``Labels.events`` rather than on any one frame.

The class hierarchy mirrors the detection-modality pattern:
    - `EventType` -- a catalog / controlled-vocabulary entry (the "ethogram"),
      lightweight and name-matched like `Track` / `Identity`.
    - `Event` -- abstract base carrying the interval, participants, and metadata.
    - `UserEvent` -- a human-annotated event (ground truth).
    - `PredictedEvent` -- a model-predicted event with optional confidence score(s).

Frame convention:
    ``start_frame`` and ``end_frame`` are both **inclusive** -- an event covers every
    frame in ``[start_frame, end_frame]`` and spans ``end_frame - start_frame + 1``
    frames. An event with ``end_frame == start_frame`` (or ``end_frame=None``, which is
    filled to ``start_frame``) is *instantaneous* (a single frame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field
from attrs.validators import instance_of

if TYPE_CHECKING:
    from sleap_io.model.identity import Identity
    from sleap_io.model.instance import Track
    from sleap_io.model.video import Video


@define(eq=False)
class EventType:
    """A type of event in the catalog (controlled vocabulary).

    A lightweight catalog object like `Track` / `Identity`: it is referenced by
    `Event`s and matched across separately-loaded files and merges by ``name``.
    Examples: a behavior (``"attack"``, ``"rear"``), a stimulus (``"light_on"``), a
    physiological event (``"seizure"``), or any labeled span.

    Attributes:
        name: Human-readable name for this event type (e.g. ``"attack"``). Not
            required to be unique, but ``name`` is how event types are matched across
            files and merges.
        description: Optional longer human-readable description of what this event
            type represents. Empty by default.
        metadata: Arbitrary string-keyed, string-valued metadata (e.g.
            ``{"color": "#e6194b", "directed": "true"}``). Empty by default.

    Notes:
        `EventType` objects use object-identity equality (``eq=False``), matching
        `Track` / `Identity`. Use `matches()` (default ``method="name"``) to compare
        event types across files, where Python object identity is not meaningful.
    """

    name: str = field(default="", validator=instance_of(str))
    description: str = field(default="", validator=instance_of(str))
    metadata: dict[str, str] = field(factory=dict, validator=instance_of(dict))

    def matches(self, other: "EventType", method: str = "name") -> bool:
        """Check if this event type matches another event type.

        Args:
            other: Another event type to compare with.
            method: Matching method:

                - ``"name"`` (default): match by the `name` attribute, which
                  survives serialization and cross-file merges.
                - ``"identity"``: match by Python object identity (same object).

        Returns:
            True if the event types match according to the specified method.

        Raises:
            ValueError: If `method` is not one of the supported values.
        """
        if method == "name":
            return self.name == other.name
        elif method == "identity":
            return self is other
        else:
            raise ValueError(f"Unknown matching method: {method}")

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return f'EventType(name="{self.name}")'


def _as_event_type(value: "EventType | str") -> "EventType":
    """Coerce a bare string into an `EventType`, passing `EventType`s through.

    A string ``"attack"`` is auto-promoted to ``EventType(name="attack")`` so
    ``UserEvent(type="attack", ...)`` works without constructing the catalog entry by
    hand. `Labels` collection later dedupes these by name onto a canonical catalog.

    Args:
        value: An `EventType` or a bare string name.

    Returns:
        An `EventType`.
    """
    if isinstance(value, str):
        return EventType(name=value)
    return value


def _as_scores(value: "np.ndarray | None") -> "np.ndarray | None":
    """Coerce a framewise score trace to a 1-D ``float32`` array (or ``None``).

    Args:
        value: A 1-D array-like of per-frame scores, or ``None`` for no trace.

    Returns:
        A 1-D ``float32`` ``np.ndarray``, or ``None`` if ``value`` is ``None``.

    Raises:
        ValueError: If ``value`` is not 1-dimensional.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(
            f"Framewise event scores must be 1-dimensional, got shape {arr.shape}."
        )
    return arr


@define(eq=False)
class Event:
    """A labeled event spanning a range of frames in a video.

    A frame-spanning annotation for anything with a temporal extent: behavior bouts,
    stimulus epochs, physiological events, review flags, etc. Unlike per-frame
    annotations, events live on ``Labels.events`` (not on a single `LabeledFrame`),
    since an event may cover frames that carry no pose labels.

    The interval is **inclusive** on both ends: the event covers every frame in
    ``[start_frame, end_frame]``. ``end_frame`` defaults to ``start_frame`` (an
    instantaneous, single-frame event) when not given.

    Participants are optional and each may be a `Track` (a within-video trajectory) or
    an `Identity` (a cross-video animal):

    - ``subject`` -- who the event is *about*. ``None`` means a frame-level event with
      no individual (e.g. a stimulus epoch).
    - ``target`` -- who the event is *directed at*. ``None`` means the event is
      non-directed / individual (``"self"`` in behavior-scoring terms).

    Attributes:
        type: The `EventType` catalog entry for this event. A bare string is
            auto-promoted to ``EventType(name=...)``.
        video: The `Video` the event occurs in.
        start_frame: First frame of the event (inclusive).
        end_frame: Last frame of the event (inclusive). Defaults to ``start_frame``
            (an instantaneous event) when ``None``. Must be ``>= start_frame``.
        subject: Optional `Track` or `Identity` the event is about. ``None`` means a
            frame-level event with no individual.
        target: Optional `Track` or `Identity` the event is directed at. ``None`` means
            a non-directed / individual event (``"self"``).
        name: Optional human-readable name for this specific event instance.
        source: Annotation source identifier.
        metadata: Arbitrary string-keyed, string-valued metadata. Empty by default.

    Notes:
        Events use object-identity equality (two `Event` objects are only equal if they
        are the same object in memory).

        This class is abstract. Use `UserEvent` or `PredictedEvent` instead.
    """

    type: "EventType" = field(
        converter=_as_event_type, validator=instance_of(EventType)
    )
    video: "Video" = field()
    start_frame: int = field(converter=int)
    end_frame: "int | None" = field(default=None)
    subject: "Track | Identity | None" = field(default=None)
    target: "Track | Identity | None" = field(default=None)
    name: str = field(default="", validator=instance_of(str))
    source: str = field(default="", validator=instance_of(str))
    metadata: dict[str, str] = field(factory=dict, validator=instance_of(dict))

    def __attrs_post_init__(self):
        """Guard abstract instantiation, fill ``end_frame``, and validate the span."""
        if type(self) is Event:
            raise TypeError("Event is abstract. Use UserEvent or PredictedEvent.")
        if self.end_frame is None:
            self.end_frame = self.start_frame
        else:
            self.end_frame = int(self.end_frame)
        if self.end_frame < self.start_frame:
            raise ValueError(
                "Expected end_frame >= start_frame, got "
                f"start_frame={self.start_frame}, end_frame={self.end_frame}."
            )

    @property
    def is_predicted(self) -> bool:
        """Whether this event is a prediction."""
        return isinstance(self, PredictedEvent)

    @property
    def is_directed(self) -> bool:
        """Whether this event is directed at a `target` (vs. non-directed / self)."""
        return self.target is not None

    @property
    def is_instantaneous(self) -> bool:
        """Whether this event spans a single frame (``start_frame == end_frame``)."""
        return self.end_frame == self.start_frame

    @property
    def n_frames(self) -> int:
        """Number of frames spanned (inclusive: ``end_frame - start_frame + 1``)."""
        return self.end_frame - self.start_frame + 1

    @property
    def frames(self) -> range:
        """The inclusive range of frame indices covered by this event.

        Returns:
            A ``range(start_frame, end_frame + 1)``. Its length equals `n_frames` and,
            for a `PredictedEvent`, aligns element-wise with a framewise `scores` trace.
            Returned lazily as a ``range`` (not a materialized array) so events spanning
            hundreds of thousands of frames stay cheap.
        """
        return range(self.start_frame, self.end_frame + 1)

    def contains(self, frame_idx: int) -> bool:
        """Whether a frame index falls within this event's inclusive span.

        Args:
            frame_idx: A frame index to test.

        Returns:
            True if ``start_frame <= frame_idx <= end_frame``.
        """
        return self.start_frame <= frame_idx <= self.end_frame

    def overlaps(self, other: "Event") -> bool:
        """Whether this event overlaps another in the same video.

        Two events overlap when they occur in the same `Video` (compared by object
        identity) and their inclusive frame spans intersect. Events in different videos
        never overlap, regardless of their frame indices.

        Args:
            other: Another event to test against.

        Returns:
            True if both events share a video and their spans intersect.
        """
        if self.video is not other.video:
            return False
        return (
            self.start_frame <= other.end_frame and other.start_frame <= self.end_frame
        )


@define(eq=False)
class UserEvent(Event):
    """A human-annotated event (ground truth).

    Inherits all fields from `Event`. Has no additional fields.

    See `Event` for attribute documentation.
    """

    pass


@define(eq=False)
class PredictedEvent(Event):
    """A model-predicted event.

    Adds two independent, optional confidence fields. A predictor sets whichever it
    produces (a framewise trace, an event-level scalar, both, or neither); neither is
    derived from the other.

    Attributes:
        scores: Optional framewise confidence trace of shape ``(n_frames,)``, aligned
            element-wise to `frames` (i.e. ``[start_frame, end_frame]``). Stored as
            ``float32``. Validated to have length `n_frames` when set. ``None`` if the
            predictor produced no per-frame trace.
        score: Optional scalar event-level confidence. ``None`` if unset. **Not**
            derived from `scores`.

    See `Event` for other attribute documentation.
    """

    scores: "np.ndarray | None" = field(default=None, converter=_as_scores, repr=False)
    score: "float | None" = field(default=None)

    def __attrs_post_init__(self):
        """Validate the framewise `scores` length against `n_frames`."""
        super().__attrs_post_init__()
        if self.scores is not None and len(self.scores) != self.n_frames:
            raise ValueError(
                "Framewise event scores must have length n_frames "
                f"({self.n_frames}), got {len(self.scores)}."
            )
