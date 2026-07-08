"""Tests for the frame-spanning Event data model."""

import numpy as np
import pytest

from sleap_io.model.event import (
    Event,
    EventType,
    PredictedEvent,
    UserEvent,
)
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Track

# --- EventType ---------------------------------------------------------------


def test_event_type_defaults():
    """EventType has empty name/description and an empty metadata dict."""
    et = EventType()
    assert et.name == ""
    assert et.description == ""
    assert et.metadata == {}


def test_event_type_fields():
    """EventType stores name, description, and metadata."""
    et = EventType(
        name="attack",
        description="One mouse attacks another.",
        metadata={"color": "#e6194b"},
    )
    assert et.name == "attack"
    assert et.description == "One mouse attacks another."
    assert et.metadata == {"color": "#e6194b"}


def test_event_type_identity_equality():
    """Two EventTypes with the same name are not equal (object-identity eq)."""
    a = EventType(name="rear")
    b = EventType(name="rear")
    assert a is not b
    assert a != b
    assert a == a


def test_event_type_matches():
    """matches() supports name (default), identity, and rejects unknown methods."""
    a = EventType(name="rear")
    b = EventType(name="rear")
    c = EventType(name="freeze")
    assert a.matches(b)  # default method="name"
    assert a.matches(b, method="name")
    assert not a.matches(c)
    assert a.matches(a, method="identity")
    assert not a.matches(b, method="identity")
    with pytest.raises(ValueError, match="Unknown matching method"):
        a.matches(b, method="bogus")


def test_event_type_repr():
    """EventType has a compact repr showing only the name."""
    assert repr(EventType(name="attack")) == 'EventType(name="attack")'


def test_event_type_validators():
    """EventType validates field types."""
    with pytest.raises(TypeError):
        EventType(name=5)
    with pytest.raises(TypeError):
        EventType(description=5)
    with pytest.raises(TypeError):
        EventType(metadata="notadict")


# --- Event (abstract base) ---------------------------------------------------


def test_event_abstract():
    """Event cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Event is abstract"):
        Event(type="attack", video="v", start_frame=0)


def test_event_type_auto_promotion():
    """A bare-string type is auto-promoted to an EventType(name=...)."""
    e = UserEvent(type="attack", video="v", start_frame=0)
    assert isinstance(e.type, EventType)
    assert e.type.name == "attack"


def test_event_type_object_passthrough():
    """An EventType object passed as type is stored as-is (not copied)."""
    et = EventType(name="rear")
    e = UserEvent(type=et, video="v", start_frame=0)
    assert e.type is et


def test_event_type_invalid():
    """A non-string, non-EventType type is rejected by the validator."""
    with pytest.raises(TypeError):
        UserEvent(type=5, video="v", start_frame=0)


def test_event_defaults():
    """Event defaults: end_frame=start_frame, no participants, empty metadata."""
    e = UserEvent(type="rear", video="v", start_frame=10)
    assert e.end_frame == 10
    assert e.subject is None
    assert e.target is None
    assert e.name == ""
    assert e.source == ""
    assert e.metadata == {}


def test_event_end_frame_fill():
    """Omitting end_frame yields an instantaneous single-frame event."""
    e = UserEvent(type="rear", video="v", start_frame=42)
    assert e.end_frame == 42
    assert e.is_instantaneous
    assert e.n_frames == 1
    assert list(e.frames) == [42]


def test_event_span():
    """A multi-frame span reports the right length, frames, and containment."""
    e = UserEvent(type="rear", video="v", start_frame=5, end_frame=9)
    assert e.end_frame == 9
    assert not e.is_instantaneous
    assert e.n_frames == 5
    assert list(e.frames) == [5, 6, 7, 8, 9]
    assert e.frames == range(5, 10)
    # Inclusive containment at both ends.
    assert e.contains(5)
    assert e.contains(9)
    assert e.contains(7)
    assert not e.contains(4)
    assert not e.contains(10)


def test_event_end_before_start():
    """end_frame < start_frame is rejected."""
    with pytest.raises(ValueError, match="end_frame >= start_frame"):
        UserEvent(type="rear", video="v", start_frame=10, end_frame=3)


def test_event_frame_indices_coerced_to_int():
    """Numpy integer frame indices are coerced to Python ints."""
    e = UserEvent(
        type="rear",
        video="v",
        start_frame=np.int64(5),
        end_frame=np.int64(9),
    )
    assert isinstance(e.start_frame, int)
    assert isinstance(e.end_frame, int)
    assert e.n_frames == 5


def test_event_large_frame_indices():
    """Frame indices well beyond int16/int32 are handled (int64 range)."""
    e = UserEvent(type="rear", video="v", start_frame=587_000, end_frame=587_937)
    assert e.n_frames == 938
    assert e.contains(587_937)


def test_event_directedness():
    """is_directed reflects whether a target is set."""
    m1, m2 = Track(name="mouse1"), Track(name="mouse2")
    directed = UserEvent(
        type="attack", video="v", start_frame=0, end_frame=2, subject=m1, target=m2
    )
    assert directed.is_directed
    assert directed.subject is m1
    assert directed.target is m2

    non_directed = UserEvent(type="rear", video="v", start_frame=0, subject=m1)
    assert not non_directed.is_directed
    assert non_directed.target is None


def test_event_participants_track_or_identity():
    """subject/target accept either a Track or an Identity."""
    track = Track(name="mouse1")
    ident = Identity(name="mouse_A")
    e = UserEvent(type="attack", video="v", start_frame=0, subject=ident, target=track)
    assert e.subject is ident
    assert e.target is track


def test_event_overlaps_same_video():
    """Events in the same video overlap iff their inclusive spans intersect."""
    video = object()
    a = UserEvent(type="x", video=video, start_frame=0, end_frame=10)
    touching = UserEvent(type="x", video=video, start_frame=10, end_frame=20)
    disjoint = UserEvent(type="x", video=video, start_frame=11, end_frame=20)
    contained = UserEvent(type="x", video=video, start_frame=3, end_frame=4)

    assert a.overlaps(touching)  # share frame 10
    assert touching.overlaps(a)  # symmetric
    assert not a.overlaps(disjoint)
    assert a.overlaps(contained)


def test_event_overlaps_different_video():
    """Events in different videos never overlap, even at the same frames."""
    a = UserEvent(type="x", video=object(), start_frame=0, end_frame=10)
    b = UserEvent(type="x", video=object(), start_frame=0, end_frame=10)
    assert not a.overlaps(b)


def test_event_identity_equality():
    """Events use object-identity equality."""
    a = UserEvent(type="x", video="v", start_frame=0, end_frame=5)
    b = UserEvent(type="x", video="v", start_frame=0, end_frame=5)
    assert a is not b
    assert a != b
    assert a == a


def test_event_metadata_and_strings():
    """name/source/metadata are stored and validated."""
    e = UserEvent(
        type="x",
        video="v",
        start_frame=0,
        name="bout_1",
        source="hand_scored",
        metadata={"scorer": "alice"},
    )
    assert e.name == "bout_1"
    assert e.source == "hand_scored"
    assert e.metadata == {"scorer": "alice"}
    with pytest.raises(TypeError):
        UserEvent(type="x", video="v", start_frame=0, name=5)
    with pytest.raises(TypeError):
        UserEvent(type="x", video="v", start_frame=0, metadata="notadict")


# --- UserEvent / PredictedEvent ----------------------------------------------


def test_user_event_not_predicted():
    """UserEvent is not a prediction and carries no score fields."""
    e = UserEvent(type="x", video="v", start_frame=0)
    assert not e.is_predicted
    assert not hasattr(e, "score")
    assert not hasattr(e, "scores")


def test_predicted_event_is_predicted():
    """PredictedEvent reports is_predicted True."""
    e = PredictedEvent(type="x", video="v", start_frame=0)
    assert e.is_predicted


def test_predicted_event_scores_none_by_default():
    """Both score fields default to None and are independent."""
    e = PredictedEvent(type="x", video="v", start_frame=0, end_frame=5)
    assert e.scores is None
    assert e.score is None


def test_predicted_event_framewise_scores():
    """Framewise scores are stored as a float32 array aligned to frames."""
    e = PredictedEvent(
        type="x",
        video="v",
        start_frame=0,
        end_frame=4,
        scores=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    assert isinstance(e.scores, np.ndarray)
    assert e.scores.dtype == np.float32
    assert len(e.scores) == e.n_frames == 5
    np.testing.assert_allclose(e.scores, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)
    # Framewise trace aligns element-wise with the frame range.
    assert len(e.scores) == len(e.frames)


def test_predicted_event_scalar_score():
    """A scalar event-level score is stored independently of any trace."""
    e = PredictedEvent(type="x", video="v", start_frame=0, end_frame=4, score=0.9)
    assert e.score == pytest.approx(0.9)
    assert e.scores is None


def test_predicted_event_both_scores():
    """A predictor may set both the scalar and the framewise scores."""
    e = PredictedEvent(
        type="x",
        video="v",
        start_frame=0,
        end_frame=2,
        scores=[0.5, 0.6, 0.7],
        score=0.6,
    )
    assert e.score == pytest.approx(0.6)
    assert len(e.scores) == 3


def test_predicted_event_scores_length_mismatch():
    """Framewise scores must have length n_frames."""
    with pytest.raises(ValueError, match="length n_frames"):
        PredictedEvent(
            type="x", video="v", start_frame=0, end_frame=4, scores=[0.1, 0.2]
        )


def test_predicted_event_scores_not_1d():
    """Framewise scores must be 1-dimensional."""
    with pytest.raises(ValueError, match="1-dimensional"):
        PredictedEvent(
            type="x",
            video="v",
            start_frame=0,
            end_frame=1,
            scores=[[0.1, 0.2], [0.3, 0.4]],
        )


def test_predicted_event_instantaneous_scores():
    """An instantaneous predicted event accepts a length-1 framewise trace."""
    e = PredictedEvent(type="x", video="v", start_frame=7, scores=[0.8])
    assert e.n_frames == 1
    assert len(e.scores) == 1
    assert e.scores.dtype == np.float32


def test_top_level_exports():
    """Event classes are importable from the package root."""
    import sleap_io as sio

    assert sio.Event is Event
    assert sio.EventType is EventType
    assert sio.UserEvent is UserEvent
    assert sio.PredictedEvent is PredictedEvent
