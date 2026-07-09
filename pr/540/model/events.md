# Events

An [`Event`][sleap_io.Event] is the first sleap-io annotation with a **temporal extent**.
Every other annotation type — [`Instance`](poses.md), [`Centroid`](centroids.md),
[`BoundingBox`](boxes.md), [`SegmentationMask`](segmentation.md), `LabelImage`,
[`ROI`](rois.md) — is strictly *per-frame* and lives on a single
[`LabeledFrame`](labels.md). An `Event` instead spans a range of frames: a
`(video, start_frame, end_frame, type)` interval. It models anything with a duration —
behavior bouts, stimulus epochs, physiological events, feeding bouts, review flags.

A behavior bout is just one kind of event, so the vocabulary is deliberately generic
([`EventType`][sleap_io.EventType]) rather than behavior-specific.

`Event` is abstract — use [`UserEvent`][sleap_io.UserEvent] (human-annotated) or
[`PredictedEvent`][sleap_io.PredictedEvent] (model output).

## Event types (the catalog)

An [`EventType`][sleap_io.EventType] is a lightweight controlled-vocabulary entry — the
"ethogram" for behavior, or any labeled span more generally. Like [`Track`](poses.md)
and [`Identity`](embedding.md) it is name-matched across separately-loaded files and
merges, and carries free-form string metadata (e.g. a UI color):

```pycon
>>> import sleap_io as sio
>>> attack = sio.EventType(
...     name="attack",
...     description="One mouse attacks another.",
...     metadata={"color": "#e6194b"},
... )
>>> print(attack.name)
attack
>>> print(attack.matches(sio.EventType(name="attack")))  # matched by name
True

```

## Frame convention (inclusive)

`start_frame` and `end_frame` are **both inclusive**: the event covers every frame in
`[start_frame, end_frame]` and spans `end_frame - start_frame + 1` frames. Omitting
`end_frame` yields an *instantaneous* single-frame event (`end_frame` is filled to
`start_frame`):

```pycon
>>> import sleap_io as sio
>>> video = sio.Video(filename="clip.mp4")
>>> bout = sio.UserEvent(type="attack", video=video, start_frame=100, end_frame=140)
>>> print(bout.n_frames)      # inclusive: 140 - 100 + 1
41
>>> print(bout.contains(140))  # inclusive at both ends
True
>>> print(bout.frames)         # a lazy range aligned to any framewise scores
range(100, 141)
>>> point = sio.UserEvent(type="light_on", video=video, start_frame=200)
>>> print(point.is_instantaneous, point.end_frame)
True 200

```

A string passed as `type` is auto-promoted to `EventType(name=...)`, so you do not have
to build the catalog entry by hand for quick construction.

## Participants: subject and target

Each event optionally names up to two participants, each a [`Track`](poses.md) (a
within-video trajectory) or an [`Identity`](embedding.md) (a cross-video animal):

- `subject` — who the event is *about*. `None` means a frame-level event with no
  individual (e.g. a stimulus epoch).
- `target` — who the event is *directed at*. `None` means the event is non-directed /
  individual (`"self"` in behavior-scoring terms).

`is_directed` reports whether a `target` is set:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video(filename="clip.mp4")
>>> mouse1, mouse2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")
>>> attack = sio.UserEvent(
...     type="attack", video=video, start_frame=100, end_frame=140,
...     subject=mouse1, target=mouse2,
... )
>>> print(attack.is_directed)
True
>>> rear = sio.UserEvent(type="rear", video=video, start_frame=50, subject=mouse1)
>>> print(rear.is_directed)  # target is None -> non-directed ("self")
False

```

## Overlapping events

Events are a flat *set of intervals*, not a per-frame partition: they may overlap in
time, even for the same subject. [`overlaps`][sleap_io.Event.overlaps] tests whether two
events share a video and their inclusive spans intersect (events in different videos
never overlap):

```pycon
>>> import sleap_io as sio
>>> video = sio.Video(filename="clip.mp4")
>>> a = sio.UserEvent(type="rear", video=video, start_frame=0, end_frame=10)
>>> b = sio.UserEvent(type="freeze", video=video, start_frame=8, end_frame=20)
>>> print(a.overlaps(b))
True

```

## User vs. predicted events

[`PredictedEvent`][sleap_io.PredictedEvent] adds two **independent, optional** confidence
fields. A predictor sets whichever it produces (a framewise trace, an event-level scalar,
both, or neither); neither is derived from the other:

- `scores` — a framewise confidence trace of shape `(n_frames,)`, stored as `float32`
  and aligned element-wise to `frames`.
- `score` — a scalar event-level confidence.

```pycon
>>> import sleap_io as sio
>>> video = sio.Video(filename="clip.mp4")
>>> pred = sio.PredictedEvent(
...     type="attack", video=video, start_frame=100, end_frame=104,
...     scores=[0.6, 0.8, 0.9, 0.7, 0.5],  # per-frame, len == n_frames
...     score=0.74,                         # event-level scalar
... )
>>> print(pred.is_predicted, pred.scores.dtype, len(pred.scores))
True float32 5
>>> user = sio.UserEvent(type="attack", video=video, start_frame=100, end_frame=104)
>>> print(user.is_predicted)
False

```

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `type` | [`EventType`](#event-types-the-catalog) | Catalog entry; a bare string is auto-promoted |
| `video` | [`Video`](video.md) | The video the event occurs in |
| `start_frame` | `int` | First frame (inclusive) |
| `end_frame` | `int \| None` | Last frame (inclusive); defaults to `start_frame` |
| `subject` | [`Track`](poses.md) `\|` [`Identity`](embedding.md) `\| None` | Who the event is about |
| `target` | [`Track`](poses.md) `\|` [`Identity`](embedding.md) `\| None` | Who the event is directed at (`None` = self) |
| `name` | `str` | Human-readable name for this event instance |
| `source` | `str` | Annotation source identifier |
| `metadata` | `dict[str, str]` | Arbitrary string-keyed metadata |
| `scores` | `np.ndarray \| None` | *(PredictedEvent)* framewise `(n_frames,)` trace, `float32` |
| `score` | `float \| None` | *(PredictedEvent)* scalar event-level confidence |

## On `Labels`

Events and their catalog live on the top-level container as two lists,
[`Labels.events`][sleap_io.Labels] and `Labels.event_types` — siblings of `videos` /
`tracks` / `suggestions`, **not** nested on any [`LabeledFrame`](labels.md) (an event
may cover frames that carry no pose labels). Constructing a `Labels` auto-collects the
catalog (deduped by name) and any `Track` / `Identity` used as a participant, exactly
like tracks are collected from instances:

```pycon
>>> import sleap_io as sio
>>> video = sio.Video(filename="clip.mp4")
>>> mouse1, mouse2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")
>>> labels = sio.Labels(
...     videos=[video],
...     tracks=[mouse1, mouse2],
...     events=[
...         sio.UserEvent(type="attack", video=video, start_frame=100,
...                       end_frame=140, subject=mouse1, target=mouse2),
...         sio.UserEvent(type="rear", video=video, start_frame=120, subject=mouse2),
...     ],
... )
>>> [et.name for et in labels.event_types]   # catalog auto-collected from events
['attack', 'rear']
>>> len(labels.get_events(type="attack"))
1
>>> len(labels.events_at(video, 130))        # events whose span covers frame 130
1

```

[`get_events`][sleap_io.Labels.get_events] filters by `video`, `subject`, `type` (an
`EventType` or a bare name), `frame_idx` (events whose span *covers* that frame), and
`predicted`. [`events_at`][sleap_io.Labels.events_at] is the shorthand for "what is
happening at this frame?". `copy()` and `merge()` carry events across too: merging
concatenates events, dedupes the catalog by name, and rebinds each event's
video / subject / target / type onto the merged catalogs.

## SLP persistence

Events persist to SLP in **format 2.6+** via two additive, presence-guarded HDF5 groups
(older readers ignore them; event-free files are byte-identical):

- `/event_types` — the catalog (a `name` string dataset, an optional `description`, and
  an optional entity-attribute-value `meta_*` table), mirroring `/identity`.
- `/events` — a columnar struct-of-arrays (one dataset per field, like `/bboxes`) plus a
  ragged CSR pair (`scores` flat float32 + `score_offsets`) for the optional framewise
  `PredictedEvent.scores`. Participants are stored as a `(kind, idx)` pair
  (`0`=none/self, `1`=track, `2`=identity). The scalar `score` column and both trace
  datasets are written only when some event uses them, so unused features cost zero
  bytes.

```python
labels.save("behavior.slp")            # writes /event_types + /events (format 2.6)
loaded = sio.load_slp("behavior.slp")  # events + catalog fully reconstructed
loaded.events, loaded.event_types
```

---

!!! note "See also"

    - **[Poses](poses.md)** and **[Embeddings](embedding.md)**: the `Track` and
      `Identity` catalogs that events reference as participants.
    - **[Labels & Frames](labels.md)**: the top-level container holding
      `labels.events` / `labels.event_types` and the `get_events()` / `events_at()`
      queries.

---

## API reference

::: sleap_io.EventType

::: sleap_io.Event

::: sleap_io.UserEvent

::: sleap_io.PredictedEvent
