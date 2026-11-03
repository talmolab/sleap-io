# Categories

A [`Category`][sleap_io.Category] names the **class** an individual belongs to — a group of
detections that share some attribute, typically assigned by a classifier or retrieved via
re-ID (e.g. `"female_fly"`, `"male_fly"`, `"fur_shaved"`, `"mouse"`). It is the third grouping
axis alongside `Track` and `Identity`:

| Concept | Scope | Question it answers |
| --- | --- | --- |
| [`Track`](poses.md) | within one video | *which trajectory* is this over time? (ephemeral) |
| [`Identity`](3d.md#identity) | across videos / sessions | *which specific individual* is this? (persistent) |
| [`Category`](category.md) | across individuals | *which class* does this belong to? (classification / re-ID) |

Where an `Identity` names one specific animal, a `Category` names a *set* of animals that share a
property. Many individuals map to one category.

## The `Category` class

A `Category` has exactly two fields — the same shape as [`Identity`][sleap_io.Identity] and
[`Track`](poses.md):

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Human-readable class name (e.g. `"female_fly"`). Not required to be unique, but `name` is how categories are matched across separately-loaded files and merges. |
| `metadata` | `dict[str, str]` | Arbitrary string-keyed, string-valued metadata (e.g. `{"color": "#e6194b", "supercategory": "insect"}`). Empty by default. |

```pycon
>>> import sleap_io as sio
>>> female = sio.Category(name="female_fly", metadata={"color": "#e6194b"})
>>> male = sio.Category(name="male_fly")
>>> print(female.name)
female_fly

```

Like `Track` and `Identity`, `Category` uses **object-identity equality** (`eq=False`), so two
`Category` objects with the same `name` are distinct objects but still *match* by name — the key
that survives serialization and cross-file merges. Compare with `matches()` (default
`method="name"`); pass `method="identity"` to instead require the same Python object:

```pycon
>>> import sleap_io as sio
>>> a1 = sio.Category(name="female_fly")
>>> a2 = sio.Category(name="female_fly")
>>> print(a1.matches(a2))  # same name -> same class
True
>>> print(a1.matches(a2, method="identity"))  # distinct objects -> no match
False

```

!!! note "No dedicated color"
    There is no `color` field on a `Category`. If a visualization color is desired, store it as a
    conventional metadata entry such as `metadata["color"] = "#e6194b"`; it persists like any other
    metadata key. Coloring *by category* uses the palette index into `Labels.categories` order
    (identical to color-by-identity), not a per-category color.

## Per-detection slots

Every detection modality — [`Instance`](poses.md), [`Centroid`](centroids.md),
[`SegmentationMask`](segmentation.md), [`BoundingBox`](boxes.md), [`ROI`](rois.md) — carries a
trio of category slots, mirroring the identity trio (`identity` / `identity_score` /
`identity_embedding`):

| Slot | Type | Description |
| --- | --- | --- |
| `category` | `Category \| None` | The assigned class. |
| `category_score` | `float \| None` | Classification / assignment confidence. |
| `category_embedding` | [`Embedding`](embedding.md) `\| None` | The appearance vector the class was predicted from. |

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> female = sio.Category(name="female_fly")
>>> inst = sio.Instance.from_numpy(
...     np.array([[0, 1], [2, 3]]),
...     skeleton=skeleton,
...     category=female,
...     category_score=0.97,
...     category_embedding=sio.Embedding(np.ones(64, dtype="float32")),
... )
>>> print(inst.category.name, inst.category_score, inst.category_embedding.dim)
female_fly 0.97 64

```

The trio is propagated when converting between detection modalities (e.g.
`Instance.to_centroid()`, `Centroid.to_bbox()`), exactly like the identity trio.

### Promotion of the legacy `category` string

Older code set a **free-form `category: str`** class label directly on bounding boxes, centroids,
ROIs, and masks (the object-detection class, e.g. `category="mouse"`). That field is now the
first-class `Category` slot, with a `str -> Category` converter so existing call sites keep
working — the empty-string "unset" sentinel maps to `None`:

```pycon
>>> import sleap_io as sio
>>> bbox = sio.UserBoundingBox(x1=0, y1=0, x2=10, y2=10, category="mouse")
>>> print(bbox.category)  # promoted to a Category
Category(name="mouse")
>>> unset = sio.UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
>>> print(unset.category)  # "" / omitted -> None
None

```

## The catalog: `Labels.categories`

[`Labels.categories`][sleap_io.Labels] is the top-level catalog of `Category` objects — a list,
like [`Labels.identities`][sleap_io.Labels] and [`Labels.tracks`][sleap_io.Labels] — auto-collected
in first-seen order from the detections on save:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> female = sio.Category(name="female_fly")
>>> male = sio.Category(name="male_fly")
>>> video = sio.Video(filename="clip.mp4", open_backend=False)
>>> inst_f = sio.Instance.from_numpy(
...     np.array([[0, 1], [2, 3]]), skeleton=skeleton, category=female
... )
>>> inst_m = sio.Instance.from_numpy(
...     np.array([[4, 5], [6, 7]]), skeleton=skeleton, category=male
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst_f, inst_m])
>>> labels = sio.Labels(labeled_frames=[lf], categories=[female, male])
>>> print([c.name for c in labels.categories])
['female_fly', 'male_fly']

```

## Save / load round-trip

The category catalog, the per-detection `category` / `category_score` links, and the
`category_embedding` appearance vectors persist to SLP in **format 2.7+** (additive — older
readers ignore them and category-free files round-trip unchanged):

```pycon
>>> import os
>>> import tempfile
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> female = sio.Category(name="female_fly")
>>> inst = sio.Instance.from_numpy(
...     np.array([[0, 1], [2, 3]]),
...     skeleton=skeleton,
...     category=female,
...     category_embedding=sio.Embedding(np.ones(64, dtype="float32")),
... )
>>> video = sio.Video(filename="clip.mp4", open_backend=False)
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf], categories=[female])
>>> path = os.path.join(tempfile.mkdtemp(), "cats.slp")
>>> sio.save_slp(labels, path, save_embedding_vectors=True)
>>> loaded = sio.load_slp(path)
>>> print([c.name for c in loaded.categories])
['female_fly']
>>> print(loaded[0][0].category.name, loaded[0][0].category_embedding.dim)
female_fly 64

```

Pass `save_slp(..., save_embedding_vectors=False)` to persist the category *links* (which
detection is which class) while skipping the large appearance vectors — the same gate used for
identity embeddings. See [Formats → SLP](../formats/slp.md#categories) and
[Embeddings](embedding.md).

## Merging: deduping the catalog

When merging files, the category catalog is deduped by a [`CategoryMatcher`][sleap_io.Labels.merge]
— by default matching on `name`, so two files that both use `"female_fly"` collapse to a single
catalog entry:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> def make():
...     female = sio.Category(name="female_fly")
...     inst = sio.Instance.from_numpy(
...         np.array([[0, 1], [2, 3]]), skeleton=skeleton, category=female
...     )
...     video = sio.Video(filename="clip.mp4", open_backend=False)
...     lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
...     return sio.Labels(labeled_frames=[lf], categories=[female])
...
>>> base, other = make(), make()
>>> _ = base.merge(other, category="name", frame="keep_both")
>>> print([c.name for c in base.categories])  # same-named categories deduped
['female_fly']

```

Pass `category="identity"` to instead require the same Python object (no name-based dedup). See
[Merging](../merging.md).

## Coloring by category

`render_image` / `render_video` accept `color_by="category"`, which assigns one palette color per
category by its index in `Labels.categories` order (identical plumbing to `color_by="identity"`).
Detections without a category fall back to index 0:

```python
import sleap_io as sio

labels = sio.load_slp("classified.slp")
img = sio.render_image(labels[0], color_by="category")
sio.render_video(labels, "by_category.mp4", color_by="category")
```

From the CLI:

```bash
sio render classified.slp --color-by category -o by_category.mp4
```

See [Rendering](../rendering.md) and the [CLI reference](../cli.md#sio-render).

---

## API reference

::: sleap_io.Category
