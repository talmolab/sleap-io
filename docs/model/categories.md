# Categories

A **category** is a named categorical label attached to a detection (or to an
[`Identity`](3d.md#identity)) along some dimension — for example `"sex"`, `"strain"`, or
`"behavior"`. Categories are how you tag instances with discrete attributes for grouping,
filtering, and balanced sampling.

Categories are stored in a `categories: dict` mapping keyed by **dimension name**. Values are
typically plain strings, but may be any JSON-serializable value (e.g. a list encoding a one-hot
or class-probability distribution):

| Dimension | Example value |
| --- | --- |
| `"sex"` | `"M"` |
| `"strain"` | `"C57BL/6"` |
| `"behavior"` | `"grooming"` |
| `"sex_probs"` | `[0.2, 0.8]` |

!!! note "Plural `categories` vs. scalar `category`"
    This plural `categories` mapping is **distinct** from the singular scalar `category` attribute
    on the geometry primitives ([`Centroid`](centroids.md), [`BoundingBox`](boxes.md),
    [`SegmentationMask`](segmentation.md), [`ROI`](rois.md)), which names the *detector class* of a
    single shape. The two never collide — categories are addressed through the `cat` alias, not
    `category`.

## Accessing categories

[`Instance`](poses.md), [`PredictedInstance`](poses.md), and [`Identity`](3d.md#identity) carry a
`categories` mapping plus a convenience `cat` alias and `set_category()` / `set_categories()`
helpers (mirroring the [`embeddings`](embedding.md) accessor pattern):

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> inst = sio.Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=skeleton)
>>> inst.set_category("sex", "M")                       # single dimension
>>> inst.set_categories({"strain": "C57BL/6", "age": "adult"})  # merge several
>>> inst.cat["behavior"] = "grooming"                   # `cat` is the live mapping
>>> sorted(inst.categories)
['age', 'behavior', 'sex', 'strain']
```

The `cat` property is a short alias for the live `categories` dict, so you can read or assign
entries in place. Assigning to `cat` directly replaces the whole mapping.

## Identity categories

An [`Identity`](3d.md#identity) carries categories at the **entity level** — attributes of the
animal itself rather than of one detection (e.g. the known sex or strain of `"mouse_A"`):

```pycon
>>> import sleap_io as sio
>>> identity = sio.Identity(name="mouse_A")
>>> identity.set_categories({"species": "mouse", "sex": "M"})
>>> identity.categories
{'species': 'mouse', 'sex': 'M'}
```

## Scope

In this iteration, categories are attached to **instances and `Identity`** only. The geometry
primitives keep their singular scalar `category` and do not yet carry the plural mapping; this can
be extended later, exactly like the [embeddings](embedding.md) rollout.

## SLP persistence

Categories persist to SLP in **format 2.7+**. Per-instance categories live in the additive
`/instance_categories` dataset — a 1-D array of `{"instance_id", "categories"}` JSON rows, one per
instance carrying any categories, joined back to instances by the global `instance_id` (the same
id space as the identity/embedding side-tables). Because the mapping is variable-width, each row
is encoded as JSON rather than a fixed structured array. Entity-level `Identity` categories ride
under a reserved `categories` key in `/identities_json`. Both are purely additive: older readers
ignore them, and category-free files round-trip unchanged at `format_id <= 2.6`. Per-instance
categories load lazily alongside the rest of the lazy store. See [Formats → SLP](../formats/slp.md#per-instance-categories).

## Sampling via DataFrames

[`Labels.to_dataframe(format="instances")`](labels.md) emits one exploded `cat.<dim>` column per
category dimension (vector-valued dims explode to `cat.<dim>.<i>`), alongside an `identity` column
(and `identity_score`), so categories drive ordinary pandas/polars filtering and balanced
sampling. Columns are uniform across all rows — a dimension is `NaN`/`null` on instances that lack
it — so sparse categories sample cleanly:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> video = sio.Video(filename="fake.mp4")
>>> frames = []
>>> for i in range(4):
...     inst = sio.Instance.from_numpy(np.zeros((2, 2)), skeleton=skeleton)
...     inst.set_category("sex", "M" if i % 2 else "F")
...     frames.append(sio.LabeledFrame(video=video, frame_idx=i, instances=[inst]))
>>> labels = sio.Labels(frames)
>>> df = labels.to_dataframe(format="instances")
>>> males = df[df["cat.sex"] == "M"]
>>> int((df["cat.sex"] == "M").sum())
2
```

The DataFrame rows are flat (no live `Instance` handle survives the row), so to map sampled rows
back to objects, key off the `video`/`frame_idx` locator columns.

!!! note "Dimension naming"
    Because export flattens each dimension to a `cat.<dim>` column (and vectors to
    `cat.<dim>.<i>`), keep dimension names simple. A name containing `.` or one that collides with a
    skeleton node name can produce overlapping columns. The SLP side-table itself preserves any
    keys faithfully — this only affects the flattened DataFrame view.

---

## API reference

::: sleap_io.model.categories.CategoriesMixin
