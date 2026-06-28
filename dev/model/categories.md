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

---

## API reference

::: sleap_io.model.categories.CategoriesMixin
