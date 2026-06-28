# Embeddings

An [`Embedding`][sleap_io.Embedding] is a per-detection appearance / re-identification feature
vector. It is the bridge for **multi-object tracking (MOT)** and **re-ID** workflows: a model
produces one vector per detection crop, and those vectors are compared (typically by cosine
similarity) to link detections to a global [`Identity`](3d.md#identity).

`Embedding` is a small value object — the vector is the only heavy part:

| Field | Type | Description |
| --- | --- | --- |
| `vector` | `np.ndarray` `(D,)` | The feature vector. Floating dtype is preserved (e.g. `float32` from a network, `float64` from JABS prototypes); non-floating input is cast to `float32`. |
| `name` | `str` | Embedding-space name (e.g. `"reid"`, `"jabs"`). Multiple spaces may coexist on one detection. |
| `normalized` | `bool` | Whether the vector is L2-normalized (cosine geometry). Defaults to `True`. |
| `source` | `str \| None` | Optional provenance / model identifier. |
| `centroid_xy` | `np.ndarray` `(2,)` `\| None` | Optional source location the crop was sampled at. |
| `metadata` | `dict` | Arbitrary metadata. |

## Per-modality slots

Every detection modality — [`Instance`](poses.md), [`Centroid`](centroids.md),
[`SegmentationMask`](segmentation.md), [`BoundingBox`](boxes.md), [`ROI`](rois.md) — as well as
[`Identity`](3d.md#identity) carries an `embeddings: dict[str, Embedding]` mapping keyed by space
name, plus a convenience `embedding` accessor and a `set_embedding()` helper:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> inst = sio.Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=skeleton)
>>> _ = inst.set_embedding(np.ones(128, dtype="float32"), name="reid", source="reid_v1")
>>> print(inst.embedding.dim)

```

The `embedding` accessor returns the `"reid"` space if present, otherwise the sole space if there
is exactly one, otherwise `None`. A second named space coexists without conflict:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> inst = sio.Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=skeleton)
>>> _ = inst.set_embedding(np.ones(128, dtype="float32"))           # -> "reid"
>>> _ = inst.set_embedding(np.ones(64, dtype="float64"), name="jabs")
>>> print(sorted(inst.embeddings))

```

## Identity prototypes

An [`Identity`](3d.md#identity) carries gallery / prototype embeddings (e.g. the cluster centroid
of its member instances) in the same `embeddings` mapping. A detection is resolved to a global
identity by comparing its `embedding` against the identity prototypes by cosine similarity.

## SLP persistence

Embeddings persist to SLP in **format 2.6+** via the additive `/embeddings` group, with one
subgroup per named space holding the stacked `vectors` `(n, D)` (dtype preserved per space), the
`owner_type`/`owner_id` join columns, and a per-row `meta_json`. The large float vectors live in
their own gzipped numeric datasets — never in any JSON blob and never in the fixed instance row
layout — so the format stays additive: older readers ignore the group and embedding-free files
round-trip unchanged. Per-instance embeddings are loaded lazily alongside the rest of the lazy
store.

!!! note "Persistence scope"
    Instance and Identity embeddings are written today. Embeddings attached to centroid / mask /
    bounding-box / ROI detections are not yet persisted; saving emits a warning so they are not
    silently dropped.

---

## API reference

::: sleap_io.Embedding

::: sleap_io.model.embedding.EmbeddingMixin
