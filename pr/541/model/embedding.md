# Embeddings

An [`Embedding`][sleap_io.Embedding] is a per-detection appearance / re-identification feature
vector. It is the bridge for **multi-object tracking** and **re-ID** workflows: a model produces
one vector per detection crop, and those vectors are compared (typically by cosine similarity) to
link detections to a global [`Identity`](3d.md#identity).

`Embedding` is a bare value object — the vector is its only field:

| Field | Type | Description |
| --- | --- | --- |
| `vector` | `np.ndarray` `(D,)` | The feature vector. Floating dtype is preserved (e.g. `float32` from a network); non-floating input is cast to `float32`. |

`Embedding` compares by **value** (two embeddings are equal when their vectors are element-wise
equal) and is therefore unhashable. It exposes a `dim` property for the vector length. What the
vector *represents* is implied by the slot it fills on the detection.

## Per-detection slot

Every detection modality — [`Instance`](poses.md), [`Centroid`](centroids.md),
[`SegmentationMask`](segmentation.md), [`BoundingBox`](boxes.md), [`ROI`](rois.md) — carries a
single `identity_embedding: Embedding | None` slot (the appearance vector used for
re-identification):

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> from sleap_io import Embedding
>>> skeleton = sio.Skeleton(["head", "tail"])
>>> inst = sio.Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=skeleton)
>>> inst.identity_embedding = Embedding(np.ones(128, dtype="float32"))
>>> print(inst.identity_embedding.dim)
128

```

Because detection objects use object-identity equality, the (potentially large) embedding vector
is never compared when two detections are compared. The `identity_embedding` is also propagated
when converting between detection modalities (e.g. `Instance.to_centroid()`).

## SLP persistence

Embeddings persist to SLP in **format 2.5+** via the additive `/embeddings` group, stored as a
single columnar struct-of-arrays:

```
/embeddings/
  vectors     (N, D) float32   # chunked so whole rows stay within a chunk, gzip compressed
  owner_type  (N,)   uint8      # OWNER_INSTANCE / CENTROID / MASK / BBOX / ROI
  owner_id    (N,)   int64      # global instance_id or per-modality list index
```

Row `i` of all three datasets describes the same detection. Keeping `owner_type`/`owner_id` as
separate parallel datasets means more per-embedding attributes can be added later without
re-laying-out `vectors`. The large float vectors live in their own chunked, gzipped dataset —
never in a JSON blob or the fixed instance row layout — so the format stays additive: older
readers ignore the group and embedding-free files round-trip unchanged.

All embedding vectors in a file must share the same dimensionality `D` (they come from one re-ID
model); a mixed-`D` save raises `ValueError`.

!!! tip "Skipping appearance vectors on disk"
    Appearance vectors are large. Pass `labels.save(path, save_embedding_vectors=False)` to skip
    the `/embeddings` group entirely while still persisting identity *links* (the `/identity`
    group); the vectors stay in memory (e.g. to build identity prototypes). This is distinct from
    `embed`, which embeds *video frames*.

---

## API reference

::: sleap_io.Embedding
