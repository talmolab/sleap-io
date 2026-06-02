# Virtual cropping

sleap-io can expose a **virtual, on-read crop** of a video — a cropped view whose
frames are produced by decoding the source and slicing in memory, without copying or
re-encoding any pixels on disk. It is the lazy, non-destructive counterpart of the
materializing [Transforms](transforms.md) pipeline: a virtually-cropped frame is
byte-identical to what baking a `Transform(crop=...)` would write.

---

## Quick start

```python
import sleap_io as sio

full = sio.load_video("session.mp4")              # (1000, 1080, 1920, 3)

# A cropped view. crop = (x1, y1, x2, y2), with x2/y2 EXCLUSIVE.
view = full.crop((320, 200, 576, 456))
view.shape            # (1000, 256, 256, 3)  -- cropped
view[0].shape         # (256, 256, 3)        -- a cropped frame
view.crop             # not a thing; use view._crop_tuple() -> (320, 200, 576, 456)
view.source_video is full   # True  -- provenance to the uncropped original
```

`Video.from_crop` opens a file and crops it in one call:

```python
view = sio.Video.from_crop("session.mp4", crop=(320, 200, 576, 456))
```

The returned object is a normal [`Video`](model/video.md): `shape`, `len()`, `grayscale`,
NumPy-style indexing, and matching all report the **cropped** view.

---

## The crop convention

A crop is `(x1, y1, x2, y2)` in **source pixel coordinates**, with `x2`/`y2`
**exclusive** — exactly the convention used by [`Transform`](transforms.md) and
`crop_frame`. The cropped size is `(y2 - y1, x2 - x1)`.

Coordinates may be **negative or extend past the source** — out-of-bounds regions are
**padded** with `fill` (default `0`), never clamped, so the output shape is always
exactly `(y2 - y1, x2 - x1)`. This makes fixed-size, centroid-following windows easy:

```python
# Fixed 128x128 window centered on a point (may run off the frame edge -> padded).
view = full.crop(center=(cx, cy), size=(128, 128), fill=0)
view.shape            # (n_frames, 128, 128, 3)
```

`Video.crop` accepts one region spec — an explicit `crop` rect, a `bbox=(x1,y1,x2,y2)`,
an `roi` (anything exposing shapely-style `.bounds`, expanded by `margin`), or a
`center`/`size` pair:

```python
full.crop((x1, y1, x2, y2))                  # explicit rect
full.crop(bbox=(x1, y1, x2, y2))             # same, named
full.crop(roi=my_roi, margin=8)              # axis-aligned bounds of an ROI + margin
full.crop(center=(cx, cy), size=(w, h))      # fixed-size window
```

---

## Coordinates

A crop is a pure integer translation by `(x1, y1)`, so mapping landmark coordinates
between source and cropped frames is exact and NaN-preserving:

```python
pts_crop   = view.to_crop_coords(pts_source)     # subtract (x1, y1)
pts_source = view.to_source_coords(pts_crop)      # add (x1, y1)
```

On an uncropped video these are identity passthroughs, so the same call works
regardless of whether a video happens to be cropped. The underlying functions live in
`sleap_io.transform.points` as `crop_points` / `uncrop_points`.

!!! note "Coordinates are never rewritten on disk"
    Virtual cropping never mutates stored `instance.points`. These helpers are
    read-time conveniences for presenting/ingesting coordinates in cropped-frame space.

---

## Mosaics: many crops, one decode

Multiple differently-cropped views of one physical file can share a single decoder, so
the source frame is decoded once per read rather than once per tile:

```python
full = sio.load_video("session.mp4")
tiles = [
    full.crop((x, y, x + 128, y + 128))       # share_decode=True (default)
    for y in range(0, 1080 - 128, 128)
    for x in range(0, 1920 - 128, 128)
]
labels = sio.Labels(videos=tiles)
```

Each tile reuses `full`'s backend as its inner reader. The tiles do **not** own that
shared decoder, so closing one tile does not tear down its siblings; the owning source
`Video` manages the decoder's lifetime. (Decoder sharing is intentionally not preserved
across `pickle`/`deepcopy`/`open()` — each reconstruction rebuilds its own reader.)

Two crops of the same file with **different** crops are kept distinct through merge,
append, and matching; two crops with the **same** rect dedup to one view.

---

## Saving & loading (SLP round-trip)

Crops round-trip through `.slp` without breaking older readers:

```python
sio.save_file(labels, "mosaic.slp")
labels2 = sio.load_file("mosaic.slp")
labels2.videos[0]._crop_tuple()         # (0, 0, 128, 128)        -- preserved
labels2.videos[0].shape                 # (1000, 128, 128, 3)
labels2.videos[0].source_video.shape    # (1000, 1080, 1920, 3)
len(labels2.videos)                     # all tiles preserved (not collapsed)
```

- The crop rects are stored in a dedicated top-level `/video_crops` dataset, written
  **only when a crop is present**; the `videos_json` entry describes the **uncropped
  source**.
- An older reader that does not understand `/video_crops` simply loads the uncropped
  source video — a graceful, lossy degrade, never an error.
- Files with no crops are byte-identical to before this feature existed (no
  `/video_crops`, no format-version bump).

---

## Baking a crop to disk

A virtual crop is the lazy dual of the materializing pipeline. To write a real cropped
file later, feed the same rect to [`transform_video`](transforms.md) /
`transform_labels`; the baked pixels are identical to the virtual view:

```python
sio.transform_video(full, "baked.mp4", sio.Transform(crop=(320, 200, 576, 456)))
```

---

## Performance expectations

The crop is applied **after** a full-frame decode for every backend except raw,
sub-frame-chunked HDF5, where it can push the region read down to the storage layer:

| Backend | Strategy | I/O effect |
|---|---|---|
| `MediaVideo` (mp4/H.264/…) | decode full frame, slice | **No decode/I/O savings** — inter-frame codecs must decode the whole frame; the slice is a free in-memory view. Saves resident array size only. |
| `HDF5Video` raw rank-4, **sub-frame chunked** | hyperslab region read (`ds[i, y1:y2, x1:x2, :]`) | **Real I/O reduction** — only the overlapping chunks are read/decompressed. The one case where a crop saves disk work. |
| `HDF5Video` raw rank-4, per-frame chunked | region read (whole chunk still fetched) | Modest — skips chunk reassembly, not I/O. |
| `HDF5Video` embedded PNG/JPEG (`.pkg.slp`) | decode full image, slice | **No savings** — the whole image must be decoded before any spatial selection. |
| `ImageVideo`, `TiffVideo`, `SeqVideo` | decode full frame, slice | **No savings** with the current decoders. |

Pushdown for raw HDF5 is automatic and gated on the dataset's actual chunking; it falls
back to a full decode plus slice (byte-identical) whenever it would not help.

---

## Non-goals

Virtual cropping is a pure translate-and-clip view. It deliberately does **not** do:

- **Rotation, scale, pad, or flip on read** — those remain the domain of the
  materializing [`Transform`](transforms.md) pipeline.
- **Decode-cost savings for compressed video** — only sub-frame-chunked raw HDF5 sees
  real I/O savings; everywhere else the crop is a free post-decode view.
- **Lossless export through non-SLP writers** (NWB, COCO, JABS, Ultralytics) — those
  formats have no crop concept; exporting a cropped `Labels` through them is acceptably
  lossy (the cropped frame and its coordinates are emitted as-is).
- **Rewriting on-disk point coordinates** — the source labels are never mutated.

---

## See also

- [Transforms](transforms.md): the materializing crop/scale/rotate/pad/flip pipeline.
- [Video](model/video.md): the `Video` facade and its backends.
