# Virtual cropping

sleap-io can expose a **virtual, on-read crop** of a video — a cropped view whose
frames are produced by decoding the source and slicing in memory, without copying or
re-encoding any pixels on disk. It is the lazy, non-destructive counterpart of the
materializing [Transforms](transforms.md) pipeline: a virtually-cropped frame is
byte-identical to what baking a `Transform(crop=...)` would write.

---

## Quick start

```pycon
>>> import sleap_io as sio
>>>
>>> full = sio.load_video("tests/data/videos/centered_pair_low_quality.mp4")
>>> print(full.shape)
>>>
>>> # A cropped view. crop = (x1, y1, x2, y2), with x2/y2 EXCLUSIVE.
>>> view = full.crop((64, 64, 192, 192))
>>> print(view.shape)                  # cropped view
>>> print(view[0].shape)               # a single cropped frame
>>> print(view._crop_tuple())          # (x1, y1, x2, y2) -- view.crop is not the rect
>>> print(view.source_video is full)   # provenance to the uncropped original
```

`Video.from_crop` opens a file and crops it in one call:

```pycon
>>> import sleap_io as sio
>>> view = sio.Video.from_crop(
...     "tests/data/videos/centered_pair_low_quality.mp4", crop=(64, 64, 192, 192)
... )
>>> print(view.shape)
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

```pycon
>>> import sleap_io as sio
>>> full = sio.load_video("tests/data/videos/centered_pair_low_quality.mp4")
>>> cx, cy = 192, 192
>>> # Fixed 128x128 window centered on a point (may run off the edge -> padded).
>>> view = full.crop(center=(cx, cy), size=(128, 128), fill=0)
>>> print(view.shape)            # (n_frames, 128, 128, channels)
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

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> full = sio.load_video("tests/data/videos/centered_pair_low_quality.mp4")
>>> view = full.crop((64, 64, 192, 192))
>>> pts_source = np.array([[100.0, 120.0]])
>>> pts_crop = view.to_crop_coords(pts_source)      # subtract (x1, y1)
>>> print(pts_crop)
>>> pts_source = view.to_source_coords(pts_crop)    # add (x1, y1)
>>> print(pts_source)
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

```pycon
>>> import sleap_io as sio
>>> full = sio.load_video("tests/data/videos/centered_pair_low_quality.mp4")
>>> tiles = [
...     full.crop((x, y, x + 128, y + 128))       # share_decode=True (default)
...     for y in range(0, full.shape[1] - 128, 128)
...     for x in range(0, full.shape[2] - 128, 128)
... ]
>>> labels = sio.Labels(videos=tiles)
>>> print(len(labels.videos))
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

## Applying (baking) a crop to disk

A virtual crop can be **materialized** to a real video file — the cropped pixels become
physical and the crop is no longer a read-time view. This is coordinate-neutral: a virtual
crop already presents cropped-frame coordinates, so baking the pixels leaves all point
coordinates unchanged.

`Video.apply_crop` bakes one cropped video and returns a new `Video` for the baked file,
preserving provenance (`source_video` is the uncropped original):

```python
view = full.crop((320, 200, 576, 456))
baked = view.apply_crop("crop.mp4")
baked.shape                    # (1000, 256, 256, 3)  — cropped, now physical
baked.source_video.shape       # (1000, 1080, 1920, 3) — uncropped original
baked._crop_tuple()            # None — the crop is materialized, not virtual
```

`Labels.apply_crops` bakes every virtually-cropped video in a `Labels` and rewires all
references (labeled frames, ROIs, suggestions) to the baked files; uncropped videos are
untouched and coordinates are unchanged:

```python
labels.apply_crops(video_dir="baked_videos/")   # one file per tile, unique names
```

From the command line, `sio apply-crops` materializes every virtual crop in an SLP,
writing baked videos to a directory next to the output and updating the references:

```bash
sio apply-crops mosaic.slp -o baked.slp --video-dir baked_videos/
```

!!! note "`apply_crop` vs `sio transform --crop`"
    `apply_crop` materializes an **existing** virtual crop (no coordinate change).
    `sio transform --crop` applies a **new** crop and adjusts coordinates — that is the
    materializing [`transform_video`](transforms.md) / `transform_labels` path:

    ```python
    sio.transform_video(full, "baked.mp4", sio.Transform(crop=(320, 200, 576, 456)))
    ```

!!! info "Encoder padding"
    The H.264 encoder pads frame dimensions up to a multiple of 16 (bottom/right only,
    preserving the top-left content and coordinate alignment). A baked video whose cropped
    width/height are not multiples of 16 is padded on those edges.

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
