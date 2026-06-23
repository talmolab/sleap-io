# Remote loading

sleap-io can load `.slp`/`.pkg.slp` labels and media video directly from
`http`/`https`, cloud storage (S3, GCS, Azure), and Google Drive URLs — with
lazy, range-based streaming by default, so only the bytes you actually read are
pulled over the network.

---

## Quick start

[`load_slp`][sleap_io.load_slp] and the universal
[`load_file`][sleap_io.load_file] accept a URL anywhere a local path is
accepted. HTTP/HTTPS works with a base install:

```python
import sleap_io as sio

# http/https works out of the box
labels = sio.load_slp("https://example.com/labels.slp")

# load_file dispatches by extension and also accepts URLs
labels = sio.load_file("https://example.com/labels.slp")
```

`.pkg.slp` files with embedded frames work too — the embedded
[`Video`][sleap_io.Video] backends reopen the remote file lazily when you read
frames (see [Embedded pkg.slp streaming](#embedded-pkgslp-streaming)).

When you pass a local path, all of the URL keyword arguments below are no-ops,
so the same call works for local and remote files.

!!! info "What is and isn't supported"
    URL loading covers `.slp`/`.pkg.slp` labels and remote *media video* over
    `http`/`https` only. Every other labels format (NWB, COCO, Label Studio,
    JABS, DLC, CSV, TrackMate, LEAP, GeoJSON, Ultralytics) raises
    `NotImplementedError` over a URL — download the file locally first.

### Command-line interface

The `sio` read commands accept a URL anywhere they accept a local input file —
the input is streamed over the network just like the Python loaders. This works
for `show`, `filenames`, `convert`, `split`, `render`, `export`, `fix`, `embed`,
and `unembed`:

```bash
# Inspect a remote labels file
sio show https://example.com/labels.slp

# Convert a remote file to a local output (output paths stay local)
sio convert s3://bucket/labels.slp -o labels.nwb

# The -i/--input option accepts URLs too
sio filenames -i https://example.com/labels.slp
```

Output paths and commands that re-encode video locally (`trim`, `reencode`,
`transform`, `apply-crops`) require local filesystem paths.

---

## Supported schemes & install matrix

HTTP/HTTPS needs nothing beyond the base install. Cloud schemes require the
`cloud` extra, which pulls in the per-provider fsspec adapters:

| Scheme | Requires | Notes |
|--------|----------|-------|
| `http`, `https` | nothing extra | Works with a plain `pip install sleap-io` |
| `s3` | `sleap-io[cloud]` | Amazon S3 (via `s3fs`) |
| `gs`, `gcs` | `sleap-io[cloud]` | Google Cloud Storage (via `gcsfs`) |
| `az`, `abfs` | `sleap-io[cloud]` | Azure Blob / ADLS (via `adlfs`) |

```bash
pip install sleap-io               # http/https only
pip install "sleap-io[cloud]"      # + s3, gs/gcs, az/abfs
pip install "sleap-io[pyav]"       # remote media video (provides av)
pip install "sleap-io[all]"        # everything (cloud + pyav included)
```

```python
# Cloud schemes need the [cloud] extra (s3fs / gcsfs / adlfs)
labels = sio.load_slp("s3://my-bucket/path/labels.slp")
labels = sio.load_slp("gs://my-bucket/path/labels.slp")
```

!!! warning "Missing cloud extra"
    Using a cloud scheme without the `[cloud]` extra raises an `ImportError`
    whose message names the missing package and the
    `pip install 'sleap-io[cloud]'` install hint.

---

## Streaming modes & caching

The `stream_mode` keyword argument controls how bytes are fetched:

| `stream_mode` | Backing strategy | Memory | Disk cache | Revalidation | Best for |
|---------------|------------------|--------|------------|--------------|----------|
| `"auto"` (default) | fsspec `blockcache` | Low (LRU of `max_blocks`) | None | n/a | One-off lazy reads, low memory |
| `"blockcache"` | fsspec `blockcache` | Low | None | n/a | Same as `auto` (explicit) |
| `"cache"` | fsspec `simplecache` | Whole file on disk | Persistent | None | Repeated opens of an immutable file |
| `"filecache"` | fsspec `filecache` | Whole file on disk | Persistent | ETag / `Last-Modified` after `cache_expiry` | Repeated opens of a file that may change |
| `"download"` | Full read into memory | Whole file in RAM | None | n/a | Small files, ephemeral environments |

```python
# Default: lazy range reads via blockcache, low memory
labels = sio.load_slp("https://example.com/labels.slp")

# Persistent on-disk cache with daily ETag revalidation
labels = sio.load_slp(
    "https://example.com/labels.slp",
    stream_mode="filecache",
    cache_storage="~/.cache/sleap-io",
    cache_expiry=86400,  # revalidate after a day
)

# Ephemeral full download into memory (no disk cache)
labels = sio.load_slp("https://example.com/labels.slp", stream_mode="download")
```

The `"auto"`/`"blockcache"` reads can be tuned with `block_size` (range block
size, default 1 MiB) and `max_blocks` (in-memory LRU cap per open file,
default 32 → ~32 MiB per file). For `"filecache"`, `cache_expiry` defaults to
3600 seconds (1 hour) when not given.

!!! tip "CI and ephemeral environments"
    In CI prefer the default `stream_mode="auto"` (no persistent cache to
    manage), or scope a per-run cache to a temporary directory you control:

    ```python
    import os
    import sleap_io as sio

    cache_dir = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "sio-cache")
    labels = sio.load_slp(url, stream_mode="filecache", cache_storage=cache_dir)
    ```

### Clearing the cache

For `"cache"` and `"filecache"` modes, downloaded files live in the directory
you pass as `cache_storage=`. To clear them, call
[`clear_remote_cache`][sleap_io.clear_remote_cache] with the **same**
`cache_storage` you loaded with:

```python
import sleap_io as sio

# Delete every sleap-io cache file in the directory
sio.clear_remote_cache(cache_storage="~/.cache/sleap-io")

# Or only files older than an hour (older_than is in seconds)
sio.clear_remote_cache(cache_storage="~/.cache/sleap-io", older_than=3600)
```

!!! note "An explicit `cache_storage` is required"
    [`clear_remote_cache`][sleap_io.clear_remote_cache] only operates on a
    directory that contains the sleap-io marker file, and only deletes files
    matching fsspec's cache-key naming pattern — so it never touches unrelated
    files even in a shared directory. It refuses to run on a directory with no
    marker, or on forbidden paths like `/` or `$HOME`. Because fsspec's default
    cache directory is a per-process temporary location, you must pass the
    explicit `cache_storage` you used when loading.

---

## Authentication & security

Pass HTTP headers (such as a bearer token) with `headers=`:

```python
labels = sio.load_slp(
    "https://my-org.example/private/labels.slp",
    headers={"Authorization": "Bearer <token>"},
)
```

Cloud schemes (`s3://`, `gs://`, …) ignore `headers=` and use their own
per-provider credential chains (environment variables, credential files,
instance metadata).

!!! warning "Headers are stripped on cross-origin redirect"
    For security, sensitive headers (`Authorization`, `Cookie`,
    `Proxy-Authorization`) are **dropped automatically** if a request is
    redirected to a different origin (scheme/host/port). This prevents leaking
    credentials to a third-party host. If a download redirects cross-origin
    (e.g. to a pre-signed CDN URL), put the credentials in the redirect
    target's query string rather than in `headers`.

Other security guarantees: TLS is always on; URLs are redacted (userinfo and
token-like query parameters stripped) in error messages and tracebacks so
credentials never leak into logs; and an identity `Accept-Encoding` is forced
so range reads stay byte-exact. Remote loading needs `aiohttp >= 3.13.5` for
the cross-origin header stripping — an older version emits a `RuntimeWarning`
at import time.

---

## Embedded pkg.slp streaming

A `.pkg.slp` URL streams its embedded frames lazily: opening the file reads
only metadata, and each embedded image is fetched on demand with a
range-request (`blockcache`) when you index into the video.

```python
# Embedded frames stream over the network on access
labels = sio.load_slp("https://example.com/project.pkg.slp")
frame = labels.videos[0][0]  # range-reads just the bytes for frame 0
```

This means you can inspect a remote packaged project without downloading the
whole archive — the embedded video backends reopen the remote file using the
same streaming configuration and authentication `headers=` as the initial load,
so an auth-gated `.pkg.slp` streams its frames without re-authenticating.

---

## Remote video

[`load_video`][sleap_io.load_video] (and [`load_file`][sleap_io.load_file] for
video extensions) reads a media video directly from an `http`/`https` URL.
Frames are decoded on demand:

```python
# Reads frames lazily over the network; needs the [pyav] extra
video = sio.load_video("https://example.com/video.mp4")
frame = video[0]  # decoded on demand
```

Supported container extensions match local media videos (`mp4`, `avi`, `mov`,
`mj2`, `mkv`). Only `http`/`https` URLs are accepted — cloud schemes and Google
Drive are **not** supported for video. The query string and fragment are
ignored for extension detection, so pre-signed URLs like
`https://host/video.mp4?token=...` route correctly. Remote video requires the
`pyav` extra (auto-selected as the backend); without it, `load_video(url)`
raises an `ImportError` with the install hint.

!!! danger "Security: remote video hands untrusted data to FFmpeg"
    Decoding a remote video streams bytes from the URL into FFmpeg (via pyav).
    FFmpeg's demuxers and decoders are a large, historically
    vulnerability-prone attack surface, so a malicious URL or stream can attempt
    to exploit the decoder running in your process. sleap-io only passes
    `http`/`https` URLs through to the decoder (no other schemes), but you
    should:

    - **Load remote video only from sources you trust** — treat an arbitrary
      third-party URL the same as running untrusted code.
    - **Sandbox untrusted inputs** — decode from an untrusted source only in an
      isolated environment (container/VM with no credentials, restricted
      network, non-privileged user) and keep FFmpeg/pyav up to date.

---

## Google Drive

Google Drive **file** share links are recognized and resolved to a direct
download, so you can pass a Drive URL straight to
[`load_slp`][sleap_io.load_slp] or [`load_file`][sleap_io.load_file]:

```python
# Any of these Drive file-share shapes resolve to a direct download:
labels = sio.load_slp("https://drive.google.com/file/d/<FILE_ID>/view")
labels = sio.load_slp("https://drive.google.com/uc?id=<FILE_ID>&export=download")
labels = sio.load_slp("https://drive.google.com/open?id=<FILE_ID>")

# load_file resolves the link, sniffs the bytes to detect the format, and routes
# it. The sniffed bytes are reused, so the file is downloaded only once:
labels = sio.load_file("https://drive.google.com/file/d/<FILE_ID>/view")
```

The file must be shared as **"Anyone with the link"** (no sign-in required).
Because Drive download links carry no extension and reject the `HEAD`/range
requests that lazy streaming relies on, a Drive file is **fully downloaded into
memory** during resolution — the `stream_mode`/cache keyword arguments do not
apply. The two-hop confirmation page Drive serves for larger files is handled
transparently, and the resolver only ever follows Google download hosts.

Some limitations:

- **Folder links are not supported** — pass a single-file share link
  (`…/file/d/<FILE_ID>/view`), not a `…/drive/folders/<ID>` URL. A folder link
  raises a `ValueError`.
- **Drive videos are not supported** — download the video file first, then load
  it locally.
- **Quota / permission errors** — if Drive returns its "too many users have
  viewed or downloaded this file recently" page, a
  [`RemoteIOError`][sleap_io.RemoteIOError] is raised; retry later or re-check
  the file's sharing settings.
- **Large files** — the in-memory prefetch is capped (8 GiB by default); a file
  exceeding the cap raises a [`RemoteIOError`][sleap_io.RemoteIOError] instead of
  exhausting memory.

For [`load_file`][sleap_io.load_file], the format is detected from the
downloaded bytes, and those bytes are reused for the load — so a Drive `.slp` is
downloaded only once whether you call `load_slp` or `load_file` (an explicit
`format=` is optional, and skips the format-detection step).

---

## Error handling

Remote HTTP/cloud failures surface as
[`RemoteIOError`][sleap_io.RemoteIOError] (a subclass of `OSError`). It carries
a `status` (HTTP code or `None`) and a credential-redacted `url`, so tokens
never leak into logs or tracebacks:

```python
import sleap_io as sio

try:
    labels = sio.load_slp("https://example.com/labels.slp")
except sio.RemoteIOError as e:
    print(e.status)  # e.g. 404, 416, 503, or None for connection errors
    print(e.url)     # redacted URL (tokens/userinfo stripped)
```

Only transient statuses (`429`, `500`, `502`, `503`, `504`) are retried (with
exponential backoff, honoring an integer `Retry-After`); the retry count is
controlled by `retries=` (default 3).

---

## Troubleshooting

- **[`RemoteIOError`][sleap_io.RemoteIOError]** — HTTP-level failures (404 not
  found, 416 range past end of file, 5xx after retries, connection errors,
  timeouts). Carries `status` and a redacted `url`.
- **`ImportError` for cloud schemes** — install the cloud adapters with
  `pip install 'sleap-io[cloud]'` (covers `s3`, `gs`/`gcs`, `az`/`abfs`).
- **`ImportError` from `load_video(url)`** — remote video needs the `pyav`
  extra; install with `pip install 'sleap-io[pyav]'`. Only `http`/`https` URLs
  are supported for video.
- **`RuntimeWarning` about `aiohttp`** — remote loading needs
  `aiohttp >= 3.13.5` for safe cross-origin header stripping. Upgrade with
  `pip install --upgrade 'aiohttp>=3.13.5'`.
- **`ValueError` for a URL** — an ambiguous-extension URL (`.h5`/`.json`/`.csv`)
  with `sniff=False` and no explicit `format=`, or a Google Drive folder /
  unparsable link.

!!! note "See also"
    - [`load_slp`][sleap_io.load_slp]: Full URL keyword-argument reference
    - [`load_file`][sleap_io.load_file]: Universal loader with URL sniffing
    - [`load_video`][sleap_io.load_video]: Loads remote media video over http/https
    - [`clear_remote_cache`][sleap_io.clear_remote_cache]: Cache cleanup helper
    - [`RemoteIOError`][sleap_io.RemoteIOError]: Remote I/O error surface
    - [SLP Format](formats/slp.md): The on-disk `.slp` layout that URL loading streams
