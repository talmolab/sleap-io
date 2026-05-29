"""Internals for remote URL loading via fsspec.

This module contains all URL-handling primitives used by the high-level loaders
in :mod:`sleap_io.io.main` (and the helper refactors in
:mod:`sleap_io.io.slp` / :mod:`sleap_io.io.utils`). The public surface exposed
to users is :func:`clear_remote_cache` and :class:`RemoteIOError` (re-exported
at the package top level); everything else is private (underscore-prefixed).

Heavy third-party dependencies (``fsspec``, ``aiohttp``) are imported lazily
inside the functions that need them so that ``import sleap_io`` stays fast and
the cheap, pure-stdlib helpers (:func:`_is_url`, :func:`_redact_url`,
:func:`_identify_magic`) remain usable with zero heavy imports on the local
hot path.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import pathlib
import re
import time
import urllib.parse
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: URL schemes recognized as remote (vs. local filesystem paths). Single-letter
#: schemes (e.g. Windows drive letters like ``c``) are intentionally excluded.
_URL_SCHEMES = frozenset({"http", "https", "s3", "gs", "gcs", "az", "abfs"})

#: Alias for clarity at call sites.
_REMOTE_SCHEMES = _URL_SCHEMES

#: Cloud schemes that require an fsspec adapter package from the ``[cloud]``
#: extra.
_CLOUD_SCHEMES = frozenset({"s3", "gs", "gcs", "az", "abfs"})

#: Maps each cloud scheme to the fsspec adapter package that provides it.
_CLOUD_PROTOCOL_TO_PACKAGE = {
    "s3": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "az": "adlfs",
    "abfs": "adlfs",
}

#: HTTP headers stripped on cross-origin redirect (case-insensitive).
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization"})

#: Query-string parameter names whose values are redacted by :func:`_redact_url`.
_SENSITIVE_QUERY_PARAMS = frozenset(
    {"token", "access_token", "x-amz-security-token", "sas", "sig"}
)

#: Magic-byte prefixes mapped to a format family (checked in order).
_FORMAT_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89HDF\r\n\x1a\n", "hdf5"),
    (b"{", "json"),
    (b"[", "json"),
    (b"PK\x03\x04", "zip"),
)

#: Minimum ``aiohttp`` version for safe cross-origin redirect header stripping.
_MIN_AIOHTTP = "3.13.5"

#: Filename written into a cache directory to mark it as managed by sleap-io.
_CACHE_MARKER_NAME = ".sleap-io-cache-marker"

#: HTTP status codes that are retried (transient server / rate-limit errors).
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Base backoff (seconds) for exponential retry: attempt N waits
#: ``_RETRY_BACKOFF_BASE * 2**N`` (200/400/800 ms for the default 3 retries),
#: unless the server sends a ``Retry-After`` header which takes precedence.
_RETRY_BACKOFF_BASE = 0.2

#: Upper bound (seconds) on any single retry sleep, including a server-provided
#: ``Retry-After`` value. Caps pathological ``Retry-After`` values from blocking
#: a load indefinitely.
_RETRY_BACKOFF_MAX = 30.0

#: Pattern matching fsspec cache filenames (sha hex hashes, optional ``.tags``).
_CACHE_KEY_PATTERN = re.compile(r"^[0-9a-f]{32,64}(\.tags)?$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RemoteIOError(OSError):
    """Raised for HTTP-level failures during remote loading.

    Subclasses :class:`OSError` so that callers which already handle
    ``OSError`` (e.g. ``HDF5Video.__attrs_post_init__``) degrade gracefully.

    Attributes:
        url: Redacted URL (credentials stripped if present), or None.
        status: HTTP status code, or None for connection-level errors.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Build a RemoteIOError with a redacted, composed message.

        Args:
            message: Human-readable description of the failure.
            url: Raw URL associated with the failure. It is redacted before
                being stored or surfaced in the message.
            status: HTTP status code, if known.
            cause: The underlying exception, preserved as ``__cause__``.
        """
        self.url = _redact_url(url) if url else None
        self.status = status
        if cause is not None:
            self.__cause__ = cause
        parts = [message]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.url:
            parts.append(f"url={self.url}")
        super().__init__("; ".join(parts))


# ---------------------------------------------------------------------------
# URL detection + redaction (pure stdlib)
# ---------------------------------------------------------------------------


def _is_url(filename: str | os.PathLike) -> bool:
    r"""Return True if ``filename`` is a remote URL (vs. a local path).

    Robust against Windows-style absolute paths (e.g. ``C:\foo``): urlparse
    reports ``scheme='c'`` for those, but single-letter schemes are not in
    :data:`_URL_SCHEMES`.

    Args:
        filename: Candidate path or URL.

    Returns:
        True if the scheme is a recognized remote scheme, else False.
    """
    if not isinstance(filename, (str, os.PathLike)):
        return False
    s = os.fspath(filename) if isinstance(filename, os.PathLike) else filename
    if not s:
        return False
    scheme = urllib.parse.urlparse(s).scheme.lower()
    return scheme in _URL_SCHEMES


def _redact_url(url: str) -> str:
    """Strip credentials from a URL for safe logging/error display.

    ``https://user:pass@host/path?token=xyz`` becomes
    ``https://***:***@host/path?token=%2A%2A%2A``. Userinfo is replaced and
    token-like query parameters have their values redacted.

    Args:
        url: The URL to redact.

    Returns:
        The redacted URL. If parsing fails, returns the input unchanged.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except (ValueError, TypeError):  # pragma: no cover - urlparse is permissive
        return url
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = "***:***@" + netloc.split("@", 1)[1]
    if parsed.query:
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted_params = []
        for k, v in params:
            if k.lower() in _SENSITIVE_QUERY_PARAMS:
                redacted_params.append((k, "***"))
            else:
                redacted_params.append((k, v))
        new_query = urllib.parse.urlencode(redacted_params)
    else:
        new_query = ""
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc, query=new_query))


# ---------------------------------------------------------------------------
# Cloud scheme gating
# ---------------------------------------------------------------------------


def _require_package(package_name: str, *, scheme: str, extra: str = "cloud") -> None:
    """Import ``package_name``, raising a user-friendly ImportError if absent.

    Factored out of :func:`_ensure_cloud_extra` so the missing-package branch
    can be exercised with a genuinely-absent package name (no mocking).

    Args:
        package_name: The importable module name to require.
        scheme: The URL scheme that triggered the requirement (for messaging).
        extra: The sleap-io extra that provides the package.

    Raises:
        ImportError: If ``package_name`` cannot be imported. The message
            includes the ``pip install 'sleap-io[<extra>]'`` hint.
    """
    try:
        importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(
            f"Loading {scheme}:// URLs requires the {package_name!r} package. "
            f"Install with: pip install 'sleap-io[{extra}]' "
            "(covers s3, gs/gcs, az/abfs)."
        ) from e


def _ensure_cloud_extra(scheme: str) -> None:
    """Raise a user-friendly ImportError for a cloud scheme missing its adapter.

    Args:
        scheme: The URL scheme being opened.

    Raises:
        ImportError: If ``scheme`` is a cloud scheme whose fsspec adapter
            package is not installed.
    """
    if scheme not in _CLOUD_SCHEMES:
        return
    _require_package(_CLOUD_PROTOCOL_TO_PACKAGE[scheme], scheme=scheme)


# ---------------------------------------------------------------------------
# aiohttp version check
# ---------------------------------------------------------------------------


def _warn_if_old_aiohttp(version_str: str) -> None:
    """Warn if ``version_str`` is below the safe minimum aiohttp version.

    Factored from :func:`_check_aiohttp_version` so the warning branch is
    testable by passing an old version string directly (no monkeypatch).

    Args:
        version_str: The installed aiohttp version (e.g. ``"3.13.0"``).
    """
    from packaging.version import Version

    if Version(version_str) < Version(_MIN_AIOHTTP):
        warnings.warn(
            f"aiohttp {version_str} is below sleap-io's minimum {_MIN_AIOHTTP} "
            "for safe cross-origin redirect header stripping. Upgrade with: "
            "pip install --upgrade 'aiohttp>=3.13.5'. The belt-and-suspenders "
            "redirect hook will still apply, but the built-in aiohttp safety "
            "is not guaranteed on this version.",
            RuntimeWarning,
            stacklevel=2,
        )


def _check_aiohttp_version() -> None:
    """Check the installed aiohttp version and warn if it is too old.

    Called at module import time. If aiohttp is not importable (it is a base
    dependency, so this should not happen in practice), the check is skipped.
    """
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is a base dependency
        return
    _warn_if_old_aiohttp(aiohttp.__version__)


# ---------------------------------------------------------------------------
# fsspec filesystem builder
# ---------------------------------------------------------------------------


def _strip_cross_origin_headers(prev_url, new_url, headers) -> None:
    """Drop sensitive headers in-place when a redirect crosses origins.

    Factored out of the :func:`_safe_get_client` trace hook so the
    credential-stripping decision is directly unit-testable (the hook itself
    runs inside aiohttp's event-loop thread). Mutates ``headers`` in place,
    removing :data:`_SENSITIVE_HEADERS` (case-insensitive) only when the origin
    of ``new_url`` differs from ``prev_url``.

    Args:
        prev_url: The URL the request was redirected from (an ``aiohttp.URL``),
            or None if there is no prior URL (in which case nothing is done).
        new_url: The redirect target URL (an ``aiohttp.URL``).
        headers: A mutable, case-insensitive header mapping to scrub in place.
    """
    if prev_url is None:
        return
    if prev_url.origin() != new_url.origin():
        for header in list(headers):
            if header.lower() in _SENSITIVE_HEADERS:
                headers.pop(header, None)


async def _safe_get_client(**client_kwargs: Any):
    """Build an aiohttp.ClientSession that strips sensitive headers on redirect.

    Installs an ``on_request_redirect`` trace hook that drops
    ``Authorization`` / ``Cookie`` / ``Proxy-Authorization`` headers when a
    redirect crosses origins. aiohttp >= 3.13.5 already does this; the hook is
    belt-and-suspenders.

    Args:
        **client_kwargs: Forwarded to ``aiohttp.ClientSession``.

    Returns:
        A configured ``aiohttp.ClientSession``.
    """
    import aiohttp

    trace_config = aiohttp.TraceConfig()

    async def on_redirect(session, ctx, params):
        history = params.response.history if params.response else []
        prev_url = history[-1].url if history else None
        _strip_cross_origin_headers(prev_url, params.url, params.headers)

    trace_config.on_request_redirect.append(on_redirect)
    return aiohttp.ClientSession(trace_configs=[trace_config], **client_kwargs)


def _identity_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Merge user headers but force ``Accept-Encoding: identity`` to win.

    Identity encoding is required on every ranged HTTP request to avoid the
    gzip/content-length ambiguity. Any user-supplied ``Accept-Encoding`` (in any
    casing, since aiohttp coalesces header keys case-insensitively) is dropped so
    the identity guarantee cannot be silently weakened.

    Args:
        headers: Optional user-supplied HTTP headers.

    Returns:
        A new header mapping with ``Accept-Encoding`` forced to ``identity``.
    """
    merged = {
        k: v for k, v in (headers or {}).items() if k.lower() != "accept-encoding"
    }
    merged["Accept-Encoding"] = "identity"
    return merged


def _build_fsspec_filesystem(
    scheme: str,
    *,
    headers: dict[str, str] | None = None,
    block_size: int = 1 << 20,
    max_blocks: int = 32,
):
    """Build an fsspec ``AbstractFileSystem`` for ``scheme`` with our defaults.

    ``block_size`` / ``max_blocks`` are accepted for signature symmetry but are
    applied at ``fs.open(...)`` time (see :func:`open_url`), not on the
    filesystem constructor.

    Args:
        scheme: The URL scheme (``http``, ``https``, or a cloud scheme).
        headers: HTTP headers to send (HTTP/HTTPS only).
        block_size: Range block size in bytes (used by the caller).
        max_blocks: Max in-memory blocks per file (used by the caller).

    Returns:
        The configured fsspec filesystem.

    Raises:
        ImportError: If a cloud scheme's adapter package is missing.
    """
    import fsspec

    _ensure_cloud_extra(scheme)

    if scheme in ("http", "https"):
        from fsspec.implementations.http import HTTPFileSystem

        # Always set identity encoding to avoid the gzip/content-length
        # ambiguity for ranged reads (user headers cannot override it).
        merged = _identity_headers(headers)
        return HTTPFileSystem(
            client_kwargs={"headers": merged},
            get_client=_safe_get_client,
        )

    # s3 / gs / gcs / az / abfs: fsspec's per-scheme registry handles auth via
    # per-scheme kwargs. PR 1 does not forward HTTP headers here.
    return fsspec.filesystem(scheme)


def _http_inner_options(headers: dict[str, str] | None) -> dict[str, Any]:
    """Build the inner-http options for chained ``simplecache::``/``filecache::``.

    Args:
        headers: Optional HTTP headers to forward.

    Returns:
        A mapping suitable for the ``http=`` kwarg of ``fsspec.open``.
    """
    merged = _identity_headers(headers)
    return {"client_kwargs": {"headers": merged}, "get_client": _safe_get_client}


# ---------------------------------------------------------------------------
# Open a URL
# ---------------------------------------------------------------------------


def _retry_after_seconds(e: BaseException) -> float | None:
    """Extract a ``Retry-After`` delay (seconds) from an HTTP error, if present.

    Only the integer-seconds form of ``Retry-After`` is honored (the
    HTTP-date form is ignored and falls back to exponential backoff).

    Args:
        e: The caught exception (its chain is searched for a response error).

    Returns:
        The non-negative ``Retry-After`` value in seconds, or None if the
        header is absent or not an integer.
    """
    resp = _find_response_error(e)
    if resp is None or resp.headers is None:
        return None
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(int(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    return max(0.0, value)


def _retry_sleep_seconds(attempt: int, exc: BaseException) -> float:
    """Compute the sleep before the next attempt.

    Honors a server ``Retry-After`` header if present, else uses exponential
    backoff (``_RETRY_BACKOFF_BASE * 2**attempt``). The result is clamped to
    ``_RETRY_BACKOFF_MAX``.

    Args:
        attempt: Zero-based index of the attempt that just failed.
        exc: The exception from the failed attempt.

    Returns:
        Seconds to sleep before retrying.
    """
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, _RETRY_BACKOFF_MAX)
    return min(_RETRY_BACKOFF_BASE * (2**attempt), _RETRY_BACKOFF_MAX)


def _open_with_retries(open_fn, *, url: str, retries: int):
    """Call ``open_fn()`` with retries on transient HTTP errors.

    Retries are attempted only for the statuses in :data:`_RETRYABLE_STATUSES`
    (429/500/502/503/504). All other errors (and a final retryable error after
    the budget is exhausted) are converted via :func:`_raise_remote`.

    Args:
        open_fn: A zero-argument callable that performs the open and returns a
            file-like object.
        url: The URL involved (for error redaction).
        retries: Maximum number of retries (so up to ``retries + 1`` attempts).

    Returns:
        Whatever ``open_fn()`` returns on success.

    Raises:
        RemoteIOError: If all attempts fail.
    """
    attempt = 0
    while True:
        try:
            return open_fn()
        except Exception as e:  # noqa: BLE001 - mapped via _raise_remote below
            status = _status_from_exception(e)
            if status in _RETRYABLE_STATUSES and attempt < retries:
                time.sleep(_retry_sleep_seconds(attempt, e))
                attempt += 1
                continue
            _raise_remote(e, url=url)


def open_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream_mode: str = "auto",
    cache_storage: str | os.PathLike | None = None,
    cache_expiry: float | None = None,
    block_size: int = 1 << 20,
    max_blocks: int = 32,
    retries: int = 3,
):
    """Open ``url`` as a file-like object using the configured strategy.

    Args:
        url: The remote URL to open.
        headers: HTTP headers (HTTP/HTTPS only).
        stream_mode: One of ``"auto"`` (alias for ``"blockcache"``),
            ``"blockcache"``, ``"cache"`` (simplecache), ``"filecache"``, or
            ``"download"`` (ephemeral full read into BytesIO).
        cache_storage: Override the cache directory for cache/filecache modes.
        cache_expiry: TTL (seconds) for ``filecache`` revalidation. Defaults to
            3600 when not given.
        block_size: Range block size in bytes for ``blockcache``.
        max_blocks: Max in-memory LRU blocks per open file for ``blockcache``.
        retries: Maximum retries for transient HTTP errors (429/500/502/503/504),
            with exponential backoff honoring ``Retry-After``. Default: 3.

    Returns:
        A file-like object (fsspec buffered file or ``io.BytesIO``) ready to be
        passed to ``h5py.File(...)`` or a reader.

    Raises:
        RemoteIOError: For HTTP / connection failures.
        ImportError: For cloud schemes when the extra is not installed.
        ValueError: For an unrecognized ``stream_mode``.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    fs = _build_fsspec_filesystem(
        scheme, headers=headers, block_size=block_size, max_blocks=max_blocks
    )

    if stream_mode == "auto":
        stream_mode = "blockcache"

    if stream_mode not in ("blockcache", "cache", "filecache", "download"):
        raise ValueError(
            f"Invalid stream_mode={stream_mode!r}; expected one of "
            "{'auto', 'blockcache', 'cache', 'filecache', 'download'}."
        )

    if stream_mode == "blockcache":

        def _open():
            return fs.open(
                url,
                mode="rb",
                cache_type="blockcache",
                block_size=block_size,
                # fsspec's BlockCache uses ``maxblocks`` (no underscore).
                cache_options={"maxblocks": max_blocks},
            )

    elif stream_mode == "cache":

        def _open():
            import fsspec

            simplecache_opts: dict[str, Any] = {}
            if cache_storage is not None:
                _mark_cache_dir(cache_storage)
                simplecache_opts["cache_storage"] = str(cache_storage)
            return fsspec.open(
                f"simplecache::{url}",
                mode="rb",
                simplecache=simplecache_opts,
                http=_http_inner_options(headers),
            ).open()

    elif stream_mode == "filecache":

        def _open():
            import fsspec

            options: dict[str, Any] = {
                "expiry_time": cache_expiry if cache_expiry is not None else 3600
            }
            if cache_storage is not None:
                _mark_cache_dir(cache_storage)
                options["cache_storage"] = str(cache_storage)
            return fsspec.open(
                f"filecache::{url}",
                mode="rb",
                filecache=options,
                http=_http_inner_options(headers),
            ).open()

    else:  # stream_mode == "download"

        def _open():
            with fs.open(url, mode="rb") as src:
                return io.BytesIO(src.read())

    return _open_with_retries(_open, url=url, retries=retries)


def open_remote_h5(url: str, *, headers: dict[str, str] | None = None):
    """Open a remote HDF5 file for membership/existence probing.

    Thin convenience wrapper around :func:`open_url` using ``blockcache`` mode,
    used by ``Video.exists`` to probe a remote ``.slp``/``pkg.slp`` URL.

    Args:
        url: The remote HDF5 URL.
        headers: Optional HTTP headers.

    Returns:
        A file-like object suitable for ``h5py.File(...)``.
    """
    return open_url(url, headers=headers, stream_mode="blockcache")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _find_response_error(e: BaseException):
    """Find a wrapped ``aiohttp.ClientResponseError`` in an exception chain.

    fsspec wraps the underlying ``aiohttp.ClientResponseError`` as the
    ``__cause__`` (or ``__context__``) of a ``FileNotFoundError`` for HTTP error
    responses, so the HTTP status is not visible on the top-level exception.
    This walks the chain (bounded) to recover it.

    Args:
        e: The caught exception.

    Returns:
        The first ``aiohttp.ClientResponseError`` found in ``e``'s cause/context
        chain (including ``e`` itself), or None if there is none.
    """
    import aiohttp

    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, aiohttp.ClientResponseError):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _status_from_exception(e: BaseException) -> int | None:
    """Return the HTTP status of an exception (unwrapping fsspec's wrapper).

    Args:
        e: The caught exception.

    Returns:
        The HTTP status code if an ``aiohttp.ClientResponseError`` is found in
        the chain, else None.
    """
    resp = _find_response_error(e)
    return resp.status if resp is not None else None


def _raise_remote(e: BaseException, *, url: str) -> None:
    """Convert an aiohttp/fsspec exception into a :class:`RemoteIOError`.

    Args:
        e: The caught exception.
        url: The URL involved (redacted before display).

    Raises:
        RemoteIOError: Always (this function never returns normally).
    """
    import aiohttp

    if isinstance(e, RemoteIOError):
        raise e

    # fsspec wraps HTTP error responses (e.g. as FileNotFoundError); recover the
    # underlying ClientResponseError from the chain so the status is preserved.
    resp = _find_response_error(e)
    if resp is not None:
        msg = {
            404: "file not found",
            416: "range past end of file",
            412: "file changed since cached (ETag mismatch)",
        }.get(resp.status, f"HTTP {resp.status}")
        raise RemoteIOError(msg, url=url, status=resp.status, cause=e) from e
    if isinstance(e, aiohttp.ClientConnectorError):
        raise RemoteIOError("connection error", url=url, cause=e) from e
    if isinstance(e, aiohttp.ClientPayloadError):
        raise RemoteIOError("truncated body", url=url, cause=e) from e
    if isinstance(e, asyncio.TimeoutError):
        raise RemoteIOError("timeout", url=url, cause=e) from e
    raise RemoteIOError(
        f"unexpected error: {type(e).__name__}", url=url, cause=e
    ) from e


# ---------------------------------------------------------------------------
# Magic-byte sniffing
# ---------------------------------------------------------------------------


def _identify_magic(head: bytes) -> str:
    """Classify a byte prefix into a format family.

    Args:
        head: The first bytes of a file (typically 16).

    Returns:
        One of ``"hdf5"``, ``"json"``, ``"csv"``, ``"zip"``, or ``"unknown"``.
    """
    for prefix, fmt in _FORMAT_MAGIC:
        if head.startswith(prefix):
            return fmt
    # CSV: non-empty, ASCII-printable (plus tab/CR/LF), contains a comma.
    if head and all(32 <= b < 127 or b in (9, 10, 13) for b in head) and b"," in head:
        return "csv"
    return "unknown"


def _sniff_format(url: str, *, headers: dict | None = None) -> str:
    """Fetch the first 16 bytes of ``url`` and identify the format family.

    Args:
        url: The remote URL to sniff.
        headers: Optional HTTP headers.

    Returns:
        One of ``"hdf5"``, ``"json"``, ``"csv"``, ``"zip"``, ``"unknown"``.

    Raises:
        RemoteIOError: For HTTP / connection failures.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    fs = _build_fsspec_filesystem(scheme, headers=headers)
    try:
        with fs.open(url, mode="rb") as f:
            head = f.read(16)
    except Exception as e:
        _raise_remote(e, url=url)
    return _identify_magic(head)


def _head_or_range_probe(url: str, *, headers: dict[str, str] | None = None) -> bool:
    """Check whether ``url`` exists via a HEAD request, falling back to a Range.

    Some servers reject HEAD (405); in that case a single-byte
    ``Range: bytes=0-0`` GET is used.

    Args:
        url: The remote URL to probe.
        headers: Optional HTTP headers.

    Returns:
        True if the resource appears to exist and is reachable, else False.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    try:
        fs = _build_fsspec_filesystem(scheme, headers=headers)
        try:
            return bool(fs.exists(url))
        except Exception:  # pragma: no cover - defensive: fsspec's _exists
            # swallows HTTP/ClientError internally and returns a bool, so this
            # only fires for unusual non-HTTP failures. Fall back to a tiny
            # ranged read in case HEAD is unsupported.
            with fs.open(url, mode="rb") as f:
                f.read(1)
            return True
    except Exception:  # pragma: no cover - defensive: filesystem construction
        # or the ranged-read fallback failing is not reachable via real servers.
        return False


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _mark_cache_dir(cache_storage: str | os.PathLike) -> None:
    """Write the sleap-io marker file into ``cache_storage`` (idempotent).

    The marker opts a directory into :func:`clear_remote_cache`, preventing
    accidental deletion of files in unrelated directories.

    Args:
        cache_storage: The cache directory to mark.
    """
    cache_dir = pathlib.Path(cache_storage).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / _CACHE_MARKER_NAME
    if not marker.exists():
        marker.touch()


def clear_remote_cache(
    *,
    older_than: float | None = None,
    cache_storage: str | os.PathLike | None = None,
) -> int:
    """Clear sleap-io's fsspec remote cache.

    Only deletes files whose names match fsspec's cache-key pattern (sha-style
    hex hashes, optionally with a ``.tags`` sidecar), preventing accidental
    deletion of unrelated files if ``cache_storage`` points at a shared dir.

    Args:
        older_than: If set, only delete files whose modification time is older
            than this many seconds.
        cache_storage: The cache directory to clear. Required to contain the
            sleap-io marker file. fsspec's built-in default cache directory is a
            per-process temporary directory (the ``"TMP"`` sentinel), which is
            not a stable, clearable location, so an explicit path is required
            here.

    Returns:
        The number of files deleted.

    Raises:
        RuntimeError: If ``cache_storage`` is None, a forbidden path (root,
            ``$HOME``), or does not contain the sleap-io cache marker file.
    """
    if cache_storage is None:
        raise RuntimeError(
            "clear_remote_cache requires an explicit cache_storage path: "
            "the same path passed to load_slp(..., cache_storage=...). fsspec's "
            "default cache directory is a per-process temporary directory and "
            "cannot be cleared reliably."
        )
    cache_dir = pathlib.Path(cache_storage).expanduser().resolve()

    forbidden = {pathlib.Path("/").resolve(), pathlib.Path.home().resolve()}
    if cache_dir in forbidden:
        raise RuntimeError(
            f"Refusing to clear cache: cache_storage={str(cache_dir)!r} is a "
            "forbidden path (root, $HOME, etc.)."
        )

    marker = cache_dir / _CACHE_MARKER_NAME
    if not marker.exists():
        raise RuntimeError(
            f"Refusing to clear cache: {str(cache_dir)!r} does not contain a "
            f"sleap-io cache marker file ({_CACHE_MARKER_NAME!r}). It was not "
            "written by sleap-io."
        )

    deleted = 0
    now = time.time()
    for p in cache_dir.iterdir():
        if not p.is_file():
            continue
        if not _CACHE_KEY_PATTERN.match(p.name):
            continue
        if older_than is not None:
            if (now - p.stat().st_mtime) < older_than:
                continue
        p.unlink()
        deleted += 1

    return deleted


# Run the aiohttp version check at import time (cheap; warns on downgrade).
_check_aiohttp_version()
