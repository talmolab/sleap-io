"""Unit and network tests for sleap_io.io._remote.

The first half of this module exercises the pure / local-filesystem logic
(URL detection, redaction, magic-byte sniffing, error mapping, cache helpers).
The second half drives real HTTP requests through ``open_url`` /
``_build_fsspec_filesystem`` / ``_sniff_format`` against a loopback
``pytest-httpserver`` so the cross-origin credential-stripping trace hook,
retry/backoff logic, and HTTP-status mapping are exercised end-to-end. No
test in this module requires live internet access; the lone optional
live-internet test is marked ``@pytest.mark.live``.
"""

import asyncio
import os
import time
import traceback
import urllib.parse
import warnings
from pathlib import Path

import aiohttp
import pytest
from aiohttp.client_reqrep import ConnectionKey
from multidict import CIMultiDict
from werkzeug.wrappers import Response
from yarl import URL

from sleap_io.io._gdrive import (
    _is_gdrive_url,
    _open_gdrive,
    _parse_gdrive,
    _url_from_confirmation,
)
from sleap_io.io._remote import (
    _CACHE_MARKER_NAME,
    _RETRY_BACKOFF_BASE,
    RemoteIOError,
    _build_fsspec_filesystem,
    _find_response_error,
    _head_or_range_probe,
    _http_inner_options,
    _identify_magic,
    _identity_headers,
    _is_url,
    _mark_cache_dir,
    _raise_remote,
    _redact_url,
    _redacted_cause_summary,
    _redirect_target,
    _require_package,
    _retry_after_seconds,
    _retry_sleep_seconds,
    _sniff_format,
    _status_from_exception,
    _strip_cross_origin_headers,
    _warn_if_old_aiohttp,
    clear_remote_cache,
    open_remote_h5,
    open_url,
)

#: HDF5 superblock signature followed by filler; serves as a sniffable body.
_HDF5_BODY = b"\x89HDF\r\n\x1a\n" + b"\x00" * 128


def _make_range_handler(body: bytes):
    """Return a werkzeug handler that honors HTTP Range requests for ``body``.

    fsspec's ``blockcache`` cache type issues ``Range`` requests; a plain
    ``respond_with_data`` server (200 with no ranged support) is rejected by
    fsspec. This handler answers ``HEAD`` (with ``Content-Length`` /
    ``Accept-Ranges``) and ranged / full ``GET`` requests.

    Args:
        body: The full response body to serve.

    Returns:
        A callable suitable for ``HTTPServer.respond_with_handler``.
    """
    import re

    total = len(body)

    def _handler(request):
        if request.method == "HEAD":
            resp = Response(b"", status=200)
            resp.headers["Content-Length"] = str(total)
            resp.headers["Accept-Ranges"] = "bytes"
            return resp
        rng = request.headers.get("Range")
        if rng:
            match = re.match(r"bytes=(\d+)-(\d*)", rng)
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else total - 1
            end = min(end, total - 1)
            chunk = body[start : end + 1]
            resp = Response(chunk, status=206)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{total}"
            resp.headers["Accept-Ranges"] = "bytes"
            return resp
        resp = Response(body, status=200)
        resp.headers["Content-Length"] = str(total)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    return _handler


# ---------------------------------------------------------------------------
# _is_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        # Windows absolute paths -> not URLs (scheme is the drive letter).
        ("C:/Users/foo.slp", False),
        ("C:\\Users\\foo.slp", False),
        ("D:\\data\\labels.slp", False),
        # Relative / bare local paths.
        ("./labels.slp", False),
        ("../data/labels.slp", False),
        ("labels.slp", False),
        ("/abs/unix/path.slp", False),
        # Remote schemes.
        ("http://example.com/x.slp", True),
        ("https://slp.sh/4M16VD/labels.slp", True),
        ("s3://bucket/key.slp", True),
        ("gs://bucket/key.slp", True),
        ("gcs://bucket/key.slp", True),
        ("az://container/key.slp", True),
        ("abfs://container/key.slp", True),
        # Unsupported scheme.
        ("ftp://host/x.slp", False),
        ("file:///tmp/x.slp", False),
        # Edge cases.
        ("", False),
    ],
)
def test_is_url_boundaries(value, expected):
    """_is_url distinguishes remote URLs from local/Windows paths."""
    assert _is_url(value) is expected


def test_is_url_case_insensitive():
    """Scheme matching is case-insensitive."""
    assert _is_url("S3://bucket/key") is True
    assert _is_url("HTTPS://host/x") is True
    assert _is_url("Https://host/x") is True


def test_is_url_pathlike():
    """os.PathLike inputs are handled and treated as local paths."""
    assert _is_url(Path("C:/Users/foo.slp")) is False
    assert _is_url(Path("labels.slp")) is False


def test_is_url_non_string_non_pathlike():
    """Non str/PathLike inputs return False rather than raising."""
    assert _is_url(123) is False  # type: ignore[arg-type]
    assert _is_url(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _redact_url
# ---------------------------------------------------------------------------


def test_redact_url_userinfo():
    """Userinfo (user:pass@) is replaced with ***:***."""
    out = _redact_url("https://user:pass@host.example/path/labels.slp")
    assert "user" not in out
    assert "pass" not in out
    assert "***:***@host.example" in out
    assert out.startswith("https://")
    assert out.endswith("/path/labels.slp")


def test_redact_url_token_query_params():
    """Token-like query parameters have their values redacted."""
    out = _redact_url("https://host/x?token=SECRET&access_token=ABC&sig=XYZ&keep=ok")
    assert "SECRET" not in out
    assert "ABC" not in out
    assert "XYZ" not in out
    # Non-sensitive params are preserved.
    assert "keep=ok" in out


def test_redact_url_no_credentials_unchanged():
    """A URL with no credentials is returned essentially unchanged."""
    url = "https://host.example/path/labels.slp"
    assert _redact_url(url) == url


def test_redact_url_userinfo_and_token_combined():
    """Both userinfo and token query params are redacted together."""
    out = _redact_url("https://u:p@host/x?x-amz-security-token=TOK&sas=SAS")
    assert "u:p" not in out
    assert "TOK" not in out
    assert "SAS" not in out
    assert "***:***@host" in out


# ---------------------------------------------------------------------------
# _identify_magic
# ---------------------------------------------------------------------------


def test_identify_magic_hdf5():
    """The HDF5 superblock signature classifies as hdf5."""
    assert _identify_magic(b"\x89HDF\r\n\x1a\n\x00\x00\x00\x00") == "hdf5"


def test_identify_magic_json_object():
    """A leading '{' classifies as json."""
    assert _identify_magic(b'{"version": 1}') == "json"


def test_identify_magic_json_array():
    """A leading '[' classifies as json."""
    assert _identify_magic(b"[1, 2, 3]") == "json"


def test_identify_magic_zip():
    r"""The PK\x03\x04 signature classifies as zip."""
    assert _identify_magic(b"PK\x03\x04rest-of-zip") == "zip"


def test_identify_magic_csv():
    """ASCII-printable bytes with a comma classify as csv."""
    assert _identify_magic(b"a,b,c\n1,2,3") == "csv"


def test_identify_magic_garbage_unknown():
    """Binary garbage classifies as unknown."""
    assert _identify_magic(b"\x00\x01\x02\xff\xfe") == "unknown"


def test_identify_magic_empty_unknown():
    """Empty head is unknown (not spuriously csv)."""
    assert _identify_magic(b"") == "unknown"


def test_identify_magic_ascii_no_comma_unknown():
    """ASCII without a comma is not csv."""
    assert _identify_magic(b"hello world\n") == "unknown"


# ---------------------------------------------------------------------------
# RemoteIOError
# ---------------------------------------------------------------------------


def test_remote_io_error_is_oserror():
    """RemoteIOError subclasses OSError for graceful degradation."""
    assert issubclass(RemoteIOError, OSError)


def test_remote_io_error_carries_status_and_redacted_url():
    """RemoteIOError stores the status and a redacted URL."""
    err = RemoteIOError(
        "file not found",
        url="https://user:pass@host/x?token=SECRET",
        status=404,
    )
    assert err.status == 404
    assert err.url is not None
    assert "user" not in err.url
    assert "pass" not in err.url
    assert "SECRET" not in err.url
    msg = str(err)
    assert "status=404" in msg
    assert "SECRET" not in msg
    assert "pass" not in msg


def test_remote_io_error_no_url_no_status():
    """RemoteIOError with neither url nor status has a clean message."""
    err = RemoteIOError("boom")
    assert err.url is None
    assert err.status is None
    assert str(err) == "boom"


def test_remote_io_error_stores_redacted_cause_summary():
    """A redacted cause summary is stored and surfaced without chaining."""
    err = RemoteIOError("wrap", cause_summary="ValueError: inner")
    # The raw exception is never chained (no credential leak via __cause__).
    assert err.__cause__ is None
    assert err.cause_summary == "ValueError: inner"
    assert "cause=ValueError: inner" in str(err)


def test_redacted_cause_summary_redacts_embedded_url():
    """A URL embedded in an exception message is redacted in the summary."""
    e = FileNotFoundError(
        "GET https://user:s3cr3t@host/missing.slp?token=TOPSECRET -> 404"
    )
    summary = _redacted_cause_summary(e)
    assert "s3cr3t" not in summary
    assert "TOPSECRET" not in summary
    assert summary.startswith("FileNotFoundError:")


def test_redacted_cause_summary_empty_message():
    """An exception with no message degrades to just its type name."""
    assert _redacted_cause_summary(ValueError()) == "ValueError"


# ---------------------------------------------------------------------------
# _raise_remote mapping
# ---------------------------------------------------------------------------


def _make_response_error(
    status: int, *, headers: dict[str, str] | None = None
) -> aiohttp.ClientResponseError:
    """Construct a real aiohttp.ClientResponseError with the given status.

    Args:
        status: HTTP status code to set on the error.
        headers: Optional response headers (e.g. ``{"Retry-After": "5"}``).
    """
    return aiohttp.ClientResponseError(
        request_info=None,
        history=(),
        status=status,
        message="err",
        headers=headers,
    )


def test_raise_remote_maps_404():
    """Aiohttp 404 maps to RemoteIOError(status=404)."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(_make_response_error(404), url="https://host/x")
    assert ei.value.status == 404
    assert "file not found" in str(ei.value)


def test_raise_remote_maps_416():
    """Aiohttp 416 maps to RemoteIOError(status=416)."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(_make_response_error(416), url="https://host/x")
    assert ei.value.status == 416
    assert "range past end of file" in str(ei.value)


def test_raise_remote_maps_generic_5xx():
    """An unmapped status maps to a generic RemoteIOError carrying the status."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(_make_response_error(503), url="https://host/x")
    assert ei.value.status == 503
    assert "HTTP 503" in str(ei.value)


def test_raise_remote_maps_connector_error():
    """aiohttp.ClientConnectorError maps to a connection-error RemoteIOError."""
    key = ConnectionKey("host.example", 443, True, None, None, None, None)
    cce = aiohttp.ClientConnectorError(key, OSError("dns fail"))
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(cce, url="https://host/x")
    assert ei.value.status is None
    assert "connection error" in str(ei.value)


def test_raise_remote_maps_timeout():
    """asyncio.TimeoutError maps to a timeout RemoteIOError."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(asyncio.TimeoutError(), url="https://host/x")
    assert ei.value.status is None
    assert "timeout" in str(ei.value)


def test_raise_remote_maps_payload_error():
    """aiohttp.ClientPayloadError maps to a truncated-body RemoteIOError."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(aiohttp.ClientPayloadError("short"), url="https://host/x")
    assert ei.value.status is None
    assert "truncated body" in str(ei.value)


def test_raise_remote_passthrough_remote_io_error():
    """An existing RemoteIOError is re-raised unchanged."""
    original = RemoteIOError("already wrapped", status=418)
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(original, url="https://host/x")
    assert ei.value is original


def test_raise_remote_unknown_exception():
    """An unrecognized exception is wrapped as an unexpected RemoteIOError."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(KeyError("nope"), url="https://host/x")
    assert ei.value.status is None
    assert "unexpected error" in str(ei.value)


def test_raise_remote_redacts_url_in_error():
    """Credentials in the URL are redacted in the raised error."""
    with pytest.raises(RemoteIOError) as ei:
        _raise_remote(
            _make_response_error(404),
            url="https://user:pass@host/x?token=SECRET",
        )
    assert "pass" not in str(ei.value)
    assert "SECRET" not in str(ei.value)


# ---------------------------------------------------------------------------
# _find_response_error / _status_from_exception (fsspec wrapper unwrapping)
# ---------------------------------------------------------------------------


def test_find_response_error_unwraps_cause_chain():
    """A ClientResponseError wrapped as __cause__ is recovered from the chain."""
    inner = _make_response_error(404)
    wrapper = FileNotFoundError("url")
    wrapper.__cause__ = inner
    assert _find_response_error(wrapper) is inner


def test_find_response_error_unwraps_context_chain():
    """A ClientResponseError reachable only via __context__ is recovered."""
    inner = _make_response_error(500)
    wrapper = OSError("boom")
    wrapper.__context__ = inner
    assert _find_response_error(wrapper) is inner


def test_find_response_error_none_when_absent():
    """An exception chain with no ClientResponseError returns None."""
    assert _find_response_error(ValueError("nope")) is None


def test_find_response_error_handles_cycle():
    """A cyclic cause/context chain does not loop forever."""
    a = ValueError("a")
    b = ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _find_response_error(a) is None


def test_status_from_exception_unwraps():
    """_status_from_exception returns the wrapped status, or None."""
    wrapper = FileNotFoundError("url")
    wrapper.__cause__ = _make_response_error(416)
    assert _status_from_exception(wrapper) == 416
    assert _status_from_exception(ValueError("x")) is None


# ---------------------------------------------------------------------------
# Retry backoff helpers
# ---------------------------------------------------------------------------


def test_retry_after_seconds_integer():
    """An integer Retry-After header is parsed to seconds."""
    err = _make_response_error(429, headers={"Retry-After": "5"})
    assert _retry_after_seconds(err) == 5.0


def test_retry_after_seconds_absent_header():
    """A response error without a Retry-After header returns None."""
    assert _retry_after_seconds(_make_response_error(429)) is None


def test_retry_after_seconds_non_integer_returns_none():
    """A non-integer (HTTP-date) Retry-After value falls back to None."""
    err = _make_response_error(429, headers={"Retry-After": "Wed, 21 Oct 2025 GMT"})
    assert _retry_after_seconds(err) is None


def test_retry_after_seconds_no_response_error_returns_none():
    """A non-HTTP exception has no Retry-After."""
    assert _retry_after_seconds(ValueError("x")) is None


def test_retry_sleep_seconds_exponential_backoff():
    """Without Retry-After, sleep grows exponentially from the base."""
    exc = _make_response_error(500)
    assert _retry_sleep_seconds(0, exc) == _RETRY_BACKOFF_BASE
    assert _retry_sleep_seconds(1, exc) == _RETRY_BACKOFF_BASE * 2
    assert _retry_sleep_seconds(2, exc) == _RETRY_BACKOFF_BASE * 4


def test_retry_sleep_seconds_honors_retry_after():
    """A Retry-After header overrides the exponential backoff."""
    exc = _make_response_error(429, headers={"Retry-After": "2"})
    assert _retry_sleep_seconds(5, exc) == 2.0


# ---------------------------------------------------------------------------
# _mark_cache_dir
# ---------------------------------------------------------------------------


def test_mark_cache_dir_creates_marker(tmp_path):
    """_mark_cache_dir creates the directory and writes the marker file."""
    target = tmp_path / "cache" / "nested"
    _mark_cache_dir(target)
    assert (target / _CACHE_MARKER_NAME).exists()


def test_mark_cache_dir_idempotent(tmp_path):
    """Calling _mark_cache_dir twice does not error and keeps one marker."""
    _mark_cache_dir(tmp_path)
    _mark_cache_dir(tmp_path)
    assert (tmp_path / _CACHE_MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# _require_package
# ---------------------------------------------------------------------------


def test_require_package_missing_raises_with_cloud_hint():
    """A genuinely-absent package raises ImportError with the cloud hint."""
    pkg = "definitely_not_installed_pkg_xyz_12345"
    with pytest.raises(ImportError) as ei:
        _require_package(pkg, scheme="s3")
    msg = str(ei.value)
    assert pkg in msg
    assert "sleap-io[cloud]" in msg
    assert "s3://" in msg


def test_require_package_present_no_raise():
    """An installed package imports without raising."""
    # 'json' is always present in the stdlib.
    _require_package("json", scheme="s3")


# ---------------------------------------------------------------------------
# _warn_if_old_aiohttp
# ---------------------------------------------------------------------------


def test_warn_if_old_aiohttp_warns_on_old_version():
    """An aiohttp version below the minimum triggers a RuntimeWarning."""
    with pytest.warns(RuntimeWarning, match="below sleap-io's minimum"):
        _warn_if_old_aiohttp("3.13.0")


def test_warn_if_old_aiohttp_silent_on_new_version():
    """A version at or above the minimum produces no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_old_aiohttp("3.13.5")
        _warn_if_old_aiohttp("4.0.0")


# ---------------------------------------------------------------------------
# clear_remote_cache
# ---------------------------------------------------------------------------


def test_clear_remote_cache_requires_explicit_storage():
    """clear_remote_cache requires an explicit cache_storage path."""
    with pytest.raises(RuntimeError) as ei:
        clear_remote_cache()
    assert "cache_storage" in str(ei.value)


def test_clear_remote_cache_refuses_without_marker(tmp_path):
    """clear_remote_cache refuses a directory lacking the marker file."""
    with pytest.raises(RuntimeError) as ei:
        clear_remote_cache(cache_storage=tmp_path)
    assert "marker" in str(ei.value).lower()


def test_clear_remote_cache_refuses_home_dir():
    """clear_remote_cache refuses to operate on $HOME (forbidden path)."""
    with pytest.raises(RuntimeError) as ei:
        clear_remote_cache(cache_storage=Path.home())
    assert "forbidden" in str(ei.value).lower()


def test_clear_remote_cache_deletes_only_cache_key_files(tmp_path):
    """Only files matching the cache-key pattern are deleted; others remain."""
    (tmp_path / _CACHE_MARKER_NAME).touch()
    # Cache-key-pattern files (sha-style hex hashes, optional .tags sidecar).
    cache_a = tmp_path / ("a" * 32)
    cache_b = tmp_path / ("b" * 64)
    cache_tags = tmp_path / (("c" * 40) + ".tags")
    for p in (cache_a, cache_b, cache_tags):
        p.write_bytes(b"x")
    # Unrelated files that must NOT be deleted.
    unrelated_named = tmp_path / "important.txt"
    unrelated_short = tmp_path / ("d" * 8)  # too short to match pattern
    unrelated_named.write_bytes(b"keep")
    unrelated_short.write_bytes(b"keep")
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    deleted = clear_remote_cache(cache_storage=tmp_path)

    assert deleted == 3
    assert not cache_a.exists()
    assert not cache_b.exists()
    assert not cache_tags.exists()
    assert unrelated_named.exists()
    assert unrelated_short.exists()
    assert subdir.exists()
    # Marker itself is preserved (does not match the cache-key pattern).
    assert (tmp_path / _CACHE_MARKER_NAME).exists()


def test_clear_remote_cache_honors_older_than(tmp_path):
    """older_than only deletes files older than the threshold."""
    (tmp_path / _CACHE_MARKER_NAME).touch()
    old_file = tmp_path / ("a" * 32)
    new_file = tmp_path / ("b" * 32)
    old_file.write_bytes(b"x")
    new_file.write_bytes(b"x")
    # Backdate old_file's mtime by 1 hour.
    old_time = time.time() - 3600
    os.utime(old_file, (old_time, old_time))

    deleted = clear_remote_cache(cache_storage=tmp_path, older_than=1800)

    assert deleted == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_clear_remote_cache_empty_dir_with_marker(tmp_path):
    """A marked dir with no cache files deletes nothing and returns 0."""
    (tmp_path / _CACHE_MARKER_NAME).touch()
    assert clear_remote_cache(cache_storage=tmp_path) == 0


# ---------------------------------------------------------------------------
# _strip_cross_origin_headers (the trace-hook decision, unit-tested directly)
# ---------------------------------------------------------------------------


def test_strip_cross_origin_headers_strips_on_different_origin():
    """Sensitive headers are removed when the origin changes."""
    headers = CIMultiDict(
        {
            "Authorization": "Bearer secret",
            "Cookie": "sid=abc",
            "Proxy-Authorization": "Basic xyz",
            "Accept": "application/json",
        }
    )
    _strip_cross_origin_headers(
        URL("https://a.example/x"), URL("https://b.example/x"), headers
    )
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert "Proxy-Authorization" not in headers
    # Non-sensitive headers are preserved.
    assert headers.get("Accept") == "application/json"


def test_strip_cross_origin_headers_keeps_on_same_origin():
    """Sensitive headers survive a same-origin redirect."""
    headers = CIMultiDict({"Authorization": "Bearer secret"})
    _strip_cross_origin_headers(
        URL("https://a.example/start"), URL("https://a.example/final"), headers
    )
    assert headers.get("Authorization") == "Bearer secret"


def test_strip_cross_origin_headers_noop_without_prev_url():
    """With no prior URL (prev_url=None) nothing is stripped."""
    headers = CIMultiDict({"Authorization": "Bearer secret"})
    _strip_cross_origin_headers(None, URL("https://b.example/x"), headers)
    assert headers.get("Authorization") == "Bearer secret"


def test_strip_cross_origin_headers_port_change_is_cross_origin():
    """A different port is a different origin, so headers are stripped."""
    headers = CIMultiDict({"Authorization": "Bearer secret"})
    _strip_cross_origin_headers(
        URL("http://host:8001/x"), URL("http://host:8002/x"), headers
    )
    assert "Authorization" not in headers


def test_strip_cross_origin_headers_noop_without_new_url():
    """With no resolvable target (new_url=None) nothing is stripped."""
    headers = CIMultiDict({"Authorization": "Bearer secret"})
    _strip_cross_origin_headers(URL("https://a.example/x"), None, headers)
    assert headers.get("Authorization") == "Bearer secret"


# ---------------------------------------------------------------------------
# _redirect_target (resolve a redirect's Location header against the source)
# ---------------------------------------------------------------------------


class _FakeRedirectResponse:
    """Minimal stand-in for an aiohttp redirect response with headers."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = CIMultiDict(headers)


def test_redirect_target_absolute_location():
    """An absolute Location header resolves to its own origin (cross-origin)."""
    prev = URL("https://a.example/start")
    resp = _FakeRedirectResponse({"Location": "https://b.example/final"})
    target = _redirect_target(prev, resp)
    assert target == URL("https://b.example/final")
    assert target.origin() != prev.origin()


def test_redirect_target_relative_location_is_same_origin():
    """A relative Location header resolves against the source (same origin)."""
    prev = URL("https://a.example/dir/start")
    resp = _FakeRedirectResponse({"Location": "/dir/final"})
    target = _redirect_target(prev, resp)
    assert target.origin() == prev.origin()


def test_redirect_target_missing_location_returns_none():
    """No Location header yields None (caller leaves headers untouched)."""
    prev = URL("https://a.example/start")
    assert _redirect_target(prev, _FakeRedirectResponse({})) is None


def test_redirect_target_no_response_returns_none():
    """A None response yields None."""
    assert _redirect_target(URL("https://a.example/start"), None) is None


def test_strip_cross_origin_via_redirect_target_strips():
    """End-to-end of our logic: cross-origin Location strips sensitive headers.

    Exercises ``_redirect_target`` + ``_strip_cross_origin_headers`` together
    independent of aiohttp's native redirect handling.
    """
    prev = URL("https://a.example/start")
    resp = _FakeRedirectResponse({"Location": "https://b.example/final"})
    headers = CIMultiDict({"Authorization": "Bearer secret", "Accept": "*/*"})
    _strip_cross_origin_headers(prev, _redirect_target(prev, resp), headers)
    assert "Authorization" not in headers
    assert headers.get("Accept") == "*/*"


def test_strip_cross_origin_via_redirect_target_keeps_same_origin():
    """Same-origin (relative) Location keeps sensitive headers."""
    prev = URL("https://a.example/dir/start")
    resp = _FakeRedirectResponse({"Location": "/dir/final"})
    headers = CIMultiDict({"Authorization": "Bearer secret"})
    _strip_cross_origin_headers(prev, _redirect_target(prev, resp), headers)
    assert headers.get("Authorization") == "Bearer secret"


# ---------------------------------------------------------------------------
# Network: cross-origin redirect credential stripping (S2)
#
# These drive real HTTP requests through open_url / _sniff_format against a
# loopback pytest-httpserver, exercising the _safe_get_client trace hook.
# ---------------------------------------------------------------------------


def test_strips_auth_on_cross_origin_redirect(httpserver_dual):
    """Authorization is dropped when a redirect crosses origins (S2).

    Server A 302-redirects to server B (a different port = different origin).
    The Authorization header must NOT reach B.
    """
    server_a, server_b = httpserver_dual
    auth_seen_at_b: list[str | None] = []

    def _b_handler(request):
        auth_seen_at_b.append(request.headers.get("Authorization"))
        return Response(_HDF5_BODY, status=200)

    server_b.expect_request("/labels.slp").respond_with_handler(_b_handler)
    b_url = server_b.url_for("/labels.slp")
    server_a.expect_request("/labels.slp").respond_with_response(
        Response(status=302, headers={"Location": b_url})
    )
    a_url = server_a.url_for("/labels.slp")

    fmt = _sniff_format(a_url, headers={"Authorization": "Bearer secret"})

    assert fmt == "hdf5"
    assert auth_seen_at_b, "redirect target B was never reached"
    assert all(a is None for a in auth_seen_at_b), (
        "Authorization leaked to the cross-origin redirect target"
    )


def test_strips_cookie_on_cross_origin_redirect(httpserver_dual):
    """Cookie is dropped when a redirect crosses origins (S2)."""
    server_a, server_b = httpserver_dual
    cookie_seen_at_b: list[str | None] = []

    def _b_handler(request):
        cookie_seen_at_b.append(request.headers.get("Cookie"))
        return Response(_HDF5_BODY, status=200)

    server_b.expect_request("/labels.slp").respond_with_handler(_b_handler)
    b_url = server_b.url_for("/labels.slp")
    server_a.expect_request("/labels.slp").respond_with_response(
        Response(status=302, headers={"Location": b_url})
    )
    a_url = server_a.url_for("/labels.slp")

    fmt = _sniff_format(a_url, headers={"Cookie": "session=abc123"})

    assert fmt == "hdf5"
    assert cookie_seen_at_b, "redirect target B was never reached"
    assert all(c is None for c in cookie_seen_at_b), (
        "Cookie leaked to the cross-origin redirect target"
    )


def test_same_origin_redirect_keeps_auth(httpserver):
    """Authorization survives a same-origin redirect (we do not over-strip)."""
    auth_seen_at_final: list[str | None] = []

    def _final_handler(request):
        auth_seen_at_final.append(request.headers.get("Authorization"))
        return Response(_HDF5_BODY, status=200)

    httpserver.expect_request("/final.slp").respond_with_handler(_final_handler)
    httpserver.expect_request("/start.slp").respond_with_response(
        Response(status=302, headers={"Location": httpserver.url_for("/final.slp")})
    )
    url = httpserver.url_for("/start.slp")

    fmt = _sniff_format(url, headers={"Authorization": "Bearer secret"})

    assert fmt == "hdf5"
    assert auth_seen_at_final, "redirect target was never reached"
    assert all(a == "Bearer secret" for a in auth_seen_at_final), (
        "Authorization was wrongly stripped on a same-origin redirect"
    )


# ---------------------------------------------------------------------------
# Network: credential redaction in errors (S4)
# ---------------------------------------------------------------------------


def test_credentials_redacted_in_errors(httpserver):
    """A 404 against a userinfo URL surfaces no credentials in RemoteIOError (S4)."""
    httpserver.expect_request("/missing.slp").respond_with_data("nope", status=404)
    # Inject userinfo + a token query param into the URL host part.
    plain = httpserver.url_for("/missing.slp")
    scheme, rest = plain.split("://", 1)
    creds_url = f"{scheme}://user:s3cr3t@{rest}?token=TOPSECRET"

    with pytest.raises(RemoteIOError) as exc_info:
        open_url(creds_url, retries=0)

    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message
    assert "user:" not in message
    # The redacted URL attribute is also scrubbed.
    assert exc_info.value.url is not None
    assert "s3cr3t" not in exc_info.value.url
    assert "TOPSECRET" not in exc_info.value.url

    # The FULL formatted traceback (which walks __cause__/__context__) must not
    # leak credentials either: the credential-bearing aiohttp/fsspec exception is
    # broken out of the displayed chain (raised `from None`).
    e = exc_info.value
    formatted = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    assert "s3cr3t" not in formatted
    assert "TOPSECRET" not in formatted
    # The explicit cause is dropped and the implicit context is suppressed so
    # the formatter never reaches the raw exception.
    assert e.__cause__ is None
    assert e.__suppress_context__ is True


# ---------------------------------------------------------------------------
# Network: HTTP-status mapping through open_url
# ---------------------------------------------------------------------------


def test_remote_io_error_maps_404(httpserver):
    """A 404 maps to RemoteIOError(status=404) through open_url."""
    httpserver.expect_request("/missing.slp").respond_with_data("nope", status=404)
    url = httpserver.url_for("/missing.slp")

    with pytest.raises(RemoteIOError) as exc_info:
        open_url(url, retries=0)

    assert exc_info.value.status == 404
    assert "file not found" in str(exc_info.value)


def test_remote_io_error_maps_416(httpserver):
    """A 416 maps to RemoteIOError(status=416) through open_url."""
    httpserver.expect_request("/short.slp").respond_with_data("x", status=416)
    url = httpserver.url_for("/short.slp")

    with pytest.raises(RemoteIOError) as exc_info:
        open_url(url, retries=0)

    assert exc_info.value.status == 416
    assert "range past end of file" in str(exc_info.value)


def test_remote_io_error_maps_500_after_retries(httpserver):
    """A persistent 500 is retried, then mapped to RemoteIOError(status=500).

    With ``retries=2`` there should be exactly 3 GET attempts (one initial plus
    two retries) before the error is surfaced.
    """
    get_count = {"n": 0}

    def _handler(request):
        if request.method == "GET":
            get_count["n"] += 1
        return Response("boom", status=500)

    httpserver.expect_request("/err.slp").respond_with_handler(_handler)
    url = httpserver.url_for("/err.slp")

    with pytest.raises(RemoteIOError) as exc_info:
        open_url(url, retries=2)

    assert exc_info.value.status == 500
    # 1 initial attempt + 2 retries = 3 GETs.
    assert get_count["n"] == 3, f"expected 3 GET attempts, saw {get_count['n']}"


def test_retries_honor_retry_after(httpserver):
    """A 429 with Retry-After is retried and the next attempt succeeds.

    A ``Retry-After: 0`` keeps the test sub-second while still routing through
    the Retry-After parsing/honoring path.
    """
    get_count = {"n": 0}

    def _handler(request):
        if request.method != "GET":
            return Response(status=200)
        get_count["n"] += 1
        if get_count["n"] == 1:
            return Response("slow down", status=429, headers={"Retry-After": "0"})
        return Response(_HDF5_BODY, status=200)

    httpserver.expect_request("/rate.slp").respond_with_handler(_handler)
    url = httpserver.url_for("/rate.slp")

    file_like = open_url(url, retries=3)
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()
    # First GET 429'd, second GET succeeded.
    assert get_count["n"] == 2


# ---------------------------------------------------------------------------
# Network: HEAD 405 fallback + sniff over HTTP
# ---------------------------------------------------------------------------


def test_head_405_fallback(httpserver):
    """A server that rejects HEAD with 405 still resolves via a ranged GET.

    ``_head_or_range_probe`` must return True even though HEAD is unsupported.
    """
    methods_seen: list[str] = []

    def _handler(request):
        methods_seen.append(request.method)
        if request.method == "HEAD":
            return Response(status=405)
        return Response(_HDF5_BODY, status=200)

    httpserver.expect_request("/probe.slp").respond_with_handler(_handler)
    url = httpserver.url_for("/probe.slp")

    assert _head_or_range_probe(url) is True
    # The probe issued a GET (fsspec uses GET for existence), proving it did not
    # give up on the HEAD 405.
    assert "GET" in methods_seen


def test_head_or_range_probe_false_on_404(httpserver):
    """A 404 makes the existence probe return False (not raise)."""
    httpserver.expect_request("/missing.slp").respond_with_data("no", status=404)
    url = httpserver.url_for("/missing.slp")

    assert _head_or_range_probe(url) is False


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"\x89HDF\r\n\x1a\n" + b"\x00" * 16, "hdf5"),
        (b'{"version": 1, "skeleton": []}', "json"),
        (b"[1, 2, 3, 4, 5, 6, 7, 8, 9]", "json"),
        (b"a,b,c,d\n1,2,3,4\n", "csv"),
        (b"PK\x03\x04rest-of-zip-archive", "zip"),
        (b"\x00\x01\x02\x03\xff\xfe\xfd\xfc", "unknown"),
    ],
)
def test_sniff_over_http(httpserver, body, expected):
    """`_sniff_format` fetches the first bytes over HTTP and classifies them."""
    httpserver.expect_request("/data").respond_with_data(
        body, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data")

    assert _sniff_format(url) == expected


def test_sniff_over_http_404_raises(httpserver):
    """`_sniff_format` maps an HTTP error to RemoteIOError with the status."""
    httpserver.expect_request("/missing").respond_with_data("no", status=404)
    url = httpserver.url_for("/missing")

    with pytest.raises(RemoteIOError) as exc_info:
        _sniff_format(url)

    assert exc_info.value.status == 404


def test_sniff_over_http_via_open_remote_h5(httpserver):
    """`open_remote_h5` opens a served HDF5 body via blockcache."""
    httpserver.expect_request("/data.slp").respond_with_handler(
        _make_range_handler(_HDF5_BODY)
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_remote_h5(url)
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()


# ---------------------------------------------------------------------------
# Cloud-scheme filesystem construction + invalid stream mode
# ---------------------------------------------------------------------------


def test_build_fsspec_filesystem_cloud_scheme():
    """A cloud scheme routes through `_ensure_cloud_extra` + `fsspec.filesystem`.

    CI installs the cloud extras, so `s3` resolves to an `s3fs` filesystem
    rather than raising.
    """
    import s3fs

    fs = _build_fsspec_filesystem("s3")
    assert isinstance(fs, s3fs.S3FileSystem)


def test_open_url_invalid_stream_mode_raises_value_error(httpserver):
    """`open_url` raises ValueError (not RemoteIOError) for a bad stream_mode."""
    httpserver.expect_request("/data.slp").respond_with_data(_HDF5_BODY)
    url = httpserver.url_for("/data.slp")

    with pytest.raises(ValueError, match="Invalid stream_mode"):
        open_url(url, stream_mode="not-a-real-mode")


# ---------------------------------------------------------------------------
# Network: open_url stream modes + filesystem builder over HTTP
# ---------------------------------------------------------------------------


def test_build_fsspec_filesystem_http_sets_identity_encoding(httpserver):
    """`_build_fsspec_filesystem` sets Accept-Encoding: identity on requests (S5)."""
    encodings_seen: list[str | None] = []

    def _handler(request):
        encodings_seen.append(request.headers.get("Accept-Encoding"))
        return Response(_HDF5_BODY, status=200)

    httpserver.expect_request("/data").respond_with_handler(_handler)
    url = httpserver.url_for("/data")

    fs = _build_fsspec_filesystem("https", headers={"X-Custom": "1"})
    with fs.open(url, mode="rb") as f:
        assert f.read(8) == b"\x89HDF\r\n\x1a\n"

    assert encodings_seen
    assert all(enc == "identity" for enc in encodings_seen)


def test_identity_headers_user_cannot_override_accept_encoding():
    """User-supplied Accept-Encoding (any casing) cannot weaken identity (S5)."""
    # Exact-case override.
    assert _identity_headers({"Accept-Encoding": "gzip"}) == {
        "Accept-Encoding": "identity"
    }
    # Different casing (aiohttp coalesces header keys case-insensitively).
    merged = _identity_headers({"accept-encoding": "gzip, br", "X-Custom": "1"})
    assert merged["Accept-Encoding"] == "identity"
    assert "accept-encoding" not in merged
    assert merged["X-Custom"] == "1"
    # None headers still yields the identity guarantee.
    assert _identity_headers(None) == {"Accept-Encoding": "identity"}


def test_http_inner_options_forces_identity_encoding():
    """`_http_inner_options` forces identity even if the user passes gzip (S5)."""
    opts = _http_inner_options({"Accept-Encoding": "gzip"})
    assert opts["client_kwargs"]["headers"]["Accept-Encoding"] == "identity"


def test_open_url_download_mode_returns_bytesio(httpserver):
    """`stream_mode='download'` returns an in-memory BytesIO with the full body."""
    httpserver.expect_request("/data.slp").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(url, stream_mode="download")
    data = file_like.read()
    assert data == _HDF5_BODY


def test_open_url_cache_mode_writes_marker(httpserver, tmp_path):
    """`stream_mode='cache'` (simplecache) writes the sleap-io cache marker."""
    httpserver.expect_request("/data.slp").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(url, stream_mode="cache", cache_storage=tmp_path)
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()
    assert (tmp_path / _CACHE_MARKER_NAME).exists()


def test_open_url_filecache_mode_writes_marker(httpserver, tmp_path):
    """`stream_mode='filecache'` writes the marker and serves the content."""
    httpserver.expect_request("/data.slp").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(
        url, stream_mode="filecache", cache_storage=tmp_path, cache_expiry=10
    )
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()
    assert (tmp_path / _CACHE_MARKER_NAME).exists()


def test_open_url_cache_mode_default_storage(httpserver):
    """`stream_mode='cache'` without an explicit cache_storage uses fsspec's default."""
    httpserver.expect_request("/data.slp").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(url, stream_mode="cache")
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()


def test_open_url_filecache_mode_default_expiry(httpserver):
    """`stream_mode='filecache'` without cache_storage/cache_expiry serves content."""
    httpserver.expect_request("/data.slp").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(url, stream_mode="filecache")
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()


def test_open_url_blockcache_reads_body(httpserver):
    """`stream_mode='auto'` -> blockcache reads the served body via Range GETs.

    Uses a Range-capable handler because fsspec's blockcache relies on ranged
    reads.
    """
    httpserver.expect_request("/data.slp").respond_with_handler(
        _make_range_handler(_HDF5_BODY)
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(url)  # auto -> blockcache
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()


def test_open_url_blockcache_explicit_max_blocks(httpserver):
    """Explicit `block_size`/`max_blocks` are accepted by the blockcache open."""
    httpserver.expect_request("/data.slp").respond_with_handler(
        _make_range_handler(_HDF5_BODY)
    )
    url = httpserver.url_for("/data.slp")

    file_like = open_url(
        url, stream_mode="blockcache", block_size=1 << 16, max_blocks=4
    )
    try:
        assert file_like.read(8) == b"\x89HDF\r\n\x1a\n"
    finally:
        file_like.close()


# ---------------------------------------------------------------------------
# Google Drive: URL detection + file-id parsing (pure, no server)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://drive.google.com/file/d/ABC123/view", True),
        ("https://drive.google.com/uc?id=ABC123&export=download", True),
        ("https://docs.google.com/uc?id=ABC123", True),
        ("https://DRIVE.GOOGLE.COM/file/d/ABC/view", True),
        ("https://example.com/file/d/ABC/view", False),
        ("https://storage.googleapis.com/bucket/x.slp", False),
        ("s3://bucket/key.slp", False),
        ("not a url at all", False),
    ],
)
def test_is_gdrive_url(url, expected):
    """`_is_gdrive_url` recognizes drive/docs hostnames only."""
    assert _is_gdrive_url(url) is expected


@pytest.mark.parametrize(
    "url,expected_id",
    [
        # /file/d/<ID>/view (+ ?usp=sharing in the query).
        ("https://drive.google.com/file/d/FILEID/view", "FILEID"),
        ("https://drive.google.com/file/d/FILEID/view?usp=sharing", "FILEID"),
        ("https://drive.google.com/file/d/FILEID/edit", "FILEID"),
        # /file/u/<n>/d/<ID>/view per-account variant.
        ("https://drive.google.com/file/u/0/d/FILEID/view", "FILEID"),
        # id= query param, both orderings + /open?id=.
        ("https://drive.google.com/uc?id=FILEID&export=download", "FILEID"),
        ("https://drive.google.com/uc?export=download&id=FILEID", "FILEID"),
        ("https://drive.google.com/open?id=FILEID", "FILEID"),
    ],
)
def test_parse_gdrive_extracts_file_id(url, expected_id):
    """`_parse_gdrive` extracts the file ID from every supported URL shape."""
    file_id, is_folder = _parse_gdrive(url)
    assert file_id == expected_id
    assert is_folder is False


@pytest.mark.parametrize(
    "url",
    [
        "https://drive.google.com/drive/folders/FOLDERID",
        "https://drive.google.com/folders/FOLDERID",
        "https://drive.google.com/drive/folders/FOLDERID?usp=sharing",
    ],
)
def test_parse_gdrive_folder_raises(url):
    """Folder share links raise a clear ValueError."""
    with pytest.raises(ValueError, match="folder URLs are not supported"):
        _parse_gdrive(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://drive.google.com/",
        "https://drive.google.com/something/else",
        "https://drive.google.com/file/d/ABC",  # missing /view|/edit suffix
    ],
)
def test_parse_gdrive_unparseable_raises(url):
    """A URL with no recoverable file ID raises a clear ValueError."""
    with pytest.raises(ValueError, match="Could not parse a Google Drive file ID"):
        _parse_gdrive(url)


def test_parse_gdrive_redacts_credentials_in_error():
    """A token in an unparseable URL is redacted in the raised error."""
    with pytest.raises(ValueError) as ei:
        _parse_gdrive("https://drive.google.com/nope?token=SECRET")
    assert "SECRET" not in str(ei.value)


# ---------------------------------------------------------------------------
# Google Drive: confirmation-page scraping (pure, no server)
# ---------------------------------------------------------------------------


def test_url_from_confirmation_download_form_merges_params():
    """The #download-form action is merged with all hidden inputs."""
    html = (
        '<html><body><form id="download-form" '
        'action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="hidden" name="id" value="FILEID">'
        '<input type="hidden" name="export" value="download">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="u-1-2-3">'
        "</form></body></html>"
    )
    url = _url_from_confirmation(html)
    parsed = urllib.parse.urlparse(url)
    assert parsed.netloc == "drive.usercontent.google.com"
    assert parsed.path == "/download"
    params = dict(urllib.parse.parse_qsl(parsed.query))
    assert params == {
        "id": "FILEID",
        "export": "download",
        "confirm": "t",
        "uuid": "u-1-2-3",
    }


def test_url_from_confirmation_download_form_skips_nameless_input():
    """Hidden inputs without a `name` attribute are ignored, not crashed on."""
    html = (
        '<form id="download-form" '
        'action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="hidden" value="orphan">'  # no name= attribute
        '<input type="hidden" name="id" value="FILEID">'
        "</form>"
    )
    url = _url_from_confirmation(html)
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert params == {"id": "FILEID"}


def test_url_from_confirmation_href_variant():
    """The small-file href variant is absolutized and unescaped."""
    html = '<a href="/uc?export=download&amp;id=ABC&amp;confirm=t">Download</a>'
    url = _url_from_confirmation(html)
    assert url == "https://docs.google.com/uc?export=download&id=ABC&confirm=t"


def test_url_from_confirmation_json_variant():
    r"""The "downloadUrl":"…" JSON variant unescapes =/&."""
    html = r'window.x = {"downloadUrl":"https://h/p?a=1&b=2=x"};'
    url = _url_from_confirmation(html)
    assert url == "https://h/p?a=1&b=2=x"


def test_url_from_confirmation_quota_error_raises():
    """A uc-error-subcaption quota page raises a clear RemoteIOError."""
    html = (
        '<p class="uc-error-subcaption">Too many users have viewed or '
        "downloaded this file recently.</p>"
    )
    with pytest.raises(RemoteIOError, match="Google Drive refused the download"):
        _url_from_confirmation(html)


def test_url_from_confirmation_no_link_raises():
    """A page with no recognizable download link raises ValueError."""
    with pytest.raises(ValueError, match="Could not find a Google Drive"):
        _url_from_confirmation("<html><body>nothing here</body></html>")


def test_url_from_confirmation_href_precedence_over_form():
    """When both an href and a form are present, the href wins (gdown order)."""
    html = (
        '<a href="/uc?export=download&amp;id=HREFID">link</a>'
        '<form id="download-form" action="https://drive.usercontent.google.com/'
        'download"><input name="id" value="FORMID"></form>'
    )
    url = _url_from_confirmation(html)
    assert "HREFID" in url
    assert "FORMID" not in url


# ---------------------------------------------------------------------------
# Google Drive: end-to-end two-hop resolution against pytest-httpserver
#
# The resolver hardcodes the start host (drive.google.com); these tests repoint
# it at the loopback test server by overriding the `_UC_URL_TEMPLATE` constant
# (the documented test seam) so NO request ever reaches real Google Drive.
# ---------------------------------------------------------------------------


def _serve_gdrive_two_hop(
    httpserver,
    *,
    file_bytes: bytes,
    set_cookie: str | None = None,
    cookie_seen: list[str | None] | None = None,
):
    """Wire a two-hop Drive flow (`/uc` interstitial -> `/download`) on a server.

    Args:
        httpserver: The `pytest-httpserver` instance to configure.
        file_bytes: The bytes served at the `/download` (usercontent) hop.
        set_cookie: If given, a `Set-Cookie` value emitted by the `/uc` hop.
        cookie_seen: If given, the `Cookie` header value received at `/download`
            is appended to this list (to assert cookie carry across the hop).

    Returns:
        The `/uc?id=...` start URL (a Drive-looking template is installed on
        `_gdrive._UC_URL_TEMPLATE` separately by the caller via the fixture).
    """
    download_url = httpserver.url_for("/download")
    form = (
        f'<html><body><form id="download-form" action="{download_url}" '
        'method="get">'
        '<input type="hidden" name="id" value="FILEID">'
        '<input type="hidden" name="export" value="download">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="u-1-2-3">'
        "</form></body></html>"
    )

    def _uc_handler(request):
        resp = Response(form, status=200, content_type="text/html")
        if set_cookie is not None:
            resp.headers["Set-Cookie"] = set_cookie
        return resp

    def _download_handler(request):
        if cookie_seen is not None:
            cookie_seen.append(request.headers.get("Cookie"))
        resp = Response(file_bytes, status=200)
        resp.headers["Content-Disposition"] = 'attachment; filename="data.bin"'
        return resp

    httpserver.expect_request("/uc").respond_with_handler(_uc_handler)
    httpserver.expect_request("/download").respond_with_handler(_download_handler)


@pytest.fixture
def gdrive_uc_template(monkeypatch, httpserver):
    """Repoint the Drive resolver's start URL at the loopback test server.

    Overrides the `_gdrive._UC_URL_TEMPLATE` module constant so
    `_open_gdrive(...)` begins its GET loop at the local `pytest-httpserver`'s
    `/uc` endpoint instead of `https://drive.google.com/uc`. This is the
    documented test seam from the resolver spec; no real Drive request is made.

    Yields:
        The httpserver (already installed as the resolver start host).
    """
    import sleap_io.io._gdrive as gdrive_mod

    template = httpserver.url_for("/uc") + "?id={file_id}"
    monkeypatch.setattr(gdrive_mod, "_UC_URL_TEMPLATE", template)
    return httpserver


def test_open_gdrive_two_hop_download_form(gdrive_uc_template):
    """`_open_gdrive` follows the interstitial #download-form to the bytes."""
    _serve_gdrive_two_hop(gdrive_uc_template, file_bytes=_HDF5_BODY)

    bio = _open_gdrive("https://drive.google.com/file/d/FILEID/view")
    assert bio.read() == _HDF5_BODY


def test_open_gdrive_carries_cookie_across_hop(gdrive_uc_template):
    """The interstitial Set-Cookie is echoed back on the second (download) hop."""
    cookie_seen: list[str | None] = []
    _serve_gdrive_two_hop(
        gdrive_uc_template,
        file_bytes=_HDF5_BODY,
        set_cookie="download_warning=xyz; Path=/",
        cookie_seen=cookie_seen,
    )

    bio = _open_gdrive("https://drive.google.com/file/d/FILEID/view")
    assert bio.read() == _HDF5_BODY
    assert cookie_seen, "download hop was never reached"
    assert cookie_seen[0] is not None
    assert "download_warning=xyz" in cookie_seen[0]


def test_open_gdrive_direct_content_disposition_first_hop(gdrive_uc_template):
    """A first hop that already carries Content-Disposition returns immediately."""

    def _handler(request):
        resp = Response(_HDF5_BODY, status=200)
        resp.headers["Content-Disposition"] = 'attachment; filename="small.bin"'
        return resp

    gdrive_uc_template.expect_request("/uc").respond_with_handler(_handler)

    bio = _open_gdrive("https://drive.google.com/uc?id=FILEID")
    assert bio.read() == _HDF5_BODY


def test_open_gdrive_non_html_first_hop(gdrive_uc_template):
    """A non-HTML first response is treated as the direct file."""
    gdrive_uc_template.expect_request("/uc").respond_with_data(
        _HDF5_BODY, content_type="application/octet-stream"
    )

    bio = _open_gdrive("https://drive.google.com/uc?id=FILEID")
    assert bio.read() == _HDF5_BODY


def test_open_gdrive_quota_page_raises(gdrive_uc_template):
    """A quota interstitial surfaces as a RemoteIOError, not file bytes."""
    quota_html = (
        '<html><body><p class="uc-error-subcaption">Too many users have '
        "viewed or downloaded this file recently.</p></body></html>"
    )
    gdrive_uc_template.expect_request("/uc").respond_with_data(
        quota_html, content_type="text/html"
    )

    with pytest.raises(RemoteIOError, match="Google Drive refused the download"):
        _open_gdrive("https://drive.google.com/file/d/FILEID/view")


def test_open_gdrive_folder_url_raises():
    """`_open_gdrive` on a folder URL raises ValueError before any request."""
    with pytest.raises(ValueError, match="folder URLs are not supported"):
        _open_gdrive("https://drive.google.com/drive/folders/FID")


def test_open_gdrive_http_error_maps_to_remote_io_error(gdrive_uc_template):
    """A non-2xx first hop is normalized to a RemoteIOError with the status."""
    gdrive_uc_template.expect_request("/uc").respond_with_data("nope", status=404)

    with pytest.raises(RemoteIOError) as ei:
        _open_gdrive("https://drive.google.com/file/d/FILEID/view")
    assert ei.value.status == 404


def test_open_gdrive_non_convergent_raises(gdrive_uc_template):
    """If every hop returns interstitial HTML, resolution fails after the bound.

    The `/uc` interstitial's #download-form points back at `/uc` itself (a
    self-referential interstitial), so the loop runs out its hop budget and
    raises rather than spinning forever.
    """
    uc_url = gdrive_uc_template.url_for("/uc")
    loop_html = (
        f'<form id="download-form" action="{uc_url}" method="get">'
        '<input type="hidden" name="id" value="FILEID"></form>'
    )

    def _handler(request):
        return Response(loop_html, status=200, content_type="text/html")

    gdrive_uc_template.expect_request("/uc").respond_with_handler(_handler)

    with pytest.raises(RemoteIOError, match="did not converge"):
        _open_gdrive("https://drive.google.com/file/d/FILEID/view")


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("SLEAP_IO_RUN_LIVE"),
    reason="live-internet test; set SLEAP_IO_RUN_LIVE=1 to enable",
)
def test_sniff_over_http_live():
    """Optional: sniff a real public `.slp` URL.

    Skipped by default (CI has no guaranteed network); opt in with
    ``SLEAP_IO_RUN_LIVE=1`` and select ``-m live``.
    """
    assert _sniff_format("https://slp.sh/4M16VD/labels.slp") == "hdf5"
