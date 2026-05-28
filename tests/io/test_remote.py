"""Non-network unit tests for sleap_io.io._remote.

Network and security (cross-origin redirect) tests live in a later step; these
tests exercise only the pure / local-filesystem logic.
"""

import asyncio
import os
import time
import warnings
from pathlib import Path

import aiohttp
import pytest
from aiohttp.client_reqrep import ConnectionKey

from sleap_io.io._remote import (
    _CACHE_MARKER_NAME,
    RemoteIOError,
    _identify_magic,
    _is_url,
    _raise_remote,
    _redact_url,
    _require_package,
    _warn_if_old_aiohttp,
    clear_remote_cache,
)

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


def test_remote_io_error_preserves_cause():
    """The underlying cause is preserved as __cause__."""
    cause = ValueError("inner")
    err = RemoteIOError("wrap", cause=cause)
    assert err.__cause__ is cause


# ---------------------------------------------------------------------------
# _raise_remote mapping
# ---------------------------------------------------------------------------


def _make_response_error(status: int) -> aiohttp.ClientResponseError:
    """Construct a real aiohttp.ClientResponseError with the given status."""
    return aiohttp.ClientResponseError(
        request_info=None, history=(), status=status, message="err"
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
