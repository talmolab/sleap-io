"""Fixtures for testing that a URL's credentials stay out of saved files."""

from __future__ import annotations

import gc
import re
from functools import partial
from pathlib import Path

import h5py
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response


class GatedFiles:
    """Files on a local server that only answers a request carrying a credential.

    Any one of these gets through: ``?token=SECRET``, the header ``Authorization: Bearer
    HEADER-SECRET``, or the password ``me:PASSWORD-SECRET`` in the URL. Everything else
    gets a 401, so a file that reads from here has really used its credential.

    Attributes:
        server: The running ``pytest_httpserver`` server.
        refused: How many requests have been turned away so far.
    """

    def __init__(self, server):
        """Serve nothing yet from `server`."""
        self.server = server
        self.refused = 0

    def serve(self, route: str, path: str | Path) -> str:
        """Serve a local file at `route`.

        Args:
            route: The route to serve it at (e.g. ``"/labels.slp"``).
            path: The local file.

        Returns:
            Its URL, with the token.
        """
        body = Path(path).read_bytes()
        self.server.expect_request(route).respond_with_handler(
            partial(self._answer, body)
        )
        return f"{self.server.url_for(route)}?token=SECRET"

    def _answer(self, body: bytes, request) -> Response:
        """Answer a request, if it has what it takes, with Range support."""
        auth = request.authorization
        if not (
            request.args.get("token") == "SECRET"
            or request.headers.get("Authorization") == "Bearer HEADER-SECRET"
            or (auth and (auth.username, auth.password) == ("me", "PASSWORD-SECRET"))
        ):
            self.refused += 1
            return Response(b"", status=401)

        total = len(body)
        if request.method == "HEAD":
            response = Response(b"", status=200)
            response.headers["Content-Length"] = str(total)
        else:
            asked = re.match(r"bytes=(\d+)-(\d*)", request.headers.get("Range", ""))
            if asked:
                start = int(asked.group(1))
                end = min(int(asked.group(2) or total - 1), total - 1)
                response = Response(body[start : end + 1], status=206)
                response.headers["Content-Range"] = f"bytes {start}-{end}/{total}"
            else:
                response = Response(body, status=200)
        response.headers["Accept-Ranges"] = "bytes"
        return response


@pytest.fixture
def gated_files():
    """A local server that refuses any request without a credential.

    It answers on several threads. FFmpeg keeps its connection open with a video half
    read, which on a server with one thread leaves the next request waiting behind it.

    The garbage collector stays off for the test. Opening a URL with h5py holds h5py's
    lock while fsspec's thread does the reading, and if the collector runs on that
    thread and frees an h5py object left over from an earlier test, it waits for that
    same lock. Neither can go on, and the test hangs for good.
    """
    gc.collect()
    gc.disable()
    server = HTTPServer(host=HTTPServer.DEFAULT_LISTEN_HOST, port=0, threaded=True)
    server.start()
    try:
        yield GatedFiles(server)
    finally:
        server.clear()
        server.stop()
        gc.enable()
        gc.collect()


def _strings_in_hdf5(path: Path) -> list[bytes]:
    """Every string in an HDF5 file, read through HDF5 so compression can't hide one."""
    found: list[bytes] = []

    def visit(name, obj):
        found.append(name.encode())
        for key, value in obj.attrs.items():
            found.append(key.encode())
            found.append(repr(value).encode())
        if isinstance(obj, h5py.Dataset) and (
            obj.dtype.kind in "OSU" or obj.dtype.names
        ):
            found.append(repr(obj[()]).encode())

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found


@pytest.fixture
def saved_bytes_without():
    """A function that searches what a save wrote for a credential.

    It takes a file or a folder and the credentials to look for (``SECRET`` if none is
    given). It searches every file's raw bytes, since HDF5 leaves what it frees in the
    file, then every string in an HDF5 file through HDF5, then the names of the files.
    It returns the bytes it searched, so a test can check that what was masked is there
    to see.
    """

    def search(path: str | Path, *tokens: str) -> bytes:
        path = Path(path)
        tokens = tokens or ("SECRET",)
        paths = [path] if path.is_file() else sorted(path.rglob("*"))
        assert paths, f"Nothing was saved at {path}."
        saved = b""
        for token in tokens:
            for found in paths:
                name = found.relative_to(path).as_posix() if found != path else ""
                assert token not in name
                if not found.is_file():
                    continue
                data = found.read_bytes()
                saved += data
                assert token.encode() not in data, f"{token} is in {found.name}."
                if h5py.is_hdf5(found):
                    for string in _strings_in_hdf5(found):
                        assert token.encode() not in string
        return saved

    return search
