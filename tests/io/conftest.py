"""Shared fixtures for `tests/io`.

Provides the `httpserver_dual` fixture used by the remote-loading network tests
to exercise cross-origin redirects (two distinct origins).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pytest_httpserver import HTTPServer


@pytest.fixture
def httpserver_dual() -> Generator[tuple[HTTPServer, HTTPServer], None, None]:
    """Yield two `HTTPServer` instances bound to distinct ports (two origins).

    A request that 302-redirects from server ``a`` to a URL on server ``b``
    crosses origins (different port => different origin), which is what the
    cross-origin credential-stripping tests need. Both servers are stopped on
    teardown.

    Yields:
        A ``(a, b)`` tuple of started `HTTPServer` instances on the loopback
        host with OS-assigned (distinct) ports.
    """
    host = HTTPServer.DEFAULT_LISTEN_HOST
    a = HTTPServer(host=host, port=0)
    b = HTTPServer(host=host, port=0)
    a.start()
    b.start()
    try:
        yield a, b
    finally:
        a.clear()
        if a.is_running():
            a.stop()
        b.clear()
        if b.is_running():
            b.stop()
