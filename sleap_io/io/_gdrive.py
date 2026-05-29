"""Google Drive share-link resolution for remote loading.

Google Drive share URLs (``https://drive.google.com/file/d/<ID>/view``,
``https://drive.google.com/uc?id=<ID>``, ``…/open?id=<ID>``) do not point at the
file bytes directly: Drive serves an HTML "interstitial" confirmation page for
anything large enough to skip its virus scan, and the real download URL lives
inside that page's ``#download-form`` (or, for small files, behind a
``Content-Disposition`` header on the first response). This module ports the
minimal subset of `gdown <https://github.com/wkentaro/gdown>`_'s resolution
logic needed by sleap-io, using only the standard library
(:mod:`html.parser` + :mod:`re`) for HTML parsing -- no ``gdown`` or
``beautifulsoup4`` dependency.

The resolver (:func:`_open_gdrive`) is **notebook-safe**: it performs its HTTP
GETs on fsspec's background event-loop thread via :func:`fsspec.asyn.sync`, so
it does not call :func:`asyncio.run` (which raises "event loop is already
running" inside a Jupyter kernel). Because Drive rejects ``HEAD`` with 405
(which breaks fsspec's lazy size-probe) and its download quota can interrupt a
transfer mid-stream, the resolver fully prefetches the resolved bytes within a
single cookie-carrying session and returns an :class:`io.BytesIO`.
"""

from __future__ import annotations

import html
import io
import re
import urllib.parse
from html.parser import HTMLParser

from sleap_io.io._remote import (
    _GDRIVE_HOSTS,
    RemoteIOError,
    _is_gdrive_url,
    _redact_url,
)

# Re-export the pure-stdlib Drive detection helpers (defined in ``_remote`` so
# they are importable without this heavier resolver module). Keeping them in the
# ``_gdrive`` namespace preserves the import sites in ``main``/``video_reading``.
__all__ = ["_GDRIVE_HOSTS", "_is_gdrive_url"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base used to absolutize a relative ``href="/uc?export=download…"`` link.
_DOCS_BASE = "https://docs.google.com"

#: Browser-like User-Agent. Drive serves the interstitial HTML (rather than a
#: terse bot response) when it believes a browser is asking.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Template for the initial resolution URL given a file ID. Exposed as a
#: module-level constant so tests can repoint the resolver at a local
#: ``pytest-httpserver`` without touching real Google Drive.
_UC_URL_TEMPLATE = "https://drive.google.com/uc?id={file_id}"

#: Maximum number of interstitial hops to follow before giving up.
_MAX_HOPS = 4

#: Path regex that yields a Drive file ID. Accepts the ``/file/d/<ID>/…`` share
#: path with an optional ``/u/<n>/`` per-account prefix and an optional trailing
#: action segment (``/view``, ``/edit``, ``/preview``, or none). Using
#: ``[^/]+`` for the ID segment keeps multi-segment paths from being misparsed.
_FILE_PATH_RE = re.compile(
    r"^/file/(?:u/[0-9]+/)?d/(?P<id>[^/]+)(?:/(?:edit|view|preview))?/?$"
)

#: Marks a folder share link (unsupported -- sleap-io loads single files).
_FOLDER_RE = re.compile(r"/(?:drive/)?folders/")

#: ``href="/uc?export=download…"`` small-file variant.
_HREF_RE = re.compile(r'href="(/uc\?export=download[^"]+)"')

#: ``"downloadUrl":"…"`` JSON variant (escaped ``=``/``&``).
_DOWNLOAD_URL_RE = re.compile(r'"downloadUrl":"([^"]+)"')

#: ``<p class="uc-error-subcaption">…</p>`` quota / permission error caption.
_ERROR_SUBCAPTION_RE = re.compile(
    r'<p class="uc-error-subcaption">(.*?)</p>', re.DOTALL
)


# ---------------------------------------------------------------------------
# URL detection + file-id parsing
#
# ``_is_gdrive_url`` / ``_GDRIVE_HOSTS`` are imported from ``_remote`` above and
# re-exported here (see ``__all__``).
# ---------------------------------------------------------------------------


def _parse_gdrive(url: str) -> tuple[str, bool]:
    """Extract the file ID from a Google Drive share URL.

    Supports the ``id=`` query parameter (``/open?id=…``,
    ``/uc?id=…&export=download``) and the ``/file/d/<ID>/…`` path form. The
    trailing action segment is optional and may be ``/view``, ``/edit``,
    ``/preview``, or absent (a bare ``/file/d/<ID>``); the ``/file/u/<n>/d/<ID>``
    per-account prefix is also accepted.

    Args:
        url: A Google Drive share URL.

    Returns:
        A ``(file_id, is_folder)`` tuple. ``is_folder`` is always False on
        return because folder URLs raise (see below); it is part of the
        signature for callers that want to branch before catching.

    Raises:
        ValueError: If the URL is a folder share link, or if no file ID can be
            parsed from it.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path

    if _FOLDER_RE.search(path):
        raise ValueError(
            "Google Drive folder URLs are not supported; pass a single-file "
            "share link of the form "
            "https://drive.google.com/file/d/<FILE_ID>/view "
            f"(got: {_redact_url(url)})."
        )

    # 1) ``id`` query parameter (covers /open?id=, /uc?id=…, both orderings).
    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0], False

    # 2) Path-based forms.
    match = _FILE_PATH_RE.match(path)
    if match:
        return match.group("id"), False

    raise ValueError(
        "Could not parse a Google Drive file ID from the URL; expected an "
        "'id=' query parameter or a '/file/d/<FILE_ID>/view' path "
        f"(got: {_redact_url(url)})."
    )


# ---------------------------------------------------------------------------
# Interstitial HTML scraping (stdlib only -- no bs4)
# ---------------------------------------------------------------------------


class _DownloadFormParser(HTMLParser):
    """Extract the ``#download-form`` action + hidden inputs from Drive HTML.

    Drive's large-file interstitial wraps the real download in a
    ``<form id="download-form" action="https://drive.usercontent.google.com/
    download" method="get">`` whose hidden ``<input>`` elements carry the
    ``id``/``export``/``confirm``/``uuid`` parameters that must be merged into
    the action's query string.

    Attributes:
        action: The form's ``action`` URL, or None if no download form was seen.
        params: Hidden input ``{name: value}`` pairs collected from the form.
    """

    def __init__(self) -> None:
        """Initialize the parser with no form found yet."""
        super().__init__(convert_charrefs=True)
        self.action: str | None = None
        self.params: dict[str, str] = {}
        self._in_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track the download form and collect its hidden inputs.

        Args:
            tag: The lowercased tag name.
            attrs: The tag's ``(name, value)`` attribute pairs.
        """
        attr = {k: v for k, v in attrs}
        if tag == "form" and attr.get("id") == "download-form":
            self._in_form = True
            self.action = attr.get("action")
        elif tag == "input" and self._in_form:
            name = attr.get("name")
            if name is not None:
                self.params[name] = attr.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        """Close the form-tracking state on the form's end tag.

        Args:
            tag: The lowercased tag name.
        """
        if tag == "form" and self._in_form:
            self._in_form = False


def _url_from_download_form(form_html: str) -> str | None:
    """Resolve the ``#download-form`` to a direct download URL, if present.

    Args:
        form_html: The interstitial HTML.

    Returns:
        The merged download URL (form action query + hidden inputs), or None if
        the HTML has no ``#download-form`` with an action.
    """
    parser = _DownloadFormParser()
    parser.feed(form_html)
    if not parser.action:
        return None

    parsed_action = urllib.parse.urlparse(parser.action)
    # Start from any params already on the action URL, then overlay the hidden
    # inputs (id/export/confirm/uuid) which take precedence.
    merged: dict[str, str] = dict(
        urllib.parse.parse_qsl(parsed_action.query, keep_blank_values=True)
    )
    merged.update(parser.params)
    new_query = urllib.parse.urlencode(merged)
    return urllib.parse.urlunparse(parsed_action._replace(query=new_query))


def _url_from_confirmation(confirmation_html: str) -> str:
    """Scrape the next download URL out of a Drive interstitial page.

    Ports gdown's ``download.py`` confirmation logic with the standard library.
    Tries, in order: the small-file ``href="/uc?export=download…"`` link, the
    large-file ``#download-form``, and the ``"downloadUrl":"…"`` JSON variant.
    If none match but the page carries a ``uc-error-subcaption`` (quota /
    permission failure), a clear :class:`~sleap_io.io._remote.RemoteIOError` is
    raised.

    Args:
        confirmation_html: The interstitial HTML returned by Drive.

    Returns:
        The next URL to GET in the resolution loop.

    Raises:
        RemoteIOError: If the page is a quota/permission error page.
        ValueError: If no download URL could be extracted from the page.
    """
    # (a) Small-file relative href.
    href_match = _HREF_RE.search(confirmation_html)
    if href_match:
        return _DOCS_BASE + href_match.group(1).replace("&amp;", "&")

    # (b) Large-file #download-form (the dominant current path).
    form_url = _url_from_download_form(confirmation_html)
    if form_url is not None:
        return form_url

    # (c) "downloadUrl":"…" JSON variant.
    json_match = _DOWNLOAD_URL_RE.search(confirmation_html)
    if json_match:
        return json_match.group(1).replace("\\u003d", "=").replace("\\u0026", "&")

    # (d) Quota / permission error page.
    error_match = _ERROR_SUBCAPTION_RE.search(confirmation_html)
    if error_match:
        caption = re.sub(r"<[^>]+>", "", error_match.group(1)).strip()
        caption = html.unescape(caption)
        raise RemoteIOError(
            "Google Drive refused the download: "
            f"{caption} "
            "(if this is a quota error, try again later; if it is a permission "
            "error, set the file's sharing to 'Anyone with the link')."
        )

    raise ValueError(
        "Could not find a Google Drive download link in the confirmation page; "
        "the file may be inaccessible or Drive's page format may have changed."
    )


# ---------------------------------------------------------------------------
# Notebook-safe resolution loop (runs on fsspec's event-loop thread)
# ---------------------------------------------------------------------------


async def _resolve_and_fetch(file_id: str, headers: dict[str, str] | None) -> bytes:
    """Run the Drive GET loop and return the resolved file bytes.

    This coroutine is driven from :func:`_open_gdrive` via
    :func:`fsspec.asyn.sync` on fsspec's background event loop, so it is safe to
    call from inside a running asyncio loop (e.g. a Jupyter kernel). A single
    :class:`aiohttp.ClientSession` is used so Drive's interstitial cookies are
    carried across the hop automatically.

    Args:
        file_id: The Google Drive file ID.
        headers: Optional extra HTTP headers (merged over the browser default).

    Returns:
        The fully-downloaded file bytes.

    Raises:
        RemoteIOError: For quota/permission pages or if resolution does not
            converge on a downloadable response within :data:`_MAX_HOPS`.
        ValueError: If a confirmation page yields no download link.
    """
    import aiohttp

    request_headers = {"User-Agent": _BROWSER_UA, **(headers or {})}
    url = _UC_URL_TEMPLATE.format(file_id=file_id)

    # ``unsafe=True`` lets the jar keep cookies set by IP-address hosts (Drive
    # uses a domain so this is a no-op there, but it is needed to carry the
    # interstitial cookie across the hop against a loopback test server).
    cookie_jar = aiohttp.CookieJar(unsafe=True)
    async with aiohttp.ClientSession(
        headers=request_headers, cookie_jar=cookie_jar
    ) as session:
        for _ in range(_MAX_HOPS):
            async with session.get(url, allow_redirects=True) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                has_disposition = "Content-Disposition" in resp.headers
                is_html = "text/html" in content_type.lower()

                # A Content-Disposition header (or any non-HTML body) means we
                # have reached the file itself: read it fully in this session.
                if has_disposition or not is_html:
                    return await resp.read()

                # Otherwise this is the interstitial HTML; scrape the next URL.
                page = await resp.text()
            url = _url_from_confirmation(page)

    raise RemoteIOError(
        "Google Drive resolution did not converge on a downloadable file "
        f"within {_MAX_HOPS} hops; the link may be invalid or require sign-in.",
        url=url,
    )


def _open_gdrive(url: str, *, headers: dict[str, str] | None = None) -> io.BytesIO:
    """Resolve a Google Drive share URL and return its bytes as a BytesIO.

    The Drive download URL carries no file extension and Drive rejects ``HEAD``
    with 405, both of which break the lazy fsspec open path; the resolved bytes
    are therefore fully prefetched here (within one cookie-carrying session) and
    returned as an in-memory :class:`io.BytesIO`, mirroring ``open_url``'s
    ``stream_mode="download"`` shape.

    Args:
        url: A Google Drive / Docs share URL.
        headers: Optional extra HTTP headers to forward.

    Returns:
        An :class:`io.BytesIO` positioned at the start of the downloaded bytes.

    Raises:
        ValueError: For folder URLs, unparseable file IDs, or confirmation
            pages with no download link.
        RemoteIOError: For HTTP failures, quota/permission pages, or
            non-convergent resolution.
    """
    import fsspec.asyn

    file_id, _is_folder = _parse_gdrive(url)

    loop = fsspec.asyn.get_loop()
    try:
        data = fsspec.asyn.sync(loop, _resolve_and_fetch, file_id, headers)
    except (RemoteIOError, ValueError):
        raise
    except Exception as e:  # noqa: BLE001 - normalize transport errors
        from sleap_io.io._remote import _raise_remote

        _raise_remote(e, url=url)
    return io.BytesIO(data)
