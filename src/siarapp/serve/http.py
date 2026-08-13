# Vixen Intelligence c.2026
"""The socket half of ``siar-app serve``: a read-only window onto one output folder.

``GET``, ``HEAD`` and ``OPTIONS``. There is no ``do_POST``, no ``do_PUT``, no ``do_DELETE`` — the
read-only guarantee is not a check somewhere that could be got round, it is the absence of any
handler that could write, and nothing in this subpackage opens a file for writing.

The intended path is an ordinary ssh tunnel: the daemon binds ``127.0.0.1`` on the box that did the
scanning, and the reader forwards a port to their laptop. That is why the default bind is loopback
and why reaching it any other way needs ``--allow-remote`` rather than a warning nobody sees, and it
is why there is no TLS here — the tunnel already has it, and a self-signed certificate on a survey
box would train people to click through certificate warnings.

**The token is not optional.** It is minted per invocation and accepted three ways, which is one
more than looks necessary: ``?t=`` because an ``<img>`` or an ``<audio>`` element cannot send a
header, and ``X-Siar-Token`` / ``Authorization: Bearer`` because a ``fetch`` should not have to put
a secret in a URL. The shipped page uses both, so neither path can quietly rot.

Everything else worth knowing is a caching decision. The two root documents are rewritten while a
scan runs, so anything derived from them is ``no-store`` — a thirty-second-stale index of an
overnight run is a wrong index. Anything derived from one recording's own artefacts gets a weak
ETag from ``(mtime_ns, size)`` and a short ``max-age``, so scrolling a lane strip of ten thousand
thumbnails costs almost nothing while a ``--resume`` rerun still invalidates what it rewrote.
"""
from __future__ import annotations

import hmac
import json
import mimetypes
import os
import re
import secrets
import socket
import threading
from collections import OrderedDict
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from siarapp import __version__, docs
from siarapp.serve.folder import INDEX_MAX_LIMIT, ServedFolder
from siarapp.serve.preview import (
    PREVIEW_DEFAULT_BINS,
    PREVIEW_DEFAULT_WIDTH,
    PREVIEW_HEADER_BYTES,
    PREVIEW_MAGIC,
    PREVIEW_MAX_BINS,
    PREVIEW_MAX_WIDTH,
    encode_preview,
    preview_png,
    read_preview,
)

__all__ = [
    "DEFAULT_PORT",
    "ServeOptions",
    "build_server",
    "local_host_hint",
    "mint_token",
    "page_url",
    "serve_folder",
]

#: Port the daemon takes when nobody says otherwise. Unremarkable, above 1024, and unlikely to
#: collide with something a survey box is already running.
DEFAULT_PORT = 8420

#: The three files the page is made of. A fixed set, resolved by name through
#: :func:`siarapp.docs.local_web_path`, so a request never reaches a path join.
_PAGE_FILES = {
    "/": ("viewer.html", "text/html; charset=utf-8"),
    "/viewer.html": ("viewer.html", "text/html; charset=utf-8"),
    "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
}

#: Previews computed at once. Two, because each is a few thousand seeks and a browser opening a
#: folder can fire thirty at a time; the rest wait rather than turning the box into a fan heater.
_PREVIEW_SLOTS = 2

#: Encoded previews held in memory, newest first. Panning the same lane should not recompute it.
_PREVIEW_CACHE_MAX = 32

#: Bytes per chunk when streaming a recording. Big enough that a 900 MB file is not a million
#: syscalls, small enough that a cancelled download stops promptly.
_STREAM_CHUNK = 256 * 1024

#: ``Range: bytes=a-b``, the only form a browser's ``<audio>`` element ever sends.
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")

#: Response headers the ``X-Siar-*`` set adds, named for a cross-origin reader that would otherwise
#: be unable to see them.
_EXPOSED = (
    "Content-Range, Accept-Ranges, ETag, X-Siar-Width, X-Siar-Height, "
    "X-Siar-Duration-Sec, X-Siar-Nyquist-Hz, X-Siar-Db-Floor"
)


class ServeOptions:
    """How to serve, as distinct from what.

    Attributes:
        port: TCP port. ``0`` binds a free one, which is what the tests use.
        bind: Address to listen on. Loopback unless the caller has been explicit.
        token: The secret every API request must carry. Empty means mint one.
        audio: Serve the recordings themselves. ``False`` leaves everything else working.
        allow_origins: Web origins allowed to read this daemon cross-origin. Empty — the default —
            means no browser page from anywhere else can read it at all.
        log_requests: One line per request, rather than only per failure.
    """

    __slots__ = ("port", "bind", "token", "audio", "allow_origins", "log_requests")

    def __init__(self, *, port: int = DEFAULT_PORT, bind: str = "127.0.0.1", token: str = "",
                 audio: bool = True, allow_origins: tuple[str, ...] = (),
                 log_requests: bool = False) -> None:
        self.port = int(port)
        self.bind = str(bind)
        self.token = token or mint_token()
        self.audio = bool(audio)
        self.allow_origins = tuple(allow_origins)
        self.log_requests = bool(log_requests)

    @property
    def loopback(self) -> bool:
        """Whether this bind is only reachable from the machine itself."""
        return self.bind in ("127.0.0.1", "::1", "localhost")


def mint_token() -> str:
    """A fresh URL-safe token. Per invocation, so closing the daemon revokes it."""
    return secrets.token_urlsafe(32)


def page_url(server: ThreadingHTTPServer, token: str, *, host: str = "") -> str:
    """The URL to open, with the token in it.

    Args:
        server: The bound server — asked for its port, so ``--port 0`` prints something usable.
        token: The access token.
        host: Override the host part, for the line a reader pastes on their laptop.

    Returns:
        A complete ``http://`` URL.
    """
    bound_host, port = server.server_address[0], server.server_address[1]
    shown = host or ("127.0.0.1" if bound_host in ("0.0.0.0", "::", "") else bound_host)
    if ":" in shown and not shown.startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/?t={token}"


def build_server(folder: ServedFolder, options: ServeOptions) -> ThreadingHTTPServer:
    """Bind a server for one folder without starting it.

    Separate from :func:`serve_folder` because the port is not known until the bind when
    ``--port 0`` was asked for, and because a test wants the address before the loop starts.

    Args:
        folder: The output folder to serve.
        options: How to serve it.

    Returns:
        A bound, unstarted :class:`http.server.ThreadingHTTPServer`.

    Raises:
        OSError: If the address is in use, or cannot be bound.
    """
    state = _State(folder, options)

    class _Bound(_Handler):
        served = state

    server = ThreadingHTTPServer((options.bind, options.port), _Bound, bind_and_activate=False)
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.server_bind()
    server.server_activate()
    return server


def serve_folder(
    folder: ServedFolder,
    options: ServeOptions | None = None,
    *,
    announce: Callable[[ThreadingHTTPServer, ServeOptions], None] | None = None,
) -> int:
    """Serve one folder until the caller is interrupted.

    Args:
        folder: The output folder.
        options: How to serve it. Defaults to loopback, a fresh token and :data:`DEFAULT_PORT`.
        announce: Called once with the bound server, before the loop starts. Printing lives in
            :mod:`siarapp.cli.commands`, and this is how it gets the port it has to print.

    Returns:
        ``0`` when the loop ends. A deliberate stop is not a failure.

    Raises:
        OSError: If the address cannot be bound.
        KeyboardInterrupt: Passed through, so the caller decides what a Ctrl-C means.
    """
    options = options or ServeOptions()
    server = build_server(folder, options)
    if announce is not None:
        announce(server, options)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


class _State:
    """What every request for one folder shares: the reader, the options, the preview cache."""

    __slots__ = ("folder", "options", "_slots", "_cache", "_lock")

    def __init__(self, folder: ServedFolder, options: ServeOptions) -> None:
        self.folder = folder
        self.options = options
        self._slots = threading.BoundedSemaphore(_PREVIEW_SLOTS)
        self._cache: OrderedDict[tuple, tuple[bytes, str, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def preview(self, key: tuple, build: Callable[[], tuple[bytes, str, dict] | None]):
        """A cached preview, or one computed under the concurrency limit.

        Args:
            key: Everything the picture depends on, the file's ``(mtime_ns, size)`` included, so a
                rerun of the same recording is a different key rather than a stale hit.
            build: Computes ``(body, content_type, headers)``, or ``None`` if it cannot.

        Returns:
            The tuple, or ``None``.
        """
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        with self._slots:
            # Another thread may have built it while this one waited for a slot.
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return hit
            built = build()
        if built is None:
            return None
        with self._lock:
            self._cache[key] = built
            self._cache.move_to_end(key)
            while len(self._cache) > _PREVIEW_CACHE_MAX:
                self._cache.popitem(last=False)
        return built


class _Handler(BaseHTTPRequestHandler):
    """One request. Subclassed per server so the folder can be reached without a global."""

    #: Replaced by :func:`build_server`.
    served: _State = None  # type: ignore[assignment]

    protocol_version = "HTTP/1.1"
    server_version = f"siar-app/{__version__}"
    sys_version = ""  # not the Python build; nobody needs to know what interpreter this is

    # -- the routing table -----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        """Answer a request, body included."""
        self._body_wanted = True
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        """Answer a request with its headers only."""
        self._body_wanted = False
        self._handle()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Answer a CORS preflight — with nothing at all unless the origin is allowed."""
        self._body_wanted = False
        origin = self.headers.get("Origin") or ""
        if origin and origin in self.served.options.allow_origins:
            extra = {
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "X-Siar-Token, Authorization, Range, If-None-Match",
                "Access-Control-Max-Age": "600",
            }
            if self.headers.get("Access-Control-Request-Private-Network"):
                # Chrome asks this before a public page may read 127.0.0.1. Answering it here is
                # one line; discovering it later is an afternoon.
                extra["Access-Control-Allow-Private-Network"] = "true"
            self._respond(204, b"", "", extra=extra)
            return
        self._respond(204, b"", "")

    def send_error(self, code, message=None, explain=None) -> None:
        """Map the stdlib's "unsupported method" onto the answer that is actually true.

        There is no ``do_POST`` here, and that absence is the read-only guarantee — so the base
        class answers an unknown method with ``501 Not Implemented``. What is true is narrower:
        this daemon *has* methods, and yours is not one of them. Rewriting it here keeps the
        guarantee and still says ``405`` with an ``Allow`` header.
        """
        if code == 501:
            self._body_wanted = self.command != "HEAD"
            self._error(405, "method_not_allowed")
            return
        super().send_error(code, message, explain)

    def _handle(self) -> None:
        """Route one request, and turn anything unexpected into a JSON error."""
        try:
            split = urlsplit(self.path)
            path = split.path
            query = parse_qs(split.query, keep_blank_values=True)

            page = _PAGE_FILES.get(path)
            if page is not None:
                self._send_page(*page)
                return

            route = _ROUTES.get(path)
            if route is None:
                self._error(404, "not_found")
                return
            if not self._authorised(query):
                self._error(401, "unauthorized")
                return
            route(self, query)
        except (BrokenPipeError, ConnectionResetError):
            # The reader navigated away mid-download. Not an error, and not worth a line.
            return
        except Exception:  # noqa: BLE001 - a handler bug must not take the daemon down
            self._error(500, "server_error")

    # -- the static shell ------------------------------------------------------------------

    def _send_page(self, name: str, content_type: str) -> None:
        """One of the three files the page is made of.

        Deliberately not behind the token: they ship in the wheel, they carry no survey data, and
        gating them would mean putting the token in the ``<link>`` and ``<script>`` hrefs of a
        static file.
        """
        try:
            path = docs.local_web_path(name)
        except docs.DocsError:
            self._error(500, "page_missing")
            return
        with open(path, "rb") as fh:
            body = fh.read()
        extra = {"Cache-Control": "no-cache", "ETag": _etag(self.served.folder.stat_of(path))}
        if content_type.startswith("text/html"):
            extra["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; media-src 'self'; "
                "connect-src 'self'; style-src 'self'; script-src 'self'; base-uri 'none'"
            )
        if self._not_modified(extra["ETag"]):
            return
        self._respond(200, body, content_type, extra=extra)

    # -- the API ---------------------------------------------------------------------------

    def _api_meta(self, _query: dict) -> None:
        """What this folder is, and how far its run got."""
        meta = self.served.folder.meta(include_source_root=self.served.options.loopback)
        meta["siar_app"] = __version__
        meta["capabilities"] = {
            "audio": self.served.options.audio,
            "preview": True,
            "thumbnails": True,
        }
        meta["preview"] = {
            "max_width": PREVIEW_MAX_WIDTH,
            "max_bins": PREVIEW_MAX_BINS,
            "default_width": PREVIEW_DEFAULT_WIDTH,
            "default_bins": PREVIEW_DEFAULT_BINS,
            "raw_magic": PREVIEW_MAGIC.decode("ascii"),
            "raw_header_bytes": PREVIEW_HEADER_BYTES,
        }
        self._json(200, meta)

    def _api_manifest(self, _query: dict) -> None:
        """The run manifest, as written."""
        manifest = self.served.folder.manifest()
        if manifest is None:
            self._error(503, "no_manifest")
            return
        self._json(200, manifest)

    def _api_index(self, query: dict) -> None:
        """A page of the corpus, filtered and sorted where the data is."""
        if self.served.folder.manifest() is None:
            self._error(503, "no_manifest")
            return
        self._json(200, self.served.folder.index(
            offset=_first(query, "offset", 0),
            limit=_first(query, "limit", 200),
            status=str(_first(query, "status", "")),
            shape=str(_first(query, "shape", "")),
            query=str(_first(query, "q", "")),
            sort=str(_first(query, "sort", "path")),
            order=str(_first(query, "order", "asc")),
        ))

    def _api_performance(self, query: dict) -> None:
        """The performance report, its per-file array left out unless asked for."""
        wanted = str(_first(query, "files", "0")) in ("1", "true", "yes")
        document = self.served.folder.performance(
            files=wanted,
            offset=_first(query, "offset", 0),
            limit=_first(query, "limit", INDEX_MAX_LIMIT),
        )
        if document is None:
            self._error(404, "no_performance")
            return
        self._json(200, document)

    def _api_structures(self, query: dict) -> None:
        """One recording's boxes."""
        rel = str(_first(query, "path", ""))
        limit = _int_or_none(_first(query, "limit", None))
        document = self.served.folder.structures(rel, limit=limit)
        if document is None:
            self._error(404, "unknown_path")
            return
        path = self.served.folder.artefact(rel, "sidecar")
        tag = _etag(self.served.folder.stat_of(path)) if path else ""
        if tag and self._not_modified(tag):
            return
        self._json(200, document, extra={"Cache-Control": "no-cache", "ETag": tag} if tag else None)

    def _api_thumbnail(self, query: dict) -> None:
        """The lane preview the scan already rendered."""
        path = self.served.folder.artefact(str(_first(query, "path", "")), "thumbnail")
        if path is None:
            self._error(404, "no_thumbnail")
            return
        tag = _etag(self.served.folder.stat_of(path))
        if self._not_modified(tag):
            return
        with open(path, "rb") as fh:
            body = fh.read()
        self._respond(200, body, "image/png",
                      extra={"Cache-Control": "private, max-age=30", "ETag": tag})

    def _api_preview(self, query: dict) -> None:
        """A recording, reduced to a picture small enough to send."""
        rel = str(_first(query, "path", ""))
        path = self.served.folder.artefact(rel, "audio")
        if path is None:
            self._error(404, "unknown_path")
            return

        width = _first(query, "w", None)
        height = _first(query, "h", None)
        raw = str(_first(query, "fmt", "png")).lower() == "raw"
        floor = _first(query, "db", None)
        stamp = self.served.folder.stat_of(path)
        key = (path, stamp, width, height, floor, raw)

        def build():
            channel = str((self.served.folder.manifest() or {}).get("channel") or "mix")
            kwargs = {"width": width, "height": height, "channel": channel}
            if floor is not None:
                kwargs["db_floor"] = _clamp_float(floor, -160.0, -20.0)
            preview = read_preview(path, **kwargs)
            if preview is None:
                return None
            head = preview.header()
            headers = {
                "Cache-Control": "private, max-age=300",
                "X-Siar-Width": str(head["width"]),
                "X-Siar-Height": str(head["height"]),
                "X-Siar-Duration-Sec": str(head["duration_sec"]),
                "X-Siar-Nyquist-Hz": str(head["nyquist_hz"]),
                "X-Siar-Db-Floor": str(head["db_floor"]),
            }
            if raw:
                return encode_preview(preview), "application/octet-stream", headers
            return preview_png(preview), "image/png", headers

        built = self.served.preview(key, build)
        if built is None:
            self._error(404, "no_preview")
            return
        body, content_type, headers = built
        headers = dict(headers)
        headers["ETag"] = _etag(stamp, f"{width}:{height}:{floor}:{raw}")
        if self._not_modified(headers["ETag"]):
            return
        self._respond(200, body, content_type, extra=headers)

    def _api_audio(self, query: dict) -> None:
        """The recording itself, in whole or in part. The one route that moves real bytes."""
        if not self.served.options.audio:
            self._error(404, "audio_disabled")
            return
        path = self.served.folder.artefact(str(_first(query, "path", "")), "audio")
        if path is None:
            self._error(404, "unknown_path")
            return
        self._send_file(path)

    def _api_runs(self, _query: dict) -> None:
        """The other output folders this machine has produced, so they are one click away."""
        from siarapp.config import run_history

        rows = []
        for row in run_history():
            out = str(row.get("out") or "")
            if not out:
                continue
            rows.append({
                "out": out,
                "name": os.path.basename(out.rstrip("/")),
                "at": row.get("at") or "",
                "algorithm": row.get("algorithm") or "",
                "files": row.get("files") or 0,
                "structures": row.get("structures") or 0,
                "current": os.path.realpath(out) == self.served.folder.root,
                "exists": os.path.isdir(out),
            })
        self._json(200, {"runs": rows, "serving": self.served.folder.root})

    # -- sending ---------------------------------------------------------------------------

    def _send_file(self, path: str) -> None:
        """Stream a file, honouring one byte range.

        The stdlib's file handler has no Range support, and without it a browser's ``<audio>``
        element cannot seek — so this is hand-rolled, deliberately for the single-range form that
        is the only one anything sends. A multi-range ask is answered with the whole file, which is
        allowed and is simpler than a multipart body nobody would read.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            self._error(404, "unknown_path")
            return

        tag = _etag(self.served.folder.stat_of(path))
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        extra = {"Accept-Ranges": "bytes", "ETag": tag, "Cache-Control": "private, max-age=0"}

        header = self.headers.get("Range")
        if header and "," not in header:
            parsed = _parse_range(header, size)
            if parsed is None:
                self._respond(416, b"", "application/json",
                              extra={"Content-Range": f"bytes */{size}",
                                     "Accept-Ranges": "bytes"})
                return
            start, end = parsed
            status = 206
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"

        length = end - start + 1
        self._respond_head(status, content_type, length, extra)
        if not self._body_wanted:
            return
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(_STREAM_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except OSError:
            # The headers, Content-Length included, have already gone. There is no way to turn
            # this into an error the reader can read, so close and let them retry.
            self.close_connection = True

    def _json(self, status: int, payload: dict, *, extra: dict | None = None) -> None:
        """A JSON body, compact but not minified — a human curls these."""
        body = json.dumps(payload, indent=1, default=str).encode("utf-8")
        headers = {"Cache-Control": "no-store"}
        headers.update(extra or {})
        self._respond(status, body, "application/json; charset=utf-8", extra=headers)

    def _error(self, status: int, code: str) -> None:
        """A failure, as one of a fixed set of codes.

        Never an exception message and never a filesystem path: a reader on the other end of a
        tunnel has no business learning the box's layout from a 404.
        """
        extra = {"Cache-Control": "no-store"}
        if status == 405:
            extra["Allow"] = "GET, HEAD, OPTIONS"
        self._respond(status, json.dumps({"error": code}).encode("utf-8"),
                      "application/json; charset=utf-8", extra=extra)
        if self.served.options.log_requests or status >= 500:
            self.log_error("%s %s -> %d %s", self.command, urlsplit(self.path).path, status, code)

    def _respond(self, status: int, body: bytes, content_type: str, *,
                 extra: dict | None = None) -> None:
        """Send a complete response, and the body unless this is a HEAD."""
        self._respond_head(status, content_type, len(body), extra)
        if self._body_wanted and body:
            self.wfile.write(body)

    def _respond_head(self, status: int, content_type: str, length: int,
                      extra: dict | None = None) -> None:
        """Status line and headers, with the ones every response carries."""
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        # Exact, always: this is HTTP/1.1 with keep-alive, and a wrong length hangs the browser.
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        # Keeps `?t=<token>` out of any Referer this page's links might send.
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in self._cors().items():
            self.send_header(name, value)
        for name, value in (extra or {}).items():
            if value:
                self.send_header(name, value)
        self.end_headers()

    def _cors(self) -> dict:
        """Cross-origin headers, for an exactly-matching allowed origin and nobody else.

        Never ``Access-Control-Allow-Credentials``: the token is the authentication, there is no
        cookie to send, and a credentialed wildcard is the one CORS mistake that matters.
        """
        origin = self.headers.get("Origin") or ""
        if not origin or origin not in self.served.options.allow_origins:
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Expose-Headers": _EXPOSED,
            "Vary": "Origin",
        }

    def _not_modified(self, tag: str) -> bool:
        """Answer 304 when the reader already has this exact version. True when it did."""
        if not tag or self.headers.get("If-None-Match") != tag:
            return False
        self._respond_head(304, "", 0, {"ETag": tag})
        return True

    # -- the token -------------------------------------------------------------------------

    def _authorised(self, query: dict) -> bool:
        """Whether this request carried the token, in any of the three accepted places."""
        expected = self.served.options.token
        offered = str(_first(query, "t", "") or "")
        if not offered:
            offered = self.headers.get("X-Siar-Token") or ""
        if not offered:
            header = self.headers.get("Authorization") or ""
            if header.lower().startswith("bearer "):
                offered = header[7:].strip()
        return bool(offered) and hmac.compare_digest(offered, expected)

    # -- logging ---------------------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        """Say nothing by default.

        The inherited version writes the whole request line — ``?t=<token>`` included — for every
        one of the hundreds of thumbnail requests a scroll makes. Failures are logged by
        :meth:`_error`, and ``--verbose`` turns on one line per request with the token removed.
        """
        if not self.served.options.log_requests:
            return
        split = urlsplit(self.path)
        query = "&".join(
            part for part in split.query.split("&") if part and not part.startswith("t=")
        )
        route = split.path + (f"?{query}" if query else "")
        self.log_error("%s %s %s", self.command or "?", route, fmt % args if args else fmt)

    def log_error(self, fmt: str, *args) -> None:
        """One line on stderr, with no address in it — a tunnel makes every client 127.0.0.1."""
        import sys

        print(f"serve: {fmt % args}", file=sys.stderr, flush=True)


#: Path -> handler. A dict, so an unknown path cannot fall through to a filesystem lookup.
_ROUTES = {
    "/api/meta": _Handler._api_meta,
    "/api/manifest": _Handler._api_manifest,
    "/api/index": _Handler._api_index,
    "/api/performance": _Handler._api_performance,
    "/api/structures": _Handler._api_structures,
    "/api/thumbnail": _Handler._api_thumbnail,
    "/api/preview": _Handler._api_preview,
    "/api/audio": _Handler._api_audio,
    "/api/runs": _Handler._api_runs,
}


def _first(query: dict, name: str, default):
    """The first value of a query parameter, or a default."""
    values = query.get(name)
    if not values:
        return default
    return values[0]


def _etag(stamp: tuple[int, int] | None, salt: str = "") -> str:
    """A weak ETag from ``(mtime_ns, size)`` and whatever else the answer depended on."""
    if stamp is None:
        return ""
    body = f"{stamp[0]:x}-{stamp[1]:x}" + (f"-{abs(hash(salt)):x}" if salt else "")
    return f'W/"{body}"'


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Resolve one ``Range: bytes=`` header against a known size.

    Returns:
        ``(start, end)`` inclusive, or ``None`` when the range cannot be satisfied — which is a
        416, not a silent whole-file answer.
    """
    match = _RANGE.match(header.strip())
    if match is None or size == 0:
        return None
    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None
    if not first:  # bytes=-N — the final N bytes
        length = min(int(last), size)
        return (size - length, size - 1) if length else None
    start = int(first)
    if start >= size:
        return None
    end = min(int(last), size - 1) if last else size - 1
    return (start, end) if end >= start else None


def _int_or_none(value) -> int | None:
    """An integer from the wire, or ``None`` when it was absent or unparseable."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_float(value, low: float, high: float) -> float:
    """One float from the wire, forced into range. Junk becomes ``low``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def local_host_hint() -> str:
    """The address a reader would use to reach this box, for the ssh line the CLI prints.

    ``$SSH_CONNECTION``'s fourth field is literally the address this box was reached on, which is
    a better guess than a hostname that may only resolve inside the machine. Falls back to the
    fully-qualified name.
    """
    connection = os.environ.get("SSH_CONNECTION", "").split()
    if len(connection) >= 3:
        return connection[2]
    try:
        return socket.getfqdn() or "this-host"
    except OSError:
        return "this-host"
