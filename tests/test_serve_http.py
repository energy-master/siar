# Vixen Intelligence c.2026
"""The daemon over a real socket.

Driven with ``urllib`` — the package's own HTTP idiom — against a server bound to port 0 in a
thread, over an output folder the real pipeline wrote. What is worth pinning here is everything a
reader on the other end of a tunnel could do to the box: ask for a path the run never wrote, ask
for a picture the size of a datacentre, ask without the token, ask with a method that writes.

The last test in this file is the one that matters most: nothing in the folder changes, whatever is
requested. That is checked by comparing a snapshot of every file's size and mtime, because reading
the handler and concluding "it looks read-only" is not the same claim.
"""
from __future__ import annotations

import json
import os
import re
import struct
import threading
import urllib.error
import urllib.request

import pytest

from siarapp.loader import load_local
from siarapp.runner import RunOptions, run_folder
from siarapp.serve.folder import ServedFolder
from siarapp.serve.http import ServeOptions, build_server, page_url
from siarapp.serve.preview import PREVIEW_HEADER_BYTES, PREVIEW_MAGIC
from test_runner import STUB, _write_package, _write_wav


@pytest.fixture()
def scan(tmp_path):
    """A real output folder, written by the stub pipeline."""
    src = tmp_path / "src"
    (src / "station-b").mkdir(parents=True)
    _write_wav(src / "loud.wav", seconds=2.0, amplitude=1.0)
    _write_wav(src / "station-b" / "quiet.wav", amplitude=1e-6)
    _write_wav(src / "tiny.wav", seconds=0.01)

    handle = load_local(_write_package(tmp_path / "algo", "stub_algo", STUB), "stub")
    out = tmp_path / "out"
    manifest = run_folder(handle, str(src), str(out), RunOptions())
    return out, manifest


def _serve(out, **kwargs):
    """A bound, running server for one folder. Returns ``(base, token, stop)``."""
    options = ServeOptions(port=0, **kwargs)
    server = build_server(ServedFolder(str(out)), options)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]

    def stop():
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    return f"http://{host}:{port}", options.token, stop


@pytest.fixture()
def served(scan):
    out, manifest = scan
    base, token, stop = _serve(out)
    try:
        yield base, token, out, manifest
    finally:
        stop()


def _fetch(url, *, headers=None, method="GET", data=None):
    """``(status, headers, body)``, with a 4xx as a return value rather than an exception."""
    request = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _api(base, token, route, **params):
    """A tokenised API URL. `route` is the path on the daemon; `path=` is a query parameter."""
    from urllib.parse import urlencode

    params["t"] = token
    return f"{base}{route}?{urlencode(params)}"


def _json(base, token, route, **params):
    status, _headers, body = _fetch(_api(base, token, route, **params))
    assert status == 200, body[:200]
    return json.loads(body)


# -- the token -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/meta", "/api/manifest", "/api/index", "/api/performance",
    "/api/structures", "/api/thumbnail", "/api/preview", "/api/audio", "/api/runs",
])
def test_every_api_route_needs_the_token(served, path):
    base, _token, _out, _manifest = served
    status, _headers, body = _fetch(f"{base}{path}?path=loud.wav")
    assert status == 401
    assert json.loads(body)["error"] == "unauthorized"


def test_a_wrong_token_is_refused(served):
    base, token, _out, _manifest = served
    assert _fetch(f"{base}/api/meta?t=")[0] == 401
    assert _fetch(f"{base}/api/meta?t={token[:-1]}x")[0] == 401
    assert _fetch(f"{base}/api/meta?t={token}")[0] == 200


def test_the_token_is_accepted_three_ways(served):
    """The query form because an <img> cannot send a header; the header forms so a fetch need not
    put a secret in a URL. The page uses both, so neither can rot unnoticed."""
    base, token, _out, _manifest = served
    assert _fetch(f"{base}/api/meta?t={token}")[0] == 200
    assert _fetch(f"{base}/api/meta", headers={"X-Siar-Token": token})[0] == 200
    assert _fetch(f"{base}/api/meta", headers={"Authorization": f"Bearer {token}"})[0] == 200


def test_the_page_itself_needs_no_token(served):
    """Three files out of the wheel with no survey data in them. Gating them would put the token
    into the href of a static file."""
    base, _token, _out, _manifest = served
    for path in ("/", "/viewer.css", "/viewer.js"):
        status, headers, _body = _fetch(f"{base}{path}")
        assert status in (200, 500), path  # 500 only if the page has not been written yet
        if status == 200 and path == "/":
            assert "text/html" in headers["Content-Type"]
            assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


def test_a_panel_hidden_by_the_attribute_stays_hidden(served):
    """The page shows and hides every panel through the ``hidden`` attribute, and the attribute
    only carries the user agent's ``display: none`` — which any author rule with a ``display`` in
    it outranks. ``.nokey`` is one: it is ``display: grid``, so without the reset below the "this
    page needs the token" curtain covers the viewer on every load, token or no token."""
    base, _token, _out, _manifest = served
    status, _headers, body = _fetch(f"{base}/viewer.css")
    if status != 200:
        pytest.skip("the page is not in this tree")
    css = body.decode()
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), css[:400]


def test_a_lane_reads_its_path_the_way_it_is_spelled(served):
    """``.lane-path`` is right-to-left so a long path loses its start rather than its filename.
    That reorders the path's own punctuation — ``134250533.191114211719.wav`` renders as
    ``wav.134250533…`` — unless the text is isolated from the container's direction."""
    base, _token, _out, _manifest = served
    status, _headers, body = _fetch(f"{base}/viewer.css")
    if status != 200:
        pytest.skip("the page is not in this tree")
    css = body.decode()
    script = _fetch(f"{base}/viewer.js")[2].decode()
    assert "direction: rtl" in css
    assert "createElement('bdi')" in script


# -- what it will not do -------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_nothing_that_writes_is_answered(served, method):
    base, token, _out, _manifest = served
    status, headers, body = _fetch(_api(base, token, "/api/meta"), method=method,
                                  data=b"{}" if method != "DELETE" else None)
    assert status == 405
    assert headers.get("Allow") == "GET, HEAD, OPTIONS"
    assert json.loads(body)["error"] == "method_not_allowed"


@pytest.mark.parametrize("rel", [
    "../../etc/passwd",
    "/etc/passwd",
    "station-b/../../etc/passwd",
    "..",
    "loud.wav.tmp-999",
    "secret.wav",
    "",
])
def test_a_path_the_run_never_wrote_is_refused(served, rel):
    base, token, out, _manifest = served
    (out / "secret.wav").write_bytes(b"RIFF" + b"\0" * 64)
    for route in ("/api/audio", "/api/thumbnail", "/api/structures", "/api/preview"):
        status, _headers, body = _fetch(_api(base, token, route, path=rel))
        assert status == 404, f"{route} {rel!r} -> {status}"
        assert b"root:" not in body


def test_an_unknown_route_is_a_json_404(served):
    base, token, _out, _manifest = served
    status, headers, body = _fetch(_api(base, token, "/api/secrets"))
    assert status == 404
    assert "application/json" in headers["Content-Type"]
    assert json.loads(body)["error"] == "not_found"


def test_the_page_is_served_from_a_whitelist(served):
    base, _token, _out, _manifest = served
    for path in ("/quickstart.html", "/../config.py", "/viewer.js.map", "/index.php"):
        assert _fetch(f"{base}{path}")[0] == 404, path


# -- the index and the metadata ------------------------------------------------------------


def test_the_index_matches_the_manifest(served):
    base, token, _out, manifest = served
    index = _json(base, token, "/api/index", limit=500)
    assert index["total"] == manifest["files"]
    rows = {row["path"]: row for row in index["files"]}
    for source in manifest["manifest"]:
        assert rows[source["path"]]["status"] == source["status"]
        assert rows[source["path"]]["structures"] == source["structures"]


def test_meta_says_what_the_run_was_and_what_it_can_serve(served):
    base, token, _out, manifest = served
    meta = _json(base, token, "/api/meta")
    assert meta["algorithm"]["slug"] == "stub"
    assert meta["totals"]["by_status"] == manifest["by_status"]
    assert meta["state"] == "complete"
    assert meta["capabilities"] == {"audio": True, "preview": True, "thumbnails": True}
    assert meta["preview"]["raw_magic"] == PREVIEW_MAGIC.decode()
    assert meta["index_source"] == "run-manifest"
    assert meta["siar_app"]


def test_the_index_is_no_store_because_a_run_may_still_be_writing(served):
    base, token, _out, _manifest = served
    for path in ("/api/meta", "/api/index", "/api/manifest", "/api/performance"):
        _status, headers, _body = _fetch(_api(base, token, path))
        assert headers.get("Cache-Control") == "no-store", path


def test_performance_leaves_its_per_file_array_out_by_default(served):
    base, token, _out, manifest = served
    assert "files" not in _json(base, token, "/api/performance")
    assert _json(base, token, "/api/performance", files=1)["files"]
    assert _json(base, token, "/api/performance")["files_total"] == manifest["files"]


def test_an_empty_folder_says_no_manifest_rather_than_failing(tmp_path):
    base, token, stop = _serve(tmp_path)
    try:
        status, _headers, body = _fetch(_api(base, token, "/api/index"))
        assert status == 503
        assert json.loads(body)["error"] == "no_manifest"
        # ...and meta still answers, so a page can say what is going on.
        assert _json(base, token, "/api/meta")["state"] == "no-manifest"
    finally:
        stop()


# -- one recording -------------------------------------------------------------------------


def test_structures_are_the_sidecar_as_written(served):
    base, token, out, _manifest = served
    document = _json(base, token, "/api/structures", path="loud.wav")
    assert document == json.loads((out / "loud.structures.json").read_text())


def test_structures_can_be_truncated_without_lying_about_the_count(served):
    base, token, _out, _manifest = served
    limited = _json(base, token, "/api/structures", path="loud.wav", limit=0)
    assert limited["structures"] == []
    assert limited["truncated"] is True
    assert limited["count"] == _json(base, token, "/api/structures", path="loud.wav")["count"]


def test_a_thumbnail_is_the_png_on_disk(served):
    base, token, out, _manifest = served
    status, headers, body = _fetch(_api(base, token, "/api/thumbnail", path="loud.wav"))
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == (out / "loud.png").read_bytes()
    # tiny.wav is too short to have one, and says so rather than 500ing.
    assert _fetch(_api(base, token, "/api/thumbnail", path="tiny.wav"))[0] == 404


def test_a_preview_is_a_png_by_default_and_raw_on_request(served):
    base, token, _out, _manifest = served
    status, headers, body = _fetch(_api(base, token, "/api/preview", path="loud.wav", w=200, h=64))
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    assert headers["X-Siar-Width"] == "200"
    assert float(headers["X-Siar-Duration-Sec"]) == pytest.approx(2.0, abs=0.05)

    status, headers, body = _fetch(
        _api(base, token, "/api/preview", path="loud.wav", w=200, h=64, fmt="raw"))
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert body[:8] == PREVIEW_MAGIC
    width, height = struct.unpack("<II", body[8:16])
    assert len(body) == PREVIEW_HEADER_BYTES + width * height
    assert (width, height) == (200, int(headers["X-Siar-Height"]))


def test_a_silly_preview_request_is_clamped(served):
    base, token, _out, _manifest = served
    _status, headers, body = _fetch(
        _api(base, token, "/api/preview", path="loud.wav", w=999999, h=99999, fmt="raw"))
    assert int(headers["X-Siar-Width"]) <= 4000
    assert int(headers["X-Siar-Height"]) <= 513
    assert len(body) <= PREVIEW_HEADER_BYTES + 4000 * 513


def test_a_recording_too_short_to_picture_says_so(served):
    base, token, _out, _manifest = served
    status, _headers, body = _fetch(_api(base, token, "/api/preview", path="tiny.wav"))
    assert status == 404
    assert json.loads(body)["error"] == "no_preview"


# -- audio ---------------------------------------------------------------------------------


def test_audio_is_ranged(served):
    base, token, out, _manifest = served
    url = _api(base, token, "/api/audio", path="loud.wav")
    whole = (out / "loud.wav").read_bytes()

    status, headers, body = _fetch(url)
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert int(headers["Content-Length"]) == len(whole)
    assert body == whole

    status, headers, body = _fetch(url, headers={"Range": "bytes=0-15"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 0-15/{len(whole)}"
    assert body == whole[:16]

    status, headers, _body = _fetch(url, headers={"Range": f"bytes={len(whole)}-"})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{len(whole)}"

    # The tail form, which is how a player finds a trailing chunk.
    _status, _headers, body = _fetch(url, headers={"Range": "bytes=-8"})
    assert body == whole[-8:]


def test_audio_can_be_refused_without_breaking_anything_else(scan):
    out, _manifest = scan
    base, token, stop = _serve(out, audio=False)
    try:
        status, _headers, body = _fetch(_api(base, token, "/api/audio", path="loud.wav"))
        assert status == 404
        assert json.loads(body)["error"] == "audio_disabled"
        assert _json(base, token, "/api/meta")["capabilities"]["audio"] is False
        # The pictures still work, which is the point of the flag.
        assert _fetch(_api(base, token, "/api/thumbnail", path="loud.wav"))[0] == 200
        assert _fetch(_api(base, token, "/api/preview", path="loud.wav"))[0] == 200
    finally:
        stop()


def test_head_gives_the_headers_without_the_body(served):
    base, token, out, _manifest = served
    status, headers, body = _fetch(_api(base, token, "/api/audio", path="loud.wav"), method="HEAD")
    assert status == 200
    assert body == b""
    assert int(headers["Content-Length"]) == os.path.getsize(out / "loud.wav")


# -- caching -------------------------------------------------------------------------------


def test_a_reader_that_already_has_it_gets_a_304(served):
    base, token, _out, _manifest = served
    for path, params in (("/api/thumbnail", {"path": "loud.wav"}),
                         ("/api/preview", {"path": "loud.wav", "w": 100, "h": 64}),
                         ("/api/structures", {"path": "loud.wav"})):
        _status, headers, _body = _fetch(_api(base, token, path, **params))
        tag = headers.get("ETag")
        assert tag, path
        status, _headers, body = _fetch(_api(base, token, path, **params),
                                       headers={"If-None-Match": tag})
        assert status == 304, path
        assert body == b""


# -- cross-origin --------------------------------------------------------------------------


def test_no_cors_headers_unless_an_origin_was_allowed(served):
    base, token, _out, _manifest = served
    _status, headers, _body = _fetch(_api(base, token, "/api/meta"),
                                     headers={"Origin": "https://goident.ai"})
    assert "Access-Control-Allow-Origin" not in headers


def test_an_allowed_origin_is_echoed_and_nothing_else_is(scan):
    out, _manifest = scan
    base, token, stop = _serve(out, allow_origins=("https://goident.ai",))
    try:
        _status, headers, _body = _fetch(_api(base, token, "/api/meta"),
                                         headers={"Origin": "https://goident.ai"})
        assert headers["Access-Control-Allow-Origin"] == "https://goident.ai"
        assert headers["Vary"] == "Origin"
        assert "X-Siar-Width" in headers["Access-Control-Expose-Headers"]
        # Never credentialed: the token is the authentication and there is no cookie.
        assert "Access-Control-Allow-Credentials" not in headers

        _status, headers, _body = _fetch(_api(base, token, "/api/meta"),
                                         headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in headers

        status, headers, _body = _fetch(
            f"{base}/api/meta", method="OPTIONS",
            headers={"Origin": "https://goident.ai",
                     "Access-Control-Request-Private-Network": "true"})
        assert status == 204
        assert headers["Access-Control-Allow-Methods"] == "GET, HEAD, OPTIONS"
        # Chrome asks before a public page may read 127.0.0.1.
        assert headers["Access-Control-Allow-Private-Network"] == "true"
    finally:
        stop()


# -- the guarantee -------------------------------------------------------------------------


def test_nothing_in_the_folder_is_touched(served):
    """The read-only claim, measured rather than reasoned about."""
    base, token, out, _manifest = served

    def snapshot():
        seen = {}
        for root, _dirs, names in os.walk(out):
            for name in names:
                path = os.path.join(root, name)
                info = os.stat(path)
                seen[os.path.relpath(path, out)] = (info.st_size, info.st_mtime_ns)
        return seen

    before = snapshot()
    for url in (
        _api(base, token, "/api/meta"),
        _api(base, token, "/api/manifest"),
        _api(base, token, "/api/index", limit=500),
        _api(base, token, "/api/performance", files=1),
        _api(base, token, "/api/structures", path="loud.wav"),
        _api(base, token, "/api/thumbnail", path="loud.wav"),
        _api(base, token, "/api/preview", path="loud.wav", w=300, h=128),
        _api(base, token, "/api/preview", path="loud.wav", w=300, h=128, fmt="raw"),
        _api(base, token, "/api/audio", path="loud.wav"),
        _api(base, token, "/api/runs"),
        f"{base}/",
    ):
        _fetch(url)
    _fetch(_api(base, token, "/api/audio", path="loud.wav"), headers={"Range": "bytes=0-31"})

    after = snapshot()
    assert after == before, "serving a folder changed something in it"
    assert set(after) == set(before)


def test_page_url_reports_the_port_it_actually_bound(scan):
    """`--port 0` has to print something that works, which means asking the socket."""
    out, _manifest = scan
    options = ServeOptions(port=0)
    server = build_server(ServedFolder(str(out)), options)
    try:
        url = page_url(server, options.token)
        assert url.startswith("http://127.0.0.1:")
        assert str(server.server_address[1]) in url
        assert options.token in url
    finally:
        server.server_close()
