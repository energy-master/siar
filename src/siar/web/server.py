# Vixen Intelligence c.2026
"""The local dashboard: a stdlib HTTP server, a JSON API, and a vanilla-JS front end.

No Flask, no PHP, no build step. ``http.server`` is enough for a single-user, read-mostly,
localhost tool, and it means ``pip install siar`` gives you a working dashboard with nothing else
to install. A tool that fails at first run with "PHP not found" is a broken tool.

**It binds to 127.0.0.1 and nothing else.** The API hands out absolute paths from the results
database and streams files off local disk; it has no authentication because it is not supposed to
be reachable. Do not "helpfully" change the bind address to 0.0.0.0.
"""
from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from siar import __version__, paths
from siar.store.db import Store
from siar.store.export import export_run

__all__ = ["serve"]

_STATIC = Path(__file__).parent / "static"

#: (method, compiled pattern) -> handler name. Checked in order.
_ROUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("GET", re.compile(r"^/api/health$"), "api_health"),
    ("GET", re.compile(r"^/api/detectors$"), "api_detectors"),
    ("GET", re.compile(r"^/api/models$"), "api_models"),
    ("GET", re.compile(r"^/api/runs$"), "api_runs"),
    ("GET", re.compile(r"^/api/runs/([^/]+)$"), "api_run"),
    ("GET", re.compile(r"^/api/runs/([^/]+)/files$"), "api_run_files"),
    ("GET", re.compile(r"^/api/runs/([^/]+)/export\.json$"), "api_run_export"),
    ("GET", re.compile(r"^/api/files/(\d+)/detections$"), "api_file_detections"),
    ("GET", re.compile(r"^/api/files/(\d+)/spectrogram\.png$"), "api_file_png"),
]


class _Handler(BaseHTTPRequestHandler):
    """Routes one request. One :class:`Store` per request — SQLite connections are cheap and
    are not safe to share across threads."""

    server_version = f"siar/{__version__}"

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102 - quieten the default logger
        pass

    # --- plumbing -----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200, download: str | None = None) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        extra = {"Content-Disposition": f'attachment; filename="{download}"'} if download else {}
        self._send(status, body, "application/json", extra)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        path = self.path.split("?", 1)[0]

        for method, pattern, handler in _ROUTES:
            if method != "GET":
                continue
            match = pattern.match(path)
            if match:
                try:
                    getattr(self, handler)(*match.groups())
                except KeyError as exc:
                    self._error(404, str(exc))
                except Exception as exc:  # a broken API call should not kill the server
                    self._error(500, f"{type(exc).__name__}: {exc}")
                return

        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        """Serve the SPA, defaulting to index.html."""
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (_STATIC / rel).resolve()
        # Refuse to serve anything outside the static directory, whatever the URL claims.
        if not target.is_file() or _STATIC.resolve() not in target.parents:
            self._error(404, f"not found: {path}")
            return
        ctype, _ = mimetypes.guess_type(target.name)
        self._send(200, target.read_bytes(), ctype or "application/octet-stream")

    # --- API ----------------------------------------------------------------

    def api_health(self) -> None:
        self._json({"ok": True, "version": __version__, "db": str(paths.db_path())})

    def api_detectors(self) -> None:
        from siar.models import DETECTORS

        self._json(
            [
                {
                    "name": name,
                    "format": cls.format,
                    "summary": (cls.__doc__ or "").strip().splitlines()[0],
                }
                for name, cls in sorted(DETECTORS.items())
            ]
        )

    def api_models(self) -> None:
        with Store() as store:
            self._json(
                [
                    {
                        "model_uid": r["model_uid"],
                        "name": r["name"],
                        "detector": r["detector"],
                        "threshold": r["threshold"],
                        "n_params": r["n_params"],
                        "spec": json.loads(r["spec_json"]),
                        "created_at": r["created_at"],
                    }
                    for r in store.models()
                ]
            )

    def api_runs(self) -> None:
        with Store() as store:
            self._json(
                [
                    {
                        "run_uid": r["run_uid"],
                        "name": r["name"],
                        "status": r["status"],
                        "model_uid": r["model_uid"],
                        "model_name": r["model_name"],
                        "detector": r["detector"],
                        "input_path": r["input_path"],
                        "threshold": r["threshold"],
                        "n_files": r["n_files"],
                        "n_detections": r["n_detections"],
                        "created_at": r["created_at"],
                    }
                    for r in store.runs()
                ]
            )

    def api_run(self, uid: str) -> None:
        with Store() as store:
            run = store.run_by_uid(uid)
            if run is None:
                raise KeyError(f"no run matching {uid!r}")
            model = json.loads(run["model_json"]) if "model_json" in run.keys() else {}
            self._json(
                {
                    "run_uid": run["run_uid"],
                    "name": run["name"],
                    "status": run["status"],
                    "input_path": run["input_path"],
                    "threshold": run["threshold"],
                    "n_files": run["n_files"],
                    "n_detections": run["n_detections"],
                    "created_at": run["created_at"],
                    "finished_at": run["finished_at"],
                    "model_uid": run["model_uid"],
                    "model_name": run["model_name"],
                    "detector": run["detector"],
                    "spec": json.loads(run["spec_json"]),
                    "provenance": model.get("provenance", {}),
                }
            )

    def api_run_files(self, uid: str) -> None:
        with Store() as store:
            run = store.run_by_uid(uid)
            if run is None:
                raise KeyError(f"no run matching {uid!r}")
            self._json(
                [
                    {
                        "file_id": f["id"],
                        "name": f["name"],
                        "path": f["path"],
                        "duration_s": f["duration_s"],
                        "sample_rate": f["sample_rate"],
                        "n_detections": f["n_detections"],
                        "max_score": f["max_score"],
                        "has_png": bool(f["png_path"]),
                        # The image's geometry, so the browser can place boxes with no maths of
                        # its own. One PNG pixel is exactly one grid cell.
                        "frames": f["frames"],
                        "n_bins": f["n_bins"],
                        "t_min": f["t_min"],
                        "t_max": f["t_max"],
                        "f_min": f["f_min"],
                        "f_max": f["f_max"],
                    }
                    for f in store.files_for_run(int(run["id"]))
                ]
            )

    def api_file_detections(self, file_id: str) -> None:
        with Store() as store:
            self._json(
                [
                    {
                        "id": d["id"],
                        "t_start": d["t_start"],
                        "t_end": d["t_end"],
                        "f_low": d["f_low"],
                        "f_high": d["f_high"],
                        "score": d["score"],
                        "peak_score": d["peak_score"],
                        "area": d["area"],
                        "fill": d["fill"],
                        "frame_lo": d["frame_lo"],
                        "frame_hi": d["frame_hi"],
                        "bin_lo": d["bin_lo"],
                        "bin_hi": d["bin_hi"],
                    }
                    for d in store.detections_for_file(int(file_id))
                ]
            )

    def api_file_png(self, file_id: str) -> None:
        with Store() as store:
            row = store.file_by_id(int(file_id))
            if row is None or not row["png_path"]:
                raise KeyError(f"no spectrogram for file {file_id}")
            run = store.conn.execute(
                "SELECT run_uid FROM siar_runs WHERE id = ?", (row["run_id"],)
            ).fetchone()
        png = paths.run_dir(run["run_uid"]) / row["png_path"]
        if not png.is_file():
            raise KeyError(f"spectrogram missing on disk: {png}")
        self._send(200, png.read_bytes(), "image/png", {"Cache-Control": "max-age=31536000"})

    def api_run_export(self, uid: str) -> None:
        with Store() as store:
            payload = export_run(store, uid)
        self._json(payload, download=f"siar-{payload['run']['run_uid']}.json")


def serve(*, host: str = "127.0.0.1", port: int = 8420, open_browser: bool = False) -> None:
    """Run the dashboard until interrupted.

    Args:
        host: Bind address. Left at localhost unless you know exactly what you are doing.
        port: Bind port.
        open_browser: Open the dashboard in the default browser once the server is up.

    Raises:
        OSError: If the port is already in use.
    """
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"siar dashboard on {url}")
    print(f"  database {paths.db_path()}")
    print("  ctrl-c to stop")

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
