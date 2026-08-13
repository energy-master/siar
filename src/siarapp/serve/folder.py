# Vixen Intelligence c.2026
"""Reading a finished — or unfinished — output folder, safely and from the outside.

This is the half of ``siar-app serve`` that has no socket in it. Everything the daemon answers
comes from here, which is what lets the awkward parts be tested by calling a function: an output
folder is a live document, and reading one correctly has more edge cases than serving it does.

**The run manifest is the index.** :func:`siarapp.runner.run_folder` already writes a recursive,
POSIX-keyed row per recording — status, duration, structure count, per-shape counts — and rewrites
the whole document after every file. So this module never walks the directory: the walk was done
during the scan, it is already sorted, and reading it again is one JSON parse instead of ten
thousand ``stat`` calls.

**The manifest is also the allowlist**, and that is the stronger half of the path story. A request
for a path the manifest does not list is not merely refused — it is *unknown*, and no amount of
cleverness with dots and slashes conjures a row into it. The syntactic refusal in :func:`check_rel`
and the ``realpath`` containment check in :meth:`ServedFolder.artefact` are the two belts either
side of it.

One thing the manifest is *not* is a census of the folder. It describes the run that wrote it: a
resumed run lists the files it skipped, so the usual case is complete, but a folder whose most
recent run used ``--limit`` — or two runs over different subsets — has more recordings on disk than
in its manifest. :meth:`ServedFolder.meta` says so in ``index_source`` and ``index_covers`` rather
than quietly showing a third of a survey as if it were all of it.
"""
from __future__ import annotations

import json
import os
import re
import threading

from siarapp.io.output import RUN_MANIFEST_NAME, SIDECAR_SUFFIX
from siarapp.io.performance import PERFORMANCE_NAME, realtime_factor

__all__ = [
    "ARTEFACT_KINDS",
    "INDEX_MAX_LIMIT",
    "FolderError",
    "PathRefused",
    "ServedFolder",
    "check_rel",
]

#: What a recording has beside it, and the only kinds of file this module will ever resolve.
ARTEFACT_KINDS = ("audio", "sidecar", "thumbnail")

#: Rows one request may ask for. A page of the corpus, not the corpus: a 12,000-file survey has
#: to arrive in pieces or the daemon spends its life serialising an index nobody scrolled to.
INDEX_MAX_LIMIT = 500

#: How often the manifest is re-``stat``ed, at most. A scan flushes its root documents every couple
#: of seconds at best, so asking the filesystem more often than this buys nothing.
_STAT_EVERY_SEC = 0.5

#: The atomic writer's transients (``io.output._write_atomic``). They exist for microseconds and
#: hold half a document; a reader that served one would be serving a truncated JSON.
_TRANSIENT = re.compile(r"\.tmp-\d+$")

#: Sort keys ``index`` accepts, mapped to the manifest field they read.
_SORTS = {
    "path": "path",
    "structures": "structures",
    "duration": "duration_sec",
    "elapsed": "elapsed_sec",
    "status": "status",
}


class FolderError(RuntimeError):
    """The folder is not one this can serve — no manifest, or an unreadable one."""


class PathRefused(ValueError):
    """A relative path this will not look up, whatever is or is not on disk."""


def check_rel(rel: str) -> str:
    """Refuse anything that is not a plain POSIX relative path inside the folder.

    Syntax only — whether the path *exists*, and whether the run wrote it, are
    :meth:`ServedFolder.artefact`'s business. Done first because it is the cheap check and because
    it is the one that can be reasoned about without a filesystem.

    Args:
        rel: The path as it arrived from the wire.

    Returns:
        The path, unchanged, when it is acceptable.

    Raises:
        PathRefused: For an empty path, an absolute one (POSIX or drive-lettered), any ``..`` or
            ``.`` segment, a backslash, a NUL, a doubled slash, or one of the atomic writer's
            ``*.tmp-<pid>`` transients.
    """
    if not rel or not isinstance(rel, str):
        raise PathRefused("empty path")
    if "\x00" in rel:
        raise PathRefused("path contains a NUL")
    if "\\" in rel:
        raise PathRefused("path contains a backslash")
    if rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        raise PathRefused("path is absolute")
    segments = rel.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise PathRefused("path has an empty, current or parent segment")
    if _TRANSIENT.search(rel):
        raise PathRefused("path is a half-written temporary file")
    return rel


class ServedFolder:
    """One output folder, read-only, as the daemon sees it.

    Args:
        root: The output folder. Resolved with ``realpath`` once, at construction, so every later
            containment check compares against a path with no symlinks left in it.

    Raises:
        FolderError: If ``root`` is not a directory.
    """

    __slots__ = ("root", "_lock", "_manifest", "_stamp", "_checked_at", "_paths")

    def __init__(self, root: str) -> None:
        resolved = os.path.realpath(os.path.abspath(os.path.expanduser(root)))
        if not os.path.isdir(resolved):
            raise FolderError(f"{root} is not a folder")
        self.root = resolved
        self._lock = threading.Lock()
        self._manifest: dict | None = None
        self._stamp: tuple[int, int] | None = None
        self._checked_at = 0.0
        self._paths: frozenset[str] = frozenset()

    # -- the root documents ----------------------------------------------------------------

    def manifest(self, *, now: float | None = None) -> dict | None:
        """The run manifest, re-read whenever the file on disk has moved.

        Memoised on ``(st_mtime_ns, st_size)`` and re-``stat``ed at most every half second, so the
        index of a run in progress is never more than that stale while a browsing session costs
        one parse rather than one per request.

        Args:
            now: Override the clock, for tests.

        Returns:
            The parsed manifest, or ``None`` when the folder has none yet — a run that has not
            reached its first flush, which is a state the daemon reports rather than an error.
        """
        import time

        at = time.time() if now is None else now
        with self._lock:
            if self._manifest is not None and at - self._checked_at < _STAT_EVERY_SEC:
                return self._manifest
            self._checked_at = at
            path = os.path.join(self.root, RUN_MANIFEST_NAME)
            stamp = _stamp_of(path)
            if stamp is None:
                self._manifest, self._stamp, self._paths = None, None, frozenset()
                return None
            if stamp == self._stamp and self._manifest is not None:
                return self._manifest
            document = _read_json(path)
            if document is None:
                # Mid-rename, or genuinely corrupt. Keep whatever was last good rather than
                # blanking a working page; the next poll picks up the new one.
                return self._manifest
            self._manifest = document
            self._stamp = stamp
            self._paths = frozenset(
                str(row.get("path") or "") for row in document.get("manifest") or ()
            )
            return document

    def performance(self, *, files: bool = False, offset: int = 0,
                    limit: int = INDEX_MAX_LIMIT) -> dict | None:
        """The performance report, with its per-file array left out unless asked for.

        At a hundred thousand recordings that array is tens of megabytes of phase dictionaries and
        the panel reading it wants the totals, so ``files=False`` is the default and the caller has
        to say when it wants the rest.

        Args:
            files: Include the per-file rows.
            offset: First per-file row to include.
            limit: How many, capped at :data:`INDEX_MAX_LIMIT`.

        Returns:
            The document, or ``None`` when the folder has no performance report.
        """
        document = _read_json(os.path.join(self.root, PERFORMANCE_NAME))
        if document is None:
            return None
        rows = document.get("files") or []
        document = dict(document)
        if files:
            start = _clamp(offset, 0, max(0, len(rows)))
            document["files"] = rows[start:start + _clamp(limit, 1, INDEX_MAX_LIMIT)]
            document["files_total"] = len(rows)
        else:
            document.pop("files", None)
            document["files_total"] = len(rows)
        return document

    def meta(self, *, include_source_root: bool = True) -> dict:
        """What this folder is, how far its run got, and what the daemon can offer for it.

        Args:
            include_source_root: Include the absolute source path from the manifest. False on a
                non-loopback bind, where it would publish the box's layout to whoever asked.

        Returns:
            The ``/api/meta`` document. ``state`` is ``"no-manifest"`` when the run has not
            flushed yet, and every other field is then absent rather than zero.
        """
        manifest = self.manifest()
        if manifest is None:
            return {"folder": os.path.basename(self.root), "state": "no-manifest",
                    "index_source": "run-manifest"}

        rows = manifest.get("manifest") or []
        by_status = manifest.get("by_status") or {}
        progress = manifest.get("progress") or {}
        meta = {
            "folder": os.path.basename(self.root),
            "state": progress.get("state") or "complete",
            "algorithm": manifest.get("algorithm") or {},
            "stft": manifest.get("stft") or {},
            "params": manifest.get("params") or {},
            "channel": manifest.get("channel") or "mix",
            "started_at": manifest.get("started_at") or "",
            "elapsed_sec": manifest.get("elapsed_sec") or 0.0,
            "workers": int(manifest.get("workers") or 1),
            "progress": progress,
            "totals": {
                "files": int(manifest.get("files") or 0),
                "by_status": by_status,
                "structures": int(manifest.get("structures") or 0),
                "shapes": manifest.get("shapes") or {},
                "audio_sec": float(manifest.get("audio_sec") or 0.0),
                "phases": manifest.get("phases") or {},
            },
            # Stated rather than implied: this index is the run's own record, and a folder written
            # by more than one run can hold recordings no manifest here lists.
            "index_source": "run-manifest",
            "index_covers": len(rows),
            "performance": os.path.isfile(os.path.join(self.root, PERFORMANCE_NAME)),
        }
        if include_source_root and manifest.get("source_root"):
            meta["source_root"] = manifest["source_root"]
        return meta

    # -- the index -------------------------------------------------------------------------

    def rel_paths(self) -> frozenset[str]:
        """Every path this run wrote — the allowlist every artefact lookup is checked against."""
        self.manifest()
        with self._lock:
            return self._paths

    def index(self, *, offset: int = 0, limit: int = 200, status: str = "", shape: str = "",
              query: str = "", sort: str = "path", order: str = "asc") -> dict:
        """A page of the corpus, filtered and sorted here rather than in the browser.

        A hundred thousand rows cannot be filtered client-side, and the fields worth filtering on
        are all in the manifest already.

        Args:
            offset: First row of the filtered set to return.
            limit: How many rows, capped at :data:`INDEX_MAX_LIMIT`.
            status: Keep only this status.
            shape: Keep only recordings whose per-shape counts include this shape.
            query: Keep only paths containing this substring, case-insensitively.
            sort: One of ``path``, ``structures``, ``duration``, ``elapsed``, ``status``.
            order: ``asc`` or ``desc``.

        Returns:
            ``{total, offset, limit, files: [...]}``, where ``total`` counts the filtered set and
            each row carries what a lane strip needs and nothing it does not — the boxes
            themselves come one recording at a time from :meth:`structures`.
        """
        manifest = self.manifest()
        rows = list((manifest or {}).get("manifest") or [])

        if status:
            rows = [r for r in rows if r.get("status") == status]
        if shape:
            rows = [r for r in rows if shape in (r.get("shapes") or {})]
        if query:
            needle = query.lower()
            rows = [r for r in rows if needle in str(r.get("path") or "").lower()]

        key = _SORTS.get(sort, "path")
        rows.sort(key=lambda r: _sort_value(r, key))
        if order == "desc":
            rows.reverse()

        total = len(rows)
        start = _clamp(offset, 0, max(0, total))
        count = _clamp(limit, 1, INDEX_MAX_LIMIT)
        page = rows[start:start + count]
        return {
            "total": total,
            "offset": start,
            "limit": count,
            "files": [self._row(r) for r in page],
        }

    def _row(self, row: dict) -> dict:
        """One index row, with the three ``stat`` calls only a returned row is worth."""
        rel = str(row.get("path") or "")
        duration = float(row.get("duration_sec") or 0.0)
        elapsed = float(row.get("elapsed_sec") or 0.0)
        audio = self.artefact(rel, "audio")
        out = {
            "path": rel,
            "name": rel.rsplit("/", 1)[-1],
            "status": row.get("status") or "",
            "duration_sec": duration,
            "elapsed_sec": elapsed,
            "realtime_factor": realtime_factor(duration, elapsed),
            "structures": int(row.get("structures") or 0),
            "shapes": row.get("shapes") or {},
            "thumbnail": self.artefact(rel, "thumbnail") is not None,
            "sidecar": self.artefact(rel, "sidecar") is not None,
            "audio_bytes": os.path.getsize(audio) if audio else 0,
        }
        if row.get("error"):
            out["error"] = row["error"]
        return out

    # -- one recording ---------------------------------------------------------------------

    def structures(self, rel: str, *, limit: int | None = None) -> dict | None:
        """One recording's sidecar document, as written.

        Args:
            rel: The recording's path, relative to the folder root.
            limit: Keep at most this many boxes, marking the document ``truncated``. ``count``
                is left alone — it is what the run found, not what this reply carries.

        Returns:
            The document, or ``None`` when there is no sidecar for that recording.
        """
        path = self.artefact(rel, "sidecar")
        if path is None:
            return None
        document = _read_json(path)
        if document is None:
            return None
        if limit is not None:
            boxes = document.get("structures") or []
            if len(boxes) > max(0, limit):
                document = dict(document)
                document["structures"] = boxes[: max(0, limit)]
                document["truncated"] = True
        return document

    def artefact(self, rel: str, kind: str) -> str | None:
        """The absolute path of one of a recording's artefacts, or ``None``.

        Three refusals, and any one of them alone would be enough: the path must be syntactically
        plain (:func:`check_rel`), the run must have written that recording (the manifest is the
        allowlist), and the resolved file must still be inside the root — which is what catches a
        symlink planted in the folder pointing at ``/etc``.

        Args:
            rel: The recording's path, relative to the folder root.
            kind: One of :data:`ARTEFACT_KINDS`.

        Returns:
            The path when it exists and passes every check, else ``None``. Never raises for bad
            input: a refusal and a missing file are the same answer to a caller serving HTTP.
        """
        if kind not in ARTEFACT_KINDS:
            return None
        try:
            check_rel(rel)
        except PathRefused:
            return None
        if rel not in self.rel_paths():
            return None

        joined = os.path.join(self.root, *rel.split("/"))
        if kind == "sidecar":
            joined = os.path.splitext(joined)[0] + SIDECAR_SUFFIX
        elif kind == "thumbnail":
            joined = os.path.splitext(joined)[0] + ".png"

        resolved = os.path.realpath(joined)
        try:
            if os.path.commonpath([self.root, resolved]) != self.root:
                return None
        except ValueError:  # different drives on Windows
            return None
        return resolved if os.path.isfile(resolved) else None

    def stat_of(self, path: str) -> tuple[int, int] | None:
        """``(st_mtime_ns, st_size)`` for an ETag, or ``None`` if it has gone."""
        return _stamp_of(path)


def _stamp_of(path: str) -> tuple[int, int] | None:
    """What identifies a version of a file, cheaply."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _read_json(path: str) -> dict | None:
    """Parse a JSON document, or ``None`` if it is missing, unreadable or half-written."""
    try:
        with open(path, "rb") as fh:
            document = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _sort_value(row: dict, key: str):
    """A sort key that never raises on a row missing the field, or holding the wrong type."""
    value = row.get(key)
    if key in ("structures", "duration_sec", "elapsed_sec"):
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return str(value or "")


def _clamp(value, low: int, high: int) -> int:
    """An integer from the wire, forced into range. Junk becomes ``low``."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))
