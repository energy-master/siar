# Vixen Intelligence c.2026
"""Carrying a model built here to a machine that has siar-app and nothing else.

A model evolved with ``siar-build`` exists in two places at once and neither is portable on its
own. The runnable half is a generated Python package on disk — self-contained, unobfuscated,
and the only thing :func:`siarapp.loader.load_local` ever needs. The *knowable* half is a row in
siar-build's ``models.db``: what it scored, which bots the search settled on, what corpus it came
off. Copy the package and the model runs but arrives anonymous; copy the database and the paths
inside it point at directories the other machine has never had.

So a bundle carries both, and neither by reference: the package as files, the row and its
programs as JSON beside them. One file to move, ``.siarmodel``, and it lands on the far side
listed in ``siar-app lib`` with its bots intact and runnable by name.

**Import never writes to siar-build's database.** That file belongs to another program, which is
why :mod:`siarapp.library` opens it ``mode=ro`` and why an import that inserted a row into it
would be this package reaching into a workspace it does not own — one whose schema versions
separately, and whose owner would then be reading rows it never wrote. Imported models live in
this package's own workspace instead, under ``~/.siar-app/models``, and :func:`installed_bundles`
is how they are found again. A machine can therefore hold a model built here, a model imported
from a vessel, and a downloaded bundle, and know which is which.

Nothing here loads or executes what it moves. Export reads files and import writes them; the
first line of the model's code runs when somebody asks for a scan, through the same loader every
other algorithm goes through.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from siarapp import __version__
from siarapp.config import imported_dir, read_json, write_json

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SUFFIX",
    "MANIFEST_NAME",
    "TransferError",
    "bundle_manifest",
    "bundle_name",
    "export_model",
    "import_bundle",
    "installed_bundles",
    "remove_bundle",
]

#: What an exported model is called. A suffix of its own rather than ``.tar.gz`` so that the
#: browser in ``siar-app lib`` can offer exactly the files that are one, and so that a folder of
#: archives says which of them are models.
BUNDLE_SUFFIX = ".siarmodel"

#: The manifest inside a bundle: the build row, its programs, and where the bundle came from.
MANIFEST_NAME = "siar-model.json"

#: Bundle layout version. Read on import and refused if unknown — a bundle from a later siar-app
#: is a clear message here rather than a partial extraction and a confusing failure later.
BUNDLE_FORMAT = 1

#: Where the package sits inside the bundle. Its own directory name is kept underneath, because
#: :func:`siarapp.loader.load_local` imports a package *by its directory name* — flattening two
#: models to a common name would make them the same module on the far side.
_PACKAGE_ROOT = "package"

#: Where the provenance sits inside the bundle: the calibration document, the runners-up, the
#: held-out tables. Not needed to run the model and never assumed present.
_DOCS_ROOT = "docs"

#: Files beside the package that are worth carrying, by name. A whitelist and not a copy of the
#: directory: siar-build writes its scans and working files under the same tree, and a bundle
#: that swept those up would be a gigabyte of someone else's spectrograms.
_DOC_FILES = ("model.json", "report.txt", "manifest.json", "run.sh")

#: Directories beside the package worth carrying, and the suffix to take out of them.
_DOC_DIRS = (("runners", ".json"),)

#: Refused before anything is written. A bundle is a model, not a corpus: the package is under a
#: megabyte and the documents a few hundred kilobytes, so anything at this scale is either a
#: mistake or an archive built to fill a disk.
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 20_000


class TransferError(Exception):
    """An export or an import that could not be done, with a sentence saying why."""


def bundle_name(slug: str, version: str = "") -> str:
    """The filename an export defaults to.

    Args:
        slug: The model's name.
        version: Its version, if it has one.

    Returns:
        ``<slug>-<version>.siarmodel``, or ``<slug>.siarmodel`` when there is no version.
    """
    safe = _safe(slug)
    tag = _safe(version)
    return f"{safe}-{tag}{BUNDLE_SUFFIX}" if tag else f"{safe}{BUNDLE_SUFFIX}"


def export_model(model: Any, dest: str = "") -> str:
    """Write one model to a bundle that can be moved to another machine.

    Args:
        model: A :class:`siarapp.library.Model` — built here, or imported from somewhere else and
            being passed on. Taken by duck typing rather than by import so that
            :mod:`siarapp.library` can depend on this module and not the other way round.
        dest: Where to write. A directory takes the default filename; anything else is used as
            given. Defaults to :func:`bundle_name` in the working directory.

    Returns:
        The path written.

    Raises:
        TransferError: If the model has no package on this disk — a downloaded bundle, or a build
            that was never packaged. Both are refusals with a way forward rather than an empty
            archive.
    """
    source = str(getattr(model, "source", "") or "")
    if source == "downloaded":
        raise TransferError(
            f"{model.slug} is a downloaded bundle, not a model built here — it is licensed per "
            f"machine and platform. Run `siar-app login` there and `siar-app run -a "
            f"{model.slug}`, which fetches the build for that machine."
        )
    # Checked before it is made absolute: ``abspath("")`` is the working directory, and a build
    # that was never packaged would otherwise export whatever the reader happened to be standing
    # in.
    recorded = str(getattr(model, "path", "") or "").strip()
    package = os.path.abspath(recorded) if recorded else ""
    if not package or not os.path.isdir(package):
        raise TransferError(
            f"{model.slug} has nothing packaged on this disk"
            + (f" — {model.note}" if getattr(model, "note", "") else "")
        )

    path = _destination(dest, model)
    manifest = _manifest(model, package)
    docs = _documents(os.path.dirname(package), manifest)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Written to a temporary name in the destination directory and moved into place, so an
    # interrupted export leaves no half-archive that looks importable.
    handle, staging = tempfile.mkstemp(prefix=".siarmodel-", dir=os.path.dirname(path) or ".")
    os.close(handle)
    try:
        with tarfile.open(staging, "w:gz") as tar:
            tar.add(package, arcname=f"{_PACKAGE_ROOT}/{os.path.basename(package)}",
                    filter=_exclude_debris)
            for relative, absolute in docs:
                tar.add(absolute, arcname=f"{_DOCS_ROOT}/{relative}")
            tar.addfile(*_json_member(MANIFEST_NAME, manifest))
        os.replace(staging, path)
    except OSError as e:
        _forget(staging)
        raise TransferError(f"could not write {path}: {e.strerror or e}") from e
    except Exception:
        _forget(staging)
        raise
    return path


def bundle_manifest(path: str) -> dict:
    """Read a bundle's manifest without extracting anything.

    What a listing needs, and what an import checks before it writes a byte.

    Args:
        path: The ``.siarmodel`` file.

    Returns:
        The manifest.

    Raises:
        TransferError: If the file is not a readable bundle of a format this understands.
    """
    try:
        with tarfile.open(path, "r:*") as tar:
            member = tar.getmember(MANIFEST_NAME)
            if not member.isfile() or member.size > 1 << 22:
                raise TransferError(f"{path} has no readable {MANIFEST_NAME}")
            extracted = tar.extractfile(member)
            payload = json.loads(extracted.read().decode("utf-8")) if extracted else {}
    except KeyError as e:
        raise TransferError(
            f"{os.path.basename(path)} is a tar archive but not a siar-app model — "
            f"there is no {MANIFEST_NAME} in it"
        ) from e
    except (tarfile.TarError, OSError, ValueError, UnicodeDecodeError) as e:
        raise TransferError(f"{os.path.basename(path)} is not a readable model bundle: {e}") from e

    if not isinstance(payload, dict) or payload.get("kind") != "siar-model":
        raise TransferError(f"{os.path.basename(path)} is not a siar-app model bundle")
    version = int(payload.get("format") or 0)
    if version > BUNDLE_FORMAT:
        raise TransferError(
            f"{os.path.basename(path)} was written by a newer siar-app (bundle format {version}, "
            f"this one reads {BUNDLE_FORMAT}) — upgrade with `pip install -U siar-app`"
        )
    return payload


def import_bundle(path: str, *, into: str = "") -> dict:
    """Unpack a bundle into this machine's workspace and index it.

    Re-importing a bundle already here replaces it rather than making a second copy: the model is
    identified by the uid the build was given, so the same model arriving twice — by a second
    stick, or over a link that had to be retried — is one entry either way.

    Args:
        path: The ``.siarmodel`` file.
        into: Where to unpack. Defaults to ``~/.siar-app/models``, which is where
            :func:`installed_bundles` looks. A path here is for a test, or for putting a model on
            a shared volume; a model outside the workspace is still runnable with
            ``--algorithm-path``, but it will not be listed.

    Returns:
        The installed row, as :func:`installed_bundles` reports it.

    Raises:
        TransferError: If the bundle is unreadable, holds something a model bundle should not, or
            cannot be written where it was asked to go.
    """
    manifest = bundle_manifest(path)
    slug = str(manifest.get("slug") or "model")
    uid = str(manifest.get("uid") or "") or _uid(slug, str(manifest.get("exported_at") or ""))
    root = os.path.join(os.path.abspath(into) if into else imported_dir(), _safe(uid))

    staging = root + ".incoming"
    _forget(staging)
    try:
        os.makedirs(staging, exist_ok=True)
        with tarfile.open(path, "r:*") as tar:
            _extract_vetted(tar, staging, path)
    except (tarfile.TarError, OSError) as e:
        _forget(staging)
        raise TransferError(f"could not unpack {os.path.basename(path)}: {e}") from e
    except Exception:
        _forget(staging)
        raise

    package = _unpacked_package(staging, manifest)
    if package is None:
        _forget(staging)
        raise TransferError(
            f"{os.path.basename(path)} carries no importable package — a model bundle holds a "
            f"directory with an __init__.py under {_PACKAGE_ROOT}/"
        )

    record = dict(manifest)
    record["imported_at"] = _now()
    # Paths in the record are relative to the model's own directory, never absolute. It is
    # written under a staging name and moved into place, and a workspace can be moved or mounted
    # somewhere else entirely; a stored absolute path would be right for exactly as long as
    # neither of those happened.
    record["package"] = os.path.relpath(package, staging)
    record["programs"] = _relocated_programs(manifest)
    write_json(os.path.join(staging, "import.json"), record)

    # Only now is anything of the old copy touched: an import that failed half way leaves the
    # model that was already here exactly as it was.
    _forget(root)
    try:
        os.replace(staging, root)
    except OSError as e:
        _forget(staging)
        raise TransferError(f"could not install into {root}: {e.strerror or e}") from e
    return _row(root, record)


def installed_bundles() -> list[dict]:
    """Every model imported onto this machine, newest import first.

    Reads ``import.json`` and stats the tree. Like
    :func:`siarapp.loader.installed_algorithms`, it imports nothing: answering "what is here"
    must never run what is here.

    Returns:
        One row per imported model, with ``uid``, ``slug``, ``title``, ``version``, ``path`` (the
        package directory, ready for :func:`siarapp.loader.load_local`), ``root``, ``bytes``,
        ``imported_at``, ``exported_from``, ``build`` and ``programs``.
    """
    root_dir = imported_dir()
    rows = []
    try:
        names = sorted(os.listdir(root_dir))
    except OSError:
        return []
    for name in names:
        directory = os.path.join(root_dir, name)
        record = read_json(os.path.join(directory, "import.json"), None)
        if not isinstance(record, dict):
            continue
        rows.append(_row(directory, record))
    rows.sort(key=lambda r: (-r["imported_at"], r["slug"]))
    return rows


def remove_bundle(uid: str) -> bool:
    """Delete one imported model from the workspace.

    Args:
        uid: The model's uid, as :func:`installed_bundles` reports it.

    Returns:
        True if something was deleted. False when there was nothing by that uid, which is the
        same end state and not an error.
    """
    root = os.path.join(imported_dir(), _safe(uid))
    if not os.path.isdir(root):
        return False
    _forget(root)
    return True


# -- writing side ------------------------------------------------------------------------------


def _destination(dest: str, model: Any) -> str:
    """Where an export is written, given what the caller said."""
    default = bundle_name(str(getattr(model, "slug", "") or "model"),
                          str(getattr(model, "version", "") or ""))
    if not dest:
        return os.path.join(os.getcwd(), default)
    expanded = os.path.abspath(os.path.expanduser(dest))
    if os.path.isdir(expanded):
        return os.path.join(expanded, default)
    return expanded


def _manifest(model: Any, package: str) -> dict:
    """The JSON that travels with the files."""
    programs = [
        {name: getattr(program, name) for name in program.__slots__}
        for program in getattr(model, "programs", []) or []
    ]
    for program in programs:
        program["features"] = list(program.get("features") or ())
    return {
        "kind": "siar-model",
        "format": BUNDLE_FORMAT,
        "uid": _model_uid(model),
        "slug": str(getattr(model, "slug", "") or "model"),
        "title": str(getattr(model, "title", "") or ""),
        "version": str(getattr(model, "version", "") or ""),
        "package": os.path.basename(package),
        "source": str(getattr(model, "source", "") or ""),
        "stamped_at": int(getattr(model, "stamped_at", 0) or 0),
        "exported_at": _now(),
        "exported_from": _hostname(),
        "siarapp_version": __version__,
        "build": _jsonable(dict(getattr(model, "detail", {}) or {})),
        "programs": _jsonable(programs),
    }


def _model_uid(model: Any) -> str:
    """A stable identity for a model, so the same one imported twice stays one entry.

    siar-build stamps a ``model_uid`` on a build and that is used when it is there. A model from
    before that column, or one being passed on after an import, is identified by what it is: its
    slug and when it was made.
    """
    detail = dict(getattr(model, "detail", {}) or {})
    recorded = str(detail.get("model_uid") or detail.get("uid") or "")
    if recorded:
        return recorded
    return _uid(str(getattr(model, "slug", "") or "model"),
                str(detail.get("created_at") or getattr(model, "stamped_at", 0) or ""))


def _documents(beside: str, manifest: dict) -> list[tuple[str, str]]:
    """The provenance files worth carrying, as ``(path in bundle, path on disk)``.

    Best effort by design: a model whose report has been moved is still a model, and an export
    that failed over a missing text file would be an export nobody could rely on.
    """
    found: list[tuple[str, str]] = []
    for name in _DOC_FILES:
        candidate = os.path.join(beside, name)
        if os.path.isfile(candidate):
            found.append((name, candidate))
    for name, suffix in _DOC_DIRS:
        directory = os.path.join(beside, name)
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            continue
        for entry in entries:
            candidate = os.path.join(directory, entry)
            if entry.endswith(suffix) and os.path.isfile(candidate):
                found.append((f"{name}/{entry}", candidate))
    manifest["documents"] = [relative for relative, _absolute in found]
    return found


def _exclude_debris(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Keep bytecode caches and anything that is not a plain file or directory out of a bundle.

    Ownership is stripped with them. A bundle carrying the uid of an account on the machine that
    made it is a bundle that unpacks differently depending on who unpacks it.
    """
    base = os.path.basename(info.name)
    if base in ("__pycache__", ".DS_Store") or base.endswith((".pyc", ".pyo")):
        return None
    if "/__pycache__/" in info.name:
        return None
    if not (info.isfile() or info.isdir()):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def _json_member(name: str, payload: Any) -> tuple[tarfile.TarInfo, Any]:
    """A JSON document as a tar member, without a temporary file."""
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(blob)
    info.mtime = int(time.time())
    info.mode = 0o644
    return info, io.BytesIO(blob)


# -- reading side ------------------------------------------------------------------------------


def _extract_vetted(tar: tarfile.TarFile, into: str, path: str) -> None:
    """Extract a bundle member by member, refusing anything a model bundle would never hold.

    An archive is somebody else's data structure and this one has been carried on a stick between
    machines. Every member is checked before any of it is written: nothing absolute, nothing that
    climbs out with ``..``, no symlink or hard link (which is how an archive reaches a file
    outside the directory it was told to use), no device or fifo, and a ceiling on both the count
    and the total size so that an archive cannot fill the disk it is unpacked onto.

    What survives that is written here, byte by byte, rather than handed to
    :meth:`tarfile.TarFile.extract`. The archive then never gets to say anything about what it
    lands as: no mode out of it is honoured, no metadata is applied, and there is no path through
    this function on which a member type nobody vetted reaches the filesystem.

    Raises:
        TransferError: On the first member that fails, before it or anything after it is written.
    """
    root = os.path.abspath(into)
    total = 0
    members = 0
    for member in tar:
        members += 1
        if members > MAX_MEMBERS:
            raise TransferError(f"{os.path.basename(path)} holds more than {MAX_MEMBERS} files")
        if member.issym() or member.islnk():
            raise TransferError(
                f"{os.path.basename(path)} contains a link ({member.name}); a model bundle is "
                f"files only"
            )
        if not (member.isfile() or member.isdir()):
            raise TransferError(
                f"{os.path.basename(path)} contains {member.name}, which is not a file or a "
                f"directory"
            )
        if member.name.startswith("/") or os.path.isabs(member.name):
            raise TransferError(f"{os.path.basename(path)} contains an absolute path "
                                f"({member.name})")
        target = os.path.abspath(os.path.join(root, member.name))
        if target != root and not target.startswith(root + os.sep):
            raise TransferError(
                f"{os.path.basename(path)} contains a path that climbs out of the folder it is "
                f"unpacked into ({member.name})"
            )
        total += max(0, int(member.size))
        if total > MAX_BUNDLE_BYTES:
            raise TransferError(
                f"{os.path.basename(path)} unpacks to more than "
                f"{MAX_BUNDLE_BYTES // (1024 * 1024)} MB, which is not a model"
            )
        if member.isdir():
            os.makedirs(target, mode=0o755, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), mode=0o755, exist_ok=True)
        source = tar.extractfile(member)
        if source is None:  # pragma: no cover - a regular file always opens
            continue
        with open(target, "wb") as out:
            shutil.copyfileobj(source, out, 1 << 20)
        os.chmod(target, 0o644)


def _unpacked_package(staging: str, manifest: dict) -> str | None:
    """The importable package inside an unpacked bundle, or ``None`` if there is not one."""
    root = os.path.join(staging, _PACKAGE_ROOT)
    named = str(manifest.get("package") or "")
    candidates = [os.path.join(root, named)] if named else []
    try:
        candidates += [os.path.join(root, name) for name in sorted(os.listdir(root))]
    except OSError:
        return None
    for candidate in candidates:
        if os.path.isdir(candidate) and any(
            name.startswith("__init__.") for name in os.listdir(candidate)
        ):
            return candidate
    return None


def _relocated_programs(manifest: dict) -> list[dict]:
    """The programs, with ``saved_path`` pointing at this machine's copy of the document.

    A program's ``saved_path`` was absolute on the machine that built it and means nothing here.
    Where the document travelled with the bundle the path is rewritten to it — relative to the
    model's directory, resolved by :func:`_row` — so the library can still open the calibration
    it was given; where it did not, the field is emptied rather than left pointing at a directory
    that belongs to somebody else.
    """
    documents = {os.path.basename(str(name)): str(name)
                 for name in manifest.get("documents") or []}
    programs = []
    for program in manifest.get("programs") or []:
        row = dict(program) if isinstance(program, dict) else {}
        relative = documents.get(os.path.basename(str(row.get("saved_path") or "")))
        row["saved_path"] = f"{_DOCS_ROOT}/{relative}" if relative else ""
        programs.append(row)
    return programs


def _row(root: str, record: dict) -> dict:
    """One imported model as :func:`installed_bundles` reports it.

    The record holds paths relative to ``root``; they become absolute here and nowhere else, so
    a workspace that has been moved reads correctly without anything being rewritten.
    """
    package = os.path.join(root, str(record.get("package") or ""))
    programs = []
    for program in record.get("programs") or []:
        row = dict(program) if isinstance(program, dict) else {}
        saved = str(row.get("saved_path") or "")
        row["saved_path"] = os.path.join(root, saved) if saved else ""
        programs.append(row)
    return {
        "uid": str(record.get("uid") or os.path.basename(root)),
        "slug": str(record.get("slug") or "model"),
        "title": str(record.get("title") or ""),
        "version": str(record.get("version") or ""),
        "root": root,
        "path": package,
        "bytes": _tree_bytes(root),
        "imported_at": _epoch(record.get("imported_at")),
        "exported_at": str(record.get("exported_at") or ""),
        "exported_from": str(record.get("exported_from") or ""),
        "runnable": os.path.isdir(package),
        "build": dict(record.get("build") or {}),
        "programs": programs,
    }


# -- small shared helpers ----------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """The same structure with anything JSON cannot carry rendered as text.

    siar-build's index is read column by column and a schema this does not know about can hand
    back a date, a blob or a decimal. A bundle that failed to write because of a column nobody
    reads would be the wrong trade.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _uid(slug: str, made: str) -> str:
    """An identity for a model that was not given one, from what it is rather than where it is."""
    digest = hashlib.sha256(f"{slug}\n{made}".encode("utf-8")).hexdigest()[:16]
    return f"{_safe(slug)}-{digest}"


def _safe(name: str) -> str:
    """A filesystem-safe name — nothing from a bundle should ever escape the workspace."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name)).strip(".") or "x"


def _hostname() -> str:
    """This machine's name, for the manifest. Empty rather than an exception if it has none."""
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - a host with no name configured
        return ""


def _now() -> str:
    """This moment, ISO-8601 with an offset, so two machines' stamps can be compared."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _epoch(stamp: Any) -> int:
    """An ISO-8601 stamp as epoch seconds, or ``0`` if it will not parse."""
    try:
        return int(datetime.fromisoformat(str(stamp)).timestamp())
    except (TypeError, ValueError):
        return 0


def _tree_bytes(root: str) -> int:
    """What an imported model occupies on disk."""
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _forget(path: str) -> None:
    """Delete a file or directory if it is there, and say nothing either way."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        os.remove(path)
    except OSError:
        pass
