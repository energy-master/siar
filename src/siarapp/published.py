# Vixen Intelligence c.2026
"""Models somebody else bred, published to the installation, and cached where this can run them.

A society is bred on one box with ``siar-build`` and published to an IDent Dynamics installation
with ``siar-build soc publish``. From then on it lives in that installation's database rather than
on the box: a package, the geometry it demands, and a row saying who uploaded it. This module is
the other end of that — the account that may run it asking for it from a machine that has
``siar-app`` and nothing else, and getting the same directory back that
``siar-app run --algorithm-path`` loads on the box that bred it.

**Who may run one is the server's answer, and this module never second-guesses it.** Three ways
in, and only one of them is a grant: the account that uploaded a society always, whether or not it
has been released; every super, because they are the people who decide what is released and answer
for a bad build; and everybody else once a super has released it *and* granted it. So two people
signed in to one installation see different lists, the same slug can mean two different societies
under two different owners, and a model missing here is a grant nobody has made — not a fault to
work around. ``soc_model_list.php`` is asked, and what it says is what this machine can have.

**A cache, and separate from the other two.** Like :mod:`siarapp.loader`'s algorithm cache and
unlike :mod:`siarapp.transfer`'s imports, everything here can be fetched again — the installation
still has it. What differs from the algorithm cache is that a re-fetch depends on the account
still being entitled, so deleting one of these is a decision about somebody else's grant rather
than about disk space. It lives in its own directory for that reason, keyed by owner as well as
name.

**What arrives is a package, not an obfuscated bundle.** A society is plain Python against numpy,
so there is no platform tag to match and no licence to place: it runs wherever this CLI does. The
manifest travels with the catalogue rather than inside the zip, and is written back beside the
unpacked package so the tree on this disk is the tree its author has.

Nothing here imports or executes what it downloads. The first line of a society's code runs when
somebody asks for a scan, through the same loader every other model goes through.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Any

from siarapp.config import published_dir, read_json, write_json
from siarapp.grid import ScannerError
from siarapp.loader import unpack_bundle

__all__ = [
    "MANIFEST_NAME",
    "MEMBERS_KEPT",
    "RECORD_NAME",
    "cached_published",
    "choose_published",
    "fetch_published",
    "install_published",
    "installed_published",
    "remove_published",
]

#: What this module writes beside an unpacked package: the catalogue row it was fetched under,
#: when, from where, and the members read out of the model document once so that listing what is
#: here never has to open a 200 KB JSON file again.
RECORD_NAME = "published.json"

#: The package's own manifest, written back beside it from the catalogue. Named the same as the
#: one an algorithm bundle carries because :func:`siarapp.loader.load_unpacked` reads it by that
#: name and must not have to know which kind of model it is looking at.
MANIFEST_NAME = "manifest.json"

#: Members summarised into the record. The same ceiling :func:`siarapp.library.society_models`
#: puts on a local society's leaderboard, and for the same reason: a panel shows six rows at a
#: time and a record is not an archive of the search.
MEMBERS_KEPT = 50


def installed_published() -> list[dict]:
    """Every published model cached on this machine, newest fetch first.

    Reads records and stats trees; it imports nothing and asks the server nothing, so this
    answers on a vessel with no link. A directory with no record is skipped — that is an
    interrupted download rather than an install, and a re-fetch replaces it.

    Returns:
        One row per cached model, with ``slug``, ``owner``, ``title``, ``version``, ``path`` (the
        package directory, ready for :func:`siarapp.loader.load_local`), ``root``, ``bytes``,
        ``fetched_at``, ``runnable``, ``programs`` and ``catalogue`` — the server row as it stood
        when the model was fetched.
    """
    root_dir = published_dir()
    rows = []
    try:
        owners = sorted(os.listdir(root_dir))
    except OSError:
        return []
    for owner in owners:
        owner_dir = os.path.join(root_dir, owner)
        try:
            names = sorted(os.listdir(owner_dir))
        except OSError:
            continue
        for name in names:
            directory = os.path.join(owner_dir, name)
            record = read_json(os.path.join(directory, RECORD_NAME), None)
            if isinstance(record, dict):
                rows.append(_row(directory, record))
    rows.sort(key=lambda r: (-r["fetched_at"], r["slug"]))
    return rows


def cached_published(slug: str, owner: str = "") -> dict | None:
    """The cached copy of one published model, or ``None``.

    Args:
        slug: The model's name, matched case-insensitively.
        owner: Narrow it to one account's upload. Two accounts can publish the same name.

    Returns:
        The row :func:`installed_published` reports, most recently fetched first. A copy whose
        package has been deleted is still returned, with ``runnable`` False — "it is here but
        broken" and "it was never fetched" call for different messages.
    """
    wanted = str(slug or "").strip().lower()
    if not wanted:
        return None
    who = str(owner or "").strip().lower()
    for row in installed_published():
        if row["slug"].lower() != wanted:
            continue
        if who and row["owner"].lower() != who:
            continue
        return row
    return None


def fetch_published(client: Any, slug: str, owner: str = "", *, refresh: bool = False,
                    on_progress: Any = None) -> dict:
    """Get one published model onto this machine, from the cache or from the installation.

    Args:
        client: A :class:`siarapp.api.Client`.
        slug: The model's name.
        owner: Which account's upload, when two of them share a name.
        refresh: Fetch again even though it is cached — how somebody picks up a republished
            society, which is the normal state of one: a society republishes as its leaderboard
            moves, so the copy here is the build that was best when it was fetched.
        on_progress: Called as ``(bytes_so_far, total_or_None)`` while the zip arrives. Only ever
            called when a download actually happens.

    Returns:
        The installed row, as :func:`installed_published` reports it.

    Raises:
        ScannerError: If no model this account may run answers to that name, or what came back
            cannot be unpacked.
        siarapp.api.ApiError: If the installation cannot be reached or refuses.
    """
    if not refresh:
        cached = cached_published(slug, owner)
        if cached is not None and cached["runnable"]:
            return cached

    rows = client.published_models()
    row = choose_published(rows, slug, owner)
    if row is None:
        raise ScannerError(_nothing_called(slug, owner, rows))
    data = client.published_bundle(
        str(row.get("slug") or slug),
        model_id=int(row.get("id") or 0),
        owner=str(row.get("owner") or ""),
        on_progress=on_progress,
    )
    return install_published(row, data, base_url=str(getattr(client, "base_url", "") or ""))


def install_published(row: dict, data: bytes, *, base_url: str = "") -> dict:
    """Unpack one downloaded package into the cache and record what it is.

    Args:
        row: The catalogue row this package was fetched under.
        data: The zip bytes, rooted at the importable package directory.
        base_url: The installation it came from, for the library panel — a machine can be pointed
            at two of them, and "which one published this" is not answerable from the name.

    Returns:
        The installed row, as :func:`installed_published` reports it.

    Raises:
        ScannerError: If the zip does not hold an importable package, or its bytes do not match
            the digest the catalogue gave for them.
    """
    slug = str(row.get("slug") or "model")
    owner = str(row.get("owner") or "unknown")

    expected = str(row.get("sha256") or "")
    if expected:
        got = hashlib.sha256(data).hexdigest()
        if got != expected.lower():
            # Not a checksum for its own sake: the catalogue and the download are two endpoints
            # over one link, and a package that is not the one the row described is the one thing
            # this must not unpack and hand to the loader.
            raise ScannerError(
                f"the {slug} package that arrived is not the one {owner} published — the "
                f"installation gave sha256 {expected[:12]}… and {len(data)} bytes hashing to "
                f"{got[:12]}… came back. Try again, and report it if it happens twice."
            )

    root = published_dir(owner, slug)
    unpack_bundle(data, root)

    manifest = row.get("manifest")
    if isinstance(manifest, dict) and manifest:
        # Written beside the package rather than into it, which is where siar-build puts it and
        # where the loader looks. It carries the version, the shapes and the grid the model was
        # bred under; nothing here is load-bearing for a scan, and all of it is what a listing
        # says about the model.
        write_json(os.path.join(root, MANIFEST_NAME), manifest)

    package = _package_in(root, str(row.get("package") or ""))
    if package is None:
        shutil.rmtree(root, ignore_errors=True)
        raise ScannerError(
            f"the {slug} package from {owner} holds no importable directory — a published "
            f"society is a zip rooted at the folder with __init__.py in it"
        )

    record = {
        "catalogue": _jsonable(row),
        "base_url": base_url,
        "fetched_at": int(time.time()),
        # Relative to the model's own directory, never absolute: a workspace can be moved or
        # mounted somewhere else, and a stored absolute path would be right until it was.
        "package": os.path.relpath(package, root),
        "programs": _members(package),
    }
    write_json(os.path.join(root, RECORD_NAME), record)
    return _row(root, record)


def remove_published(slug: str, owner: str = "") -> bool:
    """Delete one cached published model from this machine.

    It can be fetched again for as long as the account is still entitled to it, which is the
    difference between this and deleting an import.

    Args:
        slug: The model's name.
        owner: Which account's upload, when two of them share a name.

    Returns:
        True if something was deleted. False when there was nothing by that name, which is the
        same end state and not an error.
    """
    row = cached_published(slug, owner)
    if row is None:
        return False
    shutil.rmtree(row["root"], ignore_errors=True)
    return True


# -- reading what arrived ----------------------------------------------------------------------


def choose_published(rows: list[dict], slug: str, owner: str = "") -> dict | None:
    """The catalogue row for this name: the caller's own upload first, then the newest.

    The server resolves a name the same way, and does so authoritatively. It is done here as well
    because this side has to name the row it downloaded — a run that says which society it used
    cannot get that from a zip.
    """
    wanted = str(slug or "").strip().lower()
    who = str(owner or "").strip().lower()
    matches = [
        row for row in rows
        if str(row.get("slug") or "").lower() == wanted
        and (not who or str(row.get("owner") or "").lower() == who)
    ]
    matches.sort(key=lambda row: 0 if str(row.get("access") or "") == "owner" else 1)
    return matches[0] if matches else None


def _nothing_called(slug: str, owner: str, rows: list[dict]) -> str:
    """Why a name did not resolve, said in terms of what this account actually has."""
    who = f" published by {owner}" if owner else ""
    if not rows:
        return (
            f"no published model called {slug!r}{who} — this account has not been granted any. "
            f"A society is granted per account: whoever published it can release it in "
            f"Admin -> Society models and tick it for you."
        )
    names = ", ".join(sorted({str(r.get("slug") or "") for r in rows})[:8])
    return (
        f"no published model called {slug!r}{who}. This account may run: {names}. "
        f"`siar-app published` lists them with who published each."
    )


def _package_in(root: str, named: str = "") -> str | None:
    """The importable package inside an unpacked download, or ``None`` if there is not one."""
    candidates = [os.path.join(root, named)] if named else []
    try:
        candidates += [os.path.join(root, name) for name in sorted(os.listdir(root))]
    except OSError:
        return None
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        try:
            if any(name.startswith("__init__.") for name in os.listdir(candidate)):
                return candidate
        except OSError:
            continue
    return None


def _members(package: str) -> list[dict]:
    """A society's members as the library's programs, best first.

    Read once, here, rather than every time a listing is drawn: ``soc.json`` carries the whole
    model — every member's expression tree and the standardiser tables behind them — and runs to
    a couple of hundred kilobytes. What a screen wants from it is a leaderboard.

    An unreadable or absent document is no members, never an error: a published model that is not
    a society, or one from a schema this does not know, is still a model that runs.
    """
    try:
        with open(os.path.join(package, "soc.json"), "rb") as fh:
            document = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return []
    members = document.get("members") if isinstance(document, dict) else None
    if not isinstance(members, list):
        return []

    programs = []
    for member in members[:MEMBERS_KEPT]:
        if not isinstance(member, dict):
            continue
        program = member.get("program") if isinstance(member.get("program"), dict) else {}
        programs.append({
            "rank": int(member.get("rank") or 0),
            "name": str(member.get("name") or ""),
            "kind": str(member.get("kind") or ""),
            # The arena score, which is what the society ranked its members by. The search's own
            # fitness is about a different split and is not comparable — the same distinction
            # :func:`siarapp.library._members` makes for a society bred on this machine.
            "fitness": _number(member.get("arena_auc")),
            "threshold": _number(program.get("threshold")),
            "polarity": int(program.get("polarity") or 1),
            "n_nodes": int(member.get("n_nodes") or program.get("n_nodes") or 0),
            "depth": int(program.get("depth") or 0),
            "features": [str(name) for name in (member.get("features_used") or [])],
            "infix": str(member.get("infix") or program.get("infix") or ""),
            # The document that calibrated this member is on the machine that bred it. Empty
            # rather than a path that would open somebody else's disk.
            "saved_path": "",
        })
    return programs


def _row(root: str, record: dict) -> dict:
    """One cached model as :func:`installed_published` reports it."""
    catalogue = dict(record.get("catalogue") or {})
    package = os.path.join(root, str(record.get("package") or ""))
    return {
        "id": int(catalogue.get("id") or 0),
        "slug": str(catalogue.get("slug") or os.path.basename(root)),
        "owner": str(catalogue.get("owner") or os.path.basename(os.path.dirname(root))),
        "title": str(catalogue.get("title") or ""),
        "version": str(catalogue.get("version") or ""),
        "target": str(catalogue.get("target") or ""),
        "root": root,
        "path": package,
        "bytes": _tree_bytes(root),
        "fetched_at": int(record.get("fetched_at") or 0),
        "base_url": str(record.get("base_url") or ""),
        "runnable": os.path.isdir(package),
        "catalogue": catalogue,
        "programs": [dict(p) for p in record.get("programs") or [] if isinstance(p, dict)],
    }


def _number(value: Any) -> float | None:
    """A figure out of a document as a float, or ``None`` when it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    """The same structure with anything JSON cannot carry rendered as text."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tree_bytes(root: str) -> int:
    """What a cached model occupies on disk."""
    total = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total
