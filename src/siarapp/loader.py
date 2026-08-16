# Vixen Intelligence c.2026
"""Getting a scanning algorithm onto this machine and into this process.

An algorithm ships as a zip: an obfuscated Python package, PyArmor's runtime, a licence key if
the build has one, and a ``manifest.json`` describing what is inside. This module downloads it
once, unpacks it under ``~/.siar-app/algorithms/<slug>/<platform>/``, makes it importable,
and calls its factory. Every later run finds the cache and needs no network at all — which is
the point for a survey vessel or an air-gapped lab.

The download/unpack/import dance is ported from the IDent Dynamics SDK's
``identdynamics/protected.py`` (``_locate_package``, ``_place_license``, ``fetch_protected``).
It looks fussier than it should because PyArmor's runtime searches for its ``.rkey`` in more
than one place depending on how the package was built, and getting that wrong surfaces as an
``ImportError`` that says nothing useful.

Two guards exist purely to turn cryptic failures into sentences:

* **Platform check before import.** A bundle is pinned to OS + arch + Python minor. Importing a
  Linux build on a Mac raises something opaque from deep inside the runtime; checking the
  manifest's tag first lets us say "ask the publisher for a darwin-arm64-cp313 build". (It is
  ``darwin``, never ``macos`` — the tag is built from ``platform.system().lower()``.) A matching
  tag on a musl system is checked too, since ``linux`` covers both libc flavours and the tag
  cannot distinguish them.
* **Contract check after construction.** An object with no ``scan`` is a build mistake, and it
  should be named as one here rather than as an ``AttributeError`` five hundred files into a run.

:func:`load_local` is the development path: point it at an unobfuscated package directory and
it skips the network, the cache and the platform check entirely.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Callable
from typing import Any

from siarapp.config import (
    algorithm_cache_dir,
    default_platform_tag,
    home,
    libc_flavour,
    platform_compatible,
    supported_python_text,
)
from siarapp.grid import ScannerError

__all__ = [
    "FACTORY_NAME",
    "AlgorithmHandle",
    "installed_algorithms",
    "load_cached",
    "load_local",
    "load_remote",
    "load_unpacked",
    "unpack_bundle",
]

#: The callable an algorithm package must expose at its top level.
FACTORY_NAME = "algorithm"

#: Marker written beside an unpacked bundle so a later run knows the cache is complete. Without
#: it, an interrupted unpack leaves a half-tree that imports and then fails oddly.
_STAMP = ".siar-app-complete"

#: The same marker under its pre-rename name. Read, never written: a cache unpacked by
#: siar-scanner is a perfectly good cache, and treating it as incomplete would re-download every
#: bundle on the machine for no reason but a change of product name.
_LEGACY_STAMP = ".siar-scanner-complete"


class AlgorithmHandle:
    """A loaded algorithm plus where it came from.

    Attributes:
        algorithm: The live object — anything with ``scan(grid)``.
        slug: Registry slug.
        manifest: The bundle's ``manifest.json`` (``{}`` for a local source directory).
        root: Directory the package was imported from.
        platform: Build tag, or ``"source"`` for a local load.
    """

    __slots__ = ("algorithm", "slug", "manifest", "root", "platform")

    def __init__(self, algorithm: Any, slug: str, manifest: dict, root: str, platform: str):
        self.algorithm = algorithm
        self.slug = slug
        self.manifest = manifest
        self.root = root
        self.platform = platform

    def describe(self) -> dict:
        """What the run manifest records about which algorithm produced its boxes."""
        return {
            "slug": self.slug,
            "version": str(getattr(self.algorithm, "version", self.manifest.get("version", ""))),
            "platform": self.platform,
            "shapes": list(getattr(self.algorithm, "shapes", ()) or []),
        }


# -- unpacking ---------------------------------------------------------------------------


def unpack_bundle(data: bytes, dest: str) -> str:
    """Unpack a bundle zip into ``dest``, replacing whatever was there.

    Args:
        data: The zip bytes.
        dest: Destination directory.

    Returns:
        ``dest``.

    Raises:
        ScannerError: If the payload is not a zip, or contains a path that would escape
            ``dest``. The second is paranoia about an archive we did not build, and costs one
            comparison per member.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ScannerError(f"the downloaded bundle is not a zip: {e}") from e

    root = os.path.abspath(dest)
    for member in zf.namelist():
        target = os.path.abspath(os.path.join(root, member))
        if target != root and not target.startswith(root + os.sep):
            raise ScannerError(f"bundle contains an out-of-tree path: {member!r}")

    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    zf.extractall(root)
    with open(os.path.join(root, _STAMP), "w", encoding="utf-8") as fh:
        fh.write("ok\n")
    return root


def _is_complete(path: str) -> bool:
    """True when ``path`` holds a fully-unpacked bundle."""
    return (os.path.isfile(os.path.join(path, _STAMP))
            or os.path.isfile(os.path.join(path, _LEGACY_STAMP)))


def _read_manifest(root: str) -> dict:
    """The bundle's ``manifest.json``, searched for anywhere under ``root``. ``{}`` if absent."""
    for dirpath, _dirs, files in os.walk(root):
        if "manifest.json" in files:
            try:
                with open(os.path.join(dirpath, "manifest.json"), "rb") as fh:
                    payload = json.loads(fh.read().decode("utf-8"))
                    return payload if isinstance(payload, dict) else {}
            except (OSError, ValueError):
                return {}
    return {}


def _locate_package(root: str, package: str = "") -> str | None:
    """Find the importable package directory under ``root``.

    Prefers a directory whose name matches the manifest's ``package``; otherwise the shallowest
    directory holding an ``__init__`` module that is not PyArmor's own runtime.
    """
    candidates: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith("pyarmor_runtime"):
            continue
        if any(f.startswith("__init__.") for f in files):
            if package and base == package:
                return dirpath
            candidates.append(dirpath)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.count(os.sep), p))
    return candidates[0]


def _place_license(root: str, parent: str, pkg_dir: str) -> None:
    """Copy any ``*.rkey`` in the bundle to both the package directory and its parent.

    PyArmor's runtime looks for its key in one of those two places depending on how the build
    was laid out. Copying to both costs a few hundred bytes and removes a whole class of
    "works on the build machine" failure. A no-op when the build has no key.
    """
    found = None
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".rkey"):
                found = os.path.join(dirpath, f)
                break
        if found:
            break
    if not found:
        return
    for target_dir in (pkg_dir, parent):
        target = os.path.join(target_dir, "pyarmor.rkey")
        if os.path.abspath(target) != os.path.abspath(found) and not os.path.exists(target):
            try:
                shutil.copyfile(found, target)
            except OSError:
                pass


# -- importing ---------------------------------------------------------------------------


def _import_from(pkg_dir: str) -> Any:
    """Put a package directory's parent on ``sys.path`` and import it.

    Returns:
        The imported module.

    Raises:
        ScannerError: If the import fails — which is also how an expired or machine-mismatched
            PyArmor licence surfaces, hence the wording.
    """
    parent = os.path.dirname(pkg_dir)
    name = os.path.basename(pkg_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - PyArmor raises its own types on a licence failure
        raise ScannerError(
            f"could not import algorithm package {name!r} from {parent} "
            f"(missing, wrong platform, or licence rejected): {e}"
        ) from e


def _construct(module: Any, manifest: dict, slug: str) -> Any:
    """Call the package's factory and check the result honours the contract."""
    make = getattr(module, FACTORY_NAME, None)
    if make is None:
        raise ScannerError(
            f"algorithm {slug!r} exposes no {FACTORY_NAME}() factory at package top level"
        )
    try:
        obj = make(manifest) if manifest else make()
    except TypeError:
        obj = make()
    except Exception as e:  # noqa: BLE001
        raise ScannerError(f"algorithm {slug!r} failed to construct: {e}") from e

    if not hasattr(obj, "scan") or not callable(obj.scan):
        raise ScannerError(
            f"algorithm {slug!r} returned an object with no scan(grid) method — "
            "this is a build error, not a configuration one"
        )
    return obj


def _py_version(cp_tag: str) -> str:
    """``"cp313"`` -> ``"3.13"``; anything unrecognised comes back unchanged."""
    digits = cp_tag[2:] if cp_tag.startswith("cp") else ""
    if len(digits) >= 2 and digits.isdigit():
        return f"{digits[0]}.{digits[1:]}"
    return cp_tag


def _mismatch_message(slug: str, build_tag: str) -> str:
    """Explain a platform mismatch in terms of the axis that actually differs.

    The three axes fail for different reasons and have different remedies, and a message that
    prints both tags and leaves the reader to diff them is the same message for all three. A
    wrong Python is the common case and the one the user can fix in a single command; a wrong OS
    or architecture is a request to the publisher.

    Args:
        slug: The algorithm being loaded.
        build_tag: The tag the bundle carries.

    Returns:
        A message naming the mismatched axis and what to do about it.
    """
    local = default_platform_tag()
    b_os, _, rest = build_tag.partition("-")
    b_arch, _, b_py = rest.rpartition("-")
    l_os, _, rest = local.partition("-")
    l_arch, _, l_py = rest.rpartition("-")

    head = f"algorithm {slug!r} is built for {build_tag} but this machine is {local}"

    if (b_os, b_arch) == (l_os, l_arch) and b_py != l_py:
        return (
            f"{head}.\n"
            f"Only the Python version differs: this bundle needs Python {_py_version(b_py)} and "
            f"this interpreter is {_py_version(l_py)}. An obfuscated bundle's bytecode is "
            f"produced by the interpreter that built it, so it runs on that version and on no "
            f"other.\n"
            f"Reinstall against the Python the algorithms are built for:\n"
            f"    uv tool install --python {supported_python_text()} siar-app"
        )
    return (
        f"{head} — an obfuscated bundle is pinned to OS, CPU architecture and Python minor "
        f"version. Ask the publisher for a {local} build, or run it on a matching machine."
    )


def _load_tree(root: str, slug: str, platform_tag: str) -> AlgorithmHandle:
    """Load an already-unpacked bundle tree."""
    manifest = _read_manifest(root)
    build_tag = str(manifest.get("platform") or "").strip()
    if build_tag and not platform_compatible(build_tag):
        raise ScannerError(_mismatch_message(slug, build_tag))
    if build_tag and libc_flavour() == "musl":
        # The tag matched, so the OS, architecture and Python all agree — but "linux" covers
        # both glibc and musl and the tag cannot say which. Without this the import fails deep
        # in the dynamic linker, naming a symbol that means nothing to the person reading it.
        raise ScannerError(
            f"algorithm {slug!r} is built for {build_tag}, which matches this machine, but its "
            f"runtime is linked against glibc and this system uses musl (Alpine). The bundle "
            f"cannot be loaded here.\n"
            f"Run siar-app under a glibc image — python:{supported_python_text()}-slim works — "
            f"or ask the publisher for a musl build."
        )
    pkg_dir = _locate_package(root, str(manifest.get("package") or ""))
    if pkg_dir is None:
        raise ScannerError(f"no importable package found in the {slug!r} bundle at {root}")
    _place_license(root, os.path.dirname(pkg_dir), pkg_dir)
    module = _import_from(pkg_dir)
    algorithm = _construct(module, manifest, slug)
    return AlgorithmHandle(algorithm, slug, manifest, root, build_tag or platform_tag)


# -- the three entry points ----------------------------------------------------------------


def load_unpacked(root: str, slug: str, platform_tag: str = "") -> AlgorithmHandle:
    """Load a bundle that is already unpacked at ``root``.

    The entry point a worker process uses (see :mod:`siarapp.parallel`): the parent has already
    downloaded and unpacked the bundle, so a worker is given the tree directly rather than a slug
    it would have to look up — no network, no cache lookup, and no chance of a second process
    resolving a different build than the one the run was started with.

    Args:
        root: The unpacked bundle directory.
        slug: The algorithm slug, for error messages and the handle.
        platform_tag: Build tag to record when the manifest does not carry one.

    Returns:
        The loaded :class:`AlgorithmHandle`.

    Raises:
        ScannerError: On any import or contract failure.
    """
    return _load_tree(root, slug, platform_tag or default_platform_tag())


def load_cached(slug: str, platform_tag: str | None = None) -> AlgorithmHandle | None:
    """Load an algorithm already in the cache, or ``None`` if it is not there."""
    tag = platform_tag or default_platform_tag()
    root = algorithm_cache_dir(slug, tag)
    if not _is_complete(root):
        return None
    return _load_tree(root, slug, tag)


def load_remote(client: Any, slug: str, platform_tag: str | None = None,
                *, refresh: bool = False,
                on_progress: Callable[[int, int | None], None] | None = None) -> AlgorithmHandle:
    """Load an algorithm, downloading it first if the cache does not have it.

    Args:
        client: A :class:`siarapp.api.Client`.
        slug: The algorithm slug.
        platform_tag: Which build; defaults to this machine's.
        refresh: Re-download even when cached — how a user picks up a republished build.
        on_progress: Reported to only when a download actually happens, so a cache hit stays
            silent rather than flashing a completed bar at someone who waited for nothing.

    Returns:
        The loaded :class:`AlgorithmHandle`.

    Raises:
        ScannerError: On any download, unpack, import or contract failure.
    """
    tag = platform_tag or default_platform_tag()
    if not refresh:
        cached = load_cached(slug, tag)
        if cached is not None:
            return cached

    data = client.bundle(slug, tag, on_progress=on_progress)
    root = unpack_bundle(data, algorithm_cache_dir(slug, tag))
    return _load_tree(root, slug, tag)


def installed_algorithms() -> list[dict]:
    """Every fully-unpacked bundle in the cache, newest download first.

    Reads ``manifest.json`` and stats the tree; it never imports anything. That is the whole
    design of this function — answering "what have I got, and which version" must not run a
    customer's obfuscated code, must not need a network, and must not fail because one of the
    bundles is built for a different machine than the one asking.

    Returns:
        One dict per build with ``slug``, ``platform``, ``version``, ``family``, ``title``,
        ``shapes``, ``bytes``, ``downloaded_at`` (epoch seconds), ``runnable`` and ``root``.
        Incomplete
        directories — an interrupted unpack — are skipped, because they are not installed.
    """
    root_dir = os.path.join(home(), "algorithms")
    if not os.path.isdir(root_dir):
        return []

    rows: list[dict] = []
    for slug_dir in sorted(os.listdir(root_dir)):
        slug_path = os.path.join(root_dir, slug_dir)
        if not os.path.isdir(slug_path):
            continue
        for platform_dir in sorted(os.listdir(slug_path)):
            tree = os.path.join(slug_path, platform_dir)
            if not os.path.isdir(tree) or not _is_complete(tree):
                continue
            manifest = _read_manifest(tree)
            build_tag = str(manifest.get("platform") or platform_dir)
            rows.append({
                "slug": str(manifest.get("slug") or slug_dir),
                "platform": build_tag,
                "version": str(manifest.get("version") or "?"),
                "family": str(manifest.get("family") or ""),
                # What the bundle says it emits. The one thing about a closed algorithm's insides
                # that it is *for* telling you, and the answer to "what does this one look for"
                # for a bundle that has no build row to read it off.
                "shapes": tuple(str(s) for s in (manifest.get("shapes") or ())
                                if isinstance(s, (str, int, float))),
                "title": str(manifest.get("title") or ""),
                "bytes": _tree_bytes(tree),
                "downloaded_at": _stamped_at(tree),
                "runnable": platform_compatible(build_tag),
                "root": tree,
            })
    rows.sort(key=lambda r: (-r["downloaded_at"], r["slug"]))
    return rows


def _tree_bytes(root: str) -> int:
    """Total size on disk of an unpacked bundle."""
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _stamped_at(root: str) -> int:
    """When the bundle finished unpacking — the stamp's mtime, or 0 if it cannot be read.

    The stamp rather than the directory, because a run that reads the tree can touch the
    directory and would otherwise make an old download look like today's.
    """
    try:
        try:
            return int(os.path.getmtime(os.path.join(root, _STAMP)))
        except OSError:
            return int(os.path.getmtime(os.path.join(root, _LEGACY_STAMP)))
    except OSError:
        return 0


def load_local(path: str, slug: str = "") -> AlgorithmHandle:
    """Load an unobfuscated algorithm package straight off disk.

    The development path, behind ``--algorithm-path``: no login, no download, no platform check.
    It is how the pipeline gets tested before any bundle exists, and how a port gets debugged
    with a readable traceback.

    Args:
        path: Either the package directory itself (the one with ``__init__.py``) or a directory
            containing exactly one such package — which is what an algorithm's ``src/`` is.
        slug: Override the slug; defaults to the package's own ``slug`` attribute, else its
            directory name.

    Returns:
        The loaded :class:`AlgorithmHandle`.

    Raises:
        ScannerError: If no importable package is found, or the contract is not met.
    """
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise ScannerError(f"--algorithm-path {path!r} is not a directory")
    pkg_dir = root if _has_init(root) else _locate_package(root)
    if pkg_dir is None:
        raise ScannerError(f"no Python package (a directory with __init__.py) under {root}")
    module = _import_from(pkg_dir)
    # A slug given here is passed to the factory as a one-key manifest, because that is the only
    # thing a preset is selected by. Without it `--algorithm-path pkg/src -a <preset>` silently
    # ran the package's DEFAULT preset and then labelled every sidecar with the preset that was
    # asked for — the one failure mode a development path must not have, since the whole point
    # of it is trying a preset before a bundle exists.
    manifest = {"slug": slug} if slug else {}
    algorithm = _construct(module, manifest, slug or os.path.basename(pkg_dir))
    resolved = slug or str(getattr(algorithm, "slug", "") or os.path.basename(pkg_dir))
    return AlgorithmHandle(algorithm, resolved, manifest, pkg_dir, "source")


def _has_init(path: str) -> bool:
    """True when ``path`` is itself an importable package directory."""
    try:
        return any(f.startswith("__init__.") for f in os.listdir(path))
    except OSError:
        return False
