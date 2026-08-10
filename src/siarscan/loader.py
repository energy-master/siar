# Vixen Intelligence c.2026
"""Getting a scanning algorithm onto this machine and into this process.

An algorithm ships as a zip: an obfuscated Python package, PyArmor's runtime, a licence key if
the build has one, and a ``manifest.json`` describing what is inside. This module downloads it
once, unpacks it under ``~/.siar-scanner/algorithms/<slug>/<platform>/``, makes it importable,
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
  manifest's tag first lets us say "ask the publisher for a macos-arm64-cp313 build".
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
from typing import Any

from siarscan.config import algorithm_cache_dir, default_platform_tag, platform_compatible
from siarscan.grid import ScannerError

__all__ = [
    "FACTORY_NAME",
    "AlgorithmHandle",
    "load_cached",
    "load_local",
    "load_remote",
    "unpack_bundle",
]

#: The callable an algorithm package must expose at its top level.
FACTORY_NAME = "algorithm"

#: Marker written beside an unpacked bundle so a later run knows the cache is complete. Without
#: it, an interrupted unpack leaves a half-tree that imports and then fails oddly.
_STAMP = ".siar-scanner-complete"


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
    return os.path.isfile(os.path.join(path, _STAMP))


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


def _load_tree(root: str, slug: str, platform_tag: str) -> AlgorithmHandle:
    """Load an already-unpacked bundle tree."""
    manifest = _read_manifest(root)
    build_tag = str(manifest.get("platform") or "").strip()
    if build_tag and not platform_compatible(build_tag):
        raise ScannerError(
            f"algorithm {slug!r} is built for {build_tag} but this machine is "
            f"{default_platform_tag()} — an obfuscated bundle is pinned to OS, CPU "
            f"architecture and Python minor version. Ask the publisher for a "
            f"{default_platform_tag()} build, or run it on a matching machine."
        )
    pkg_dir = _locate_package(root, str(manifest.get("package") or ""))
    if pkg_dir is None:
        raise ScannerError(f"no importable package found in the {slug!r} bundle at {root}")
    _place_license(root, os.path.dirname(pkg_dir), pkg_dir)
    module = _import_from(pkg_dir)
    algorithm = _construct(module, manifest, slug)
    return AlgorithmHandle(algorithm, slug, manifest, root, build_tag or platform_tag)


# -- the three entry points ----------------------------------------------------------------


def load_cached(slug: str, platform_tag: str | None = None) -> AlgorithmHandle | None:
    """Load an algorithm already in the cache, or ``None`` if it is not there."""
    tag = platform_tag or default_platform_tag()
    root = algorithm_cache_dir(slug, tag)
    if not _is_complete(root):
        return None
    return _load_tree(root, slug, tag)


def load_remote(client: Any, slug: str, platform_tag: str | None = None,
                *, refresh: bool = False) -> AlgorithmHandle:
    """Load an algorithm, downloading it first if the cache does not have it.

    Args:
        client: A :class:`siarscan.api.Client`.
        slug: The algorithm slug.
        platform_tag: Which build; defaults to this machine's.
        refresh: Re-download even when cached — how a user picks up a republished build.

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

    data = client.bundle(slug, tag)
    root = unpack_bundle(data, algorithm_cache_dir(slug, tag))
    return _load_tree(root, slug, tag)


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
    algorithm = _construct(module, {}, slug or os.path.basename(pkg_dir))
    resolved = slug or str(getattr(algorithm, "slug", "") or os.path.basename(pkg_dir))
    return AlgorithmHandle(algorithm, resolved, {}, pkg_dir, "source")


def _has_init(path: str) -> bool:
    """True when ``path`` is itself an importable package directory."""
    try:
        return any(f.startswith("__init__.") for f in os.listdir(path))
    except OSError:
        return False
