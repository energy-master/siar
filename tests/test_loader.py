# Vixen Intelligence c.2026
"""The obfuscated-bundle path: platform tags, unpacking, licence placement, and the guards.

``test_runner.py`` covers ``load_local``, which is the development path and skips all of this.
Everything a real customer's install actually executes — the tag that decides which bundle is
downloaded, the zip that arrives, the checks that refuse it — was untested until here.

The tag tests matter most. A platform tag is computed in three repositories and compared by
exact string equality; when they agree, downloads work, and when they drift, nothing raises and
the catalogue merely looks empty. Writing the expected strings down is the only way that failure
becomes visible.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

from siarapp import config
from siarapp.config import default_platform_tag, libc_flavour, platform_compatible
from siarapp.grid import ScannerError
from siarapp.loader import (
    _is_complete,
    _locate_package,
    _mismatch_message,
    _place_license,
    _py_version,
    _read_manifest,
    unpack_bundle,
)


# -- the platform tag -----------------------------------------------------------------------


@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", "darwin-arm64"),
    ("Darwin", "x86_64", "darwin-x86_64"),
    ("Linux", "x86_64", "linux-x86_64"),
    ("Linux", "aarch64", "linux-aarch64"),
    ("Windows", "AMD64", "windows-amd64"),
])
def test_tag_is_built_from_what_the_platform_module_reports(monkeypatch, system, machine,
                                                            expected):
    """macOS is ``darwin``, never ``macos``; Windows reports ``AMD64``, not ``x86_64``.

    Both spellings have been wrong in our own documentation, and a bundle published under a tag
    no machine reports is downloaded by nobody and fails silently.
    """
    monkeypatch.setattr(config.platform, "system", lambda: system)
    monkeypatch.setattr(config.platform, "machine", lambda: machine)
    tag = default_platform_tag()
    assert tag.startswith(expected + "-cp")
    assert "macos" not in tag


def test_tag_ends_with_this_interpreter():
    import sys

    assert default_platform_tag().endswith(f"-cp{sys.version_info[0]}{sys.version_info[1]}")


def test_tag_survives_a_platform_module_that_answers_nothing(monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "")
    monkeypatch.setattr(config.platform, "machine", lambda: "")
    assert default_platform_tag().startswith("any-any-cp")


# -- platform_compatible --------------------------------------------------------------------


def test_untagged_bundles_are_portable():
    """A pure-Python bundle has no platform, and refusing it for that would be wrong."""
    assert platform_compatible("") is True
    assert platform_compatible(None) is True
    assert platform_compatible("any") is True


def test_matching_is_exact_but_case_insensitive():
    assert platform_compatible("linux-x86_64-cp313", "LINUX-X86_64-CP313") is True
    assert platform_compatible("linux-x86_64-cp313", "linux-x86_64-cp312") is False
    assert platform_compatible("darwin-arm64-cp313", "linux-x86_64-cp313") is False


# -- the mismatch message -------------------------------------------------------------------


def test_python_only_mismatch_names_the_fix():
    """The common case, and the one the user can fix themselves in one command."""
    message = _mismatch_message("all_structures", "linux-x86_64-cp314")
    assert "uv tool install" in message
    assert "3.14" in message and "3.13" in message


def test_os_mismatch_asks_the_publisher_and_never_says_macos():
    message = _mismatch_message("all_structures", "darwin-arm64-cp313")
    assert "macos" not in message
    assert "publisher" in message


@pytest.mark.parametrize("tag,expected", [("cp313", "3.13"), ("cp39", "3.9"), ("nope", "nope")])
def test_py_version_reads_a_cp_tag(tag, expected):
    assert _py_version(tag) == expected


# -- unpacking ------------------------------------------------------------------------------


def _zip_bytes(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_unpack_writes_the_completion_stamp(tmp_path):
    dest = str(tmp_path / "bundle")
    unpack_bundle(_zip_bytes({"pkg/__init__.py": "x = 1\n"}), dest)
    assert os.path.isfile(os.path.join(dest, "pkg", "__init__.py"))
    assert _is_complete(dest)


def test_unpack_refuses_a_path_that_escapes_the_destination(tmp_path):
    """Zip-slip. The archive is ours, but it arrives over the network."""
    dest = str(tmp_path / "bundle")
    with pytest.raises(ScannerError, match="out-of-tree"):
        unpack_bundle(_zip_bytes({"../escaped.py": "x = 1\n"}), dest)
    assert not os.path.exists(str(tmp_path / "escaped.py"))


def test_unpack_refuses_something_that_is_not_a_zip(tmp_path):
    with pytest.raises(ScannerError, match="not a zip"):
        unpack_bundle(b"<html>404</html>", str(tmp_path / "bundle"))


def test_unpack_replaces_an_earlier_bundle(tmp_path):
    dest = str(tmp_path / "bundle")
    unpack_bundle(_zip_bytes({"pkg/old.py": "x = 1\n"}), dest)
    unpack_bundle(_zip_bytes({"pkg/new.py": "x = 2\n"}), dest)
    assert not os.path.exists(os.path.join(dest, "pkg", "old.py"))
    assert os.path.isfile(os.path.join(dest, "pkg", "new.py"))


def test_a_bundle_cached_under_the_old_product_name_is_still_complete(tmp_path):
    """The rename from siar-scanner must not force every user to re-download."""
    dest = tmp_path / "bundle"
    dest.mkdir()
    (dest / ".siar-scanner-complete").write_text("ok\n")
    assert _is_complete(str(dest))


def test_an_interrupted_unpack_is_not_complete(tmp_path):
    dest = tmp_path / "bundle"
    (dest / "pkg").mkdir(parents=True)
    assert not _is_complete(str(dest))


# -- manifest and package discovery ---------------------------------------------------------


def test_manifest_is_read_from_the_tree(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"slug": "x", "platform": "any"}))
    assert _read_manifest(str(tmp_path))["slug"] == "x"


def test_a_bundle_with_no_manifest_reads_as_empty(tmp_path):
    assert _read_manifest(str(tmp_path)) == {}


def test_package_lookup_prefers_the_manifest_name(tmp_path):
    """``package`` need not equal ``slug`` — fuzzy_sweep_scanner_beta ships as scanner_beta."""
    for name in ("scanner_beta", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("")
    assert os.path.basename(_locate_package(str(tmp_path), "scanner_beta")) == "scanner_beta"


def test_package_lookup_skips_the_pyarmor_runtime(tmp_path):
    (tmp_path / "pyarmor_runtime_000000").mkdir()
    (tmp_path / "pyarmor_runtime_000000" / "__init__.py").write_text("")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    assert os.path.basename(_locate_package(str(tmp_path), "")) == "pkg"


def test_package_lookup_returns_none_when_there_is_nothing_importable(tmp_path):
    (tmp_path / "data").mkdir()
    assert _locate_package(str(tmp_path), "") is None


# -- licence placement ----------------------------------------------------------------------


def test_licence_lands_in_both_places_the_runtime_might_look(tmp_path):
    """PyArmor searches the package directory or its parent depending on build layout.

    Copying to both is why a hardened bundle loads at all, and it is the least obvious code in
    the loader.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (tmp_path / "licence.rkey").write_bytes(b"KEY")
    _place_license(str(tmp_path), str(tmp_path), str(pkg))
    assert (pkg / "pyarmor.rkey").read_bytes() == b"KEY"
    assert (tmp_path / "pyarmor.rkey").read_bytes() == b"KEY"


def test_licence_placement_is_a_no_op_without_a_key(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _place_license(str(tmp_path), str(tmp_path), str(pkg))
    assert not (pkg / "pyarmor.rkey").exists()


# -- musl -----------------------------------------------------------------------------------


def test_libc_flavour_is_empty_off_linux(monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    assert libc_flavour() == ""


def test_musl_is_detected_when_libc_ver_says_nothing(monkeypatch):
    """``platform.libc_ver()`` reads the executable and comes back empty on Alpine."""
    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "libc_ver", lambda *a, **k: ("", ""))
    import glob as glob_module

    monkeypatch.setattr(glob_module, "glob", lambda p: ["/lib/ld-musl-x86_64.so.1"])
    assert libc_flavour() == "musl"


def test_glibc_is_reported_as_glibc(monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "libc_ver", lambda *a, **k: ("glibc", "2.39"))
    assert libc_flavour() == "glibc"
