# Vixen Intelligence c.2026
"""Reading an output folder from the outside — the half of ``serve`` with no socket in it.

Most of what could go wrong with serving a survey folder goes wrong here rather than in the HTTP
layer, and all of it is reachable by calling a function: a path that escapes the root, a file the
run never wrote, a half-written document, an index that disagrees with the manifest it came from.

The folders under test are built by the real pipeline — the same STUB algorithm ``test_runner``
uses through ``--algorithm-path`` — because a hand-made fixture would encode this module's
assumptions about the layout rather than the layout the writer actually produces.
"""
from __future__ import annotations

import json
import os

import pytest

from siarapp.loader import load_local
from siarapp.runner import RunOptions, run_folder
from siarapp.serve.folder import (
    INDEX_MAX_LIMIT,
    FolderError,
    PathRefused,
    ServedFolder,
    check_rel,
)
from test_runner import STUB, _write_package, _write_wav


@pytest.fixture()
def scan(tmp_path):
    """A real output folder: two scannable recordings, one too short, one in a subfolder."""
    src = tmp_path / "src"
    (src / "station-b").mkdir(parents=True)
    _write_wav(src / "loud.wav", amplitude=1.0)
    _write_wav(src / "station-b" / "quiet.wav", amplitude=1e-6)
    _write_wav(src / "tiny.wav", seconds=0.01)

    handle = load_local(_write_package(tmp_path / "algo", "stub_algo", STUB), "stub")
    out = tmp_path / "out"
    manifest = run_folder(handle, str(src), str(out), RunOptions())
    return ServedFolder(str(out)), manifest, out


# -- what it will and will not look up -----------------------------------------------------


@pytest.mark.parametrize("rel", [
    "../../etc/passwd",
    "/etc/passwd",
    "C:/Windows/win.ini",
    "station-b/../../escape.wav",
    "station-b/./quiet.wav",
    "..",
    "station-b//quiet.wav",
    "back\\slash.wav",
    "loud.wav\x00.png",
    "",
    "loud.wav.tmp-1234",
])
def test_check_rel_refuses_anything_that_is_not_a_plain_relative_path(rel):
    with pytest.raises(PathRefused):
        check_rel(rel)


def test_check_rel_allows_the_paths_a_run_actually_writes():
    for rel in ("loud.wav", "station-b/quiet.wav", "2026-07-03/0410.flac", "a.b/c d.wav"):
        assert check_rel(rel) == rel


def test_traversal_is_refused_at_every_kind(scan):
    folder, _manifest, _out = scan
    for rel in ("../../etc/passwd", "/etc/passwd", "station-b/../../x.wav", ".."):
        for kind in ("audio", "sidecar", "thumbnail"):
            assert folder.artefact(rel, kind) is None


def test_only_what_the_run_wrote_is_reachable(scan):
    """The manifest is the allowlist: a real file the run never listed is *unknown*, not merely
    refused, and no path arithmetic conjures a row into the manifest."""
    folder, _manifest, out = scan
    (out / "secret.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (out / "secret.structures.json").write_text('{"structures": []}')

    assert "secret.wav" not in folder.rel_paths()
    for kind in ("audio", "sidecar", "thumbnail"):
        assert folder.artefact("secret.wav", kind) is None
    assert folder.structures("secret.wav") is None
    assert all(row["path"] != "secret.wav" for row in folder.index()["files"])


def test_a_symlink_out_of_the_folder_is_refused(scan, tmp_path):
    """The one case syntax cannot catch: a listed path whose file resolves elsewhere."""
    folder, _manifest, out = scan
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF" + b"\0" * 32)
    target = out / "loud.wav"
    target.unlink()
    try:
        os.symlink(outside, target)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not do symlinks")
    assert folder.artefact("loud.wav", "audio") is None


def test_an_unknown_kind_is_refused(scan):
    folder, _manifest, _out = scan
    assert folder.artefact("loud.wav", "notes") is None
    assert folder.artefact("loud.wav", "../audio") is None


def test_the_artefacts_of_a_real_recording_resolve(scan):
    folder, _manifest, out = scan
    assert folder.artefact("loud.wav", "audio") == str(out / "loud.wav")
    assert folder.artefact("loud.wav", "sidecar") == str(out / "loud.structures.json")
    assert folder.artefact("loud.wav", "thumbnail") == str(out / "loud.png")
    assert folder.artefact("station-b/quiet.wav", "audio") == str(out / "station-b" / "quiet.wav")


# -- the index -----------------------------------------------------------------------------


def test_the_index_matches_the_manifest_row_for_row(scan):
    folder, manifest, _out = scan
    index = folder.index(limit=INDEX_MAX_LIMIT)

    assert index["total"] == manifest["files"]
    rows = {row["path"]: row for row in index["files"]}
    for source in manifest["manifest"]:
        row = rows[source["path"]]
        assert row["status"] == source["status"]
        assert row["structures"] == source["structures"]
        assert row["duration_sec"] == pytest.approx(source["duration_sec"])
        assert row["shapes"] == (source.get("shapes") or {})


def test_the_index_carries_no_boxes(scan):
    """Counts come from the manifest; the boxes themselves are a per-recording fetch. Inlining
    them would make a twelve-thousand-file survey a gigabytes-long response."""
    folder, _manifest, _out = scan
    for row in folder.index()["files"]:
        assert "structures" in row and isinstance(row["structures"], int)
        assert "boxes" not in row


def test_the_index_pages_and_never_exceeds_the_cap(scan):
    folder, manifest, _out = scan
    first = folder.index(offset=0, limit=1)
    second = folder.index(offset=1, limit=1)
    assert first["limit"] == 1 and len(first["files"]) == 1
    assert first["files"][0]["path"] != second["files"][0]["path"]
    assert first["total"] == second["total"] == manifest["files"]
    # A silly limit is clamped, not honoured, and junk falls back to the floor.
    assert folder.index(limit=10_000)["limit"] == INDEX_MAX_LIMIT
    assert folder.index(limit="banana")["limit"] == 1
    assert folder.index(offset=-5)["offset"] == 0


def test_the_index_filters_and_sorts_where_the_data_is(scan):
    folder, _manifest, _out = scan
    assert {r["status"] for r in folder.index(status="too_short")["files"]} == {"too_short"}
    assert [r["path"] for r in folder.index(query="quiet")["files"]] == ["station-b/quiet.wav"]
    assert all("blob" in r["shapes"] for r in folder.index(shape="blob")["files"])

    ascending = [r["path"] for r in folder.index(sort="path")["files"]]
    assert ascending == sorted(ascending)
    assert [r["path"] for r in folder.index(sort="path", order="desc")["files"]] == ascending[::-1]
    counts = [r["structures"] for r in folder.index(sort="structures", order="desc")["files"]]
    assert counts == sorted(counts, reverse=True)
    # An unknown sort key falls back to path rather than raising at the caller's expense.
    assert [r["path"] for r in folder.index(sort="nonsense")["files"]] == ascending


def test_a_row_says_whether_its_pictures_and_boxes_exist(scan):
    folder, _manifest, out = scan
    rows = {row["path"]: row for row in folder.index()["files"]}
    assert rows["loud.wav"]["thumbnail"] is True
    assert rows["loud.wav"]["sidecar"] is True
    assert rows["loud.wav"]["audio_bytes"] == os.path.getsize(out / "loud.wav")
    # tiny.wav is too short to picture: no thumbnail, and the row says so rather than 404ing later.
    assert rows["tiny.wav"]["thumbnail"] is False


# -- one recording -------------------------------------------------------------------------


def test_structures_come_one_recording_at_a_time(scan):
    folder, _manifest, out = scan
    document = folder.structures("loud.wav")
    assert document == json.loads((out / "loud.structures.json").read_text())
    assert document["structures"], "the loud recording should have produced a box"


def test_a_limit_truncates_the_boxes_but_not_the_count(scan):
    folder, _manifest, _out = scan
    full = folder.structures("loud.wav")
    limited = folder.structures("loud.wav", limit=0)
    assert limited["structures"] == []
    assert limited["truncated"] is True
    # `count` is what the run found, not what this reply carries.
    assert limited["count"] == full["count"]
    assert "truncated" not in full


# -- the folder as a live document ---------------------------------------------------------


def test_a_folder_with_no_manifest_says_so_rather_than_failing(tmp_path):
    folder = ServedFolder(str(tmp_path))
    assert folder.manifest() is None
    assert folder.meta()["state"] == "no-manifest"
    assert folder.index()["total"] == 0
    assert folder.rel_paths() == frozenset()
    assert folder.structures("anything.wav") is None


def test_a_folder_that_is_not_a_folder_is_refused(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(FolderError):
        ServedFolder(str(tmp_path / "file"))
    with pytest.raises(FolderError):
        ServedFolder(str(tmp_path / "nope"))


def test_a_rewritten_manifest_is_picked_up(scan):
    """The daemon serves runs that are still going, so the index cannot be read once and held."""
    folder, manifest, out = scan
    assert folder.index()["total"] == manifest["files"]

    trimmed = dict(manifest)
    trimmed["files"] = 1
    trimmed["manifest"] = manifest["manifest"][:1]
    (out / "siar-app-run.json").write_text(json.dumps(trimmed))
    os.utime(out / "siar-app-run.json", (0, 0))  # move the mtime so the stamp changes

    assert folder.manifest(now=1e12)["files"] == 1
    assert folder.index()["total"] == 1


def test_a_half_written_manifest_keeps_the_last_good_one(scan):
    """`os.replace` makes a torn read almost impossible, but a broken document must not blank a
    working page — the next poll will pick up the finished one."""
    folder, manifest, out = scan
    assert folder.index()["total"] == manifest["files"]
    (out / "siar-app-run.json").write_text('{"format": "siar-app-run-v1", "manif')
    os.utime(out / "siar-app-run.json", (0, 0))
    assert folder.manifest(now=1e12)["files"] == manifest["files"]


def test_meta_reports_the_run_and_admits_what_its_index_covers(scan):
    folder, manifest, _out = scan
    meta = folder.meta()

    assert meta["algorithm"]["slug"] == "stub"
    assert meta["stft"] == manifest["stft"]
    assert meta["state"] == "complete"
    assert meta["totals"]["by_status"] == manifest["by_status"]
    assert meta["totals"]["structures"] == manifest["structures"]
    assert meta["performance"] is True
    # The manifest describes a run, not a census of the folder, and the reply says which it is.
    assert meta["index_source"] == "run-manifest"
    assert meta["index_covers"] == manifest["files"]
    assert meta["source_root"]
    assert "source_root" not in folder.meta(include_source_root=False)


def test_the_performance_report_leaves_its_per_file_array_out_by_default(scan):
    folder, manifest, _out = scan
    totals_only = folder.performance()
    assert "files" not in totals_only
    assert totals_only["files_total"] == manifest["files"]
    assert totals_only["totals"]["phases"]

    with_files = folder.performance(files=True, limit=1)
    assert len(with_files["files"]) == 1
    assert with_files["files_total"] == manifest["files"]


def test_a_transient_write_is_invisible(scan):
    """`io.output` writes through `*.tmp-<pid>`, which holds half a document for microseconds."""
    folder, _manifest, out = scan
    (out / "loud.wav.tmp-999").write_bytes(b"RIFF")
    assert folder.artefact("loud.wav.tmp-999", "audio") is None
    assert all(not row["path"].endswith(".tmp-999") for row in folder.index()["files"])
