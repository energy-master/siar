# Vixen Intelligence c.2026
"""Moving a model to another machine, and refusing to be moved into.

Two things are pinned here. A bundle must be **enough**: what comes out of an import has to load
through the ordinary loader and carry the bots and figures that make a detection worth trusting,
because a model that arrives runnable but anonymous is one nobody can vouch for. And an import
must be **suspicious**: the file has been carried on a stick between machines, so the archive is
somebody else's data structure until every member of it has been checked.
"""
from __future__ import annotations

import json
import os
import tarfile

import pytest

from siarapp import transfer
from siarapp.library import IMPORTED, Model, Program, imported_models, local_model


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A siar-app workspace of its own, so nothing here touches the real one."""
    home = tmp_path / "workspace"
    monkeypatch.setenv("SIAR_APP_HOME", str(home))
    return home


def _package(root, name="siar_thing", slug="thing"):
    """A minimal algorithm package: importable, and honouring the loader's contract."""
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "slug = %r\n"
        "version = '1.2.3'\n"
        "class _Scanner:\n"
        "    def scan(self, grid):\n"
        "        return []\n"
        "def algorithm(manifest=None):\n"
        "    return _Scanner()\n" % slug
    )
    (root / "report.txt").write_text("held-out AUC 0.81\n")
    (root / "model.json").write_text('{"threshold": 0.5}')
    runners = root / "runners"
    runners.mkdir()
    (runners / "rank_01.json").write_text('{"rank": 1}')
    return package


@pytest.fixture()
def model(tmp_path):
    """A built model, as :func:`siarapp.library.built_models` would report one."""
    package = _package(tmp_path / "built")
    return Model(
        source="built", slug="thing", title="thing", version="0.1.0", platform="source",
        path=str(package), size_bytes=1234, stamped_at=1_760_000_000, runnable=True,
        detail={"target": "thing", "created_at": "2026-08-01T00:00:00+00:00",
                "held_out_auc": 0.81, "input_dir": "/corpus/on/the/other/machine"},
        programs=[
            Program(rank=0, kind="champion", fitness=0.9, threshold=0.5, polarity=1, n_nodes=7,
                    depth=3, features=("band_1000_2000hz",), infix="x > 1",
                    saved_path=str(tmp_path / "built" / "model.json")),
            Program(rank=1, kind="runner_up", fitness=0.8, threshold=None, n_nodes=5, depth=2,
                    features=("band_2000_3000hz",), infix="y > 2",
                    saved_path=str(tmp_path / "built" / "runners" / "rank_01.json")),
        ],
    )


# -- the round trip ----------------------------------------------------------------------------


def test_a_model_survives_the_trip_with_everything_that_makes_it_trustworthy(
        model, tmp_path, workspace):
    """The whole point: what lands on the far side is runnable *and* still knowable."""
    bundle = transfer.export_model(model, str(tmp_path))
    assert bundle.endswith(transfer.BUNDLE_SUFFIX)

    row = transfer.import_bundle(bundle)

    assert row["runnable"] and os.path.isdir(row["path"])
    assert row["slug"] == "thing"
    assert len(row["programs"]) == 2, "the runners-up travel with the champion"
    assert row["build"]["held_out_auc"] == 0.81, "the figures it was gated on travel with it"
    assert row["exported_from"], "a bundle says which machine it came off"
    assert os.path.isfile(row["programs"][0]["saved_path"]), "and the calibration it was given"


def test_an_imported_model_loads_through_the_ordinary_loader(model, tmp_path, workspace):
    """No second code path on the far side: it is a package on disk like any other."""
    from siarapp.loader import load_local

    transfer.import_bundle(transfer.export_model(model, str(tmp_path)))
    installed = imported_models()

    assert len(installed) == 1 and installed[0].source == IMPORTED
    assert installed[0].local and not installed[0].built
    handle = load_local(installed[0].path, installed[0].slug)
    assert handle.slug == "thing"
    assert handle.algorithm.scan(None) == []


def test_the_package_keeps_its_own_directory_name(model, tmp_path, workspace):
    """The loader imports a package *by its directory name*; flattening two models to one name
    would make them the same module on the far side."""
    row = transfer.import_bundle(transfer.export_model(model, str(tmp_path)))

    assert os.path.basename(row["path"]) == "siar_thing"


def test_importing_the_same_model_twice_leaves_one_of_it(model, tmp_path, workspace):
    bundle = transfer.export_model(model, str(tmp_path))
    first = transfer.import_bundle(bundle)
    second = transfer.import_bundle(bundle)

    assert first["uid"] == second["uid"]
    assert len(transfer.installed_bundles()) == 1


def test_an_imported_model_is_runnable_by_name(model, tmp_path, workspace):
    """`siar-app run -a thing` after an import, with no --algorithm-path anywhere."""
    transfer.import_bundle(transfer.export_model(model, str(tmp_path)))

    found = local_model("thing")
    assert found is not None and found.imported
    assert local_model("nothing-by-that-name") is None


def test_a_bundle_carries_the_provenance_but_not_the_survey(model, tmp_path, workspace):
    """siar-build writes scans and working files under the same tree; a bundle is a model."""
    scans = tmp_path / "built" / "scans"
    scans.mkdir()
    (scans / "huge.npy").write_bytes(b"0" * 4096)

    with tarfile.open(transfer.export_model(model, str(tmp_path))) as tar:
        names = tar.getnames()

    assert "docs/report.txt" in names and "docs/runners/rank_01.json" in names
    assert not any("scans" in name for name in names), "the corpus stays where it was"
    assert not any("__pycache__" in name for name in names)


def test_the_workspace_can_be_moved_without_breaking_what_is_in_it(model, tmp_path, workspace):
    """Paths in the record are relative, so a workspace on a stick still reads on the far side."""
    transfer.import_bundle(transfer.export_model(model, str(tmp_path)))
    record = json.loads((workspace / "models").glob("*/import.json").__next__().read_text())

    assert not os.path.isabs(record["package"])
    assert all(not os.path.isabs(p["saved_path"] or "x") for p in record["programs"])


def test_removing_an_imported_model_is_the_same_end_state_twice(model, tmp_path, workspace):
    row = transfer.import_bundle(transfer.export_model(model, str(tmp_path)))

    assert transfer.remove_bundle(row["uid"]) is True
    assert transfer.remove_bundle(row["uid"]) is False
    assert transfer.installed_bundles() == []


# -- what will not be exported ------------------------------------------------------------------


def test_a_downloaded_bundle_is_refused_with_the_way_forward(tmp_path):
    """It is licensed per machine and platform; carrying it would be carrying a dead file."""
    bundle = Model(source="downloaded", slug="all_structures", path=str(tmp_path), runnable=True)

    with pytest.raises(transfer.TransferError, match="siar-app login"):
        transfer.export_model(bundle, str(tmp_path))


def test_a_build_that_was_never_packaged_is_refused_with_its_own_reason(tmp_path):
    unpackaged = Model(source="built", slug="thing", path="", runnable=False,
                       note="stopped at calibration — nothing packaged")

    with pytest.raises(transfer.TransferError, match="nothing packaged"):
        transfer.export_model(unpackaged, str(tmp_path))


# -- what will not be imported -------------------------------------------------------------------


def _archive(path, members, manifest=None):
    """A tarball built by hand, to say things a real export never would."""
    with tarfile.open(path, "w:gz") as tar:
        if manifest is not None:
            info, blob = transfer._json_member(transfer.MANIFEST_NAME, manifest)
            tar.addfile(info, blob)
        for name, target in members:
            info = tarfile.TarInfo(name)
            if target is None:
                info.size = 4
                tar.addfile(info, __import__("io").BytesIO(b"data"))
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
    return str(path)


@pytest.fixture()
def manifest():
    return {"kind": "siar-model", "format": transfer.BUNDLE_FORMAT, "slug": "thing",
            "uid": "thing-1", "package": "siar_thing", "programs": [], "build": {}}


def test_an_archive_that_climbs_out_of_the_folder_is_refused(tmp_path, manifest, workspace):
    """The classic: a member whose path walks up and writes over something else."""
    path = _archive(tmp_path / "evil.siarmodel", [("../../escaped.py", None)], manifest)

    with pytest.raises(transfer.TransferError, match="climbs out"):
        transfer.import_bundle(path)
    assert not (tmp_path.parent / "escaped.py").exists()


def test_an_archive_carrying_a_link_is_refused(tmp_path, manifest, workspace):
    """A symlink is how an archive reaches a file outside the directory it was given."""
    path = _archive(tmp_path / "linky.siarmodel",
                    [("package/siar_thing/keys", "/etc/passwd")], manifest)

    with pytest.raises(transfer.TransferError, match="link"):
        transfer.import_bundle(path)


def test_an_absolute_member_is_refused(tmp_path, manifest, workspace):
    path = _archive(tmp_path / "abs.siarmodel", [("/etc/cron.d/thing", None)], manifest)

    with pytest.raises(transfer.TransferError, match="absolute"):
        transfer.import_bundle(path)


def test_an_archive_that_would_fill_the_disk_is_refused(tmp_path, manifest, workspace,
                                                        monkeypatch):
    monkeypatch.setattr(transfer, "MAX_BUNDLE_BYTES", 3)
    path = _archive(tmp_path / "big.siarmodel", [("package/siar_thing/__init__.py", None)],
                    manifest)

    with pytest.raises(transfer.TransferError, match="not a model"):
        transfer.import_bundle(path)


def test_a_tarball_that_is_not_a_model_says_so(tmp_path, workspace):
    path = _archive(tmp_path / "random.siarmodel", [("notes.txt", None)])

    with pytest.raises(transfer.TransferError, match="not a siar-app model"):
        transfer.import_bundle(path)


def test_a_bundle_from_a_newer_siar_app_says_to_upgrade(tmp_path, manifest, workspace):
    manifest["format"] = transfer.BUNDLE_FORMAT + 1
    path = _archive(tmp_path / "future.siarmodel", [("package/siar_thing/__init__.py", None)],
                    manifest)

    with pytest.raises(transfer.TransferError, match="newer siar-app"):
        transfer.import_bundle(path)


def test_a_bundle_with_no_package_in_it_is_refused(tmp_path, manifest, workspace):
    path = _archive(tmp_path / "empty.siarmodel", [("docs/report.txt", None)], manifest)

    with pytest.raises(transfer.TransferError, match="no importable package"):
        transfer.import_bundle(path)


def test_a_failed_import_leaves_the_model_that_was_already_here(model, tmp_path, workspace,
                                                               manifest):
    """An import that goes wrong half way must not be how a working model is lost."""
    good = transfer.import_bundle(transfer.export_model(model, str(tmp_path)))
    manifest["uid"] = good["uid"]
    bad = _archive(tmp_path / "bad.siarmodel", [("../escaped.py", None)], manifest)

    with pytest.raises(transfer.TransferError):
        transfer.import_bundle(bad)
    assert os.path.isdir(good["path"]), "the one that was here is untouched"
    assert len(transfer.installed_bundles()) == 1
