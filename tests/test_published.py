# Vixen Intelligence c.2026
"""Models another account published to the installation, fetched onto this machine.

Three things are worth pinning, and none of them is about the transport.

The first is that what arrives is checked before it is unpacked. The catalogue row and the zip
come back from two separate requests, and a package that is not the one the row described is the
one thing that must never reach the loader — so the digest is compared, and a mismatch leaves
nothing on disk to be run by accident.

The second is that a published model is somebody else's. It runs here, it is listed here, and it
cannot be exported: who may run one is a grant on the installation, and a bundle written out of
this cache would hand it to an account that was never given it.

The third is that a fetched model behaves like every other model on this disk. It resolves by
name, it loads through :func:`siarapp.loader.load_local`, and its members show up in the library —
read once at fetch time out of ``soc.json``, because that document runs to a couple of hundred
kilobytes and a listing must not open it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import pytest

from siarapp import library, published
from siarapp.grid import ScannerError

#: A catalogue row as ``soc_model_list.php`` sends one.
ROW = {
    "id": 12,
    "slug": "socmodel_recall_83vy3g",
    "package": "siar_socmodel_recall_83vy3g",
    "owner": "rahul",
    "access": "granted",
    "published": True,
    "target": "recall",
    "title": "Sonar recall — society of 20",
    "description": "Fires where 6 of 20 metrics agree.",
    "version": "0.1.0+83vy3g",
    "family": "brahma_society",
    "platform": "any",
    "vote": {"n": 20, "k": 6},
    "expects": {"fft": 32768, "hop": 8192, "window": "hann", "sample_rate": 96000},
    "unseen": {"roc_auc": 0.914, "n": 40},
    "manifest": {"slug": "socmodel_recall_83vy3g", "package": "siar_socmodel_recall_83vy3g",
                 "platform": "any", "version": "0.1.0+83vy3g", "shapes": ["recall"]},
}

#: One member of a society, as ``soc.json`` records one.
MEMBER = {
    "name": "recall_a1_vxbot", "uid": "a1", "rank": 0, "kind": "champion",
    "arena_auc": 0.88, "n_nodes": 39, "infix": "band_1000hz - peak_band_share",
    "features_used": ["band_1000hz", "peak_band_share"],
    "program": {"threshold": 0.0776, "polarity": 1, "depth": 8, "n_nodes": 39},
}


def _zip(package: str = "siar_socmodel_recall_83vy3g", *, document: dict | None = None) -> bytes:
    """A published package as the install stores one: a zip rooted at the package directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{package}/__init__.py", "def algorithm(manifest=None):\n    return 1\n")
        archive.writestr(f"{package}/soc.py", "# the scanner\n")
        archive.writestr(f"{package}/soc.json", json.dumps(
            document if document is not None
            else {"soc": "socmodel_recall_83vy3g", "target": "recall",
                  "vote": {"n": 20, "k": 6}, "members": [MEMBER]}))
    return buffer.getvalue()


def _row(**over) -> dict:
    """The catalogue row, with its digest matching the payload unless a test says otherwise."""
    row = dict(ROW)
    row["sha256"] = hashlib.sha256(over.pop("payload", _zip())).hexdigest()
    row["bytes"] = len(_zip())
    row.update(over)
    return row


class _FakeClient:
    """The install, as far as this module can tell: a catalogue and one package.

    Records what was asked for, because "did it go to the network at all" is half of what is
    being pinned here — a model already on this disk must not be fetched again.
    """

    def __init__(self, rows=None, payload=None, base_url="https://example.test"):
        self.base_url = base_url
        self.rows = rows if rows is not None else [_row()]
        self.payload = payload if payload is not None else _zip()
        self.listed = 0
        self.downloads: list[dict] = []

    def published_models(self):
        self.listed += 1
        return [dict(row) for row in self.rows]

    def published_bundle(self, slug="", *, model_id=0, owner="", on_progress=None):
        self.downloads.append({"slug": slug, "id": model_id, "owner": owner})
        if on_progress is not None:
            on_progress(len(self.payload), len(self.payload))
        return self.payload


# -- installing --------------------------------------------------------------------------------


def test_a_fetched_package_lands_runnable_and_names_who_published_it():
    row = published.install_published(_row(), _zip(), base_url="https://example.test")

    assert row["slug"] == "socmodel_recall_83vy3g"
    assert row["owner"] == "rahul"
    assert row["runnable"]
    assert os.path.isfile(os.path.join(row["path"], "__init__.py"))
    assert row["base_url"] == "https://example.test"
    assert published.installed_published() == [row]


def test_the_manifest_is_written_back_beside_the_package():
    # It travels with the catalogue rather than inside the zip, and the loader reads it off disk:
    # without this the tree here would not be the tree the box that bred it has.
    row = published.install_published(_row(), _zip())
    beside = os.path.join(row["root"], published.MANIFEST_NAME)

    with open(beside, encoding="utf-8") as fh:
        assert json.load(fh)["version"] == "0.1.0+83vy3g"


def test_a_package_that_is_not_the_one_the_catalogue_described_is_refused():
    row = _row()
    row["sha256"] = "0" * 64

    with pytest.raises(ScannerError, match="not the one rahul published"):
        published.install_published(row, _zip())
    assert published.installed_published() == []


def test_a_zip_with_no_importable_package_is_refused_and_leaves_nothing_behind():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "no package here")
    payload = buffer.getvalue()

    with pytest.raises(ScannerError, match="no importable directory"):
        published.install_published(_row(payload=payload), payload)
    assert published.installed_published() == []


def test_the_members_are_read_out_of_the_document_once_at_fetch_time():
    row = published.install_published(_row(), _zip())

    assert [p["name"] for p in row["programs"]] == ["recall_a1_vxbot"]
    assert row["programs"][0]["threshold"] == pytest.approx(0.0776)
    assert row["programs"][0]["features"] == ["band_1000hz", "peak_band_share"]
    # Listing must not have to open soc.json again: the record carries what a screen shows.
    os.remove(os.path.join(row["path"], "soc.json"))
    assert len(published.installed_published()[0]["programs"]) == 1


def test_a_model_that_is_not_a_society_still_installs():
    # A document this version cannot read is no members, never a failed fetch.
    payload = _zip(document={"schema": "something-newer"})
    row = published.install_published(_row(payload=payload), payload)

    assert row["runnable"]
    assert row["programs"] == []


def test_two_accounts_publishing_one_name_are_two_models_here():
    published.install_published(_row(), _zip())
    published.install_published(_row(owner="chandra"), _zip())

    assert len(published.installed_published()) == 2
    assert published.cached_published("socmodel_recall_83vy3g", "chandra")["owner"] == "chandra"


def test_a_download_interrupted_before_its_record_is_not_installed():
    row = published.install_published(_row(), _zip())
    os.remove(os.path.join(row["root"], published.RECORD_NAME))

    assert published.installed_published() == []


def test_removing_one_takes_it_off_the_disk_and_says_whether_there_was_anything():
    published.install_published(_row(), _zip())

    assert published.remove_published("socmodel_recall_83vy3g") is True
    assert published.remove_published("socmodel_recall_83vy3g") is False
    assert published.installed_published() == []


# -- fetching ----------------------------------------------------------------------------------


def test_fetching_asks_the_install_once_and_then_never_again():
    client = _FakeClient()

    first = published.fetch_published(client, "socmodel_recall_83vy3g")
    second = published.fetch_published(client, "socmodel_recall_83vy3g")

    assert first["path"] == second["path"]
    assert client.listed == 1 and len(client.downloads) == 1


def test_refresh_fetches_again_because_a_society_republishes_as_it_improves():
    client = _FakeClient()
    published.fetch_published(client, "socmodel_recall_83vy3g")

    published.fetch_published(client, "socmodel_recall_83vy3g", refresh=True)

    assert len(client.downloads) == 2


def test_a_download_names_the_row_by_id_rather_than_by_name():
    # (owner, slug) is the identity on the install, and the id says which exactly.
    client = _FakeClient()
    published.fetch_published(client, "socmodel_recall_83vy3g")

    assert client.downloads[0]["id"] == 12
    assert client.downloads[0]["owner"] == "rahul"


def test_a_name_this_account_has_not_been_granted_says_what_it_has():
    client = _FakeClient()

    with pytest.raises(ScannerError, match="socmodel_recall_83vy3g"):
        published.fetch_published(client, "something_else")


def test_an_account_with_no_grants_at_all_is_told_that_rather_than_offered_a_list():
    client = _FakeClient(rows=[])

    with pytest.raises(ScannerError, match="granted per account"):
        published.fetch_published(client, "anything")


def test_a_name_two_accounts_published_resolves_to_this_accounts_own_upload():
    mine = _row(owner="me", access="owner", id=99)
    theirs = _row(owner="rahul", access="granted", id=12)

    assert published.choose_published([theirs, mine], "socmodel_recall_83vy3g")["id"] == 99
    assert published.choose_published([theirs, mine], "socmodel_recall_83vy3g", "rahul")["id"] == 12


# -- what the library makes of one -------------------------------------------------------------


def test_a_fetched_model_is_listed_as_a_society_belonging_to_whoever_published_it():
    published.install_published(_row(), _zip(), base_url="https://example.test")

    model = library.published_models()[0]

    assert model.source == library.PUBLISHED
    assert model.published and model.society and model.local
    assert model.target == "recall"
    assert model.detail["owner"] == "rahul"
    assert model.detail["votes"] == "6 of 20"
    assert model.detail["install"] == "https://example.test"
    assert [p.label for p in model.programs] == ["recall_a1_vxbot"]


def test_it_runs_by_name_without_a_path_and_without_the_network():
    published.install_published(_row(), _zip())

    model = library.local_model("socmodel_recall_83vy3g")

    assert model is not None and model.published
    assert os.path.isfile(os.path.join(model.path, "__init__.py"))


def test_a_model_built_here_wins_a_name_a_published_one_also_answers_to(tmp_path, monkeypatch):
    # A cache of somebody else's model must never shadow this machine's own bench.
    package = tmp_path / "siar_recall"
    package.mkdir()
    (package / "__init__.py").write_text("def algorithm(manifest=None):\n    return None\n")
    published.install_published(_row(slug="recall"), _zip())
    monkeypatch.setattr(library, "built_models", lambda *a, **k: [
        library.Model(source=library.BUILT, slug="recall", path=str(package), runnable=True)])

    assert library.local_model("recall").source == library.BUILT


def test_loading_one_reads_the_manifest_that_came_with_it_so_the_build_is_named():
    """A society republishes as its leaderboard moves, so "which build drew these boxes" is a
    question about an output folder that only the version can answer. It is in the manifest that
    travelled with the package, which is why a published model is loaded through the tree rather
    than straight at the directory."""
    from siarapp.cli.commands import _load_disk

    package = "siar_loadtest_pub"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{package}/__init__.py",
                         "class S:\n"
                         "    shapes = ('recall',)\n"
                         "    def scan(self, grid):\n        return []\n"
                         "def algorithm(manifest=None):\n    return S()\n")
    payload = buffer.getvalue()
    manifest = dict(ROW["manifest"], package=package, slug="loadtest_pub")
    row = published.install_published(
        _row(payload=payload, slug="loadtest_pub", package=package, manifest=manifest), payload)

    handle = _load_disk(row["path"], row["slug"], published=True)

    assert handle.describe() == {"slug": "loadtest_pub", "version": "0.1.0+83vy3g",
                                 "platform": "any", "shapes": ["recall"]}


def test_it_cannot_be_exported_because_it_was_granted_rather_than_made_here():
    from siarapp.transfer import TransferError, export_model

    published.install_published(_row(), _zip())
    model = library.published_models()[0]

    assert not model.portable
    with pytest.raises(TransferError, match="published to the installation by rahul"):
        export_model(model)
