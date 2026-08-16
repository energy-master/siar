# Vixen Intelligence c.2026
"""The index behind ``siar-app lib``.

Two things are worth pinning here, and neither is cosmetic.

The first is that siar-build's database belongs to another program. This package reads it and
must never write it, must never create it, and must never fail because it is a version it has not
seen — a column added or dropped over there is a blank field on a screen here, not a traceback in
the middle of a listing.

The second is that "runnable" means something specific and different on each side of the seam: a
downloaded bundle runs if it was built for this machine, and a built model runs if its package is
still on the disk. Both are stated in the listing rather than being used to hide rows, because
"why is it not in the list" is a worse question than "why does this one say no".
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from siarapp import library


def _index(path, *, package_dir="", stopped_at="", features=("band_1000hz", "peak_band_share")):
    """Write a models.db shaped like siar-build's, with one build and two programs."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE builds (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, siarbuild_version TEXT,
            target TEXT, slug TEXT, input_dir TEXT, output_dir TEXT, package_dir TEXT,
            sample_rate INTEGER, n_fft INTEGER, hop INTEGER, n_bins INTEGER,
            fmin_hz REAL, fmax_hz REAL, held_out_auc REAL, null_auc REAL, suspect INTEGER,
            parity_ok INTEGER, stopped_at TEXT, reason TEXT
        );
        CREATE TABLE programs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, build_id INTEGER, rank INTEGER, kind TEXT,
            infix TEXT, n_nodes INTEGER, depth INTEGER, fitness REAL, threshold REAL,
            polarity INTEGER, features_used TEXT, saved_path TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO builds (created_at, siarbuild_version, target, slug, input_dir, output_dir,"
        " package_dir, sample_rate, n_fft, hop, n_bins, fmin_hz, fmax_hz, held_out_auc, null_auc,"
        " suspect, parity_ok, stopped_at, reason) VALUES"
        " ('2026-08-15T21:05:51+00:00', '0.1.0', 'recall', 'recall', '/audio', '/models', ?,"
        " 96000, 8192, 2048, 128, 5000.0, 7800.0, 0.78, 0.51, 0, 1, ?, '')",
        (package_dir or None, stopped_at),
    )
    conn.execute(
        "INSERT INTO programs (build_id, rank, kind, infix, n_nodes, depth, fitness, threshold,"
        " polarity, features_used, saved_path) VALUES"
        " (1, 0, 'champion', 'band_1000hz - peak_band_share', 15, 8, 0.99, -0.36, 1, ?, '/m.json')",
        (json.dumps(list(features)),),
    )
    conn.execute(
        "INSERT INTO programs (build_id, rank, kind, infix, n_nodes, depth, fitness, threshold,"
        " polarity, features_used, saved_path) VALUES"
        " (1, 1, 'runner_up', 'band_1000hz', 3, 2, 0.98, NULL, 1, ?, NULL)",
        (json.dumps([features[0]]),),
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def package(tmp_path):
    """A generated model package, as `siar-build package` leaves one."""
    pkg = tmp_path / "models" / "siar_recall"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("def algorithm(manifest=None):\n    return None\n")
    return pkg


def test_a_missing_database_is_an_empty_library_not_a_failure(tmp_path):
    assert library.built_models(str(tmp_path / "nothing" / "models.db")) == []


def test_a_missing_database_is_not_created_by_looking_for_it(tmp_path):
    path = tmp_path / "models.db"
    library.built_models(str(path))
    assert not path.exists(), "reading another program's index must not leave a file behind"


def test_a_database_that_is_not_one_is_an_empty_library(tmp_path):
    path = tmp_path / "models.db"
    path.write_bytes(b"this is not sqlite")
    assert library.built_models(str(path)) == []


def test_a_schema_without_the_columns_we_read_still_lists_its_builds(tmp_path):
    path = tmp_path / "models.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE builds (id INTEGER PRIMARY KEY, created_at TEXT, slug TEXT)")
    conn.execute("INSERT INTO builds (created_at, slug) VALUES ('2026-01-01T00:00:00+00:00', 'x')")
    conn.commit()
    conn.close()

    models = library.built_models(str(path))
    assert [m.slug for m in models] == ["x"]
    assert models[0].runnable is False
    assert models[0].programs == []


def test_a_packaged_build_is_runnable_and_carries_its_bots(tmp_path, package):
    models = library.built_models(_index(tmp_path / "models.db", package_dir=str(package)))

    assert len(models) == 1
    model = models[0]
    assert model.built and model.runnable and model.note == ""
    assert model.slug == "recall"
    assert model.path == str(package)
    assert model.platform == "source"
    assert model.size_bytes > 0
    assert model.detail["n_fft"] == 8192
    assert [p.kind for p in model.programs] == ["champion", "runner_up"]


def test_an_unpackaged_build_is_listed_but_says_why_it_cannot_run(tmp_path):
    models = library.built_models(_index(tmp_path / "models.db", stopped_at="evolve"))

    assert len(models) == 1, "a build that stopped early is still a build somebody will look for"
    assert models[0].runnable is False
    assert "evolve" in models[0].note


def test_a_package_that_has_been_deleted_says_so(tmp_path):
    gone = tmp_path / "went-away"
    models = library.built_models(_index(tmp_path / "models.db", package_dir=str(gone)))

    assert models[0].runnable is False
    assert str(gone) in models[0].note


def test_only_the_champion_is_calibrated(tmp_path, package):
    champion, runner_up = library.built_models(
        _index(tmp_path / "models.db", package_dir=str(package)))[0].programs

    assert champion.calibrated and champion.threshold == pytest.approx(-0.36)
    assert not runner_up.calibrated, "an uncalibrated runner-up cannot be packaged or run"


def test_a_models_features_are_every_feature_its_bots_read_champion_first(tmp_path, package):
    model = library.built_models(
        _index(tmp_path / "models.db", package_dir=str(package)))[0]

    assert model.features == ("band_1000hz", "peak_band_share")


def test_feature_usage_counts_programs_across_models_most_used_first(tmp_path, package):
    models = library.built_models(_index(tmp_path / "models.db", package_dir=str(package)))

    assert library.feature_usage(models) == [("band_1000hz", 2), ("peak_band_share", 1)]


def test_feature_usage_of_downloaded_models_is_empty(tmp_path, package):
    models = library.built_models(_index(tmp_path / "models.db", package_dir=str(package)))
    for model in models:
        model.programs = []
    assert library.feature_usage(models) == []


def test_an_index_without_the_name_columns_still_lists_its_bots(tmp_path, package):
    """siar-build's database belongs to another program; a column it has not got is a blank."""
    model = library.built_models(_index(tmp_path / "models.db", package_dir=str(package)))[0]

    assert model.run_name == "", "and nothing here invents one"
    assert [p.name for p in model.programs] == ["", ""]
    assert [p.label for p in model.programs] == ["#0", "#1"], "a rank reads as the rank it is"


def test_models_group_under_what_they_detect(tmp_path, package):
    built = library.built_models(_index(tmp_path / "models.db", package_dir=str(package)))
    bundle = library.Model(source=library.DOWNLOADED, slug="sweeps", stamped_at=10,
                           detail={"shapes": ["sweep"]})
    nameless = library.Model(source=library.DOWNLOADED, slug="mystery", stamped_at=5)

    groups = library.targets(built + [bundle, nameless])

    assert [g.name for g in groups] == ["recall", "sweep", library.UNKNOWN_TARGET]
    assert groups[0].models == built
    assert groups[0].bots == 2, "every bot across every model under that target"


def test_a_target_counts_only_what_can_actually_be_run(tmp_path):
    """"Two models for this call" is a different fact from "two I could scan with tonight"."""
    models = library.built_models(_index(tmp_path / "models.db", stopped_at="evolve"))

    group = library.targets(models)[0]

    assert len(group.models) == 1 and group.runnable == 0


def test_the_build_home_environment_variable_moves_the_index(tmp_path, monkeypatch):
    monkeypatch.setenv(library.BUILD_HOME_ENV, str(tmp_path / "elsewhere"))
    assert library.build_db_path() == str(tmp_path / "elsewhere" / "models.db")


def test_downloaded_models_come_from_the_cache_and_carry_why_they_cannot_run(monkeypatch):
    monkeypatch.setattr(library, "installed_algorithms", lambda: [
        {"slug": "all_structures", "platform": "linux-x86_64-cp313", "version": "1.0.0",
         "family": "structure_scanners", "title": "Everything", "bytes": 1024,
         "downloaded_at": 1_700_000_000, "runnable": True, "root": "/cache/all"},
        {"slug": "wrong_arch", "platform": "darwin-arm64-cp313", "version": "1.0.0",
         "family": "", "title": "", "bytes": 2048, "downloaded_at": 1_700_000_001,
         "runnable": False, "root": "/cache/wrong"},
    ])

    downloaded = library.downloaded_models()

    assert [m.slug for m in downloaded] == ["all_structures", "wrong_arch"]
    assert all(not m.built for m in downloaded)
    assert downloaded[0].note == ""
    assert "darwin-arm64-cp313" in downloaded[1].note
    assert downloaded[0].programs == [], "a bundle's insides are not knowable from here"


def test_the_library_lists_downloaded_models_before_built_ones(tmp_path, package, monkeypatch):
    monkeypatch.setattr(library, "installed_algorithms", lambda: [
        {"slug": "all_structures", "platform": "linux-x86_64-cp313", "version": "1.0.0",
         "family": "", "title": "", "bytes": 1, "downloaded_at": 1, "runnable": True, "root": "/c"},
    ])

    models = library.library(db_path=_index(tmp_path / "models.db", package_dir=str(package)))

    assert [(m.source, m.slug) for m in models] == [
        (library.DOWNLOADED, "all_structures"),
        (library.BUILT, "recall"),
    ]


# ---- societies ------------------------------------------------------------------------------


def _society(path, *, package_dir="", parity_ok=1, members=2, name="socmodel_recall_a7f3k2"):
    """Add a society and its leaderboard to an index :func:`_index` has already written.

    Two tables siar-build added after this file was first written, which is the point of the
    defensive reads: an index without them must be a listing with no societies in it, never an
    error.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS socs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT, name TEXT, created_at TEXT,
            siarbuild_version TEXT, target TEXT, input_dir TEXT, output_dir TEXT,
            positive_tags TEXT, sample_rate INTEGER, n_fft INTEGER, hop INTEGER, n_bins INTEGER,
            fmin_hz REAL, fmax_hz REAL, arena_recordings INTEGER, unseen_recordings INTEGER,
            top_n INTEGER, k INTEGER, n_members INTEGER, n_bots INTEGER, rounds INTEGER,
            evaluations INTEGER, best_arena_auc REAL, unseen_auc REAL, stability REAL,
            parity_ok INTEGER, package_dir TEXT, state TEXT, last_round_at TEXT, notes TEXT
        );
        CREATE TABLE IF NOT EXISTS soc_ranks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, soc_id INTEGER, program_id INTEGER,
            rank INTEGER, arena_auc REAL, arena_ap REAL, arena_f1 REAL, train_f1 REAL,
            threshold REAL, in_model INTEGER, seeded_rounds INTEGER, rank_history TEXT,
            evaluated_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO socs (uid, name, created_at, siarbuild_version, target, input_dir,"
        " output_dir, sample_rate, n_fft, hop, n_bins, fmin_hz, fmax_hz, arena_recordings,"
        " unseen_recordings, top_n, k, n_members, n_bots, rounds, evaluations, best_arena_auc,"
        " unseen_auc, stability, parity_ok, package_dir, state, last_round_at) VALUES"
        " ('u1', ?, '2026-08-16T09:00:00+00:00', '0.1.0', 'recall', '/audio', '/models',"
        " 96000, 8192, 2048, 128, 5000.0, 7800.0, 2, 2, 20, 13, ?, 312, 37, 666000, 0.951,"
        " 0.883, 0.85, ?, ?, 'running', '2026-08-16T11:00:00+00:00')",
        (name, members, parity_ok, package_dir or None),
    )
    for rank in range(members):
        conn.execute(
            "INSERT INTO soc_ranks (soc_id, program_id, rank, arena_auc, arena_f1, threshold,"
            " in_model, seeded_rounds) VALUES (1, ?, ?, ?, 0.84, -0.12, 1, ?)",
            (rank + 1, rank, 0.95 - rank * 0.01, 9 - rank),
        )
    conn.commit()
    conn.close()
    return str(path)


def test_an_index_without_societies_lists_none_rather_than_failing(tmp_path):
    """siar-build versions separately from this package. A table it has not got yet must cost a
    listing with nothing in it, exactly as a missing column costs a blank field."""
    path = _index(tmp_path / "models.db")
    assert library.society_models(path) == []
    assert library.library(db_path=path)          # the builds are still there


def test_a_society_is_listed_with_its_members(tmp_path, package):
    path = _index(tmp_path / "models.db")
    _society(path, package_dir=str(package))

    societies = library.society_models(path)
    assert len(societies) == 1
    soc = societies[0]
    assert soc.slug == "socmodel_recall_a7f3k2"
    assert soc.source == library.SOCIETY
    assert soc.society and soc.local and not soc.built
    assert soc.runnable and soc.note == ""
    assert soc.shapes == ("recall",)
    assert len(soc.programs) == 2
    assert soc.detail["votes"] == "13 of 2"
    assert soc.detail["stability"] == 0.85


def test_a_society_is_a_local_package_so_the_loader_takes_the_same_path(tmp_path, package):
    """``Model.local`` is what decides between ``load_local`` and a cached bundle, in the CLI and
    on the library screen alike. A society is a plain package on this disk like any built model."""
    path = _index(tmp_path / "models.db")
    _society(path, package_dir=str(package))
    assert library.society_models(path)[0].local


def test_a_society_that_has_not_published_says_so(tmp_path):
    path = _index(tmp_path / "models.db")
    _society(path, package_dir="")
    soc = library.society_models(path)[0]
    assert not soc.runnable
    assert "not finished a round" in soc.note


def test_a_society_that_failed_its_parity_gate_will_not_run(tmp_path, package):
    """The same refusal siar-build makes. Boxes from a model that does not compute what was
    measured look exactly like boxes from one that does, which makes them worse than none."""
    path = _index(tmp_path / "models.db")
    _society(path, package_dir=str(package), parity_ok=0)
    soc = library.society_models(path)[0]
    assert not soc.runnable
    assert "parity" in soc.note.lower()


def test_a_society_whose_package_is_gone_says_where_it_was(tmp_path):
    path = _index(tmp_path / "models.db")
    _society(path, package_dir="/nowhere/siar_soc")
    soc = library.society_models(path)[0]
    assert not soc.runnable and "no longer there" in soc.note


def test_a_society_is_resolvable_by_name_for_siar_app_run(tmp_path, package):
    """What ``siar-app run -a <name>`` consults before it asks the server. Without this a society
    falls through to the registry, which has never heard of it."""
    path = _index(tmp_path / "models.db")
    _society(path, package_dir=str(package))
    found = library.local_model("socmodel_recall_a7f3k2", db_path=path)
    assert found is not None and found.society
    assert library.local_model("SOCMODEL_RECALL_A7F3K2", db_path=path) is not None


def test_a_society_that_cannot_run_is_not_resolved_by_name(tmp_path, package):
    path = _index(tmp_path / "models.db")
    _society(path, package_dir=str(package), parity_ok=0)
    assert library.local_model("socmodel_recall_a7f3k2", db_path=path) is None


def test_societies_lead_the_library(tmp_path, package):
    """A society is the best of many builds, so a machine running one has its answer at the top
    rather than under forty rounds of its own workings."""
    path = _index(tmp_path / "models.db", package_dir=str(package))
    _society(path, package_dir=str(package))
    models = library.library(db_path=path)
    kinds = [m.source for m in models]
    assert library.SOCIETY in kinds and library.BUILT in kinds
    assert kinds.index(library.SOCIETY) < kinds.index(library.BUILT)
