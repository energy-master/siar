# Vixen Intelligence c.2026
"""Scanning a folder on more than one core, and getting the same folder back.

The one property that matters is here first: a parallel run and a serial run over the same corpus
must produce byte-identical sidecars. Everything else about ``--parallel`` is a speed argument,
and a speed argument that changes the answers is not an argument at all.

The rest is the plumbing that only shows up under a pool — an algorithm reloaded in a process
that never saw the parent's, a per-file failure that must stay per-file when the file is being
scanned somewhere else, and the sizing that stops a sixteen-core machine from asking for
sixteen grids it has no memory for.
"""
from __future__ import annotations

import json
import os

import pytest

from siarapp.io.performance import PHASES
from siarapp.loader import load_local
from siarapp.parallel import AlgorithmSpec, memory_ceiling, resolve_workers
from siarapp.runner import RunOptions, run_folder

from test_runner import BROKEN, STUB, _write_package, _write_wav


@pytest.fixture()
def corpus(tmp_path):
    """Enough recordings to keep several workers busy, in two folders."""
    src = tmp_path / "src"
    (src / "station-b").mkdir(parents=True)
    for i in range(6):
        _write_wav(src / f"rec{i}.wav", seconds=0.4 + i * 0.05)
    for i in range(4):
        _write_wav(src / "station-b" / f"deep{i}.wav", seconds=0.3, amplitude=1e-6)
    return src


@pytest.fixture()
def stub(tmp_path):
    return load_local(_write_package(tmp_path / "algo", "stub_algo", STUB), "stub")


def _sidecars(root):
    """Every sidecar under a folder, keyed by relative path."""
    return {
        str(p.relative_to(root)): json.loads(p.read_text())
        for p in root.rglob("*.structures.json")
    }


def test_parallel_and_serial_produce_the_same_folder(corpus, stub, tmp_path):
    one = tmp_path / "serial"
    many = tmp_path / "parallel"
    serial = run_folder(stub, str(corpus), str(one), RunOptions())
    parallel = run_folder(stub, str(corpus), str(many), RunOptions(workers=4))

    assert _sidecars(one) == _sidecars(many)
    assert parallel["files"] == serial["files"]
    assert parallel["structures"] == serial["structures"]
    assert parallel["by_status"] == serial["by_status"]
    assert parallel["workers"] == 4
    # The rows arrive in completion order, so the manifest lists the same recordings and not
    # necessarily in the same order — which is the one difference the two runs are allowed.
    assert ({r["path"] for r in parallel["manifest"]}
            == {r["path"] for r in serial["manifest"]})


def test_every_recording_is_written_exactly_once(corpus, stub, tmp_path):
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus), str(out), RunOptions(workers=3))
    paths = [r["path"] for r in manifest["manifest"]]
    assert len(paths) == len(set(paths)) == 10
    assert len(_sidecars(out)) == 10
    assert manifest["by_status"] == {"scanned": 10}


def test_a_worker_reports_its_lane_and_lanes_are_reused(corpus, stub, tmp_path):
    starts: list[tuple] = []
    idle: list[int] = []
    run_folder(
        stub, str(corpus), str(tmp_path / "out"), RunOptions(workers=3),
        on_start=lambda i, total, rel, dur, lane, size: starts.append((rel, lane, size)),
        on_idle=idle.append,
    )
    assert len(starts) == 10
    assert {lane for _rel, lane, _size in starts} == {0, 1, 2}
    # The display is told how big each recording is, not just what it is called.
    assert all(size > 0 for _rel, _lane, size in starts)
    # Every lane is told when it runs dry, so a display never leaves a finished worker drawn
    # as though it were still on a file.
    assert sorted(idle) == [0, 1, 2]


def test_a_worker_says_which_stage_it_is_on(corpus, stub, tmp_path):
    """Stages cross the process boundary, or a parallel panel has nothing to say mid-file.

    What arrives is not asserted file by file. A stage that reaches the parent after its own
    file's result is dropped rather than drawn on a lane that has already moved on, and on
    recordings this small some of the closing stages lose that race — which is the behaviour
    wanted, not a fault. The claim here is that the channel exists and reports real stages
    against real lanes; :mod:`tests.test_runner` pins the exact sequence on a serial run.
    """
    stages: list[tuple] = []
    run_folder(
        stub, str(corpus), str(tmp_path / "out"), RunOptions(workers=3),
        on_stage=lambda lane, stage: stages.append((lane, stage)),
    )
    assert stages, "a parallel run reported no stages at all"
    assert {stage for _lane, stage in stages} <= set(PHASES)
    assert {lane for lane, _stage in stages} <= {0, 1, 2}


def test_a_failing_recording_stays_one_row(corpus, tmp_path):
    """An algorithm that raises must cost one manifest row per file, pool or no pool."""
    broken = load_local(_write_package(tmp_path / "algo", "broken_algo", BROKEN), "broken")
    manifest = run_folder(broken, str(corpus), str(tmp_path / "out"), RunOptions(workers=3))
    assert manifest["by_status"] == {"error": 10}
    assert all("the port has a bug" in r["error"] for r in manifest["manifest"])


def test_resume_after_a_parallel_run(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions(workers=4))
    again = run_folder(stub, str(corpus), str(out), RunOptions(workers=4, resume=True))
    assert again["by_status"] == {"skipped": 10}


def test_the_run_records_what_it_was_run_with(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions(workers=2))
    perf = json.loads((out / "siar-app-performance.json").read_text())
    assert perf["machine"]["workers"] == 2
    assert perf["totals"]["files_scanned"] == 10


def test_the_corpus_is_announced_once_the_headers_are_read(corpus, stub, tmp_path):
    seen: list[tuple] = []
    run_folder(stub, str(corpus), str(tmp_path / "out"), RunOptions(workers=2),
               on_corpus=lambda files, audio, workers: seen.append((files, workers)))
    assert len(seen) == 1
    assert seen[0] == (10, 2)


# -- sizing the pool ---------------------------------------------------------------------------


def test_workers_never_exceed_the_recordings_there_are():
    assert resolve_workers(0, 2, 0) <= 2
    assert resolve_workers(64, 3, 0) == 3


def test_an_automatic_count_is_capped_by_memory():
    warnings: list[str] = []
    # A grid so large that only one worker fits, whatever the machine.
    workers = resolve_workers(0, 1000, 10 * 1024**4, warnings.append)
    assert workers == 1
    assert warnings and "worker" in warnings[0]


def test_an_explicit_count_is_obeyed_and_warned_about():
    warnings: list[str] = []
    workers = resolve_workers(8, 1000, 10 * 1024**4, warnings.append)
    assert workers == 8
    assert warnings and "--parallel" in warnings[0]


def test_an_unknown_memory_size_is_not_a_low_one(monkeypatch):
    monkeypatch.setattr("siarapp.parallel.total_memory_bytes", lambda: 0)
    assert memory_ceiling(10 * 1024**4) == 0
    assert resolve_workers(8, 1000, 10 * 1024**4) == 8


def test_a_spec_reloads_the_same_algorithm(stub):
    spec = AlgorithmSpec.of(stub, {"floor": 2.0})
    handle = spec.load()
    assert handle.slug == stub.slug
    assert handle.algorithm.floor == 2.0
    assert handle is not stub


def test_library_threads_are_pinned_before_a_worker_starts(monkeypatch):
    from siarapp.parallel import limit_library_threads

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    limit_library_threads()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    # A choice the user already made is left alone.
    assert os.environ["MKL_NUM_THREADS"] == "4"
