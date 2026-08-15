# Vixen Intelligence c.2026
"""End-to-end: a folder of recordings in, a folder the web app can open out.

The algorithm here is a stub written to disk and loaded through the same ``--algorithm-path``
route a developer uses on a real port, so this exercises the whole chain — package discovery,
factory call, contract check, decode, transform, scan, sidecar, thumbnail, manifest — without
needing a bundle, a token or a network.

The two tests that matter most are the unhappy ones. A corrupt file and an algorithm that
raises are both certainties over ten thousand recordings, and both must cost one manifest row.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from siarapp.loader import load_local
from siarapp.runner import RunOptions, plan_grid, run_folder

#: A conforming algorithm package: finds one box per loud frame band, and nothing else.
STUB = '''
"""A stub scanning algorithm for the test suite."""

slug = "stub"


class Stub:
    slug = "stub"
    version = "1.2.3"
    shapes = ("blob",)
    expects = {"fft": 256, "hop": 64, "window": "hann"}
    params_schema = {"floor": {"type": "number", "default": 1.0, "help": "magnitude bar"}}

    def __init__(self):
        self.floor = 1.0

    def configure(self, **params):
        if "floor" in params:
            self.floor = float(params["floor"])

    def scan(self, grid):
        import numpy as np

        mags = grid.magnitudes
        hot = np.argwhere(mags > self.floor)
        if hot.size == 0:
            return []
        f0, f1 = int(hot[:, 0].min()), int(hot[:, 0].max())
        b0, b1 = int(hot[:, 1].min()), int(hot[:, 1].max())
        return [{
            "tmin": f0 * grid.seconds_per_frame,
            "tmax": (f1 + 1) * grid.seconds_per_frame,
            "fmin": b0 * grid.hz_per_bin,
            "fmax": (b1 + 1) * grid.hz_per_bin,
            "peakHz": float(np.argmax(mags.max(axis=0))) * grid.hz_per_bin,
            "cells": int(hot.shape[0]),
            "peakSigma": float(mags.max()),
            "confidence": 0.9,
            "shape": "blob",
        }]


def algorithm(manifest=None):
    return Stub()
'''

BROKEN = '''
"""An algorithm that raises, which is what a bad port does on the file that breaks it."""

class Broken:
    slug = "broken"
    expects = {"fft": 256, "hop": 64}

    def scan(self, grid):
        raise ZeroDivisionError("the port has a bug")


def algorithm(manifest=None):
    return Broken()
'''


def _write_package(root, name, source):
    """Write a one-module package under ``root/src/<name>`` and return its src directory."""
    pkg = root / "src" / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(source)
    return str(root / "src")


def _write_wav(path, *, seconds=1.0, rate=8000, tone_hz=1000.0, amplitude=1.0):
    """A deterministic mono WAV with one tone in it."""
    t = np.arange(int(seconds * rate)) / rate
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32), rate)


@pytest.fixture()
def corpus(tmp_path):
    """Three recordings: a loud one at the top level, a quiet one in a subfolder, a tiny one."""
    src = tmp_path / "src"
    (src / "station-b").mkdir(parents=True)
    _write_wav(src / "loud.wav", amplitude=1.0)
    _write_wav(src / "station-b" / "quiet.wav", amplitude=1e-6)
    _write_wav(src / "tiny.wav", seconds=0.01)  # 80 samples: shorter than one 256-point window
    return src


@pytest.fixture()
def stub(tmp_path):
    return load_local(_write_package(tmp_path / "algo", "stub_algo", STUB), "stub")


def test_run_folder_writes_the_whole_output_folder(corpus, stub, tmp_path):
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus), str(out), RunOptions())

    assert manifest["files"] == 3
    # tiny.wav is 80 samples — shorter than one 256-point window, so two are scanned.
    assert manifest["by_status"] == {"scanned": 2, "too_short": 1}
    assert manifest["structures"] >= 1
    assert manifest["algorithm"]["slug"] == "stub"
    assert manifest["algorithm"]["version"] == "1.2.3"

    # Audio, sidecar and thumbnail for the top-level recording...
    assert (out / "loud.wav").is_file()
    assert (out / "loud.structures.json").is_file()
    assert (out / "loud.png").is_file()
    # ...and the subfolder's layout preserved.
    assert (out / "station-b" / "quiet.wav").is_file()
    assert (out / "station-b" / "quiet.structures.json").is_file()
    # ...and the run manifest at the root.
    assert (out / "siar-app-run.json").is_file()


def test_sidecar_carries_what_the_app_reads(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions())
    doc = json.loads((out / "loud.structures.json").read_text())

    assert doc["algorithm"] == "stub"
    assert doc["stft"] == {"fft": 256, "hop": 64, "window": "hann", "sample_rate": 8000.0}
    assert doc["structures"], "a 1.0-amplitude tone should produce a box"
    box = doc["structures"][0]
    for field in ("tmin", "tmax", "fmin", "fmax", "peakHz", "cells", "peakSigma",
                  "confidence", "shape"):
        assert field in box
    assert 0.0 <= box["tmin"] < box["tmax"] <= doc["duration_sec"] + 0.05


def test_the_algorithms_own_grid_is_the_default(stub):
    """A click scanner needs a 256-point window and the user should not have to know that."""
    assert plan_grid(stub.algorithm, RunOptions()) == {"fft": 256, "hop": 64, "window": "hann"}
    assert plan_grid(stub.algorithm, RunOptions(fft=1024))["fft"] == 1024
    # Overriding the FFT drops the algorithm's paired hop rather than keeping 94% overlap.
    assert plan_grid(stub.algorithm, RunOptions(fft=1024))["hop"] == 256
    assert plan_grid(stub.algorithm, RunOptions(fft=1024, hop=100))["hop"] == 100


def test_params_reach_the_algorithm(corpus, stub, tmp_path):
    """--param floor=1e9 puts the bar out of reach, so the loud file stops producing a box."""
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions(params={"floor": 1e9}))
    doc = json.loads((out / "loud.structures.json").read_text())
    assert doc["structures"] == []
    assert doc["params"] == {"floor": 1e9}


def test_a_recording_shorter_than_a_window_is_a_status_not_a_crash(corpus, stub, tmp_path):
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus), str(out), RunOptions())
    statuses = {row["path"]: row["status"] for row in manifest["manifest"]}
    assert statuses["tiny.wav"] == "too_short"
    assert (out / "tiny.wav").is_file(), "a too-short recording still belongs in the folder"


def test_an_algorithm_that_raises_costs_one_row(corpus, tmp_path):
    """The port has a bug on one file. That is a manifest row, not the end of a four-hour run."""
    broken = load_local(_write_package(tmp_path / "algo", "broken_algo", BROKEN), "broken")
    manifest = run_folder(broken, str(corpus), str(tmp_path / "out"), RunOptions())
    # Two scannable recordings, both failing; tiny.wav never reaches the algorithm.
    assert manifest["by_status"]["error"] == 2
    errors = [row for row in manifest["manifest"] if row["status"] == "error"]
    assert "the port has a bug" in errors[0]["error"]


def test_an_unreadable_file_costs_one_row(corpus, stub, tmp_path):
    (corpus / "truncated.wav").write_bytes(b"RIFF____WAVEfmt ")
    manifest = run_folder(stub, str(corpus), str(tmp_path / "out"), RunOptions())
    rows = {row["path"]: row for row in manifest["manifest"]}
    assert rows["truncated.wav"]["status"] == "error"
    assert manifest["by_status"]["scanned"] == 2


def test_resume_skips_what_is_already_written(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions())
    again = run_folder(stub, str(corpus), str(out), RunOptions(resume=True))
    assert again["by_status"]["skipped"] == 3   # too_short files write a sidecar too
    assert again["structures"] == 0  # nothing rescanned, so nothing recounted


def test_one_recording_can_be_scanned_on_its_own(corpus, stub, tmp_path):
    """A single file is a corpus of one, laid out as if its folder had been scanned.

    Which is the point: somebody trying an algorithm on one recording before committing a survey
    drive to it should be able to resume the whole folder on top of the result, and that only
    works if the one file lands exactly where the folder run would have put it.
    """
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus / "loud.wav"), str(out), RunOptions())

    assert manifest["files"] == 1
    assert manifest["by_status"] == {"scanned": 1}
    assert manifest["manifest"][0]["path"] == "loud.wav"
    assert (out / "loud.wav").is_file()
    assert (out / "loud.structures.json").is_file()


def test_a_file_that_is_not_a_recording_is_the_same_error_as_an_empty_folder(tmp_path, stub):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a recording")

    with pytest.raises(FileNotFoundError):
        run_folder(stub, str(notes), str(tmp_path / "out"), RunOptions())


def test_an_empty_folder_is_an_error_not_an_empty_output(tmp_path, stub):
    """Silently producing an empty output folder helps nobody — it is a typo, or MP3s."""
    (tmp_path / "nothing").mkdir()
    with pytest.raises(FileNotFoundError, match="no recordings"):
        run_folder(stub, str(tmp_path / "nothing"), str(tmp_path / "out"), RunOptions())


def test_mixed_sample_rates_are_warned_about(tmp_path, stub):
    src = tmp_path / "src"
    src.mkdir()
    _write_wav(src / "a.wav", rate=8000)
    _write_wav(src / "b.wav", rate=16000)
    warnings: list[str] = []
    run_folder(stub, str(src), str(tmp_path / "out"), RunOptions(), warn=warnings.append)
    assert any("sample rate" in w for w in warnings)


def test_limit_and_no_recursive_bound_the_work(corpus, stub, tmp_path):
    manifest = run_folder(stub, str(corpus), str(tmp_path / "o1"), RunOptions(limit=1))
    assert manifest["files"] == 1

    manifest = run_folder(stub, str(corpus), str(tmp_path / "o2"), RunOptions(recursive=False))
    assert all("/" not in row["path"] for row in manifest["manifest"])


def test_a_run_says_which_stage_each_recording_is_on(corpus, stub, tmp_path):
    """The live line has to have something to say during the minutes one `scan` can take."""
    seen: list[tuple] = []
    starts: list[tuple] = []
    run_folder(
        stub, str(corpus), str(tmp_path / "out"), RunOptions(),
        on_start=lambda i, total, rel, dur, lane, size: starts.append((rel, dur, size, lane)),
        on_stage=lambda lane, stage, fraction=0.0: seen.append((lane, stage)),
    )
    stages = [stage for _lane, stage in seen]
    # loud.wav in full, then quiet.wav, then tiny.wav — which is too short to reach `scan`.
    assert stages[:5] == ["decode", "fft", "scan", "write", "thumbnail"]
    assert stages[-4:] == ["decode", "fft", "write", "thumbnail"]
    assert {lane for lane, _stage in seen} == {0}, "a serial run is one lane"

    # And the display is handed the size and the length of every recording it announces.
    assert [rel for rel, _dur, _size, _lane in starts] == [
        "loud.wav", "station-b/quiet.wav", "tiny.wav",
    ]
    assert all(size > 0 for _rel, _dur, size, _lane in starts)
    assert starts[0][1] == pytest.approx(1.0)


def test_max_size_leaves_the_oversized_recording_out_of_the_run(corpus, stub, tmp_path):
    """The whole point: one enormous file costs a manifest row, not the machine's memory."""
    _write_wav(corpus / "big.wav", seconds=10.0)  # ~160 KB against ~16 KB each
    out = tmp_path / "out"
    warnings: list[str] = []
    manifest = run_folder(stub, str(corpus), str(out), RunOptions(max_bytes=64 * 1024),
                          warn=warnings.append)

    rows = {row["path"]: row for row in manifest["manifest"]}
    assert rows["big.wav"]["status"] == "too_large"
    assert "--max-size" in rows["big.wav"]["error"]
    # It is out of the run entirely, not scanned and then discarded.
    assert not (out / "big.wav").exists()
    assert not (out / "big.structures.json").exists()
    # ...and the rest of the folder went through untouched.
    assert manifest["files"] == 4
    assert manifest["by_status"]["scanned"] == 2
    assert any("--max-size" in w for w in warnings)


def test_max_size_zero_takes_whatever_is_there(corpus, stub, tmp_path):
    _write_wav(corpus / "big.wav", seconds=10.0)
    manifest = run_folder(stub, str(corpus), str(tmp_path / "out"), RunOptions(max_bytes=0))
    assert "too_large" not in manifest["by_status"]
    assert manifest["by_status"]["scanned"] == 3


def test_an_unreadable_file_is_not_reported_as_oversized(corpus, stub, tmp_path):
    """A stat that fails says nothing about size, so the file goes on to be a real error row."""
    (corpus / "truncated.wav").write_bytes(b"RIFF____WAVEfmt ")
    manifest = run_folder(stub, str(corpus), str(tmp_path / "out"), RunOptions(max_bytes=64 * 1024))
    rows = {row["path"]: row for row in manifest["manifest"]}
    assert rows["truncated.wav"]["status"] == "error"


def test_max_size_is_parsed_off_the_command_line(corpus):
    from siarapp.cli.main import build_parser
    from siarapp.runner import DEFAULT_MAX_BYTES

    def parsed(*extra):
        return build_parser().parse_args(
            ["run", str(corpus), "--out", "/tmp/x", *extra]
        ).max_size

    assert parsed() == DEFAULT_MAX_BYTES == 550 * 1024**2
    assert parsed("--max-size", "550MB") == 550 * 1024**2
    assert parsed("--max-size", "1.5G") == int(1.5 * 1024**3)
    assert parsed("--max-size", "4096") == 4096
    assert parsed("--max-size", "0") == 0
    with pytest.raises(SystemExit):
        parsed("--max-size", "huge")
    with pytest.raises(SystemExit):
        parsed("--max-size", "12PB")


def test_load_local_rejects_a_package_with_no_scan(tmp_path):
    from siarapp.grid import ScannerError

    src = _write_package(tmp_path / "algo", "no_scan", "def algorithm(manifest=None):\n    return object()\n")
    with pytest.raises(ScannerError, match="no scan"):
        load_local(src, "no_scan")


def test_load_local_rejects_a_package_with_no_factory(tmp_path):
    from siarapp.grid import ScannerError

    src = _write_package(tmp_path / "algo", "no_factory", "x = 1\n")
    with pytest.raises(ScannerError, match="factory"):
        load_local(src, "no_factory")


def test_output_folder_may_not_be_the_source(corpus, stub):
    """Writing copies of the recordings into the folder being walked would recurse."""
    from siarapp.cli.main import build_parser
    from siarapp.cli import commands

    args = build_parser().parse_args(
        ["run", str(corpus), "--out", str(corpus), "--algorithm-path", "x"]
    )
    assert commands.cmd_run(args) == 1


def test_manifest_counts_shapes(corpus, stub, tmp_path):
    manifest = run_folder(stub, str(corpus), str(tmp_path / "out"), RunOptions())
    assert manifest["shapes"].get("blob", 0) >= 1
    assert sum(manifest["shapes"].values()) == manifest["structures"]


def test_thumbnails_can_be_turned_off(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions(thumbnails=False))
    assert not (out / "loud.png").exists()
    assert os.path.isfile(out / "loud.structures.json")


def test_every_stage_of_a_recording_is_timed(corpus, stub, tmp_path):
    """The phase breakdown, end to end: per file, in the totals, and in the run manifest.

    It is the difference between "the run was slow" and "the run was waiting on the disk", so a
    stage that quietly stops being measured is worth failing a test over.
    """
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus), str(out), RunOptions())
    perf = json.loads((out / "siar-app-performance.json").read_text())

    scanned = [f for f in perf["files"] if f["status"] == "scanned"]
    assert scanned
    for row in scanned:
        assert set(row["phases"]) == {"decode", "fft", "scan", "write", "thumbnail"}
        # Every stage of a file is part of that file's own wall time, with room for the rounding.
        assert sum(row["phases"].values()) <= row["elapsed_sec"] + 0.01

    assert perf["totals"]["phases"] == manifest["phases"]
    assert manifest["phases"]["scan"] > 0
    assert sum(manifest["phases"].values()) <= manifest["elapsed_sec"] + 0.01


def test_a_stage_that_never_ran_is_absent_rather_than_zero(corpus, stub, tmp_path):
    out = tmp_path / "out"
    manifest = run_folder(stub, str(corpus), str(out), RunOptions(thumbnails=False))
    assert "thumbnail" not in manifest["phases"]
    assert "fft" in manifest["phases"]


def test_the_folder_is_openable_before_the_run_finishes(corpus, stub, tmp_path):
    """The point of the whole flush loop: drop an overnight scan into the app while it runs.

    Read from inside the progress callback, which is the only place "mid-run" exists in a
    synchronous loop — after the first recording there must already be a run manifest, a
    performance report, and a progress block that admits the run is not done.
    """
    out = tmp_path / "out"
    seen = []

    def watch(done, total, _result):
        if done < total:
            perf = json.loads((out / "siar-app-performance.json").read_text())
            run = json.loads((out / "siar-app-run.json").read_text())
            seen.append((done, perf["progress"], run["progress"]["state"], run["files"]))

    run_folder(stub, str(corpus), str(out), RunOptions(), progress=watch)

    assert seen, "no document was written until the run had finished"
    done, progress, run_state, run_files = seen[0]
    assert progress["state"] == "running"
    assert run_state == "running"
    # The manifest describes what has been done, not what will be: a partially scanned folder
    # claiming three files would make the app draw two empty lanes.
    assert run_files == done
    assert progress["files_total"] == 3
    assert 0.0 < progress["fraction"] < 1.0
    assert progress["files_remaining"] == 3 - done


def test_the_final_write_says_the_run_finished(corpus, stub, tmp_path):
    out = tmp_path / "out"
    run_folder(stub, str(corpus), str(out), RunOptions())

    progress = json.loads((out / "siar-app-performance.json").read_text())["progress"]
    assert progress["state"] == "complete"
    assert progress["fraction"] == 1.0
    assert progress["eta_sec"] == 0.0
    assert progress["files_done"] == progress["files_total"] == 3
    assert progress["files_remaining"] == 0
    assert progress["audio_total_sec"] > 0


def test_the_estimate_is_made_in_audio_seconds_not_files():
    """Half the files can be a tenth of the work, and the bar has to say so."""
    from siarapp.io.performance import progress_block

    # Ten seconds in, one short file of a hundred done: 1% of the audio, 50% of the files.
    block = progress_block(
        started=0.0, now=10.0,
        files_total=100, files_done=50, files_worked=50,
        audio_total_sec=10_000.0, audio_done_sec=100.0, audio_worked_sec=100.0,
    )
    assert block["fraction"] == 0.01
    # 9,900 s of audio left at 10 audio-seconds per wall second.
    assert block["eta_sec"] == pytest.approx(990.0)
    assert block["eta_at"]


def test_the_estimate_holds_off_until_it_has_something_to_go_on():
    """A null ETA renders as "estimating"; a zero would render as "finished"."""
    from siarapp.io.performance import progress_block

    block = progress_block(
        started=0.0, now=0.0, files_total=100, files_done=0, files_worked=0,
        audio_total_sec=10_000.0,
    )
    assert block["eta_sec"] is None
    assert block["eta_at"] == ""
    assert block["state"] == "running"


def test_resumed_files_do_not_inflate_the_estimate(corpus, stub, tmp_path):
    """A resume that skips most of the corpus must not quote the hours it is not going to spend."""
    from siarapp.io.performance import progress_block

    # 90 of 100 files handled in a second and only 9 of them actually scanned. The 100 s of
    # audio left is charged at the same 1-in-10 rate, not in full: 10 s of work at 10x.
    block = progress_block(
        started=0.0, now=1.0,
        files_total=100, files_done=90, files_worked=9,
        audio_total_sec=1000.0, audio_done_sec=900.0, audio_worked_sec=10.0,
    )
    assert block["eta_sec"] == pytest.approx(1.0)

    # The same run without the discount would quote ten times as long — a resumed scan of a
    # nearly-finished corpus reporting minutes of work that will never happen.
    naive = progress_block(
        started=0.0, now=1.0,
        files_total=100, files_done=90, files_worked=90,
        audio_total_sec=1000.0, audio_done_sec=900.0, audio_worked_sec=10.0,
    )
    assert naive["eta_sec"] == pytest.approx(10.0)


def test_a_run_records_what_the_next_run_draws_its_first_bars_from(corpus, stub, tmp_path,
                                                                   monkeypatch):
    """The history carries throughput and its stage split, or every run starts blind."""
    monkeypatch.setenv("SIAR_APP_HOME", str(tmp_path / "home"))
    from siarapp.cli import commands
    from siarapp.cli.main import build_parser
    from siarapp.config import recent_cost

    args = build_parser().parse_args([
        "run", str(corpus), "--out", str(tmp_path / "out"),
        "--algorithm-path", str(tmp_path / "algo" / "src"), "--algorithm", "stub", "-q",
    ])
    assert commands.cmd_run(args) == 0

    rate, shares = recent_cost("stub")
    assert rate > 0, "a run that scanned audio must leave a cost behind"
    assert shares.get("scan", 0) > 0
    assert recent_cost("some_other_algorithm") == (0.0, {})


def test_a_run_of_milliseconds_still_leaves_a_cost_behind(tmp_path, monkeypatch):
    """A trial over a handful of clips is exactly the run whose figures the next one needs.

    Per-file times are milliseconds on a folder of clips, and a history that rounded them to a
    hundredth of a second would record "no time at all" — after which
    :func:`~siarapp.config.recent_cost` divides by zero worker seconds, comes back with nothing,
    and the next run draws no bars until its first recording lands. Which, on a corpus of
    hour-long files, is an hour of a screen with nothing moving on it.
    """
    monkeypatch.setenv("SIAR_APP_HOME", str(tmp_path / "home"))
    from siarapp.cli import commands
    from siarapp.config import recent_cost
    from siarapp.runner import RunOptions

    manifest = {
        "started_at": "2026-08-16T09:00:00Z",
        "files": 3,
        "structures": 4,
        "workers": 1,
        "elapsed_sec": 0.012,
        "by_status": {"scanned": 3},
        "phases": {"scan": 0.006, "fft": 0.003},
        "manifest": [
            {"path": f"clip{i}.wav", "status": "scanned", "duration_sec": 1.0,
             "elapsed_sec": 0.002}
            for i in range(3)
        ],
    }
    monkeypatch.setattr(commands, "run_folder", lambda *a, **k: manifest)

    class _Reporter:
        on_result = on_start = on_corpus = on_idle = on_stage = staticmethod(lambda *a, **k: None)

        def close(self):
            pass

    written, failure = commands.perform_run(
        _Handle(), str(tmp_path / "src"), str(tmp_path / "out"), RunOptions(), _Reporter())

    assert failure == "" and written is manifest
    rate, shares = recent_cost("stub")
    assert rate > 0, "six milliseconds of work is still work, and the next run needs the number"
    assert shares.get("scan", 0) > 0


class _Handle:
    """The two things :func:`perform_run` reads off a loaded algorithm."""

    slug = "stub"
    platform = "source"
