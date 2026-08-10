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

from siarscan.loader import load_local
from siarscan.runner import RunOptions, plan_grid, run_folder

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
    assert (out / "siar-scanner-run.json").is_file()


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


def test_load_local_rejects_a_package_with_no_scan(tmp_path):
    from siarscan.grid import ScannerError

    src = _write_package(tmp_path / "algo", "no_scan", "def algorithm(manifest=None):\n    return object()\n")
    with pytest.raises(ScannerError, match="no scan"):
        load_local(src, "no_scan")


def test_load_local_rejects_a_package_with_no_factory(tmp_path):
    from siarscan.grid import ScannerError

    src = _write_package(tmp_path / "algo", "no_factory", "x = 1\n")
    with pytest.raises(ScannerError, match="factory"):
        load_local(src, "no_factory")


def test_output_folder_may_not_be_the_source(corpus, stub):
    """Writing copies of the recordings into the folder being walked would recurse."""
    from siarscan.cli.main import build_parser
    from siarscan.cli import commands

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
