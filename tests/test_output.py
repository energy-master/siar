# Vixen Intelligence c.2026
"""The output folder is the product, so its layout is a contract with the web app.

Three things here would break silently and be discovered only by a user staring at an app that
shows no boxes: the sidecar's filename (the app pairs sidecars to audio by name), the presence
of the ``structures`` key (the app sniffs on it to tell this document from a labels or decisions
one), and the preservation of relative layout (flattening would collide names from different
subfolders). Each gets a test.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from siarscan.grid import Region, ScannerError, normalise_regions
from siarscan.io.output import SIDECAR_SUFFIX, OutputFolder, sidecar_document
from siarscan.viz.png import encode_png
from siarscan.viz.thumbnail import render_rgb, thumbnail_png


@pytest.fixture()
def corpus(tmp_path):
    """A source folder with one recording at the top level and one in a subfolder."""
    src = tmp_path / "src"
    (src / "station-b").mkdir(parents=True)
    (src / "a.wav").write_bytes(b"RIFF" + b"\0" * 100)
    (src / "station-b" / "a.wav").write_bytes(b"RIFF" + b"\0" * 200)
    return src


def test_relative_layout_is_preserved(corpus, tmp_path):
    """Two recordings both called a.wav in different subfolders must stay distinct."""
    out = OutputFolder(tmp_path / "out", corpus)
    top = out.audio_path(str(corpus / "a.wav"))
    nested = out.audio_path(str(corpus / "station-b" / "a.wav"))
    assert top != nested
    assert nested.endswith(os.path.join("station-b", "a.wav"))


def test_sidecar_is_named_for_its_audio(corpus, tmp_path):
    """The app pairs a sidecar to the audio whose basename is the longest prefix of it, so
    a.wav -> a.structures.json pairs correctly and could not be claimed by a_2.wav."""
    out = OutputFolder(tmp_path / "out", corpus)
    assert out.sidecar_path(str(corpus / "a.wav")).endswith("a" + SIDECAR_SUFFIX)
    assert out.thumbnail_path(str(corpus / "a.wav")).endswith("a.png")


def test_sidecar_keeps_the_structures_key_when_empty(tmp_path):
    """"Scanned, found nothing" and "never scanned" are different statements, and the app can
    only tell them apart if the empty case still writes a document."""
    doc = sidecar_document(
        filename="a.wav", duration_sec=10.0, algorithm="all_structures", structures=[]
    )
    assert doc["structures"] == []
    assert doc["count"] == 0
    assert "structures" in json.loads(json.dumps(doc))


def test_place_audio_copies_and_resume_sees_it(corpus, tmp_path):
    out = OutputFolder(tmp_path / "out", corpus)
    src = str(corpus / "a.wav")
    dest = out.place_audio(src)
    assert os.path.getsize(dest) == os.path.getsize(src)
    assert not out.already_done(src)  # audio yes, sidecar no
    out.write_sidecar(src, sidecar_document(
        filename="a.wav", duration_sec=1.0, algorithm="x", structures=[]))
    assert out.already_done(src)


def test_link_falls_back_to_copy_across_filesystems(corpus, tmp_path):
    """--link is an optimisation, never a failure mode: an unlinkable destination copies."""
    out = OutputFolder(tmp_path / "out", corpus, link=True)
    dest = out.place_audio(str(corpus / "a.wav"))
    assert os.path.isfile(dest)


def test_numpy_scalars_survive_serialisation(corpus, tmp_path):
    """A port's region dict is full of numpy floats, and json.dumps refuses them by default."""
    out = OutputFolder(tmp_path / "out", corpus)
    doc = sidecar_document(
        filename="a.wav",
        duration_sec=np.float32(1.5),
        algorithm="x",
        structures=[{"tmin": np.float32(0.1), "tmax": np.float64(0.2), "fmin": 0.0,
                     "fmax": np.float32(500.0), "cells": np.int64(4), "shape": "click"}],
    )
    path = out.write_sidecar(str(corpus / "a.wav"), doc)
    reloaded = json.loads(open(path).read())
    assert reloaded["structures"][0]["cells"] == 4


def test_writes_are_atomic(corpus, tmp_path):
    """No .tmp debris left behind — an interrupted run must not leave the app a broken JSON."""
    out = OutputFolder(tmp_path / "out", corpus)
    src = str(corpus / "a.wav")
    out.place_audio(src)
    out.write_sidecar(src, sidecar_document(
        filename="a.wav", duration_sec=1.0, algorithm="x", structures=[]))
    leftovers = [p for p in os.listdir(os.path.dirname(out.audio_path(src))) if ".tmp-" in p]
    assert leftovers == []


# -- the seam ------------------------------------------------------------------------------


def test_normalise_accepts_regions_dicts_and_objects():
    """A port may return any of the three; the writer must not care which."""
    class Duck:
        tmin, tmax, fmin, fmax, shape = 0.0, 1.0, 100.0, 200.0, "tonal"

    rows = normalise_regions([
        Region(0.0, 1.0, 100.0, 200.0, shape="sweep"),
        {"tmin": 0.0, "tmax": 1.0, "fmin": 100.0, "fmax": 200.0, "shape": "click"},
        Duck(),
    ])
    assert [r["shape"] for r in rows] == ["sweep", "click", "tonal"]


def test_normalise_keeps_a_scanners_own_diagnostics():
    """fill/trend/cohesion are the scanner explaining itself. Nothing reads them yet, and
    throwing them away would make the sidecar useless for working out why a box is there."""
    rows = normalise_regions([
        {"tmin": 0, "tmax": 1, "fmin": 0, "fmax": 1, "fill": 0.7, "trend": -0.3},
    ])
    assert rows[0]["fill"] == 0.7 and rows[0]["trend"] == -0.3


def test_normalise_refuses_an_undrawable_box():
    """A NaN bound is a bug in a port, and a box that cannot be drawn should stop the run rather
    than land in a sidecar for the app to choke on."""
    with pytest.raises(ScannerError, match="non-finite"):
        normalise_regions([{"tmin": float("nan"), "tmax": 1, "fmin": 0, "fmax": 1}])
    with pytest.raises(ScannerError, match="missing"):
        normalise_regions([{"tmax": 1, "fmin": 0, "fmax": 1}])


# -- pictures ------------------------------------------------------------------------------


def test_thumbnail_is_the_apps_size_and_orientation():
    """200x64, low frequency at the bottom — the app's lane geometry."""
    rate = 8000
    t = np.arange(rate * 4) / rate
    signal = np.sin(2 * np.pi * 200 * t).astype(np.float32)  # low tone
    rgb = render_rgb(signal)
    assert rgb.shape == (64, 200, 3)
    # The energy is at 200 Hz of a 4 kHz Nyquist: the bottom rows must be brighter than the top.
    assert rgb[-8:].mean() > rgb[:8].mean()


def test_thumbnail_is_none_below_one_window():
    assert thumbnail_png(np.zeros(100, dtype=np.float32)) is None


def test_png_is_a_png():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    blob = encode_png(rgb)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in blob[:32] and blob[-8:-4] == b"IEND"


def test_viridis_matches_the_browsers_polynomial():
    """Not matplotlib's table: the app evaluates a polynomial fit, and the two differ by a few
    LSB. A lane rendered here sits next to one rendered by the browser."""
    from siarscan.viz.colormap import viridis_lut

    assert viridis_lut.shape == (256, 3)
    # Anchors dumped from the app's own buildLut('viridis') under node. matplotlib's viridis
    # ends (253, 231, 37) — close enough to look right on its own, wrong next to a lane the
    # browser drew.
    assert tuple(viridis_lut[0]) == (71, 1, 85)
    assert tuple(viridis_lut[128]) == (31, 145, 139)
    assert tuple(viridis_lut[255]) == (252, 231, 33)
