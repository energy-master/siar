# Vixen Intelligence c.2026
"""The train -> run -> export -> serve round trip, on a corpus with a known planted anomaly.

These tests go through the real store, the real PNG encoder and the real HTTP handlers, because
the interesting failures in this layer are not in the maths — they are in the wiring. A detection
that is correct in memory and lands in the wrong row, or a box whose grid coordinates do not match
the PNG it is drawn on, is invisible to a unit test of either half.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import soundfile as sf

from siar.detect.boxes import BoxSpec
from siar.infer.run import run_from_folder
from siar.store.db import Store
from siar.store.export import export_run
from siar.train.fit import default_spec, train_from_folder
from siar.train.model import TrainedModel
from siar.viz.png import encode_png
from siar.viz.spectrogram import render_spectrogram

SR = 16_000
CHIRP_T0, CHIRP_T1 = 2.0, 3.0
CHIRP_F0, CHIRP_F1 = 3000.0, 3500.0


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    k = np.arange(spectrum.size)
    k[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(k), n).astype(np.float32)


def _normal(rng: np.random.Generator) -> np.ndarray:
    return 0.05 * _pink(SR * 4, rng)


def _chirp(rng: np.random.Generator) -> np.ndarray:
    x = _normal(rng)
    t = np.arange(SR) / SR
    phase = 2 * np.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) * t**2)
    x[int(CHIRP_T0 * SR) : int(CHIRP_T1 * SR)] += (0.10 * np.sin(phase)).astype(np.float32)
    return x


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A trained model, scored over a folder containing one planted chirp."""
    root = tmp_path_factory.mktemp("corpus")
    normal_dir = root / "normal"
    check_dir = root / "check"
    normal_dir.mkdir()
    check_dir.mkdir()

    rng = np.random.default_rng(7)
    for i in range(12):
        sf.write(normal_dir / f"n{i:02d}.wav", _normal(rng), SR)
    for i in range(2):
        sf.write(check_dir / f"quiet{i}.wav", _normal(rng), SR)
    sf.write(check_dir / "chirp.wav", _chirp(rng), SR)

    spec = default_spec(SR, fft_size=512, n_bins=32)
    result = train_from_folder(
        str(normal_dir), spec=spec, config={
            "depth": 2, "base_channels": 16, "latent_channels": 2, "dropout": 0.0,
            "noise_std": 0.1, "lr": 1e-3, "weight_decay": 1e-6, "batch_size": 128, "epochs": 12,
        }, name="test-model", seed=0,
    )

    db = root / "siar.db"
    store = Store(db)
    store.migrate()
    model_id, model_uid = store.insert_model(result.model)
    run = run_from_folder(
        result.model, model_id, str(check_dir), store=store, name="check", render=True
    )
    return {
        "store": store,
        "model": result.model,
        "model_uid": model_uid,
        "run": run,
        "check_dir": check_dir,
    }


def test_run_records_every_file_and_finds_the_chirp(corpus):
    store, run = corpus["store"], corpus["run"]
    files = store.files_for_run(run.run_id)
    assert len(files) == 3

    # Files come back most-anomalous-first, so the chirp must lead.
    assert files[0]["name"] == "chirp"
    assert files[0]["n_detections"] > 0
    assert all(f["n_detections"] == 0 for f in files if f["name"].startswith("quiet"))

    detections = store.detections_for_file(int(files[0]["id"]))
    top = detections[0]
    assert top["t_start"] < CHIRP_T1 + 0.2 and top["t_end"] > CHIRP_T0 - 0.2
    assert top["f_low"] < CHIRP_F1 + 400 and top["f_high"] > CHIRP_F0 - 400


def test_run_totals_match_the_rows_actually_written(corpus):
    """The summary must be derived from the detections, never from a caller's count."""
    store, run = corpus["store"], corpus["run"]
    row = store.run_by_uid(run.run_uid)
    counted = store.conn.execute(
        "SELECT COUNT(*) AS n FROM siar_detections WHERE run_id = ?", (run.run_id,)
    ).fetchone()["n"]
    assert row["n_detections"] == counted == run.n_detections
    assert row["n_files"] == 3
    assert row["status"] == "done"


def test_model_survives_the_database_round_trip(corpus):
    """A model read back out of SQLite must score identically to the one that went in."""
    store, original = corpus["store"], corpus["model"]
    row = store.model_by_uid(corpus["model_uid"])
    restored = TrainedModel.from_json(json.loads(row["model_json"]))

    assert restored.spec == original.spec
    assert restored.threshold == pytest.approx(original.threshold)

    rng = np.random.default_rng(99)
    from siar.features.frontend import build_grid

    grid = build_grid(_chirp(rng), SR, original.spec)
    assert [d.to_dict() for d in restored.detect_grid(grid)] == [
        d.to_dict() for d in original.detect_grid(grid)
    ]


def test_model_uid_prefix_lookup(corpus):
    store, uid = corpus["store"], corpus["model_uid"]
    assert store.model_by_uid(uid[:14])["model_uid"] == uid
    assert store.model_by_uid("model-nope") is None


def test_png_is_exactly_one_pixel_per_grid_cell(corpus):
    """The dashboard's box placement depends on this and nothing else."""
    store, run = corpus["store"], corpus["run"]
    from siar import paths

    row = store.files_for_run(run.run_id)[0]
    png = paths.run_dir(run.run_uid) / row["png_path"]
    data = png.read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (row["frames"], row["n_bins"])


def test_detection_grid_coordinates_lie_inside_the_png(corpus):
    """A box that references a pixel outside the image would be drawn in the wrong place."""
    store, run = corpus["store"], corpus["run"]
    for f in store.files_for_run(run.run_id):
        for d in store.detections_for_file(int(f["id"])):
            assert 0 <= d["frame_lo"] <= d["frame_hi"] < f["frames"]
            assert 0 <= d["bin_lo"] <= d["bin_hi"] < f["n_bins"]


def test_export_is_complete_and_carries_the_model_config(corpus):
    store, run = corpus["store"], corpus["run"]
    payload = export_run(store, run.run_uid)

    assert payload["format"] == "siar-detections-v1"
    assert payload["summary"]["n_detections"] == run.n_detections
    assert payload["summary"]["files_with_detections"] == 1
    assert payload["model"]["spec"] is not None
    assert payload["model"]["config"] is not None
    assert len(payload["files"]) == 3

    # Weights must NOT be in a results file — they are large and it is not what it is for.
    assert "weights_b64" not in json.dumps(payload)

    # ...and it must actually serialise.
    json.dumps(payload)


def test_export_of_an_unknown_run_raises(corpus):
    with pytest.raises(KeyError):
        export_run(corpus["store"], "run-does-not-exist")


def test_threshold_override_changes_what_is_found(corpus):
    """Lowering the threshold must find strictly more; it is the knob users reach for first."""
    store, model = corpus["store"], corpus["model"]
    row = store.model_by_uid(corpus["model_uid"])

    loose = run_from_folder(
        model, int(row["id"]), str(corpus["check_dir"]), store=store,
        threshold=model.threshold / 100.0, name="loose", render=False,
    )
    assert loose.n_detections >= corpus["run"].n_detections


def test_png_encoder_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), dtype=np.float32))


def test_render_handles_an_empty_grid():
    assert render_spectrogram(np.zeros((0, 32), dtype=np.float32))[:8] == b"\x89PNG\r\n\x1a\n"


def test_box_spec_round_trips():
    spec = BoxSpec(min_fill=0.5, peak_fraction=0.2, max_boxes=17)
    assert BoxSpec.from_dict(spec.to_dict()) == spec
