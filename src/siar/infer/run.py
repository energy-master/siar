# Vixen Intelligence c.2026
"""Applying a trained model to a folder.

Decode each recording, rebuild the grid **using the model's own feature spec** (not a fresh
one — that is the whole point of carrying the spec on the model), score it, extract boxes, render
a spectrogram PNG, and write it all to the database.

Note what is *not* here: no re-training, no re-calibration, no threshold fitting. The model
arrives fully determined. Two runs of the same model over the same folder produce identical
detections, which is what makes a result citable.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from siar.data.audio import load_audio
from siar.data.dataset import scan_folder
from siar.detect.boxes import extract_boxes
from siar.features.frontend import bin_edges_hz, build_grid
from siar.store.db import Store
from siar.train.model import TrainedModel
from siar.viz.spectrogram import render_spectrogram

__all__ = ["RunResult", "run_from_folder"]

Reporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of scoring a folder.

    Attributes:
        run_id: Database row id.
        run_uid: The run's stable handle.
        n_files: Recordings scored.
        n_detections: Boxes found.
    """

    run_id: int
    run_uid: str
    n_files: int
    n_detections: int


def run_from_folder(
    model: TrainedModel,
    model_id: int,
    folder: str,
    *,
    store: Store,
    threshold: float | None = None,
    name: str | None = None,
    render: bool = True,
    report: Reporter = lambda _line: None,
) -> RunResult:
    """Score every recording in a folder with a trained model.

    Args:
        model: The trained model.
        model_id: Its row id in ``siar_models``.
        folder: The folder to score. Need not be the one the model was trained on — that is the
            point.
        store: An open :class:`~siar.store.db.Store`.
        threshold: Override the model's own z threshold. Lower finds more (and more false
            positives); higher finds less.
        name: Run name. Defaults to the folder's name.
        render: Write a spectrogram PNG per recording. Turn off for a headless bulk run.
        report: Called with progress lines.

    Returns:
        The :class:`RunResult`.

    Raises:
        FileNotFoundError: If ``folder`` is not a directory.
        ValueError: If it holds no readable audio.
    """
    from siar import paths

    scan = scan_folder(folder)
    if not scan.files:
        raise ValueError(f"no readable audio in {folder}")

    effective = float(model.threshold if threshold is None else threshold)
    run_name = name or scan.root.rstrip("/").rsplit("/", 1)[-1]
    run_id, run_uid = store.create_run(
        name=run_name, model_id=model_id, input_path=scan.root, threshold=effective
    )
    report(f"  run {run_uid}  (threshold z = {effective:.2f})")

    out_dir = paths.run_dir(run_uid)
    if render:
        out_dir.mkdir(parents=True, exist_ok=True)

    spec = model.spec
    edges = bin_edges_hz(spec)
    total_detections = 0

    try:
        for i, audio in enumerate(scan.files, start=1):
            recording = load_audio(audio.path)
            grid = build_grid(recording.samples, recording.sample_rate, spec)

            if grid.shape[0] == 0:
                report(f"  [{i}/{scan.n_files}] {audio.name}: too short, skipped")
                continue

            # Score once, then cut boxes at whatever threshold is in force for this run — which
            # may not be the model's own.
            z = model.baseline.apply(model.detector.error_map(grid))
            boxes = extract_boxes(z, spec, effective, model.box_spec)

            png_path: str | None = None
            if render:
                png_name = f"{audio.name}.png"
                (out_dir / png_name).write_bytes(render_spectrogram(grid))
                png_path = png_name

            file_id = store.insert_file(
                run_id,
                path=audio.path,
                name=audio.name,
                duration_s=audio.duration,
                sample_rate=audio.sample_rate,
                frames=int(grid.shape[0]),
                n_bins=int(grid.shape[1]),
                t_min=0.0,
                t_max=float(grid.shape[0] * spec.delta_t),
                f_min=float(edges[0]),
                f_max=float(edges[-1]),
                png_path=png_path,
            )
            store.insert_detections(run_id, file_id, boxes)
            total_detections += len(boxes)

            top = f"  top z={boxes[0].score:,.0f}" if boxes else ""
            report(f"  [{i}/{scan.n_files}] {audio.name}: {len(boxes)} detection(s){top}")

        store.finish_run(run_id, status="done")
    except Exception as exc:
        store.finish_run(run_id, status="failed", error=str(exc))
        raise

    hours = scan.total_duration / 3600.0
    rate = total_detections / hours if hours > 0 else 0.0
    report(f"  {total_detections} detection(s) over {hours:.2f} h  ({rate:.1f} per hour)")

    return RunResult(
        run_id=run_id, run_uid=run_uid, n_files=scan.n_files, n_detections=total_detections
    )
