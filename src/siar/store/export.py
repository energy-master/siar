# Vixen Intelligence c.2026
"""Turning a run into a JSON document.

One function, used by both ``siar export`` and the dashboard's download button, so a file saved
from the browser and a file written by the CLI are byte-identical. Two ways to produce the "same"
export that quietly differ is a bug waiting to be discovered by whoever tries to reconcile them.
"""
from __future__ import annotations

import json

from siar import __version__
from siar.store.db import Store, utcnow

__all__ = ["EXPORT_FORMAT", "export_run"]

EXPORT_FORMAT = "siar-detections-v1"


def export_run(store: Store, run_uid: str) -> dict:
    """Build the full JSON payload for one run.

    Args:
        store: An open store.
        run_uid: The run's uid, or a unique prefix of it.

    Returns:
        A JSON-safe dict: the model that was used, the corpus, a summary, and every detection
        grouped by recording.

    Raises:
        KeyError: If no such run exists.
    """
    run = store.run_by_uid(run_uid)
    if run is None:
        raise KeyError(f"no run matching {run_uid!r}")

    files = store.files_for_run(int(run["id"]))
    total_seconds = sum(float(f["duration_s"]) for f in files)
    hours = total_seconds / 3600.0

    scores: list[float] = []
    file_payloads = []
    for f in files:
        detections = store.detections_for_file(int(f["id"]))
        scores.extend(float(d["score"]) for d in detections)
        file_payloads.append(
            {
                "name": f["name"],
                "path": f["path"],
                "duration_s": round(float(f["duration_s"]), 3),
                "sample_rate": int(f["sample_rate"]),
                "n_detections": int(f["n_detections"]),
                "max_score": float(f["max_score"]) if f["max_score"] is not None else None,
                "detections": [
                    {
                        "t_start": round(float(d["t_start"]), 4),
                        "t_end": round(float(d["t_end"]), 4),
                        "f_low": round(float(d["f_low"]), 1),
                        "f_high": round(float(d["f_high"]), 1),
                        "score": round(float(d["score"]), 3),
                        "peak_score": round(float(d["peak_score"]), 3),
                        "area": int(d["area"]),
                        "fill": round(float(d["fill"]), 3),
                    }
                    for d in detections
                ],
            }
        )

    scores.sort()

    def pct(p: float) -> float | None:
        if not scores:
            return None
        i = min(len(scores) - 1, int(p / 100.0 * len(scores)))
        return round(scores[i], 3)

    model_doc = json.loads(run["model_json"]) if "model_json" in run.keys() else None
    if model_doc is None:
        model_row = store.conn.execute(
            "SELECT model_json FROM siar_models WHERE id = ?", (run["model_id"],)
        ).fetchone()
        model_doc = json.loads(model_row["model_json"]) if model_row else {}

    # The weights are large and useless in a results file; the model's *identity* is what matters
    # for reproducing it, and that is the uid.
    detector_doc = dict(model_doc.get("detector", {}))
    detector_doc.pop("weights_b64", None)

    return {
        "format": EXPORT_FORMAT,
        "siar_version": __version__,
        "exported_at": utcnow(),
        "run": {
            "run_uid": run["run_uid"],
            "name": run["name"],
            "status": run["status"],
            "threshold": float(run["threshold"]),
            "created_at": run["created_at"],
            "finished_at": run["finished_at"],
        },
        "model": {
            "model_uid": run["model_uid"],
            "name": run["model_name"],
            "detector": run["detector"],
            "spec": model_doc.get("detector", {}).get("spec"),
            "threshold": model_doc.get("threshold"),
            "threshold_method": model_doc.get("threshold_method"),
            "contamination": model_doc.get("contamination"),
            "box_spec": model_doc.get("box_spec"),
            "config": detector_doc.get("config"),
            "provenance": model_doc.get("provenance"),
        },
        "dataset": {
            "path": run["input_path"],
            "n_files": len(files),
            "total_seconds": round(total_seconds, 1),
        },
        "summary": {
            "n_detections": sum(len(f["detections"]) for f in file_payloads),
            "detections_per_hour": round(
                sum(len(f["detections"]) for f in file_payloads) / hours, 2
            )
            if hours > 0
            else 0.0,
            "files_with_detections": sum(1 for f in file_payloads if f["detections"]),
            "score_percentiles": {"p50": pct(50), "p90": pct(90), "p99": pct(99)},
        },
        "files": file_payloads,
    }
