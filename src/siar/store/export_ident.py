# Vixen Intelligence c.2026
"""Export SIAR detections as IDent Dynamics decision sidecar files.

IDent Dynamics auto-pairs ``<basename>.json`` sidecars with audio files when a folder is opened
in the browser.  Each sidecar is a JSON array of decision records — one per detection — carrying
the time-frequency box, a model signature, and the detection's z-score.

The format is defined by ``ident_dynamic/js/decisions.js`` and accepted by the folder-drop UI,
the Results Import panel, and the headless ``/api/idapi/`` endpoints.  Field names follow the
IDent convention (``tmin``/``tmax`` in seconds, ``fmin_hz``/``fmax_hz`` in Hz).

Usage::

    siar export-ident <run-uid> --out /tmp/ident-drop

produces one ``<recording>.json`` per scored file.  Copy or symlink the audio into the same
folder, open it in IDent Dynamics, and the detections appear.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from siar.store.db import Store

__all__ = ["export_ident_sidecars"]


def _decisions_for_file(
    detections: list,
    signature: str,
) -> list[dict]:
    """Convert SIAR detection rows to IDent Dynamics decision records.

    Args:
        detections: Rows from ``siar_detections``, strongest first.
        signature: The ``signature`` string for each decision (e.g. ``"siar:conv_ae"``).

    Returns:
        A list of decision dicts ready for ``json.dump``.
    """
    decisions: list[dict] = []
    for d in detections:
        t_start = float(d["t_start"])
        t_end = float(d["t_end"])
        decisions.append(
            {
                "signature": signature,
                "dt": round((t_start + t_end) / 2, 4),
                "decision": "detection",
                "tmin": round(t_start, 4),
                "tmax": round(t_end, 4),
                "fmin_hz": round(float(d["f_low"]), 1),
                "fmax_hz": round(float(d["f_high"]), 1),
            }
        )
    return decisions


def export_ident_sidecars(
    store: Store,
    run_uid: str,
    out_dir: str | Path,
) -> int:
    """Write one JSON sidecar per scored file into *out_dir*.

    Args:
        store: An open store.
        run_uid: The run's uid, or a unique prefix of it.
        out_dir: Destination folder.  Created if it does not exist.

    Returns:
        The number of sidecar files written (i.e. files that had at least one detection).

    Raises:
        KeyError: If no run matches *run_uid*.
    """
    run = store.run_by_uid(run_uid)
    if run is None:
        raise KeyError(f"no run matching {run_uid!r}")

    signature = f"siar:{run['detector']}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = store.files_for_run(int(run["id"]))
    written = 0
    for f in files:
        detections = store.detections_for_file(int(f["id"]))
        if not detections:
            continue

        decisions = _decisions_for_file(detections, signature)
        # Strip the audio extension and write <basename>.json so IDent pairs by name.
        stem = os.path.splitext(f["name"])[0]
        sidecar = out / f"{stem}.json"
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(decisions, fh, indent=2)
        written += 1

    return written
