# Vixen Intelligence c.2026
"""The performance report — how fast a run went, and on what.

Written at the root of every output folder as ``siar-app-performance.json``, beside the run
manifest, and read by the web app when the folder is opened so the numbers appear next to the
results they describe.

Separate from the run manifest on purpose. The manifest answers *what was found*; this answers
*what it cost*, and the two are read by different people at different times — one by whoever is
reading the boxes, the other by whoever is deciding whether a corpus can be scanned overnight.
Keeping them apart also means a future model family that finds something other than structures
still produces a performance file of exactly this shape.

**Realtime factor is the number that matters** and it is stated as a multiple: 3.0 means the
scan ran three times faster than the audio it scanned, so an hour of recording took twenty
minutes. Above 1 is faster than realtime. It is computed from audio actually *scanned*, not
from the folder's total duration — a run that skipped half its files on ``--resume`` did not
process that audio and must not take credit for it.

The document is rewritten as the run goes, not once at the end, and :func:`progress_block` is
what makes a half-written one readable: it says how far through the corpus the run is and when
it expects to finish, so a folder dropped into the web app mid-run renders a live report rather
than looking like a finished run that scanned a third of its files.
"""
from __future__ import annotations

import os
import platform
import sys
import time

__all__ = [
    "PERFORMANCE_NAME",
    "PERFORMANCE_FORMAT",
    "performance_document",
    "progress_block",
]

#: Written at the root of every output folder.
PERFORMANCE_NAME = "siar-app-performance.json"

#: Stamped into the document so a reader can tell what it is holding.
PERFORMANCE_FORMAT = "siar-app-performance-v1"


def _machine() -> dict:
    """What ran the scan, so two timings can be compared honestly.

    A realtime factor with no machine attached is close to meaningless — the same scan is 8x on
    a workstation and 0.9x on a vessel laptop, and a user comparing two runs needs to know which
    they are looking at.
    """
    return {
        "platform": f"{platform.system()} {platform.machine()}".strip(),
        "python": platform.python_version(),
        "cpus": os.cpu_count() or 0,
    }


def progress_block(
    *,
    started: float,
    files_total: int,
    files_done: int,
    files_worked: int = 0,
    audio_total_sec: float = 0.0,
    audio_done_sec: float = 0.0,
    audio_worked_sec: float = 0.0,
    complete: bool = False,
    now: float | None = None,
) -> dict:
    """How far through the corpus this run is, and when it expects to finish.

    Written into both root documents after every recording, which is what lets the web app draw
    a progress bar for a run that is still going.

    The estimate is made in **audio seconds, not files**. A survey folder is never evenly
    divided: one 40-minute drift recording among three hundred 10-second clips would make a
    file-count percentage jump from 8% to 92% and then sit there for an hour. Audio duration is
    what the scan actually costs, it is known up front from the headers, and it makes the bar
    move at a steady rate.

    Args:
        started: ``time.time()`` when the run began.
        files_total: Recordings in the run.
        files_done: Recordings handled so far, whatever their outcome.
        files_worked: Of those, the ones actually processed — ``--resume`` skips are excluded so
            they neither earn credit for speed nor count as work still to come.
        audio_total_sec: Total audio in the corpus, from the headers.
        audio_done_sec: Audio belonging to the recordings handled so far.
        audio_worked_sec: Audio belonging to the ones actually processed.
        complete: True for the final write, which fixes the state at ``"complete"`` and the
            estimate at zero.
        now: Override the clock, for tests.

    Returns:
        The ``progress`` block, with ``eta_sec`` ``None`` when there is not yet enough
        information to guess — a null a reader can render as "estimating" instead of a zero it
        would render as "finished".
    """
    now = time.time() if now is None else now
    elapsed = max(0.0, now - started)
    remaining_files = max(0, int(files_total) - int(files_done))

    if audio_total_sec > 0:
        fraction = min(1.0, audio_done_sec / audio_total_sec)
    elif files_total > 0:
        fraction = min(1.0, files_done / files_total)
    else:
        fraction = 0.0

    eta: float | None
    if complete or remaining_files <= 0:
        eta, fraction = 0.0, 1.0
    elif audio_worked_sec > 0 and elapsed > 0 and audio_total_sec > 0:
        # Audio per wall second, measured on this run only, so a resumed scan is rated on the
        # work it actually did. The remaining audio is discounted by the share of files that
        # have been skipped so far: on a corpus that is mostly done, most of what is left will
        # be skipped too, and charging full price for it would report hours that never happen.
        rate = audio_worked_sec / elapsed
        remaining_audio = max(0.0, audio_total_sec - audio_done_sec)
        if files_done > 0:
            remaining_audio *= files_worked / files_done
        eta = remaining_audio / rate if rate > 0 else None
    elif files_done > 0 and elapsed > 0:
        eta = elapsed / files_done * remaining_files
    else:
        eta = None

    return {
        "state": "complete" if complete else "running",
        "files_total": int(files_total),
        "files_done": int(files_done),
        "files_remaining": remaining_files,
        "audio_total_sec": round(float(audio_total_sec), 2),
        "audio_done_sec": round(float(audio_done_sec), 2),
        "fraction": round(fraction, 4),
        "elapsed_sec": round(elapsed, 2),
        "eta_sec": None if eta is None else round(eta, 1),
        "eta_at": "" if eta is None else _stamp(now + eta),
        "updated_at": _stamp(now),
    }


def _stamp(when: float) -> str:
    """A UTC timestamp in the format every other document here uses."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def performance_document(
    *,
    algorithm: str,
    algorithm_version: str = "",
    family: str = "",
    started: float,
    results,
    stft: dict | None = None,
    progress: dict | None = None,
) -> dict:
    """Build the performance report from the finished per-file results.

    Args:
        algorithm: The slug that was run.
        algorithm_version: Its version, so a slowdown can be pinned to a release.
        family: The model family, so the app knows what kind of run this describes.
        started: ``time.time()`` when the run began.
        results: The ``FileResult`` objects, in the order they were scanned.
        stft: The analysis grid — the single biggest lever on speed, and the first thing to
            look at when a run is slower than expected.
        progress: The block from :func:`progress_block`. Omitted only by a caller that has no
            notion of a corpus; the run loop always supplies one.

    Returns:
        The document, ready to be written at the output-folder root.
    """
    wall = max(0.0, time.time() - started)
    scanned = [r for r in results if r.status == "scanned"]
    audio_sec = sum(r.duration_sec for r in scanned)
    scan_sec = sum(r.elapsed_sec for r in scanned)

    files = [
        {
            "file": r.rel_path,
            "status": r.status,
            "audio_sec": round(r.duration_sec, 3),
            "elapsed_sec": round(r.elapsed_sec, 3),
            # Per file as well as overall: one pathological recording in a folder of a thousand
            # is invisible in an average and obvious in a sorted column.
            "realtime_factor": _factor(r.duration_sec, r.elapsed_sec),
            "count": r.count,
        }
        for r in results
    ]

    return {
        "format": PERFORMANCE_FORMAT,
        "algorithm": algorithm,
        "algorithm_version": algorithm_version,
        "family": family,
        "started_at": _stamp(started),
        # Before the totals, because on a running scan it is the only part anyone reads.
        "progress": progress or progress_block(
            started=started, files_total=len(results), files_done=len(results), complete=True,
        ),
        "machine": _machine(),
        "stft": stft or {},
        "totals": {
            "files": len(results),
            "files_scanned": len(scanned),
            "audio_sec": round(audio_sec, 2),
            # Two different times, and the gap between them is the answer to "why is this
            # slower than the sum of its parts": `wall` includes decoding, thumbnails, copying
            # the audio and writing the sidecars, while `scan_sec` is the algorithm alone.
            "wall_sec": round(wall, 2),
            "scan_sec": round(scan_sec, 2),
            "realtime_factor": _factor(audio_sec, wall),
            "scan_realtime_factor": _factor(audio_sec, scan_sec),
            "sec_per_file": round(wall / len(results), 3) if results else 0.0,
            "throughput_audio_hours_per_hour": _factor(audio_sec, wall),
        },
        "files": files,
    }


def _factor(audio_sec: float, elapsed_sec: float) -> float:
    """Audio seconds per elapsed second, to two decimals. ``0.0`` when it cannot be computed.

    Zero rather than infinity for an instant scan: the JSON has to survive a round trip, and
    ``Infinity`` is not valid JSON — a reader that got one would fail on the whole document
    rather than on one row.
    """
    if elapsed_sec <= 0 or audio_sec <= 0:
        return 0.0
    return round(audio_sec / elapsed_sec, 2)
