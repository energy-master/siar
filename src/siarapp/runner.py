# Vixen Intelligence c.2026
"""The per-file loop: decode, transform, scan, write.

This is the middle of the CLI — everything above it is argument parsing and everything below is
one of the modules it calls. It exists as its own module so the loop can be tested without a
terminal, and so the two hard-won rules about running over a big corpus live in one place:

* **One recording resident at a time.** The signal, its magnitude grid and its boxes are all
  released before the next file is opened. A ten-thousand-file survey costs the same memory as
  its largest single recording, and the run manifest is the only thing that grows.
* **A bad file is a row in the manifest, not the end of the run.** Corrupt headers, truncated
  WAVs, permissions, a recording shorter than one FFT window — every one of them is an outcome
  recorded against that file. Four hours into a scan is the wrong moment to discover that the
  loop had no opinion about a zero-byte file.
* **The output folder is complete after every recording, not at the end.** The audio, the
  sidecar and the thumbnail have always been written per file; the two root documents are now
  rewritten as the run goes as well, carrying a progress block. So an overnight scan can be
  dragged into IDent Dynamics at any point and the boxes found so far are all there, with the
  Performance panel showing how far through it is rather than pretending it finished.

The STFT defaults come from the algorithm itself (its ``expects``), not from this module. A
harbour-porpoise click scanner needs a 256-point window because a click is about a millisecond
long; a tonal survey needs 1024 or more. Making the user know that before they can run anything
would be a bad trade, so the algorithm carries its own grid and ``--fft`` / ``--hop`` override
it only when someone deliberately wants to.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

from siarapp.dsp.stft import frame_count, grid_bytes, stft
from siarapp.grid import FrameGrid, ScannerError, run_scan
from siarapp.io.audio import Recording, find_recordings, load_mono, probe
from siarapp.io.output import OutputFolder, sidecar_document
from siarapp.io.output import DEFAULT_FAMILY
from siarapp.io.performance import performance_document, progress_block
from siarapp.viz.thumbnail import thumbnail_png

__all__ = ["DEFAULT_FFT", "FileResult", "RunOptions", "plan_grid", "run_folder", "scan_one"]

#: Grid used when neither the algorithm nor the user has an opinion. The web app's own default,
#: which is what a box's coordinates were tuned against.
DEFAULT_FFT = 1024

#: Hop as a fraction of the FFT size when unspecified — 75% overlap, the app's default.
_DEFAULT_HOP_FRACTION = 4

#: Grid size past which the CLI warns before transforming. 2 GiB of float32 magnitudes is about
#: 45 minutes at 96 kHz and fft 1024; beyond it the machine is likely to swap, and a warning is
#: cheaper than a wedged laptop.
_LARGE_GRID_BYTES = 2 * 1024**3

#: Shortest gap between two rewrites of the root documents. Freshness past this is wasted: the
#: web app reads the folder when someone drops it in, not continuously.
_MIN_FLUSH_SEC = 2.0

#: The rewrite may cost at most one part in this of the run's wall time. Both root documents
#: carry a row per recording, so they grow with the corpus and a fixed interval that is free at
#: 100 files is a real tax at 100,000. Measuring the write and standing off twenty times as long
#: keeps the overhead flat at ~5% of a flush's own cost, whatever the size of the corpus.
_FLUSH_DUTY = 20.0


class RunOptions:
    """Everything the loop needs beyond "which algorithm" and "which folder".

    Attributes:
        fft: FFT size, or ``None`` to take the algorithm's.
        hop: Hop in samples, or ``None`` for a quarter of the FFT size.
        window: Window name.
        channel: Channel selection for the downmix (``mix`` / ``left`` / ``right`` / index).
        params: Run-time parameters handed to the algorithm's ``configure``.
        link: Hardlink the audio into the output folder instead of copying.
        resume: Skip recordings whose sidecar and audio are already in place.
        thumbnails: Write lane thumbnails.
        limit: Stop after this many recordings (a dry run over a big corpus).
        recursive: Descend into subfolders.
    """

    __slots__ = ("fft", "hop", "window", "channel", "params", "link", "resume",
                 "thumbnails", "limit", "recursive")

    def __init__(
        self,
        *,
        fft: int | None = None,
        hop: int | None = None,
        window: str = "hann",
        channel: str = "mix",
        params: dict | None = None,
        link: bool = False,
        resume: bool = False,
        thumbnails: bool = True,
        limit: int | None = None,
        recursive: bool = True,
    ) -> None:
        self.fft = fft
        self.hop = hop
        self.window = window
        self.channel = channel
        self.params = dict(params or {})
        self.link = link
        self.resume = resume
        self.thumbnails = thumbnails
        self.limit = limit
        self.recursive = recursive


class FileResult:
    """What happened to one recording.

    Attributes:
        rel_path: Path relative to the scanned folder.
        status: ``"scanned"``, ``"skipped"`` (resume), ``"too_short"``, or ``"error"``.
        count: Structures found.
        duration_sec: Recording length.
        elapsed_sec: Wall time spent on it.
        shapes: Per-shape counts.
        error: The message, when ``status == "error"``.
    """

    __slots__ = ("rel_path", "status", "count", "duration_sec", "elapsed_sec", "shapes", "error")

    def __init__(self, rel_path: str, status: str, *, count: int = 0, duration_sec: float = 0.0,
                 elapsed_sec: float = 0.0, shapes: dict | None = None, error: str = ""):
        self.rel_path = rel_path
        self.status = status
        self.count = count
        self.duration_sec = duration_sec
        self.elapsed_sec = elapsed_sec
        self.shapes = shapes or {}
        self.error = error

    def to_dict(self) -> dict:
        """The manifest row for this recording."""
        row = {
            "path": self.rel_path,
            "status": self.status,
            "structures": self.count,
            "duration_sec": round(self.duration_sec, 3),
            "elapsed_sec": round(self.elapsed_sec, 3),
        }
        if self.shapes:
            row["shapes"] = self.shapes
        if self.error:
            row["error"] = self.error
        return row


def plan_grid(algorithm: Any, options: RunOptions) -> dict:
    """Resolve the STFT the run will use: the user's choice, then the algorithm's, then ours.

    Args:
        algorithm: The loaded algorithm (read for ``expects``).
        options: The run options.

    An overridden FFT size drops the algorithm's hop rather than keeping it. The two are a
    pair: a scanner asking for fft 256 / hop 64 wants 75% overlap, and pinning hop 64 under a
    user's ``--fft 1024`` would silently give them 94% overlap — sixteen times the frames, and
    a scan that takes sixteen times as long for no reason anyone asked for.

    Returns:
        ``{"fft", "hop", "window"}``.

    Raises:
        ValueError: If the resulting hop is not in ``[1, fft]``, which a bad ``--hop`` produces
            and which is far clearer to say here than from inside the transform.
    """
    expects = dict(getattr(algorithm, "expects", None) or {})
    fft = int(options.fft or expects.get("fft") or DEFAULT_FFT)
    fft_is_the_algorithms = options.fft is None or int(options.fft) == int(expects.get("fft") or 0)
    default_hop = expects.get("hop") if fft_is_the_algorithms else None
    hop = int(options.hop or default_hop or max(1, fft // _DEFAULT_HOP_FRACTION))
    window = options.window or str(expects.get("window") or "hann")
    if hop < 1 or hop > fft:
        raise ValueError(f"hop {hop} is out of range [1, {fft}]")
    return {"fft": fft, "hop": hop, "window": window}


def scan_one(
    algorithm: Any,
    recording: Recording,
    plan: dict,
) -> tuple[list[dict], FrameGrid]:
    """Transform one decoded recording and scan it.

    Args:
        algorithm: The loaded algorithm.
        recording: The decoded mono recording.
        plan: The grid from :func:`plan_grid`.

    Returns:
        ``(regions, grid)``. An empty region list and a zero-frame grid when the recording is
        shorter than one analysis window.
    """
    result = stft(
        recording.samples,
        fft_size=plan["fft"],
        hop_size=plan["hop"],
        window_name=plan["window"],
    )
    grid = FrameGrid(
        result.magnitudes,
        result.frames,
        result.bins,
        recording.sample_rate,
        result.fft_size,
        result.hop_size,
        result.window_name,
    )
    if grid.frames == 0:
        return [], grid
    return run_scan(algorithm, grid), grid


def run_folder(
    handle: Any,
    source_root: str,
    out_root: str,
    options: RunOptions | None = None,
    *,
    progress: Callable[[int, int, FileResult], None] | None = None,
    on_start: Callable[[int, int, str, float], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> dict:
    """Scan every recording under ``source_root`` and build the output folder.

    Args:
        handle: A :class:`siarapp.loader.AlgorithmHandle`.
        source_root: The folder of recordings.
        out_root: Where to write the output folder.
        options: See :class:`RunOptions`.
        progress: Called ``(done, total, result)`` after each recording.
        on_start: Called ``(index, total, rel_path, duration_sec)`` *before* each recording is
            decoded. The pair exists because one recording is not a quick step: nearly all of
            its time is spent inside the algorithm's ``scan``, which is a single opaque call,
            so a caller told only about completions has nothing to say for the whole of it.
            ``duration_sec`` comes from the header probe already done above, and is ``0.0``
            for a file whose header would not read.
        warn: Called with one-line warnings (mixed sample rates, oversized grids).

    Returns:
        The run manifest, as written to ``siar-app-run.json``.

    Raises:
        FileNotFoundError: If ``source_root`` holds no recordings at all — almost always a typo
            or a folder of MP3s, and silently producing an empty output folder helps nobody.
        ValueError: If the requested grid is invalid.
    """
    options = options or RunOptions()
    warn = warn or (lambda _msg: None)
    algorithm = handle.algorithm

    files = find_recordings(source_root, recursive=options.recursive)
    if not files:
        raise FileNotFoundError(
            f"no recordings under {source_root} — expected .wav or .flac "
            "(use --no-recursive if you meant only the top level)"
        )
    if options.limit:
        files = files[: options.limit]

    plan = plan_grid(algorithm, options)
    if options.params:
        try:
            algorithm.configure(**options.params)
        except Exception as e:  # noqa: BLE001 - a closed algorithm's own validation
            raise ScannerError(f"algorithm rejected --param: {e}") from e

    durations = _probe_corpus(files, plan, warn)
    audio_total = sum(durations)

    out = OutputFolder(out_root, source_root, link=options.link)
    started = time.time()
    results: list[FileResult] = []
    audio_done = audio_worked = 0.0
    worked = 0
    next_flush_at = 0.0

    for i, path in enumerate(files, start=1):
        if on_start is not None:
            on_start(i, len(files), out.rel_path(path), durations[i - 1])
        result = _run_one_file(handle, path, out, plan, options)
        results.append(result)
        audio_done += durations[i - 1]
        if result.status != "skipped":
            worked += 1
            audio_worked += durations[i - 1]
        # Flushed before the caller is told, not after: `progress` is how anything outside this
        # loop learns a recording is finished, and whatever it does about that — print a line,
        # or go and read the folder — should not race the write that makes it true.
        #
        # The first recording always flushes, so the folder becomes openable as soon as there is
        # anything in it to open rather than two seconds later.
        if time.time() >= next_flush_at:
            cost = _write_state(
                out, handle, source_root, plan, options, results, started,
                progress_block(
                    started=started,
                    files_total=len(files),
                    files_done=i,
                    files_worked=worked,
                    audio_total_sec=audio_total,
                    audio_done_sec=audio_done,
                    audio_worked_sec=audio_worked,
                ),
            )[1]
            next_flush_at = time.time() + max(_MIN_FLUSH_SEC, cost * _FLUSH_DUTY)

        if progress is not None:
            progress(i, len(files), result)

    # The final write is not an optimisation of the last flush: it is the one that says the run
    # finished. Everything before it is stamped "running", which is exactly what an interrupted
    # scan should leave behind.
    manifest, _cost = _write_state(
        out, handle, source_root, plan, options, results, started,
        progress_block(
            started=started,
            files_total=len(files),
            files_done=len(results),
            files_worked=worked,
            audio_total_sec=audio_total,
            audio_done_sec=audio_done,
            audio_worked_sec=audio_worked,
            complete=True,
        ),
    )
    return manifest


def _write_state(
    out: OutputFolder,
    handle: Any,
    source_root: str,
    plan: dict,
    options: RunOptions,
    results: list[FileResult],
    started: float,
    progress: dict,
) -> tuple[dict, float]:
    """Rewrite both root documents from the results so far.

    Both are written every time, and both are atomic (see :mod:`siarapp.io.output`): a reader
    that opens the folder mid-flush sees the previous complete document, never half of this one.

    Returns:
        ``(manifest, seconds_the_write_cost)``. The cost is what paces the next flush.
    """
    at = time.time()
    manifest = _build_manifest(handle, source_root, plan, options, results, started, progress)
    out.write_manifest(manifest)
    # Written even when nothing was scanned: "this run did nothing, and took four minutes to
    # decide that" is a performance answer too, and the panel that reads it should not have to
    # cope with the file being absent.
    out.write_performance(performance_document(
        algorithm=handle.slug,
        algorithm_version=str(getattr(handle.algorithm, "version", "")),
        family=_family_of(handle),
        started=started,
        results=results,
        stft=plan,
        progress=progress,
    ))
    return manifest, time.time() - at


def _run_one_file(
    handle: Any,
    path: str,
    out: OutputFolder,
    plan: dict,
    options: RunOptions,
) -> FileResult:
    """Decode, scan and write one recording. Never raises for a per-file problem."""
    rel = out.rel_path(path)
    if options.resume and out.already_done(path):
        return FileResult(rel, "skipped")

    started = time.time()
    try:
        recording = load_mono(path, channel=options.channel)
    except OSError as e:
        return FileResult(rel, "error", elapsed_sec=time.time() - started, error=str(e))

    try:
        regions, grid = scan_one(handle.algorithm, recording, plan)
    except ScannerError as e:
        return FileResult(rel, "error", duration_sec=recording.duration_sec,
                          elapsed_sec=time.time() - started, error=str(e))

    status = "scanned" if grid.frames else "too_short"

    try:
        out.place_audio(path)
        out.write_sidecar(path, sidecar_document(
            filename=os.path.basename(path),
            duration_sec=recording.duration_sec,
            algorithm=handle.slug,
            algorithm_version=str(getattr(handle.algorithm, "version", "")),
            family=_family_of(handle),
            params=options.params,
            stft={
                "fft": plan["fft"],
                "hop": plan["hop"],
                "window": plan["window"],
                "sample_rate": recording.sample_rate,
            },
            structures=regions,
            source_path=path,
        ))
        if options.thumbnails:
            out.write_thumbnail(path, thumbnail_png(recording.samples))
    except OSError as e:
        return FileResult(rel, "error", count=len(regions),
                          duration_sec=recording.duration_sec,
                          elapsed_sec=time.time() - started,
                          error=f"could not write output: {e}")

    return FileResult(
        rel,
        status,
        count=len(regions),
        duration_sec=recording.duration_sec,
        elapsed_sec=time.time() - started,
        shapes=_shape_counts(regions),
    )


def _shape_counts(regions: list[dict]) -> dict:
    """Per-shape totals for one recording, so the manifest says *what* was found, not just how many."""
    counts: dict[str, int] = {}
    for r in regions:
        shape = str(r.get("shape") or "structure")
        counts[shape] = counts.get(shape, 0) + 1
    return dict(sorted(counts.items()))


def _probe_corpus(files: list[str], plan: dict, warn: Callable[[str], None]) -> list[float]:
    """Read every header once, warn about what they say, and return the durations.

    One pass for two jobs. The warnings are header-only checks worth making before spending
    hours — probing ten thousand files costs about a second, and both mixed sample rates and an
    oversized grid are the kind of problem that is obvious in advance and baffling afterwards.
    The durations are what the progress estimate is built on, and asking the same headers twice
    for them would double the only part of a run that happens before anything appears on screen.

    Returns:
        One duration in seconds per file, in the order given. ``0.0`` for a header that could
        not be read — that file is an error row later, and guessing a length for it would put
        time into the estimate that will never be spent.
    """
    infos = [probe(p) for p in files]
    _warn_about_corpus(infos, plan, warn)
    return [i.duration_sec if i is not None else 0.0 for i in infos]


def _warn_about_corpus(infos: list, plan: dict, warn: Callable[[str], None]) -> None:
    """Say what the headers imply about the run: mixed rates, oversized grids."""
    rates = sorted({i.sample_rate for i in infos if i is not None})
    if len(rates) > 1:
        shown = ", ".join(f"{r:g} Hz" for r in rates[:4])
        more = "" if len(rates) <= 4 else f" (+{len(rates) - 4} more)"
        warn(
            f"this folder holds {len(rates)} different sample rates ({shown}{more}). "
            "Every band in Hz maps to a different bin at each rate, so the scan is really "
            "several scans — consider splitting the folder."
        )

    biggest = max(
        (grid_bytes(i.frames, plan["fft"], plan["hop"]) for i in infos if i is not None),
        default=0,
    )
    if biggest > _LARGE_GRID_BYTES:
        gib = biggest / 1024**3
        frames = max(
            (frame_count(i.frames, plan["fft"], plan["hop"]) for i in infos if i is not None),
            default=0,
        )
        warn(
            f"the largest recording needs a {gib:.1f} GiB magnitude grid ({frames:,} frames "
            f"at fft {plan['fft']}, hop {plan['hop']}). Raise --hop to shrink it."
        )


def _family_of(handle) -> str:
    """Which family the loaded model belongs to.

    Falls back to structure scanners rather than to an empty string: a bundle built before the
    manifest carried a family IS a structure scanner, and a blank here would make the web app
    unable to tell what kind of results it is looking at for exactly the folders most likely to
    already exist.
    """
    return str((getattr(handle, "manifest", None) or {}).get("family_slug") or DEFAULT_FAMILY)


def _build_manifest(
    handle: Any,
    source_root: str,
    plan: dict,
    options: RunOptions,
    results: list[FileResult],
    started: float,
    progress: dict | None = None,
) -> dict:
    """Assemble ``siar-app-run.json`` from the per-file results so far.

    Called after every recording as well as at the end, so "so far" is meant literally: on a
    running scan every total here describes the part of the corpus that has been done.
    """
    total_shapes: dict[str, int] = {}
    for r in results:
        for shape, n in r.shapes.items():
            total_shapes[shape] = total_shapes.get(shape, 0) + n

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    return {
        "format": "siar-app-run-v1",
        "algorithm": handle.describe(),
        "source_root": os.path.abspath(source_root),
        "stft": plan,
        "params": options.params,
        "channel": options.channel,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_sec": round(time.time() - started, 2),
        "progress": progress or {},
        "files": len(results),
        "by_status": by_status,
        "structures": sum(r.count for r in results),
        "shapes": dict(sorted(total_shapes.items())),
        "audio_sec": round(sum(r.duration_sec for r in results), 2),
        "manifest": [r.to_dict() for r in results],
    }
