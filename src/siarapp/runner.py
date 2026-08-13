# Vixen Intelligence c.2026
"""The per-file loop: decode, transform, scan, write.

This is the middle of the CLI — everything above it is argument parsing and everything below is
one of the modules it calls. It exists as its own module so the loop can be tested without a
terminal, and so the two hard-won rules about running over a big corpus live in one place:

* **One recording resident at a time, per worker.** The signal, its magnitude grid and its boxes
  are all released before the next file is opened. A ten-thousand-file survey costs the same
  memory as its largest single recording — times the number of workers, which is why
  ``--parallel`` sizes its pool against the biggest header in the corpus — and the run manifest
  is the only thing that grows.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator

from siarapp.dsp.stft import frame_count, grid_bytes, stft
from siarapp.grid import FrameGrid, ScannerError, run_scan
from siarapp.io.audio import Recording, find_recordings, load_mono, probe
from siarapp.io.output import OutputFolder, sidecar_document
from siarapp.io.output import DEFAULT_FAMILY
from siarapp.io.performance import performance_document, phase_totals, progress_block
from siarapp.parallel import AlgorithmSpec, peak_worker_bytes, resolve_workers, scan_parallel
from siarapp.viz.thumbnail import thumbnail_png

__all__ = [
    "DEFAULT_FFT",
    "DEFAULT_MAX_BYTES",
    "FileResult",
    "RunOptions",
    "plan_grid",
    "run_folder",
    "scan_file",
    "scan_one",
]

#: Grid used when neither the algorithm nor the user has an opinion. The web app's own default,
#: which is what a box's coordinates were tuned against.
DEFAULT_FFT = 1024

#: Recordings above this many bytes are left out of the run unless ``--max-size`` says otherwise.
#:
#: The cost of a recording is not its file size, it is the magnitude grid the transform builds
#: from it — at fft 1024 / hop 256, roughly eight times the sample data, and every worker may be
#: handed the biggest file in the corpus. 550 MB of WAV is about an hour at 96 kHz stereo and
#: costs a worker something like 4 GB; the file above it in the folder, at three hours, costs 12
#: and takes the machine down with it. A default ceiling turns "the survey drive had one
#: day-long file in it" from a dead run into one skipped row in the manifest.
DEFAULT_MAX_BYTES = 550 * 1024 * 1024

#: Hop as a fraction of the FFT size when unspecified — 75% overlap, the app's default.
_DEFAULT_HOP_FRACTION = 4

#: Grid size past which the CLI warns before transforming. 2 GiB of float32 magnitudes is about
#: 45 minutes at 96 kHz and fft 1024; beyond it the machine is likely to swap, and a warning is
#: cheaper than a wedged laptop.
_LARGE_GRID_BYTES = 2 * 1024**3

#: Shortest gap between two rewrites of the root documents. Freshness past this is wasted: the
#: web app reads the folder when someone drops it in, not continuously.
_MIN_FLUSH_SEC = 2.0

#: Corpus size at which reading the headers moves onto threads. Below it the pool costs more to
#: start than the probes cost to run.
_PROBE_THREAD_FLOOR = 64

#: Threads used for that. Small on purpose: this is queue depth for a disk, not compute, and a
#: hundred threads on a spinning survey drive is slower than eight.
_PROBE_THREADS = 8

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
        workers: Recordings to scan at once. ``1`` runs everything in this process, which is the
            default and the only behaviour that existed before ``--parallel``; ``0`` means as
            many as the machine can use (see :func:`siarapp.parallel.resolve_workers`).
        max_bytes: Largest recording to ingest, in bytes (see :data:`DEFAULT_MAX_BYTES`).
            ``0`` scans whatever is there, however big.
    """

    __slots__ = ("fft", "hop", "window", "channel", "params", "link", "resume",
                 "thumbnails", "limit", "recursive", "workers", "max_bytes")

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
        workers: int = 1,
        max_bytes: int = DEFAULT_MAX_BYTES,
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
        self.workers = int(workers)
        self.max_bytes = max(0, int(max_bytes))


class FileResult:
    """What happened to one recording.

    Attributes:
        rel_path: Path relative to the scanned folder.
        status: ``"scanned"``, ``"skipped"`` (resume), ``"too_short"``, ``"too_large"``
            (``--max-size``), or ``"error"``.
        count: Structures found.
        duration_sec: Recording length.
        elapsed_sec: Wall time spent on it.
        shapes: Per-shape counts.
        error: The message, when ``status == "error"``.
        phases: Seconds spent in each of :data:`PHASES` that this file reached. Partial on an
            error — a file that failed to decode has a ``decode`` time and nothing else, which is
            itself the diagnosis.
        lane: Which worker slot scanned it, for the live display. Always ``0`` on a serial run,
            and deliberately not written to the manifest: it says which of several identical
            processes happened to take the file, which is true of this run and of nothing else.
    """

    __slots__ = ("rel_path", "status", "count", "duration_sec", "elapsed_sec", "shapes", "error",
                 "phases", "lane")

    def __init__(self, rel_path: str, status: str, *, count: int = 0, duration_sec: float = 0.0,
                 elapsed_sec: float = 0.0, shapes: dict | None = None, error: str = "",
                 phases: dict | None = None, lane: int = 0):
        self.rel_path = rel_path
        self.status = status
        self.count = count
        self.duration_sec = duration_sec
        self.elapsed_sec = elapsed_sec
        self.shapes = shapes or {}
        self.error = error
        self.phases = dict(phases or {})
        self.lane = lane

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
    phases: dict | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], FrameGrid]:
    """Transform one decoded recording and scan it.

    Args:
        algorithm: The loaded algorithm.
        recording: The decoded mono recording.
        plan: The grid from :func:`plan_grid`.
        phases: Optional dict to record timings into, as ``fft`` and ``scan`` seconds. Written
            into rather than returned so the two callers that want a breakdown can have one
            without changing what this returns for the ones that do not.
        on_stage: Called with each stage's name as it begins. The same names :data:`PHASES`
            uses, so what a display says is happening and what the closing table charges it to
            are the same thing.

    Returns:
        ``(regions, grid)``. An empty region list and a zero-frame grid when the recording is
        shorter than one analysis window.
    """
    if on_stage is not None:
        on_stage("fft")
    at = time.time()
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
    if phases is not None:
        phases["fft"] = time.time() - at
    if grid.frames == 0:
        return [], grid
    if on_stage is not None:
        on_stage("scan")
    at = time.time()
    regions = run_scan(algorithm, grid)
    if phases is not None:
        phases["scan"] = time.time() - at
    return regions, grid


def run_folder(
    handle: Any,
    source_root: str,
    out_root: str,
    options: RunOptions | None = None,
    *,
    progress: Callable[[int, int, FileResult], None] | None = None,
    on_start: Callable[[int, int, str, float, int, int], None] | None = None,
    on_corpus: Callable[[int, float, int], None] | None = None,
    on_idle: Callable[[int], None] | None = None,
    on_stage: Callable[[int, str], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> dict:
    """Scan every recording under ``source_root`` and build the output folder.

    Args:
        handle: A :class:`siarapp.loader.AlgorithmHandle`.
        source_root: The folder of recordings.
        out_root: Where to write the output folder.
        options: See :class:`RunOptions`.
        progress: Called ``(done, total, result)`` after each recording.
        on_start: Called ``(index, total, rel_path, duration_sec, lane, size_bytes)`` *before*
            each recording is decoded. The pair exists because one recording is not a quick step:
            nearly all of its time is spent inside the algorithm's ``scan``, which is a single
            opaque call, so a caller told only about completions has nothing to say for the whole
            of it. ``duration_sec`` comes from the header probe already done above, and is ``0.0``
            for a file whose header would not read; ``size_bytes`` is what it occupies on disk,
            and ``0`` when it could not be stat'ed. ``lane`` is the worker slot it went to,
            always ``0`` on a serial run.
        on_corpus: Called once, ``(files, audio_sec, workers)``, after the headers have been read
            and the pool sized — everything a display needs to draw a total before the first
            recording is opened.
        on_idle: Called ``(lane)`` when a worker slot runs out of work, at the tail of a parallel
            run.
        on_stage: Called ``(lane, stage)`` as each stage of a recording begins — the
            :data:`~siarapp.io.performance.PHASES` names. Between ``on_start`` and the result
            this is the only thing that moves, and on a long recording it is the difference
            between a display that says "still going" and one that says which part is slow.
        warn: Called with one-line warnings (mixed sample rates, oversized grids).

    Returns:
        The run manifest, as written to ``siar-app-run.json``.

    Recordings above ``options.max_bytes`` never reach the loop: they are a ``too_large`` row in
    the manifest and are gone before the headers are read, so neither the worker pool nor the
    progress estimate is sized against a file that will not be scanned.

    Raises:
        FileNotFoundError: If ``source_root`` holds no recordings at all — almost always a typo
            or a folder of MP3s, and silently producing an empty output folder helps nobody.
        ValueError: If the requested grid is invalid.
        ScannerError: If the algorithm rejects the parameters, or a worker process dies.
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
    # One stat per file, and every later question about size is answered from it: what the
    # ceiling excludes, and what the live display puts beside each recording's name.
    sizes = [_file_size(p) for p in files]
    files, sizes, oversized = _split_oversized(files, sizes, options.max_bytes, warn)

    plan = plan_grid(algorithm, options)
    if options.params:
        try:
            algorithm.configure(**options.params)
        except Exception as e:  # noqa: BLE001 - a closed algorithm's own validation
            raise ScannerError(f"algorithm rejected --param: {e}") from e

    infos = _probe_corpus(files, plan, warn)
    durations = [i.duration_sec if i is not None else 0.0 for i in infos]
    audio_total = sum(durations)

    workers = 1
    if options.workers != 1:
        workers = resolve_workers(options.workers, len(files), peak_worker_bytes(infos, plan),
                                  warn)
    if on_corpus is not None:
        on_corpus(len(files), audio_total, workers)

    out = OutputFolder(out_root, source_root, link=options.link)
    started = time.time()
    # The oversized files are outcomes already, so they go in as rows before anything is scanned:
    # a manifest that simply omitted them would say the folder held fewer recordings than it does,
    # which is the one thing a skipped file must never do.
    results: list[FileResult] = [
        FileResult(out.rel_path(path), "too_large",
                   error=f"{size:,} bytes is over the --max-size ceiling of "
                         f"{options.max_bytes:,}")
        for path, size in oversized
    ]
    audio_done = audio_worked = 0.0
    worked = 0
    next_flush_at = 0.0
    # Everything the loop below counts is offset by the rows already standing.
    total = len(files) + len(oversized)

    if workers > 1:
        stream = scan_parallel(
            AlgorithmSpec.of(handle, options.params), files, out, plan, options, durations,
            sizes, workers, on_start=on_start, on_idle=on_idle, on_stage=on_stage,
        )
    else:
        stream = _scan_serially(handle, files, out, plan, options, durations, sizes, on_start,
                                on_stage)

    for i, (index, result) in enumerate(stream, start=1):
        results.append(result)
        audio_done += durations[index]
        if result.status != "skipped":
            worked += 1
            audio_worked += durations[index]
        # Flushed before the caller is told, not after: `progress` is how anything outside this
        # loop learns a recording is finished, and whatever it does about that — print a line,
        # or go and read the folder — should not race the write that makes it true.
        #
        # The first recording always flushes, so the folder becomes openable as soon as there is
        # anything in it to open rather than two seconds later.
        if time.time() >= next_flush_at:
            cost = _write_state(
                out, handle, source_root, plan, options, results, started, workers,
                progress_block(
                    started=started,
                    files_total=total,
                    files_done=i + len(oversized),
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
        out, handle, source_root, plan, options, results, started, workers,
        progress_block(
            started=started,
            files_total=total,
            files_done=len(results),
            files_worked=worked,
            audio_total_sec=audio_total,
            audio_done_sec=audio_done,
            audio_worked_sec=audio_worked,
            complete=True,
        ),
    )
    return manifest


def _scan_serially(
    handle: Any,
    files: list[str],
    out: OutputFolder,
    plan: dict,
    options: RunOptions,
    durations: list[float],
    sizes: list[int],
    on_start: Callable[[int, int, str, float, int, int], None] | None,
    on_stage: Callable[[int, str], None] | None = None,
) -> Iterator[tuple]:
    """Scan every file in this process, yielding ``(index, FileResult)`` in folder order.

    The other half of the pair :func:`siarapp.parallel.scan_parallel` completes, and the reason
    the run loop has no idea which of the two it is consuming: one process or eight, the loop
    above sees the same stream of finished recordings and writes the same two documents.
    """
    total = len(files)
    # Lane 0, always: a serial run is one worker, and the display it feeds has one row.
    stage = None if on_stage is None else (lambda name: on_stage(0, name))
    for index, path in enumerate(files):
        if on_start is not None:
            on_start(index + 1, total, out.rel_path(path), durations[index], 0, sizes[index])
        yield index, scan_file(handle, path, out, plan, options, stage)


def _write_state(
    out: OutputFolder,
    handle: Any,
    source_root: str,
    plan: dict,
    options: RunOptions,
    results: list[FileResult],
    started: float,
    workers: int,
    progress: dict,
) -> tuple[dict, float]:
    """Rewrite both root documents from the results so far.

    Both are written every time, and both are atomic (see :mod:`siarapp.io.output`): a reader
    that opens the folder mid-flush sees the previous complete document, never half of this one.

    Returns:
        ``(manifest, seconds_the_write_cost)``. The cost is what paces the next flush.
    """
    at = time.time()
    manifest = _build_manifest(handle, source_root, plan, options, results, started, workers,
                               progress)
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
        workers=workers,
        progress=progress,
    ))
    return manifest, time.time() - at


def scan_file(
    handle: Any,
    path: str,
    out: OutputFolder,
    plan: dict,
    options: RunOptions,
    on_stage: Callable[[str], None] | None = None,
) -> FileResult:
    """Decode, scan and write one recording. Never raises for a per-file problem.

    Public because a worker process calls it directly (:mod:`siarapp.parallel`), and it is the
    whole of what a worker does: everything it touches is either read-only or this recording's
    own output paths, which is what makes running several at once safe.

    Args:
        handle: A :class:`siarapp.loader.AlgorithmHandle`.
        path: The recording.
        out: The output folder to write into.
        plan: The grid from :func:`plan_grid`.
        options: The run options.
        on_stage: Called with each :data:`PHASES` name as that stage begins, for a display that
            would otherwise have nothing to say for the several minutes one recording can spend
            inside ``scan``. A recording that is skipped or fails reports the stages it reached
            and no more.

    Returns:
        The :class:`FileResult` for this recording, whatever happened to it.
    """
    rel = out.rel_path(path)
    if options.resume and out.already_done(path):
        return FileResult(rel, "skipped")

    started = time.time()
    # Filled stage by stage, so whatever this file reached before it failed is still reported. The
    # phases are the only view into where a slow run's time actually goes; see :data:`PHASES`.
    phases: dict[str, float] = {}
    if on_stage is not None:
        on_stage("decode")
    try:
        recording = load_mono(path, channel=options.channel)
    except OSError as e:
        return FileResult(rel, "error", elapsed_sec=time.time() - started, error=str(e),
                          phases={"decode": time.time() - started})
    phases["decode"] = time.time() - started

    try:
        regions, grid = scan_one(handle.algorithm, recording, plan, phases, on_stage)
    except ScannerError as e:
        return FileResult(rel, "error", duration_sec=recording.duration_sec,
                          elapsed_sec=time.time() - started, error=str(e), phases=phases)

    status = "scanned" if grid.frames else "too_short"

    if on_stage is not None:
        on_stage("write")
    at = time.time()
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
        phases["write"] = time.time() - at
        if options.thumbnails:
            if on_stage is not None:
                on_stage("thumbnail")
            at = time.time()
            out.write_thumbnail(path, thumbnail_png(recording.samples))
            phases["thumbnail"] = time.time() - at
    except OSError as e:
        phases.setdefault("write", time.time() - at)
        return FileResult(rel, "error", count=len(regions),
                          duration_sec=recording.duration_sec,
                          elapsed_sec=time.time() - started,
                          error=f"could not write output: {e}", phases=phases)

    return FileResult(
        rel,
        status,
        count=len(regions),
        duration_sec=recording.duration_sec,
        elapsed_sec=time.time() - started,
        shapes=_shape_counts(regions),
        phases=phases,
    )


def _shape_counts(regions: list[dict]) -> dict:
    """Per-shape totals for one recording, so the manifest says *what* was found, not just how many."""
    counts: dict[str, int] = {}
    for r in regions:
        shape = str(r.get("shape") or "structure")
        counts[shape] = counts.get(shape, 0) + 1
    return dict(sorted(counts.items()))


def _file_size(path: str) -> int:
    """What a recording occupies on disk, or ``0`` when it cannot be stat'ed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _split_oversized(
    files: list[str],
    sizes: list[int],
    max_bytes: int,
    warn: Callable[[str], None],
) -> tuple[list[str], list[int], list[tuple[str, int]]]:
    """Divide the corpus into what will be scanned and what is too big to ingest.

    On file size, not on duration or frame count, and before the headers are read: the point is
    to be cheap and to be certain. ``os.stat`` costs nothing next to a decode, needs no codec to
    agree, and is the one number available for a file whose header is unreadable — which a
    truncated 40 GB WAV very often is.

    Args:
        files: Candidate recordings, in folder order.
        sizes: Their sizes, from :func:`_file_size`, one per file.
        max_bytes: The ceiling. ``0`` disables the test and returns everything.
        warn: Called once with a summary when anything was left out. Once, not once per file: a
            survey drive holding two thousand oversized files should cost two thousand rows in
            the manifest and one line on the terminal.

    Returns:
        ``(kept, kept_sizes, oversized)``, where ``oversized`` pairs each excluded path with its
        size. A file whose size would not read is kept: it goes on to be an error row with a real
        message rather than being called too big on the strength of a stat that never worked.
    """
    if max_bytes <= 0:
        return files, sizes, []

    kept: list[str] = []
    kept_sizes: list[int] = []
    oversized: list[tuple[str, int]] = []
    for path, size in zip(files, sizes):
        if size > max_bytes:
            oversized.append((path, size))
        else:
            kept.append(path)
            kept_sizes.append(size)

    if oversized:
        biggest = max(size for _p, size in oversized)
        warn(
            f"{len(oversized)} recording(s) are over the --max-size ceiling of "
            f"{max_bytes / 1024**2:.0f} MB (largest {biggest / 1024**2:.0f} MB) and will not be "
            "scanned. They are in the manifest as `too_large`. Raise --max-size to take them, "
            "or --max-size 0 for no ceiling — a single recording costs a worker several times "
            "its own size once it is a magnitude grid."
        )
    return kept, kept_sizes, oversized


def _probe_corpus(files: list[str], plan: dict, warn: Callable[[str], None]) -> list:
    """Read every header once, warn about what they say, and return what they said.

    One pass for three jobs. The warnings are header-only checks worth making before spending
    hours — both mixed sample rates and an oversized grid are the kind of problem that is obvious
    in advance and baffling afterwards. The durations drive the progress estimate. The frame
    counts size the worker pool. Asking the same headers again for any of the three would double
    the only part of a run that happens before anything appears on screen.

    Read on a few threads above :data:`_PROBE_THREAD_FLOOR` files, because a probe is one seek and
    one small read: it is latency, not work, and a three-week stream split into ten-minute files
    is a hundred thousand of them. The threads only wait on the filesystem — ``soundfile`` drops
    the GIL for the read — so this is not the parallelism ``--parallel`` provides and does not
    interact with it.

    Returns:
        One :class:`~siarapp.io.audio.AudioInfo` per file, in the order given, or ``None`` where
        the header could not be read — that file is an error row later, and inventing a length
        for it would put time into the estimate that will never be spent.
    """
    if len(files) >= _PROBE_THREAD_FLOOR:
        with ThreadPoolExecutor(max_workers=_PROBE_THREADS) as pool:
            infos = list(pool.map(probe, files))
    else:
        infos = [probe(p) for p in files]
    _warn_about_corpus(infos, plan, warn)
    return infos


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
    workers: int = 1,
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
        # How many recordings were in flight at once. A wall-clock time means something quite
        # different at 16 than at 1, and a manifest that does not say which is not comparable
        # with the next one.
        "workers": int(workers),
        "progress": progress or {},
        "files": len(results),
        "by_status": by_status,
        "structures": sum(r.count for r in results),
        "shapes": dict(sorted(total_shapes.items())),
        "audio_sec": round(sum(r.duration_sec for r in results), 2),
        # Totals only. Where the time went is a performance answer and the per-file detail belongs
        # in siar-app-performance.json — but the closing summary is printed from this document, so
        # the five sums come along rather than making the CLI open the other file to read them.
        "phases": phase_totals(results),
        "manifest": [r.to_dict() for r in results],
    }
