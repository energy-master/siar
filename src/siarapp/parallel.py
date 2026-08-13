# Vixen Intelligence c.2026
"""Scanning several recordings at once, one process per core.

A scan is compute-bound inside the algorithm's ``scan``, which is ordinary Python bytecode
however it was obfuscated, so it holds the GIL and threads buy nothing. Recordings are
independent of each other — nothing a file's scan does is visible to any other file — so the
unit of parallelism is the file, and the unit of isolation is a process.

**Processes, started by spawn.** Fork would be cheaper: the algorithm is already imported in the
parent, and children would inherit it. It is also unsafe here, because the pool grows lazily
while the progress ticker thread is running, and forking a process whose other thread holds a
lock gives a child that deadlocks the first time it touches that lock. Spawn costs each worker
one interpreter start and one bundle import, paid once at the start of a run that is measured in
hours, and it behaves identically on every platform.

So a worker cannot be handed the loaded algorithm — a PyArmor-obfuscated module is not
picklable, and would not survive the trip if it were. It is handed an :class:`AlgorithmSpec`
instead: where on disk the bundle was unpacked, and which parameters to configure it with. Every
worker loads the same tree the parent loaded, and applies the same ``--param`` values, so the
boxes do not depend on which worker happened to take the file.

**Nothing is shared and nothing is locked.** Each worker writes its own recording's audio,
sidecar and thumbnail — different paths, atomic writes — and returns a
:class:`~siarapp.runner.FileResult`. The two root documents stay the parent's alone, so there is
exactly one writer for the only two files any two recordings would contend over.

Spawn has one cost worth stating: the worker re-imports the program's entry point, so anything
calling :func:`siarapp.runner.run_folder` with workers must guard its own with
``if __name__ == "__main__":``. The CLI's console script does; an unguarded scratch script will
fork-bomb itself and report a broken pool.

**Work is kept one deep per lane.** Exactly as many files are in flight as there are workers, and
a new one is submitted as each finishes. That bounds the memory a run can occupy — the reason
:func:`resolve_workers` exists — and gives the display a stable slot per worker to report into.
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, ProcessPoolExecutor, wait
from multiprocessing import get_context
from typing import Any, Callable, Iterator

from siarapp.dsp.stft import grid_bytes
from siarapp.grid import ScannerError

__all__ = [
    "AlgorithmSpec",
    "limit_library_threads",
    "memory_ceiling",
    "peak_worker_bytes",
    "resolve_workers",
    "scan_parallel",
    "total_memory_bytes",
]

#: Share of physical memory a run may plan to occupy. The rest is the operating system's, the
#: page cache the decode reads through, and whatever else the machine is doing — a survey laptop
#: is rarely running only this.
_MEMORY_SHARE = 0.7

#: What a worker costs beyond its magnitude grid: the decoded signal, soundfile's own read
#: buffer, the interpreter and numpy itself. Deliberately generous — the cost of overestimating
#: is one fewer worker, and the cost of underestimating is a machine that swaps for six hours.
_WORKER_OVERHEAD_BYTES = 256 * 1024**2

#: The grid is built from the signal and then read alongside it, so a worker holds more than one
#: copy at its peak. Measured against the transform in :mod:`siarapp.dsp.stft`.
_GRID_PEAK_FACTOR = 1.6

#: Thread-count variables that BLAS libraries read at import time. A worker per core, each
#: running a library that also wants a thread per core, is how a parallel run ends up slower than
#: a serial one. Set in the parent before any worker starts, and only when the user has not
#: already said what they want.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class AlgorithmSpec:
    """How to load the same algorithm again in another process.

    Attributes:
        kind: ``"bundle"`` for an unpacked download, ``"source"`` for ``--algorithm-path``.
        root: The directory it was loaded from.
        slug: The algorithm slug.
        platform: The build tag, or ``"source"``.
        params: The ``--param`` values to configure each copy with.
    """

    __slots__ = ("kind", "root", "slug", "platform", "params")

    def __init__(self, kind: str, root: str, slug: str, platform: str, params: dict) -> None:
        self.kind = kind
        self.root = root
        self.slug = slug
        self.platform = platform
        self.params = dict(params)

    @classmethod
    def of(cls, handle: Any, params: dict | None = None) -> AlgorithmSpec:
        """Describe an already-loaded handle so a worker can reproduce it."""
        kind = "source" if handle.platform == "source" else "bundle"
        return cls(kind, handle.root, handle.slug, handle.platform, params or {})

    def load(self) -> Any:
        """Load and configure this algorithm in the calling process.

        Returns:
            An :class:`~siarapp.loader.AlgorithmHandle`.

        Raises:
            ScannerError: If the bundle will not import here, or rejects the parameters.
        """
        from siarapp.loader import load_local, load_unpacked

        if self.kind == "source":
            handle = load_local(self.root, self.slug)
        else:
            handle = load_unpacked(self.root, self.slug, self.platform)
        if self.params:
            try:
                handle.algorithm.configure(**self.params)
            except Exception as e:  # noqa: BLE001 - a closed algorithm's own validation
                raise ScannerError(f"algorithm rejected --param: {e}") from e
        return handle


# -- sizing the pool -------------------------------------------------------------------------


def total_memory_bytes() -> int:
    """Physical memory on this machine, or ``0`` when it cannot be read.

    ``sysconf`` covers Linux and macOS, which is where a survey corpus is scanned. Zero on
    Windows rather than a guess: an unknown ceiling must not become a low one.
    """
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 0


def peak_worker_bytes(infos: list, plan: dict) -> int:
    """What one worker will occupy at its worst, from the corpus headers.

    Sized on the *largest* recording, not the average: every worker can be given that file, and a
    pool sized for the mean is a pool that dies four hours in on the one long drift recording.

    Args:
        infos: The :class:`~siarapp.io.audio.AudioInfo` per file, ``None`` where the header would
            not read.
        plan: The grid from :func:`siarapp.runner.plan_grid`.

    Returns:
        Bytes per worker, overhead included.
    """
    biggest = max(
        (grid_bytes(i.frames, plan["fft"], plan["hop"]) for i in infos if i is not None),
        default=0,
    )
    return int(biggest * _GRID_PEAK_FACTOR) + _WORKER_OVERHEAD_BYTES


def memory_ceiling(peak_bytes: int) -> int:
    """How many workers this machine's memory will hold. ``0`` when memory is unknown."""
    total = total_memory_bytes()
    if total <= 0 or peak_bytes <= 0:
        return 0
    return max(1, int(total * _MEMORY_SHARE) // peak_bytes)


def resolve_workers(
    requested: int,
    file_count: int,
    peak_bytes: int,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Decide how many workers to run.

    Args:
        requested: What the user asked for. ``0`` means "as many as this machine can use".
        file_count: Recordings in the run — never worth more workers than files.
        peak_bytes: From :func:`peak_worker_bytes`.
        warn: Called with one-line warnings.

    Returns:
        The worker count, at least 1.

    An explicit ``--parallel N`` is obeyed even when it does not fit in memory, with a warning:
    the user may know something the headers do not, and this is their machine. An automatic count
    is capped instead, because nobody asked for it and a swapping run is worse than a slower one.
    """
    warn = warn or (lambda _msg: None)
    cpus = os.cpu_count() or 1
    automatic = requested <= 0
    want = max(1, min(cpus if automatic else int(requested), max(1, file_count)))

    ceiling = memory_ceiling(peak_bytes)
    if ceiling and want > ceiling:
        gib = peak_bytes / 1024**3
        if automatic:
            warn(
                f"using {ceiling} worker(s) rather than {want}: the largest recording needs "
                f"about {gib:.1f} GiB per worker and this machine has "
                f"{total_memory_bytes() / 1024**3:.0f} GiB. Raise --hop, or split the folder, "
                f"to use more cores."
            )
            want = ceiling
        else:
            warn(
                f"--parallel {want} needs about {want * gib:.0f} GiB — the largest recording "
                f"costs a worker {gib:.1f} GiB and this machine has "
                f"{total_memory_bytes() / 1024**3:.0f} GiB. Running it as asked; drop to "
                f"--parallel {ceiling} if it starts swapping."
            )
    return want


def limit_library_threads() -> None:
    """Pin BLAS and OpenMP to one thread each, unless the user has already chosen.

    Called in the parent before the first worker starts, so the children inherit it: a spawned
    worker reads these when it imports numpy, which is long after anything we could set from
    inside the worker itself.
    """
    for name in _THREAD_VARS:
        os.environ.setdefault(name, "1")


# -- the worker ------------------------------------------------------------------------------

#: Everything a worker needs, built once per process by :func:`_init_worker`. Module-level
#: because that is the only state a pool worker can carry between tasks, and rebuilding it per
#: file would re-import the bundle ten thousand times.
_WORKER: dict = {}


def _init_worker(spec: AlgorithmSpec, out_root: str, source_root: str, plan: dict,
                 options: Any) -> None:
    """Load the algorithm and open the output folder in a fresh worker process.

    Raising here breaks the pool, which is the right outcome: an algorithm that will not load is
    not a per-file problem, and every worker is about to fail the same way.
    """
    from siarapp.io.output import OutputFolder

    _WORKER["handle"] = spec.load()
    _WORKER["out"] = OutputFolder(out_root, source_root, link=options.link)
    _WORKER["plan"] = plan
    _WORKER["options"] = options


def _scan_path(index: int, path: str) -> tuple:
    """Scan one recording in a worker. Returns ``(index, FileResult)``.

    The import is deferred so that :mod:`siarapp.runner` may import this module at the top and
    this one need not import it back.
    """
    from siarapp.runner import scan_file

    return index, scan_file(
        _WORKER["handle"], path, _WORKER["out"], _WORKER["plan"], _WORKER["options"],
    )


# -- the pool --------------------------------------------------------------------------------


def scan_parallel(
    spec: AlgorithmSpec,
    files: list[str],
    out: Any,
    plan: dict,
    options: Any,
    durations: list[float],
    workers: int,
    *,
    on_start: Callable[[int, int, str, float, int], None] | None = None,
    on_idle: Callable[[int], None] | None = None,
) -> Iterator[tuple]:
    """Scan every file across ``workers`` processes, yielding results as they land.

    Args:
        spec: How a worker loads the algorithm.
        files: Absolute paths, in the order the manifest lists them.
        out: The parent's :class:`~siarapp.io.output.OutputFolder`, for relative paths only —
            each worker builds its own.
        plan: The grid from :func:`siarapp.runner.plan_grid`.
        options: The :class:`~siarapp.runner.RunOptions`.
        durations: One duration per file, from the headers, for the display.
        workers: How many processes to run.
        on_start: ``(index, total, rel_path, duration_sec, lane)`` as each file is handed to a
            worker. ``lane`` is a display slot, ``0`` to ``workers - 1``, held for as long as
            that file is being scanned and then reused by the next one.
        on_idle: ``(lane)`` when a lane has no more work — only at the tail of a run.

    Yields:
        ``(index, FileResult)`` in completion order, which is not input order: a two-second clip
        submitted after a forty-minute recording finishes long before it.

    Raises:
        ScannerError: If a worker process dies outright — an out-of-memory kill or a segfault
            inside the algorithm. There is no way to attribute that to one recording, so the run
            stops; the output folder holds everything finished so far and ``--resume`` continues
            from there.
    """
    total = len(files)
    limit_library_threads()
    pending = iter(list(enumerate(files)))
    free_lanes = list(range(workers - 1, -1, -1))
    inflight: dict = {}

    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(spec, out.root, out.source_root, plan, options),
    )

    def submit_next() -> bool:
        """Hand the next file to a free lane. False when there are none left."""
        try:
            index, path = next(pending)
        except StopIteration:
            return False
        lane = free_lanes.pop()
        inflight[executor.submit(_scan_path, index, path)] = lane
        if on_start is not None:
            on_start(index + 1, total, out.rel_path(path), durations[index], lane)
        return True

    try:
        for _ in range(workers):
            if not submit_next():
                # Fewer files than workers: say so rather than leave empty lanes drawn as busy.
                for lane in free_lanes:
                    if on_idle is not None:
                        on_idle(lane)
                break

        while inflight:
            done, _not_done = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                lane = inflight.pop(future)
                index, result = future.result()
                result.lane = lane
                free_lanes.append(lane)
                # Refilled before the result is handed up, so the worker starts its next file
                # rather than waiting on whatever the caller does with this one — which is a
                # manifest rewrite, and on a large corpus that is not free.
                if not submit_next() and on_idle is not None:
                    on_idle(lane)
                yield index, result
    except BrokenExecutor as e:
        raise ScannerError(_broken_message(workers, e)) from e
    finally:
        # Not `with`: on Ctrl-C or a broken pool the point is to stop, and shutting down with
        # wait=True would sit here until every worker finished the file it was on.
        executor.shutdown(wait=False, cancel_futures=True)


def _broken_message(workers: int, error: Exception) -> str:
    """What to say when a worker process dies rather than returning a result.

    Two causes, and the reader cannot tell them apart from here: the pool breaks the same way
    when a worker cannot load the algorithm at all as when one is killed mid-file. Both are named
    because the worker's own traceback has already gone to stderr above this, and between the two
    the reader has everything needed to tell which happened.
    """
    return (
        f"a scanning worker process died ({error}).\n"
        f"  Either the algorithm would not load in a worker — its traceback is above — or the "
        f"machine ran out of memory with {workers} recordings in flight at once.\n"
        f"  Everything scanned so far is already in the output folder: rerun with --resume and "
        f"a smaller --parallel to carry on."
    )
