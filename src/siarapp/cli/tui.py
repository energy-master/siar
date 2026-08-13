# Vixen Intelligence c.2026
"""``--tui``: the whole run on one screen, redrawn in place.

The default displays answer "is it alive" — a line per recording, or a row per worker. This one
answers the questions asked while a survey is *running* and nowhere else:

* Will it finish in the time available? The bar, the realtime factor and the estimate.
* Where is the time going? The stage breakdown, live, so ``--no-thumbnails`` or a coarser ``--hop``
  is a decision that can be made an hour in rather than after the run.
* Is it finding anything? Structures by shape, as they accumulate — a survey that has scanned four
  hours and found nothing is usually the wrong algorithm rather than a quiet ocean.
* Is anything stuck, and did anything fail? A fixed row per worker, and the failures kept on
  screen instead of scrolling past at three in the morning.

**It takes the screen.** The run gets the alternate screen buffer — the one ``vim`` and ``less``
use — cleared, and the frame is drawn to the terminal's full width and full height, repainted from
the top left each tick. So there is no scrollback to fight with, no arithmetic about how tall the
panel was last time, and a window resized mid-run is simply drawn again at its new size. On the way
out the buffer is handed back untouched: the shell looks exactly as it did, and the closing summary
prints under the command that started the run.

The frame always fills the window. Sections have a fixed order and the least useful are dropped
when the terminal is too short, so a small window still shows the bar and the totals; the
completions list has no natural length and is stretched to reach the bottom rule, so a tall window
is full of run rather than empty below the middle.

Warnings and errors are collected as well as shown, and reprinted as plain lines once the buffer is
handed back — everything drawn inside it is discarded, and a failure nobody can scroll back to is a
failure nobody fixed.

Stdlib only, like everything else here: ANSI escapes and box-drawing characters, no curses, no
``rich``. Off a terminal there is no screen to take, so the flag falls back to the ordinary
displays rather than emitting a frame per tick into a log.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from collections import deque

try:  # POSIX only, and only needed to read one keystroke at the end of a run.
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None
    tty = None

from siarapp.cli.format import (
    BAR_CAP,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    HIDE_CURSOR,
    RED,
    SHOW_CURSOR,
    YELLOW,
    clock,
    cost,
    duration,
    factor_text,
    fit,
    fit_path,
    named_recording,
    paint,
    share,
    stage_tag,
    visible_len,
)
from siarapp.io.performance import PHASES, progress_block, realtime_factor

__all__ = ["TuiDisplay"]

#: Gap between redraws. Slower than the one-line displays': this rebuilds a whole frame, and a
#: ten-hour survey should not spend its evening drawing boxes.
_TICK_SEC = 0.25

#: Completions remembered for the "just finished" section. As many as reach the bottom of the
#: screen are drawn, so this is a ceiling for a tall terminal rather than a layout choice; the rest
#: is history the counters already carry.
_RECENT = 60

#: Problems kept on screen. Older ones are still reprinted on the way out.
_PROBLEMS = 3

#: Widest path column, and the outcome column beside it, in the "just finished" rows.
_RECENT_PATH = 56
_RECENT_TAIL = 30

#: Lane bar width, and the stage/shape bar width inside the two-column block.
_LANE_BAR = 12
_MINI_BAR = 10

#: Terminal width below which the two-column block becomes two stacked ones. A 40-column terminal
#: cannot hold two labelled bars side by side and the attempt is unreadable.
_TWO_COLUMN_MIN = 96

#: What dismisses the finished panel: Ctrl-Q as asked for, and ``q`` because someone will try it.
_QUIT_KEYS = (b"\x11", b"q", b"Q")

#: Drawn on the last line when a run has less history than screen. True, and the thing worth
#: knowing while watching a scan that might need stopping.
_FOOTER = "Ctrl-C leaves a usable folder — --resume picks it up where it stopped."

#: And once it has finished. The panel is worth nothing if the reader cannot tell it is waiting for
#: them rather than still working.
_HELD_FOOTER = "Run complete — Ctrl-Q (or q) to close this panel."

#: Shortest terminal the frame will draw itself into. Below this it is drawn at this height anyway
#: and the terminal scrolls it: a window eight lines tall cannot show a run, and pretending
#: otherwise would mean deciding which of the two rules to leave out.
_MIN_HEIGHT = 10

#: The alternate screen buffer — what ``vim`` and ``less`` use. Entering it gives the run a blank
#: screen of its own to own completely; leaving it puts the shell's scrollback back exactly as it
#: was, so the closing summary prints under the command that started the run rather than under the
#: wreckage of a panel.
_ENTER_SCREEN = "\033[?1049h\033[H\033[2J"
_LEAVE_SCREEN = "\033[?1049l"


class TuiDisplay:
    """The full-screen-ish run display behind ``--tui``.

    Carries the same five callbacks the other displays do
    (:class:`~siarapp.cli.commands.ScanReporter`, :class:`~siarapp.cli.commands.WorkerPanel`), so
    :func:`~siarapp.cli.commands.cmd_run` picks one and does not care which.

    Args:
        algorithm: Slug to name in the frame's title.
        workers: What the user asked for, so the title is right before the corpus is probed.
        source: Folder being scanned, and
        out: folder being written — shown inside the frame because the lines that announced them
            are on the other screen for as long as the run lasts.
    """

    def __init__(self, algorithm: str = "", workers: int = 1, source: str = "",
                 out: str = "", prior_rate: float = 0.0) -> None:
        self._algorithm = algorithm or "scan"
        self._where = f"{source}  →  {out}" if source and out else (source or out)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hidden = False

        self._started = time.time()
        self._workers = max(1, workers)
        self._lanes: list[tuple | None] = [None] * self._workers
        self._files_total = 0
        self._audio_total = 0.0
        self._files_done = 0
        self._files_worked = 0
        self._audio_done = 0.0
        self._audio_worked = 0.0
        # Durations from the headers, held from on_start to on_result: a skipped or unreadable
        # recording reports no duration of its own, and the bar still has to move past it.
        self._planned: dict[str, float] = {}
        # Learned cost inside one worker — wall seconds per second of audio — which is what a
        # lane's own bar is drawn from. Seeded from the last run of this algorithm on this
        # machine so the first round of the run has bars at all; replaced by this run's own
        # figure the moment a recording lands.
        self._prior = max(0.0, float(prior_rate))
        self._scan_audio = 0.0
        self._scan_wall = 0.0
        # And per lane, so a worker slower than its neighbours reads as a number on its own row.
        self._lane_audio: dict[int, float] = {}
        self._lane_wall: dict[int, float] = {}
        self._structures = 0
        self._shapes: dict[str, int] = {}
        self._phases: dict[str, float] = {}
        self._by_status: dict[str, int] = {}
        self._recent: deque[tuple[str, str, str]] = deque(maxlen=_RECENT)
        # Set by `hold`: the run is over, the clock stops, and the closing metrics take the place of
        # the live stage block until the reader quits.
        self._finished = False
        self._ended = 0.0
        self._metrics: list[list[str]] = []
        # ``(kind, text)`` — a per-file failure and a run-wide warning read differently on the way
        # out, and both have to survive the frame being wiped off the screen.
        self._problems: list[tuple[str, str]] = []

    # -- the callbacks run_folder wants ----------------------------------------------------

    def on_corpus(self, files: int, audio_sec: float, workers: int) -> None:
        """Learn the size of the run, and start drawing."""
        with self._lock:
            self._workers = max(1, workers)
            self._lanes = [None] * self._workers
            self._files_total = files
            self._audio_total = audio_sec
            self._started = time.time()
        sys.stdout.write(_ENTER_SCREEN + HIDE_CURSOR)
        self._hidden = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def on_start(self, index: int, total: int, rel_path: str, duration_sec: float,
                 lane: int = 0, size_bytes: int = 0) -> None:
        """A worker has taken a recording."""
        with self._lock:
            self._files_total = max(self._files_total, total)
            self._planned[rel_path] = duration_sec
            self._ensure_lanes(lane)
            self._lanes[lane] = (index, rel_path, duration_sec, size_bytes, time.time(), "")

    def on_idle(self, lane: int) -> None:
        """A worker has run out of work — the tail of the run."""
        with self._lock:
            self._ensure_lanes(lane)
            self._lanes[lane] = None

    def on_stage(self, lane: int, stage: str) -> None:
        """A worker has moved on to another stage of the recording it holds.

        The one thing that changes on a lane's row between taking a recording and finishing it,
        and on a long file that is minutes of otherwise motionless screen.
        """
        with self._lock:
            self._ensure_lanes(lane)
            entry = self._lanes[lane]
            if entry is not None:
                self._lanes[lane] = entry[:5] + (stage,)

    def on_result(self, done: int, total: int, result) -> None:
        """A recording is finished: fold it into every counter the frame draws."""
        with self._lock:
            seconds = self._planned.pop(result.rel_path, result.duration_sec)
            self._files_done = done
            self._files_total = max(self._files_total, total)
            self._audio_done += seconds
            self._by_status[result.status] = self._by_status.get(result.status, 0) + 1
            if result.status != "skipped":
                self._files_worked += 1
                self._audio_worked += seconds
            if result.status == "scanned" and result.duration_sec and result.elapsed_sec:
                self._scan_audio += float(result.duration_sec)
                self._scan_wall += float(result.elapsed_sec)
                lane = int(result.lane)
                self._lane_audio[lane] = self._lane_audio.get(lane, 0.0) + result.duration_sec
                self._lane_wall[lane] = self._lane_wall.get(lane, 0.0) + result.elapsed_sec
            self._structures += result.count
            for shape, n in result.shapes.items():
                self._shapes[shape] = self._shapes.get(shape, 0) + n
            for name, spent in getattr(result, "phases", {}).items():
                self._phases[name] = self._phases.get(name, 0.0) + float(spent)

            if result.status == "error":
                self._problems.append(("error", f"{result.rel_path}: {result.error}"))
                self._recent.appendleft(("✗", result.rel_path, "ERROR"))
            elif result.status == "skipped":
                self._recent.appendleft(("·", result.rel_path, "already done"))
            elif result.status == "too_short":
                self._recent.appendleft(("·", result.rel_path, "too short"))
            else:
                found = f"{result.count} structure{'' if result.count == 1 else 's'}"
                self._recent.appendleft(("✓", result.rel_path, found))

    def note(self, message: str) -> None:
        """Record a warning. Kept on screen while the run goes, and reprinted on the way out.

        The frame owns the bottom of the terminal, so a warning printed straight to stderr would
        be drawn over within a quarter of a second — which is the same as not printing it.
        """
        with self._lock:
            self._problems.append(("warning", message))

    def hold(self, metrics: list[list[str]]) -> None:
        """Freeze the finished run on screen, add its metrics, and wait to be dismissed.

        A run that wiped its own panel the moment it finished would take the answer with it — the
        stages, the lanes, what was found, all gone half a second after the last recording landed.
        So the frame stays, the clock stops, the closing table takes the place of the live stage
        block, and nothing happens until the reader presses Ctrl-Q.

        Blocking is the point, and it is safe: this runs on the main thread after
        :func:`siarapp.runner.run_folder` has returned, so there is no work left to hold up. Ctrl-C
        still interrupts, and the summary prints on the ordinary screen either way.

        Args:
            metrics: ``[label, value, share]`` rows from the closing summary — passed in rather
                than computed here, because this module draws and the command decides what a run
                cost.
        """
        if not self._hidden:
            return
        with self._lock:
            self._metrics = [list(row) for row in metrics]
            self._finished = True
            self._ended = time.time()
            self._draw(self._render(self._ended))
        self._wait_for_quit()

    def _wait_for_quit(self) -> None:
        """Block until Ctrl-Q, ``q``, or end of input.

        The terminal is put in cbreak mode so a single keystroke arrives without a newline, and
        ``IXON`` is cleared for the duration: Ctrl-Q is XON to the line discipline, which would
        swallow it as flow control before this ever saw it. Interrupts are left enabled, so Ctrl-C
        still stops the program the way it does everywhere else.

        Returns immediately when stdin is not a terminal — a scripted run has nobody to press a
        key, and waiting forever for one would hang a survey in a cron job.
        """
        try:
            if not sys.stdin.isatty():
                return
        except (AttributeError, ValueError):
            return
        if termios is None or tty is None:
            # No POSIX terminal control (Windows): the panel has been drawn, and holding the run
            # open with no way to read a keypress would be worse than closing it.
            return

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            mode = termios.tcgetattr(fd)
            mode[0] &= ~termios.IXON
            termios.tcsetattr(fd, termios.TCSANOW, mode)
            while True:
                key = sys.stdin.buffer.read(1)
                if not key or key in _QUIT_KEYS:
                    return
        except (OSError, termios.error):
            return
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except (OSError, termios.error):
                pass

    def close(self) -> None:
        """Stop drawing, give the terminal back, and leave the problems in the scrollback."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._hidden:
            sys.stdout.write(SHOW_CURSOR + _LEAVE_SCREEN)
            sys.stdout.flush()
            self._hidden = False
        for kind, text in self._problems:
            print(f"{kind}: {text}", file=sys.stderr)
        self._problems = []

    # -- drawing ---------------------------------------------------------------------------

    def _ensure_lanes(self, lane: int) -> None:
        """Grow the lane list if a lane arrives before (or without) ``on_corpus``."""
        if lane >= len(self._lanes):
            self._lanes.extend([None] * (lane + 1 - len(self._lanes)))
            self._workers = len(self._lanes)

    def _tick(self) -> None:
        while not self._stop.wait(_TICK_SEC):
            with self._lock:
                # Redrawn even while held, so a window resized after the run still looks right.
                self._draw(self._render(time.time()))

    def _draw(self, lines: list[str]) -> None:
        """Repaint the screen from the top left. Caller holds the lock.

        Home the cursor, clear each line as it is written, then clear whatever is below — so the
        frame needs no bookkeeping about how tall it was last time, and a terminal resized mid-run
        redraws correctly on the next tick rather than tearing.

        No trailing newline: a newline written on the bottom line of the screen scrolls it, and a
        scrolled screen is one the next repaint would draw in the wrong place.
        """
        sys.stdout.write("\033[H" + "\n".join(f"\033[K{line}" for line in lines) + "\033[J")
        sys.stdout.flush()

    def _cost_rate(self) -> float:
        """Wall seconds one worker spends per second of audio, or ``0.0`` when unknowable.

        This run's own measurement once anything has finished; the last run of this algorithm
        before that. Only zero on a machine that has never run it, where there is genuinely
        nothing to draw a bar from.
        """
        if self._scan_audio <= 0 or self._scan_wall <= 0:
            return self._prior
        return self._scan_wall / self._scan_audio

    def _lane_factor(self, lane: int) -> float:
        """One worker's own realtime factor: audio seconds scanned per wall second in that lane.

        Zero — drawn as ``—`` — until that lane has finished a recording. A worker's speed is
        only knowable from a worker's completed work, and the run-wide figure divided by the pool
        would hide exactly what the rows exist to show: the lane that is slower than the rest.
        """
        return realtime_factor(self._lane_audio.get(lane, 0.0), self._lane_wall.get(lane, 0.0))

    def _lane_fraction(self, entry: tuple | None, now: float) -> float:
        """How far through its recording one lane is, ``0.0`` when there is no way to tell."""
        if entry is None:
            return 0.0
        seconds, started_at = entry[2], entry[4]
        estimate = seconds * self._cost_rate()
        if estimate <= 0:
            return 0.0
        return min(BAR_CAP / 100.0, (now - started_at) / estimate)

    def _audio_in_flight(self, now: float) -> float:
        """Audio inside the workers right now, counted by how far through it each of them is.

        The overall bar counts recordings when they land, and on a pool of sixteen workers each
        half-way through an hour-long file that is eight hours of audio the bar knows nothing
        about — it reads 2% while every row under it reads 40%. Counting the part-done work is
        an estimate, and it is drawn from the same throughput the rows are, so the bar and the
        rows now agree with each other and with the clock.
        """
        return sum(
            self._lane_fraction(entry, now) * entry[2]
            for entry in self._lanes if entry is not None
        )

    def _render(self, now: float) -> list[str]:
        """The whole screen, exactly as many lines and columns as the terminal has.

        The frame is the window: it fills the width, it fills the height, and the last section is
        stretched to reach the bottom rule rather than leaving the lower half of the screen empty.

        Sections are dropped from the least useful up when the terminal is too short for all of
        them — the bar and the totals are the two lines somebody watching actually needs.
        """
        # A finished run is reported against the moment it finished, however long the panel is then
        # left on screen: "finished in 4:12" is a fact about the run, not about the reader.
        if self._finished:
            now = self._ended
        size = shutil.get_terminal_size((100, 24))
        width = max(40, size.columns)
        inner = width - 4
        height = max(_MIN_HEIGHT, size.lines)
        elapsed = max(0.0, now - self._started)
        # Nothing is in flight once the run is over, so a held panel reports only what finished.
        in_flight = 0.0 if self._finished else self._audio_in_flight(now)

        block = progress_block(
            started=self._started,
            files_total=self._files_total,
            files_done=self._files_done,
            files_worked=self._files_worked,
            audio_total_sec=self._audio_total,
            # Part-done recordings included: see :meth:`_audio_in_flight`. A held panel counts
            # only what finished — nothing is in flight once the run is over.
            audio_done_sec=self._audio_done + in_flight,
            # In the estimate too, or the panel draws a bar for the first hour of a run and still
            # says "estimating" beside it: the ETA is computed from audio worked per wall second,
            # and until a file lands all of the work done is in flight.
            audio_worked_sec=self._audio_worked + in_flight,
            now=now,
        )

        head = f"{self._algorithm} · {self._workers} worker{'' if self._workers == 1 else 's'}"
        body = [
            _row(self._bar_line(block, elapsed, in_flight, inner), inner),
            _row(self._totals_line(block, elapsed, inner), inner),
        ]
        if self._where:
            body.append(_row(paint(fit_path(self._where, inner), DIM), inner))

        # Everything between the two rules. Sections are taken while they fit; a section that does
        # not is skipped rather than ending the loop, so a small important one still lands when a
        # large one above it could not.
        room = height - 2 - len(body)
        for section in self._sections(inner, now):
            if len(section) <= room:
                body.extend(section)
                room -= len(section)

        # A held panel keeps a line back for its footer before anything else claims the space: it is
        # the only thing on screen saying the run is waiting for a keypress rather than still
        # working, and a reader who cannot find that has a terminal that looks wedged.
        keep = 1 if self._finished else 0

        # What is left goes to the completions, which are the one section with no natural length:
        # a rule plus as many of the most recent as reach the bottom of the screen.
        if room - keep > 2 and self._recent:
            rows = self._recent_lines(inner, room - keep - 1)
            body.append(_rule("just finished", inner))
            body.extend(_row(line, inner) for line in rows)
            room -= 1 + len(rows)

        # The last line is the one worth reading at that moment: how to stop a run that is going, or
        # how to dismiss one that has finished. A small run has less history than screen, and blank
        # rows to the bottom would otherwise look like the panel gave up.
        footer = paint(_HELD_FOOTER, BOLD) if self._finished else paint(_FOOTER, DIM)
        if room >= 2:
            body.extend(_row("", inner) for _ in range(room - 1))
            body.append(_row(footer, inner))
        elif room >= 1:
            body.append(_row(footer, inner))

        clocked = f"finished in {clock(elapsed)}" if self._finished else f"{clock(elapsed)} elapsed"
        lines = [_top(head, clocked, width)]
        lines += body[: height - 2]
        lines.append(paint("╰" + "─" * (width - 2) + "╯", DIM))
        return lines

    def _sections(self, inner: int, now: float) -> list[list[str]]:
        """The fixed-length parts of the frame, in the order they are worth keeping.

        Stages and shapes first — they are the reason this display exists — then the workers, then
        the problems. What just finished is not here: it has no length of its own and is built last,
        to fill whatever these leave.

        Each is returned complete with its rule, so a section is either drawn or not: a rule with
        nothing under it is worse than a missing section.
        """
        out = []
        block = self._stage_shape_block(inner)
        if block:
            out.append(block)
        # Idle lanes on a finished run say nothing: the closing metrics have the space instead.
        lanes = [] if self._finished else self._lane_lines(inner, now)
        if lanes:
            out.append([_rule("workers", inner)] + [_row(line, inner) for line in lanes])
        if self._problems:
            shown = self._problems[-_PROBLEMS:]
            out.append([_rule("problems", inner)]
                       + [_row(paint(f"! {text}", RED if kind == "error" else YELLOW), inner)
                          for kind, text in shown])
        return out

    def _bar_line(self, block: dict, elapsed: float, in_flight: float, inner: int) -> str:
        """The overall bar: fraction of the corpus by audio, the file count, and the run's speed.

        The speed sits here rather than only on the line below because this is the line somebody
        glances at, and "how far through" and "how fast" are one question asked twice.

        It counts the audio inside the workers as well as the audio they have finished, and says
        so with a ``~``. A pool of sixteen workers an hour into their first recordings has done
        sixteen hours of real work, and a speed that reads ``—`` until the first file lands is
        not more truthful than a marked estimate — it is just less use.
        """
        fraction = float(block["fraction"])
        # A part-done recording may not carry the bar to the end. The estimate is allowed to fill
        # the bar and it is allowed to be wrong, but "100%" on a run that is still working is the
        # one thing it must never say — the same rule the lane bars follow.
        if in_flight > 0 and not self._finished:
            fraction = min(fraction, BAR_CAP / 100.0)
        counts = f"{self._files_done}/{self._files_total} files"
        about = "~" if in_flight > 0 else ""
        audio = (f"{about}{duration(self._audio_done + in_flight)} "
                 f"of {duration(self._audio_total)}")
        factor = realtime_factor(self._audio_worked + in_flight, elapsed)
        # Green above realtime, amber below it: under 1x an overnight scan does not finish
        # overnight, and that is worth seeing before the morning.
        speed = paint(f"{about if factor > 0 else ''}{factor_text(factor)} realtime",
                      GREEN if factor >= 1 else YELLOW)
        # The bar takes whatever the labels leave, so a wide terminal gets a wide bar rather
        # than a fixed one with an acre of blank beside it.
        bar_width = max(8, inner - len(counts) - len(audio) - visible_len(speed) - 14)
        filled = int(round(max(0.0, min(1.0, fraction)) * bar_width))
        drawn = paint("█" * filled, GREEN) + paint("░" * (bar_width - filled), DIM)
        return (f"{drawn} {paint(f'{fraction * 100:3.0f}%', BOLD)}  "
                f"{counts}  {paint(audio, DIM)}  {speed}")

    def _totals_line(self, block: dict, elapsed: float, inner: int) -> str:
        """The one line to read if you only read one: when it ends, and what it has found."""
        eta = block["eta_sec"]
        if self._finished:
            when = paint("finished", GREEN)
        else:
            when = "estimating" if eta is None else f"{clock(eta)} left"
        parts = [
            when,
            f"{paint(f'{self._structures:,}', BOLD)} "
            f"structure{'' if self._structures == 1 else 's'}",
        ]
        for status, label, style in (("skipped", "skipped", DIM), ("too_short", "too short", DIM),
                                     ("error", "error", RED)):
            if self._by_status.get(status):
                parts.append(paint(f"{self._by_status[status]:,} {label}", style))
        return paint("  ·  ", DIM).join(parts)

    def _stage_shape_block(self, inner: int) -> list[str]:
        """Where the time went, and what came out — side by side when the terminal is wide enough.

        Stacked below that width rather than squeezed: two bars and two labels in forty columns is
        four things competing for the same eight characters, and none of them stays readable.
        """
        stages = self._metric_lines() if self._finished else self._stage_lines()
        shapes = self._shape_lines()
        if not stages and not shapes:
            return []
        title = "performance" if self._finished else "time by stage"

        if inner >= _TWO_COLUMN_MIN and stages and shapes:
            left = inner // 2 - 1
            right = inner - left - 3
            rows = []
            for i in range(max(len(stages), len(shapes))):
                a = _pad(stages[i] if i < len(stages) else "", left)
                b = _pad(shapes[i] if i < len(shapes) else "", right)
                rows.append(_row(f"{a} │ {b}", inner))
            return [_split_rule(title, "structures found", left, inner)] + rows

        out = []
        if stages:
            out.append(_rule(title, inner))
            out += [_row(line, inner) for line in stages]
        if shapes:
            out.append(_rule("structures found", inner))
            out += [_row(line, inner) for line in shapes]
        return out

    def _stage_lines(self) -> list[str]:
        """One row per measured stage, biggest first, with its share of the time measured.

        Biggest first rather than in the order they run: the reader is looking for the stage worth
        doing something about, and on every real run that is the first row of this list.

        The shares are of measured time, not of the wall clock. A ``--parallel`` run spends more
        worker seconds than it does wall seconds, so a percentage of wall time would come to
        several hundred; a percentage of the work says the same thing and stays true either way.
        """
        measured = sum(self._phases.values())
        if measured <= 0:
            return []
        order = sorted(self._phases.items(), key=lambda kv: -kv[1])
        label_width = max(len(name) for name in PHASES)
        lines = []
        for rank, (name, spent) in enumerate(order):
            fraction = spent / measured
            filled = int(round(fraction * _MINI_BAR))
            # The largest stage carries the accent: it is the row worth reading first, and on a
            # healthy run it is `scan`.
            drawn = (paint("█" * filled, CYAN if rank == 0 else DIM)
                     + paint("·" * (_MINI_BAR - filled), DIM))
            lines.append(f"{name:<{label_width}}  {paint(f'{cost(spent):>8}', BOLD)}  {drawn} "
                         f"{paint(f'{share(spent, measured):>4}', DIM)}")
        return lines

    def _metric_lines(self) -> list[str]:
        """The closing metrics, as rows: the same table the summary prints, in the panel.

        Drawn from the rows handed to :meth:`hold`, so the panel and the summary printed underneath
        it cannot disagree — they are one table rendered twice. Indented labels (the stage rows) keep
        their indent, which is what makes them read as belonging to the wall time above them.
        """
        if not self._metrics:
            return []
        width = max(visible_len(row[0]) for row in self._metrics)
        lines = []
        for label, value, part in self._metrics:
            # A stage row is indented; it is detail under the total above it, so it is dimmed and
            # its total is not emboldened.
            stage = label.startswith("  ")
            cells = [f"{label:<{width}}", f"{value:>9}", f"{part:>4}"]
            lines.append("  ".join((
                paint(cells[0], DIM) if stage else cells[0],
                cells[1] if stage else paint(cells[1], BOLD),
                paint(cells[2], DIM),
            )))
        return lines

    def _shape_lines(self) -> list[str]:
        """Structures by shape, most numerous first, each against the most numerous."""
        if not self._shapes:
            return []
        order = sorted(self._shapes.items(), key=lambda kv: -kv[1])
        top = order[0][1] or 1
        label_width = max(len(name) for name, _ in order)
        lines = []
        for rank, (shape, n) in enumerate(order):
            filled = int(round(_MINI_BAR * n / top))
            drawn = (paint("█" * filled, CYAN if rank == 0 else DIM)
                     + paint("·" * (_MINI_BAR - filled), DIM))
            lines.append(f"{shape:<{label_width}}  {paint(f'{n:>7,}', BOLD)}  {drawn}")
        return lines

    def _lane_lines(self, inner: int, now: float) -> list[str]:
        """One row per worker, in a fixed order, with the recording it is on.

        The rows hold their position for the whole run, which is what makes a stalled lane visible:
        a worker that has been on the same file for twenty minutes is a row that stopped moving,
        and nothing else on the screen would have shown it.
        """
        if not self._lanes:
            return []
        lines = []
        for lane, entry in enumerate(self._lanes):
            label = f"{lane + 1:>3}  "
            if entry is None:
                lines.append(f"{label}{paint('·' * _LANE_BAR + '   idle', DIM)}")
                continue
            _index, rel_path, seconds, size_bytes, started_at, stage = entry
            spent = now - started_at
            fraction = self._lane_fraction(entry, now)
            if fraction > 0:
                filled = int(round(_LANE_BAR * fraction))
                stat = (paint("█" * filled, GREEN) + paint("░" * (_LANE_BAR - filled), DIM)
                        + f" {100.0 * fraction:3.0f}% {spent:5.0f}s")
            else:
                # Nothing finished and no prior for this algorithm on this machine, so there is
                # no throughput to predict with. Elapsed alone: a bar from nothing is invention.
                stat = paint("░" * _LANE_BAR, DIM) + f"   —  {spent:5.0f}s"
            # This lane's own speed, beside this lane's own bar.
            stat += "  " + paint(f"{factor_text(self._lane_factor(lane)):>7}", CYAN)
            stat += "  " + paint(stage_tag(stage), CYAN)
            room = inner - len(label) - visible_len(stat) - 2
            lines.append(f"{label}{stat}  {named_recording(rel_path, size_bytes, seconds, room)}")
        return lines

    def _recent_lines(self, inner: int, limit: int) -> list[str]:
        """The recordings that have landed, newest first, as many as ``limit`` allows.

        The outcome sits in a column just past the longest path rather than against the right edge:
        on a wide terminal a run of short names would put every count eighty characters from the
        name it belongs to.
        """
        room = min(_RECENT_PATH, max(12, inner - _RECENT_TAIL - 4))
        rows = list(self._recent)[: max(0, limit)]
        marks = {"✓": GREEN, "✗": RED, "·": DIM}
        return [f"{paint(mark, marks.get(mark, DIM))} {fit_path(rel_path, room):<{room}}  "
                f"{paint(fit_path(tail, _RECENT_TAIL), DIM if mark != '✗' else RED)}"
                for mark, rel_path, tail in rows]


def _top(title: str, right: str, width: int) -> str:
    """The frame's top rule, with a title on the left and the clock on the right.

    The title is cut before the clock is: on a narrow terminal the elapsed time is the half worth
    keeping, and a rule one character too long would wrap and break the redraw.
    """
    plain_tail = f" {right} ─╮"
    room = width - len(plain_tail) - 4
    name = fit_path(title, max(0, room))
    fill = "─" * max(0, width - len(name) - len(plain_tail) - 4)
    return (paint("╭─ ", DIM) + paint(name, BOLD) + " " + paint(fill, DIM)
            + paint(" ", DIM) + paint(right, DIM) + paint(" ─╮", DIM))


def _rule(title: str, inner: int) -> str:
    """A section rule across the frame, titled."""
    name = fit_path(title, max(0, inner - 2))
    fill = "─" * max(0, inner - len(name) - 1)
    return paint(f"├─ {name} {fill}┤", DIM)


def _split_rule(left_title: str, right_title: str, left: int, inner: int) -> str:
    """The rule over a two-column block: a title each side and a ``┬`` where they divide.

    ``left`` is the width of the left column's content, which fixes where the divider goes: a row
    is ``"│ "`` then ``left`` characters, so the ``│`` between the columns lands at index
    ``left + 3`` and the ``┬`` above it has to as well.
    """
    chars = ["├"] + ["─"] * (inner + 2) + ["┤"]
    divider = left + 3
    for text, start in ((f"─ {left_title} ", 1), (f"─ {right_title} ", divider + 1)):
        for offset, char in enumerate(text):
            if start + offset < divider:
                chars[start + offset] = char
            elif start > divider and start + offset < len(chars) - 1:
                chars[start + offset] = char
    chars[divider] = "┬"
    return paint("".join(chars), DIM)


def _row(text: str, inner: int) -> str:
    """One line of frame content, padded and clipped to the frame's inside width."""
    edge = paint("│", DIM)
    return f"{edge} {fit(text, inner)} {edge}"


def _pad(text: str, width: int) -> str:
    """Fit text to exactly ``width`` visible columns. Colour codes cost nothing."""
    return fit(text, width)
