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

**One frame, no scrolling.** Everything is drawn inside a box that is redrawn in place, sized to
the terminal each tick, so nothing the run prints can push it apart. Sections are dropped in a
fixed order when the terminal is too short for all of them, so a 24-line window still shows the bar
and the totals rather than half a frame. Warnings and errors are collected as well as shown, and
reprinted as plain lines on the way out — the frame is gone by the time the summary prints, and a
failure nobody can scroll back to is a failure nobody fixed.

Stdlib only, like everything else here: ANSI cursor movement and box-drawing characters, no curses,
no ``rich``. Off a terminal there is nothing to redraw in place, so the flag falls back to the
ordinary displays rather than emitting a frame per tick into a log.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from collections import deque

from siarapp.cli.format import (
    BAR_CAP,
    HIDE_CURSOR,
    SHOW_CURSOR,
    clip,
    clock,
    cost,
    duration,
    factor_text,
    fit_path,
    share,
)
from siarapp.cli.table import terminal_width
from siarapp.io.performance import PHASES, progress_block, realtime_factor

__all__ = ["TuiDisplay"]

#: Gap between redraws. Slower than the one-line displays': this rebuilds a whole frame, and a
#: ten-hour survey should not spend its evening drawing boxes.
_TICK_SEC = 0.25

#: Rows kept in the "just finished" section. What fits is drawn; the rest is history the counters
#: already carry.
_RECENT = 8

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

#: Lines the frame needs before any optional section: the two rules, the bar and the totals.
_FRAME_OVERHEAD = 4


class TuiDisplay:
    """The full-screen-ish run display behind ``--tui``.

    Carries the same five callbacks the other displays do
    (:class:`~siarapp.cli.commands.ScanReporter`, :class:`~siarapp.cli.commands.WorkerPanel`), so
    :func:`~siarapp.cli.commands.cmd_run` picks one and does not care which.

    Args:
        algorithm: Slug to name in the frame's title.
        workers: What the user asked for, so the title is right before the corpus is probed.
    """

    def __init__(self, algorithm: str = "", workers: int = 1) -> None:
        self._algorithm = algorithm or "scan"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drawn = 0
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
        # lane's own bar is drawn from.
        self._scan_audio = 0.0
        self._scan_wall = 0.0
        self._structures = 0
        self._shapes: dict[str, int] = {}
        self._phases: dict[str, float] = {}
        self._by_status: dict[str, int] = {}
        self._recent: deque[tuple[str, str, str]] = deque(maxlen=_RECENT)
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
        sys.stdout.write(HIDE_CURSOR)
        self._hidden = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def on_start(self, index: int, total: int, rel_path: str, duration_sec: float,
                 lane: int = 0) -> None:
        """A worker has taken a recording."""
        with self._lock:
            self._files_total = max(self._files_total, total)
            self._planned[rel_path] = duration_sec
            self._ensure_lanes(lane)
            self._lanes[lane] = (index, rel_path, duration_sec, time.time())

    def on_idle(self, lane: int) -> None:
        """A worker has run out of work — the tail of the run."""
        with self._lock:
            self._ensure_lanes(lane)
            self._lanes[lane] = None

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

    def close(self) -> None:
        """Stop drawing, give the terminal back, and leave the problems in the scrollback."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._hidden:
            with self._lock:
                self._erase()
            sys.stdout.write(SHOW_CURSOR)
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

    def _erase(self) -> None:
        """Take the frame off the screen. Caller holds the lock."""
        if self._drawn:
            sys.stdout.write(f"\033[{self._drawn}A\033[J")
            self._drawn = 0

    def _tick(self) -> None:
        while not self._stop.wait(_TICK_SEC):
            with self._lock:
                self._draw(self._render(time.time()))

    def _draw(self, lines: list[str]) -> None:
        """Redraw the frame in place. Caller holds the lock."""
        out = [f"\033[{self._drawn}A"] if self._drawn else []
        out += [f"\r\033[K{line}\n" for line in lines]
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._drawn = len(lines)

    def _cost_rate(self) -> float:
        """Wall seconds one worker spends per second of audio, or ``0.0`` before it is known."""
        if self._scan_audio <= 0 or self._scan_wall <= 0:
            return 0.0
        return self._scan_wall / self._scan_audio

    def _render(self, now: float) -> list[str]:
        """The whole frame, sized to the terminal as it is at this tick.

        Sections are added while there is room and dropped from the bottom up when there is not:
        the bar and the totals are the two lines somebody watching actually needs, and a frame that
        overflowed the window would scroll — which breaks redrawing in place and leaves torn copies
        of itself up the screen.
        """
        width = min(terminal_width(), 120)
        inner = width - 4
        height = max(8, shutil.get_terminal_size((100, 24)).lines - 1)
        elapsed = max(0.0, now - self._started)

        block = progress_block(
            started=self._started,
            files_total=self._files_total,
            files_done=self._files_done,
            files_worked=self._files_worked,
            audio_total_sec=self._audio_total,
            audio_done_sec=self._audio_done,
            audio_worked_sec=self._audio_worked,
            now=now,
        )

        head = f"{self._algorithm} · {self._workers} worker{'' if self._workers == 1 else 's'}"
        lines = [_top(head, f"{clock(elapsed)} elapsed", width)]
        lines.append(_row(self._bar_line(block, inner), inner))
        lines.append(_row(self._totals_line(block, elapsed, inner), inner))

        # What is left after the frame's own furniture, which is the two rules plus the two lines
        # above. Every section below asks for its rule as well as its rows.
        budget = height - _FRAME_OVERHEAD
        for section in self._sections(inner, now):
            if len(section) <= budget:
                lines.extend(section)
                budget -= len(section)
        lines.append("╰" + "─" * (width - 2) + "╯")
        return lines

    def _sections(self, inner: int, now: float) -> list[list[str]]:
        """The optional parts of the frame, in the order they are worth keeping.

        Stages and shapes first — they are the reason this display exists — then the workers, then
        the problems, then what just finished, which is the one section the counters above already
        summarise and so the one worth losing first on a short terminal.

        Each is returned complete with its rule, so a section is either drawn or not: a rule with
        nothing under it is worse than a missing section. A section that does not fit is skipped
        rather than ending the loop, so a small important one still lands when a large one above it
        could not.
        """
        out = []
        body = self._stage_shape_block(inner)
        if body:
            out.append(body)
        lanes = self._lane_lines(inner, now)
        if lanes:
            out.append([_rule("workers", inner)] + [_row(line, inner) for line in lanes])
        if self._problems:
            shown = self._problems[-_PROBLEMS:]
            out.append([_rule("problems", inner)]
                       + [_row(f"! {text}", inner) for _kind, text in shown])
        if self._recent:
            out.append([_rule("just finished", inner)]
                       + [_row(line, inner) for line in self._recent_lines(inner)])
        return out

    def _bar_line(self, block: dict, inner: int) -> str:
        """The overall bar: fraction of the corpus, by audio, with the file count beside it."""
        fraction = float(block["fraction"])
        counts = f"{self._files_done}/{self._files_total} files"
        audio = f"{duration(self._audio_done)} of {duration(self._audio_total)}"
        # The bar takes whatever the two labels leave, so a wide terminal gets a wide bar rather
        # than a fixed one with an acre of blank beside it.
        bar_width = max(8, inner - len(counts) - len(audio) - 12)
        filled = int(round(max(0.0, min(1.0, fraction)) * bar_width))
        drawn = "█" * filled + "░" * (bar_width - filled)
        return f"{drawn} {fraction * 100:3.0f}%  {counts}  {audio}"

    def _totals_line(self, block: dict, elapsed: float, inner: int) -> str:
        """The one line to read if you only read one: speed, when it ends, what it has found."""
        eta = block["eta_sec"]
        factor = realtime_factor(self._audio_worked, elapsed)
        parts = [
            f"{factor_text(factor)} realtime",
            "estimating" if eta is None else f"{clock(eta)} left",
            f"{self._structures:,} structure{'' if self._structures == 1 else 's'}",
        ]
        for status, label in (("skipped", "skipped"), ("too_short", "too short"),
                              ("error", "error")):
            if self._by_status.get(status):
                parts.append(f"{self._by_status[status]:,} {label}")
        return "  ·  ".join(parts)[:inner]

    def _stage_shape_block(self, inner: int) -> list[str]:
        """Where the time went, and what came out — side by side when the terminal is wide enough.

        Stacked below that width rather than squeezed: two bars and two labels in forty columns is
        four things competing for the same eight characters, and none of them stays readable.
        """
        stages = self._stage_lines()
        shapes = self._shape_lines()
        if not stages and not shapes:
            return []

        if inner >= _TWO_COLUMN_MIN and stages and shapes:
            left = inner // 2 - 1
            right = inner - left - 3
            rows = []
            for i in range(max(len(stages), len(shapes))):
                a = _pad(stages[i] if i < len(stages) else "", left)
                b = _pad(shapes[i] if i < len(shapes) else "", right)
                rows.append(_row(f"{a} │ {b}", inner))
            return [_split_rule("time by stage", "structures found", left, inner)] + rows

        out = []
        if stages:
            out.append(_rule("time by stage", inner))
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
        for name, spent in order:
            fraction = spent / measured
            filled = int(round(fraction * _MINI_BAR))
            drawn = "█" * filled + "·" * (_MINI_BAR - filled)
            lines.append(f"{name:<{label_width}}  {cost(spent):>8}  {drawn} "
                         f"{share(spent, measured):>4}")
        return lines

    def _shape_lines(self) -> list[str]:
        """Structures by shape, most numerous first, each against the most numerous."""
        if not self._shapes:
            return []
        order = sorted(self._shapes.items(), key=lambda kv: -kv[1])
        top = order[0][1] or 1
        label_width = max(len(name) for name, _ in order)
        lines = []
        for shape, n in order:
            filled = int(round(_MINI_BAR * n / top))
            drawn = "█" * filled + "·" * (_MINI_BAR - filled)
            lines.append(f"{shape:<{label_width}}  {n:>7,}  {drawn}")
        return lines

    def _lane_lines(self, inner: int, now: float) -> list[str]:
        """One row per worker, in a fixed order, with the recording it is on.

        The rows hold their position for the whole run, which is what makes a stalled lane visible:
        a worker that has been on the same file for twenty minutes is a row that stopped moving,
        and nothing else on the screen would have shown it.
        """
        if not self._lanes:
            return []
        rate = self._cost_rate()
        lines = []
        for lane, entry in enumerate(self._lanes):
            label = f"{lane + 1:>3}  "
            if entry is None:
                lines.append(f"{label}{'·' * _LANE_BAR}   idle")
                continue
            _index, rel_path, seconds, started_at = entry
            spent = now - started_at
            estimate = seconds * rate
            if estimate > 0:
                pct = min(BAR_CAP, 100.0 * spent / estimate)
                filled = int(round(_LANE_BAR * pct / 100.0))
                stat = f"{'█' * filled}{'░' * (_LANE_BAR - filled)} {pct:3.0f}% {spent:5.0f}s"
            else:
                # Nothing has finished yet, so there is no throughput to predict with. Elapsed
                # alone: an estimate drawn before the first result is invention.
                stat = f"{'░' * _LANE_BAR}   —  {spent:5.0f}s"
            lines.append(f"{label}{stat}  {fit_path(rel_path, inner - len(label + stat) - 2)}")
        return lines

    def _recent_lines(self, inner: int) -> list[str]:
        """The last few recordings to land, newest at the top.

        The outcome sits in a column just past the longest path rather than against the right edge:
        on a wide terminal a run of short names would put every count eighty characters from the
        name it belongs to.
        """
        if not self._recent:
            return []
        room = min(_RECENT_PATH, max(12, inner - _RECENT_TAIL - 4))
        return [f"{mark} {fit_path(rel_path, room):<{room}}  {clip(tail, _RECENT_TAIL)}"
                for mark, rel_path, tail in self._recent]


def _top(title: str, right: str, width: int) -> str:
    """The frame's top rule, with a title on the left and the clock on the right.

    The title is cut before the clock is: on a narrow terminal the elapsed time is the half worth
    keeping, and a rule one character too long would wrap and break the redraw.
    """
    tail = f" {right} ─╮"
    room = width - len(tail) - 4
    left = f"╭─ {clip(title, max(0, room))} "
    return f"{left}{'─' * max(0, width - len(left) - len(tail))}{tail}"


def _rule(title: str, inner: int) -> str:
    """A section rule across the frame, titled."""
    text = f"├─ {clip(title, max(0, inner - 2))} "
    return f"{text}{'─' * max(0, inner + 3 - len(text))}┤"


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
    return "".join(chars)


def _row(text: str, inner: int) -> str:
    """One line of frame content, padded and clipped to the frame's inside width."""
    return f"│ {_pad(text, inner)} │"


def _pad(text: str, width: int) -> str:
    """Fit text to exactly ``width`` columns, clipping with an ellipsis when it is too long."""
    if len(text) > width:
        return text[: max(0, width - 1)] + "…"
    return text + " " * (width - len(text))
