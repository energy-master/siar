# Vixen Intelligence c.2026
"""How numbers, times and paths are written on the terminal.

Every live display and every closing summary states the same handful of quantities — a duration,
a running clock, a realtime factor, a share of a total, a bar, a path too long for the width it
has. They are here rather than beside their first caller because there are now three displays
(:class:`~siarapp.cli.commands.ScanReporter`, :class:`~siarapp.cli.commands.WorkerPanel`,
:mod:`siarapp.cli.tui`) plus the closing table, and a run whose panel says ``38.1x`` and whose
summary says ``38.14x`` for the same number invites the reader to work out which is lying.

Formatting only: nothing here writes to a terminal, holds state or knows what a run is, so all of
it is testable by calling it.
"""
from __future__ import annotations

import os
import re
import sys
import time

from siarapp.io.performance import PHASES

__all__ = [
    "BAR_CAP",
    "BAR_WIDTH",
    "BOLD",
    "CYAN",
    "DIM",
    "GREEN",
    "HIDE_CURSOR",
    "RED",
    "RESET",
    "SHOW_CURSOR",
    "SPINNER_FRAMES",
    "YELLOW",
    "bar",
    "clip",
    "clock",
    "colour_enabled",
    "cost",
    "duration",
    "factor_text",
    "fit",
    "fit_path",
    "human_bytes",
    "named_recording",
    "set_colour",
    "share",
    "size_and_length",
    "spin",
    "stamp",
    "stage_tag",
    "paint",
    "visible_len",
]

#: Width of the download and progress bars, in characters.
BAR_WIDTH = 24

#: An estimated bar never draws past this. A bar that reaches 100% and then keeps going has told
#: the user the wrong thing twice; one that waits at 99% has only ever said "nearly", which is true.
BAR_CAP = 99.0

#: Frames of the spinner :func:`spin` cycles through. Braille dots because they are one column
#: wide in every terminal that draws the box characters the panels already use, and because a
#: rotating shape reads as "still working" where a blinking one reads as "stuck".
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Terminal control: hide the cursor while a panel is redrawing under it.
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


# -- colour ----------------------------------------------------------------------------------
#
# Sparingly, and never as the only thing carrying a meaning: a red row also says ERROR, a green bar
# is also longer. Someone reading this over ssh in a monochrome terminal, or piping it into a log,
# must lose nothing but the colour.

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

#: Cached decision from :func:`colour_enabled`. ``None`` until something first asks.
_COLOUR: bool | None = None

#: Columns a stage tag occupies, brackets included. Measured from the phase names themselves, so
#: a sixth stage would widen the column rather than tear the row it sits in.
_STAGE_WIDTH = max(len(name) for name in PHASES) + 2

#: Columns a recording's name must keep before its size and length may share the row with it.
_NAME_FLOOR = 16


def colour_enabled() -> bool:
    """Whether to emit colour: a terminal, and nobody having asked us not to.

    Honours ``NO_COLOR`` (any value, per the informal standard) and ``TERM=dumb``, because the one
    thing worse than no colour is escape codes in a log file.
    """
    global _COLOUR
    if _COLOUR is None:
        _COLOUR = bool(
            sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM", "") != "dumb"
        )
    return _COLOUR


def set_colour(enabled: bool | None) -> None:
    """Force colour on or off, or ``None`` to decide again from the environment."""
    global _COLOUR
    _COLOUR = None if enabled is None else bool(enabled)


def paint(text: str, *codes: str) -> str:
    """Wrap text in SGR codes, or return it untouched when colour is off.

    Args:
        text: What to colour. Padding belongs *inside* this argument — ``paint(f"{x:<8}", DIM)``,
            not ``f"{paint(x, DIM):<8}"`` — because a format spec counts escape characters as
            width and would pad the column short.
        codes: Style constants from this module.

    Returns:
        The text, decorated and reset, or exactly the text given.
    """
    if not codes or not colour_enabled():
        return text
    return "".join(codes) + text + RESET


_SGR = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    """How many columns text occupies once its colour codes are discounted."""
    return len(_SGR.sub("", text))


def fit(text: str, width: int) -> str:
    """Cut or pad text to exactly ``width`` visible columns, colour codes and all.

    Every row of a redrawn panel has to be the same width or the frame tears, and a naive ``len``
    counts escape sequences as if they were letters. This walks the string instead: escapes are
    copied through and cost nothing, printable characters cost one each, and anything cut is
    followed by a reset so a colour never leaks into the rest of the line.
    """
    if width <= 0:
        return ""
    coloured = bool(_SGR.search(text))
    tail = RESET if coloured else ""
    seen = visible_len(text)
    if seen <= width:
        return text + tail + " " * (width - seen)

    out: list[str] = []
    kept = 0
    i = 0
    while i < len(text) and kept < width - 1:
        match = _SGR.match(text, i)
        if match:
            out.append(match.group())
            i = match.end()
            continue
        out.append(text[i])
        kept += 1
        i += 1
    return "".join(out) + "…" + tail


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width progress bar, brackets included."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def spin(tick: int) -> str:
    """One frame of the waiting spinner.

    For the one thing a bar cannot describe: work whose end is not known yet, such as walking a
    folder nobody has counted. A bar there would have to invent a denominator, and a number that
    moves for reasons the reader cannot see is worse than a dot that only claims to be alive.

    Args:
        tick: Any counter — the frame is ``tick`` modulo the number of frames, so a caller can
            simply increment.

    Returns:
        A single character.
    """
    return SPINNER_FRAMES[int(tick) % len(SPINNER_FRAMES)]


def clip(text: str, width: int) -> str:
    """Cut a line to the terminal, so a redraw never wraps and doubles the panel's height."""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def fit_path(rel_path: str, limit: int) -> str:
    """A recording's path within ``limit`` columns, keeping the end.

    The end, not the start: survey folders are named by station and date, so the first thirty
    characters of every path in a run are identical and the last twenty are the file.
    """
    if limit <= 1 or len(rel_path) <= limit:
        return rel_path
    return "…" + rel_path[-(limit - 1):]


def clock(seconds: float) -> str:
    """A running time as ``h:mm:ss``, or ``m:ss`` under an hour."""
    whole = int(max(0.0, seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def duration(seconds: float) -> str:
    """A duration a human reads at a glance.

    Hours for a survey, minutes for a session, seconds for a handful of clips. Always printing
    hours makes a twelve-second trial run report "0.00 h of audio", which reads like a failure.
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def cost(seconds: float) -> str:
    """One stage's time. Milliseconds under a second, then whatever :func:`duration` would say.

    A trial run on a handful of clips spends milliseconds writing sidecars, and a breakdown that
    reports every stage of it as "0.0 s" has stopped being a breakdown.
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return duration(seconds)


def factor_text(factor: float) -> str:
    """A realtime factor as a multiple, at the precision the number deserves.

    Two decimals below 10x and one above, because the interesting digit moves: 0.35x and 0.40x
    are the difference between an overnight run and a two-day one, while 38.1x and 38.14x say the
    same thing about the same machine.

    Args:
        factor: Audio seconds per wall second.

    Returns:
        The multiple, or ``"—"`` when there is nothing to state — a run that scanned no audio has
        no speed, and printing ``0.00x`` for it claims one.
    """
    if factor <= 0:
        return "—"
    return f"{factor:.1f}x" if factor >= 10 else f"{factor:.2f}x"


def human_bytes(n: int | float | None) -> str:
    """A size at a human scale. KiB not kB: this is a file on a disk, not a link speed."""
    if not n:
        return "0 B"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def size_and_length(size_bytes: int, seconds: float) -> str:
    """A recording's two dimensions together, as ``36.6 MiB · 6.7 min``.

    The pair travels together on every display that names a recording, because either one alone
    invites the wrong conclusion: a large file may be a short one at 384 kHz, and a long one may
    be eight hours of 8 kHz that costs nothing.

    Args:
        size_bytes: What it occupies on disk. ``0`` leaves it out.
        seconds: Its length, from the header. ``0`` leaves it out.

    Returns:
        The pair, one of them, or ``""`` when neither is known.
    """
    parts = []
    if size_bytes:
        parts.append(human_bytes(size_bytes))
    if seconds > 0:
        parts.append(duration(seconds))
    return " · ".join(parts)


def named_recording(rel_path: str, size_bytes: int, seconds: float, room: int) -> str:
    """A recording's name with its size and length beside it, inside ``room`` columns.

    The name wins when the two will not both fit. A row that has squeezed the path down to
    ``…0410.wav`` to make space for ``3.4 GiB · 2.10 h`` has traded the only thing identifying
    the recording for two facts about it, so below :data:`_NAME_FLOOR` columns the pair is
    dropped and the path takes the width.

    Args:
        rel_path: Path relative to the scanned folder.
        size_bytes: What it occupies on disk, or ``0``.
        seconds: Its length from the header, or ``0``.
        room: Columns available for all of it.

    Returns:
        The row's tail, at most ``room`` columns wide.
    """
    tail = size_and_length(size_bytes, seconds)
    if tail and room - len(tail) - 2 >= _NAME_FLOOR:
        return f"{fit_path(rel_path, room - len(tail) - 2)}  {tail}"
    return fit_path(rel_path, room)


def stage_tag(stage: str) -> str:
    """Which stage of a recording is running, as a fixed-width ``[fft]``.

    Fixed width, padded to the longest of :data:`~siarapp.io.performance.PHASES`, because this
    sits mid-line on a display that redraws four times a second: a tag that changed width would
    shuffle the recording's name sideways every time the stage moved on.

    Args:
        stage: A phase name, or ``""`` before a recording has reached one.

    Returns:
        The tag, or the same width in spaces when there is no stage to name.
    """
    return f"[{stage}]".ljust(_STAGE_WIDTH) if stage else " " * _STAGE_WIDTH


def stamp(epoch: int | float) -> str:
    """A moment in local time, to the minute — how every listing writes "when".

    Local time rather than UTC: this is a stamp on something that happened on *this* machine, and
    the reader is sitting at it. To the minute because the second a bundle finished unpacking has
    never been the answer to anything.

    Args:
        epoch: Seconds since the epoch, or ``0`` when it is not known.

    Returns:
        ``"YYYY-MM-DD HH:MM"``, or ``"—"`` for an unknown moment — which is a statement, and
        ``"1970-01-01 01:00"`` is a lie.
    """
    if not epoch:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(epoch)))


def share(seconds: float, total: float) -> str:
    """One part's percentage of a total, or ``""`` when there is no total to divide by."""
    if total <= 0:
        return ""
    return f"{100.0 * float(seconds) / total:.0f}%"
