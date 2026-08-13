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

__all__ = [
    "BAR_CAP",
    "BAR_WIDTH",
    "HIDE_CURSOR",
    "SHOW_CURSOR",
    "bar",
    "clip",
    "clock",
    "cost",
    "duration",
    "factor_text",
    "fit_path",
    "share",
]

#: Width of the download and progress bars, in characters.
BAR_WIDTH = 24

#: An estimated bar never draws past this. A bar that reaches 100% and then keeps going has told
#: the user the wrong thing twice; one that waits at 99% has only ever said "nearly", which is true.
BAR_CAP = 99.0

#: Terminal control: hide the cursor while a panel is redrawing under it.
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width progress bar, brackets included."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


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


def share(seconds: float, total: float) -> str:
    """One part's percentage of a total, or ``""`` when there is no total to divide by."""
    if total <= 0:
        return ""
    return f"{100.0 * float(seconds) / total:.0f}%"
