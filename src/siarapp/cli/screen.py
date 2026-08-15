# Vixen Intelligence c.2026
"""Taking a terminal over, and reading keys off it.

:mod:`siarapp.cli.tui` draws a run and reads one keystroke at the end of it. The library
(:mod:`siarapp.cli.library_tui`) is *driven* by keys — it has lists to move through, fields to
type into and a run to start — and that needs three things the run panel never did: the terminal
in cbreak mode for the whole session, an arrow key that arrives as one event rather than as three
characters, and a way to hand the screen to the run panel and take it back afterwards.

Those are here, and nothing else is. What a line looks like is
:mod:`siarapp.cli.format`'s job — colours, widths and clipping are already defined there and a
second copy would eventually disagree with the first.

Stdlib only, like the rest of the package: ANSI escapes, ``termios`` and ``select``. No
``curses``. siar-build's ``siarbuild/term.py`` makes the same choice for the same reasons and is
deliberately the same shape, so someone who has read one can read the other.

Off a terminal none of this works and none of it is offered: :func:`is_tty` is the question every
caller asks first, and the answer is a sentence to the user rather than a frame written into a
log.
"""
from __future__ import annotations

import os
import select
import sys
from contextlib import contextmanager
from typing import Iterator

from siarapp.cli.format import HIDE_CURSOR, SHOW_CURSOR, fit

try:  # POSIX only. Windows gets the no-TTY path, which is a sentence rather than a crash.
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None
    tty = None

__all__ = [
    "draw",
    "enter",
    "is_tty",
    "leave",
    "read_key",
    "screen",
    "size",
]

#: The alternate screen buffer — what ``vim`` and ``less`` use, and what
#: :mod:`siarapp.cli.tui` takes for a run. Entering gives the interface a blank screen of its
#: own; leaving puts the shell's scrollback back exactly as it was.
_ALT_ON = "\033[?1049h"
_ALT_OFF = "\033[?1049l"
_HOME = "\033[H"
_CLEAR = "\033[2J"
_CLEAR_TO_EOL = "\033[K"

#: Smallest terminal a frame is drawn into. Below this it is drawn at this size anyway and the
#: terminal scrolls it — a window that narrow cannot show a library, and every render function
#: having its own opinion about the floor would be worse than one that is wrong here.
_MIN_COLUMNS = 60
_MIN_ROWS = 12

#: How long an ``\033`` waits for the rest of an escape sequence before it is taken to be the
#: Escape key. An arrow key arrives as three bytes together and Escape as one, with nothing but
#: timing to tell them apart: long enough that a terminal's arrow is never split, short enough
#: that Escape does not feel stuck.
_ESCAPE_GRACE_SEC = 0.05

#: What the final byte of a CSI sequence means. Anything else is swallowed rather than handed on,
#: because a function key arriving as a stray letter would type into whatever field is open.
_CSI = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}


def is_tty() -> bool:
    """Whether there is a terminal to draw on and read from.

    Both halves are asked. A session whose stdout is a pipe has nowhere to draw, and one whose
    stdin is not a terminal can never be told what to do — either alone makes a full-screen
    interface the wrong answer.

    Returns:
        True when both streams are terminals.
    """
    try:
        return bool(sys.stdout.isatty() and sys.stdin.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed streams
        return False


def size(default: tuple[int, int] = (100, 30)) -> tuple[int, int]:
    """``(columns, rows)`` for the terminal, clamped to something drawable.

    Args:
        default: Used when the size cannot be measured — a pty-less test, a stream that is not a
            terminal.

    Returns:
        The size, never smaller than :data:`_MIN_COLUMNS` by :data:`_MIN_ROWS`.
    """
    try:
        columns, rows = os.get_terminal_size()
    except OSError:
        columns, rows = default
    return max(_MIN_COLUMNS, int(columns)), max(_MIN_ROWS, int(rows))


def enter() -> None:
    """Take the alternate screen buffer, blank it, and hide the cursor.

    Separate from :func:`screen` because the library hands the screen to the run panel and takes
    it back: :mod:`siarapp.cli.tui` enters and leaves the buffer on its own, so what the library
    needs afterwards is exactly this, and not a second context manager wrapped around the first.
    """
    sys.stdout.write(_ALT_ON + HIDE_CURSOR + _CLEAR + _HOME)
    sys.stdout.flush()


def leave() -> None:
    """Give the buffer back and show the cursor. The inverse of :func:`enter`."""
    sys.stdout.write(SHOW_CURSOR + _ALT_OFF)
    sys.stdout.flush()


@contextmanager
def screen() -> Iterator[None]:
    """Hold the alternate screen in cbreak mode for the body of the ``with``.

    The restore runs under ``finally``, in the reverse order of the setup, so an exception inside
    the interface still leaves a usable terminal — the traceback then prints onto the normal
    buffer, where it can be read and scrolled.

    ``IXON`` is cleared for the duration. Ctrl-S and Ctrl-Q are flow control to the line
    discipline, which would swallow them before any of this saw them, and a user who presses
    Ctrl-S out of habit would find a frozen screen and nothing to press to fix it. Interrupts are
    left enabled, so Ctrl-C stops the program the way it does everywhere else.
    """
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd) if termios is not None else None
    enter()
    try:
        if tty is not None and termios is not None:
            tty.setcbreak(fd)
            mode = termios.tcgetattr(fd)
            mode[0] &= ~termios.IXON
            termios.tcsetattr(fd, termios.TCSANOW, mode)
        yield
    finally:
        if saved is not None and termios is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except (OSError, termios.error):  # pragma: no cover - terminal already gone
                pass
        leave()


def draw(lines: list[str], width: int, height: int) -> None:
    """Paint a frame from the top left, one line per row.

    Every row is padded to the width and cleared to the end of the line as it is written, rather
    than the screen being cleared first: clearing and then drawing flickers, because the terminal
    shows the blank frame in between.

    Nothing is written after the last row — a newline on the bottom line scrolls the screen, and a
    scrolled screen is one the next repaint draws in the wrong place.

    Args:
        lines: The frame. Shorter than ``height`` is fine; the rest is blanked.
        width: Columns to pad and clip each line to.
        height: Rows to write.
    """
    out = [_HOME]
    for row in range(height):
        text = lines[row] if row < len(lines) else ""
        out.append(fit(text, width) + _CLEAR_TO_EOL)
        if row < height - 1:
            out.append("\r\n")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _read_char(fd: int) -> str:
    """One character off the descriptor, continuation bytes included.

    The *descriptor*, never ``sys.stdin``. That is the whole reason this helper exists: a
    text-mode stdin reads a chunk into its own buffer and hands back one character, and every
    later :func:`select.select` then reports the descriptor as empty while the rest of the burst
    sits in that buffer. An arrow key — three bytes arriving together — would come back as
    Escape, then ``[``, then ``B``: three keystrokes the interface never received.
    """
    try:
        first = os.read(fd, 1)
    except OSError:  # pragma: no cover - descriptor closed under us
        return ""
    if not first:
        return ""
    lead = first[0]
    if lead < 0x80:
        return chr(lead)
    extra = (1 if lead >> 5 == 0b110 else
             2 if lead >> 4 == 0b1110 else
             3 if lead >> 3 == 0b11110 else 0)
    return (first + (os.read(fd, extra) if extra else b"")).decode("utf-8", "replace")


def read_key(timeout: float = 0.0) -> str:
    """One keystroke, or ``""`` if none arrived within ``timeout`` seconds.

    Args:
        timeout: Seconds to wait. Zero polls and returns immediately, which is what a loop that
            also has to redraw wants.

    Returns:
        A name for the keys an interface dispatches on — ``up``, ``down``, ``left``, ``right``,
        ``enter``, ``escape``, ``backspace``, ``tab``, ``space``, ``eof`` — or the character
        itself for anything else, so a text field can simply append what it is given.

    Raises:
        KeyboardInterrupt: On Ctrl-C, which cbreak mode delivers as a byte rather than a signal
            on some terminals; raising it here means one handler covers both.
    """
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], timeout)[0]:
        return ""
    char = _read_char(fd)
    if char == "":
        return "eof"
    if char == "\033":
        if not select.select([fd], [], [], _ESCAPE_GRACE_SEC)[0]:
            return "escape"
        if _read_char(fd) != "[":
            return "escape"
        # Read to the sequence's final byte (0x40-0x7E) so a parameterised key such as Home as
        # ``ESC [ 1 ~`` cannot leave its tail behind to be read as typing.
        final = ""
        for _ in range(8):
            final = _read_char(fd)
            if not final or "@" <= final <= "~":
                break
        return _CSI.get(final, "")
    if char in ("\r", "\n"):
        return "enter"
    if char in ("\x7f", "\b"):
        return "backspace"
    if char == "\t":
        return "tab"
    if char == " ":
        return "space"
    if char == "\x03":
        raise KeyboardInterrupt
    if char == "\x04":
        return "eof"
    return char
