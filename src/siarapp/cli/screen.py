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
    "capture_keys",
    "click_at",
    "decode_csi",
    "draw",
    "enter",
    "is_tty",
    "leave",
    "read_key",
    "release_keys",
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

#: Mouse reporting: button presses and releases only, in SGR form. Motion is deliberately not
#: asked for — nothing here reacts to a mouse moving, and a terminal that reports every pixel of
#: it over an ssh link to a vessel is spending the link on nothing.
#:
#: It is off by default and turned on only where a screen has something to click. While it is on,
#: the terminal's own text selection needs Shift held, which is a real cost to anyone reading over
#: ssh — so screens that offer no click do not pay it.
_MOUSE_ON = "\033[?1000h\033[?1006h"
_MOUSE_OFF = "\033[?1006l\033[?1000l"

#: Longest CSI sequence read before giving up. An SGR mouse report on a very large terminal is
#: about fifteen characters; anything past this is a terminal saying something this does not speak.
_CSI_MAX = 32

#: Mouse button bits, from the SGR report's first parameter.
_WHEEL = 64
_MOTION = 32
_BUTTON = 3


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


def enter(*, mouse: bool = False) -> None:
    """Take the alternate screen buffer, blank it, and hide the cursor.

    Separate from :func:`screen` because the library hands the screen to the run panel and takes
    it back: :mod:`siarapp.cli.tui` enters and leaves the buffer on its own, so what the library
    needs afterwards is exactly this, and not a second context manager wrapped around the first.

    Args:
        mouse: Also ask the terminal to report clicks (see :data:`_MOUSE_ON`). Only for a screen
            that has something to click: while it is on, selecting text needs Shift held.
    """
    sys.stdout.write(_ALT_ON + HIDE_CURSOR + _CLEAR + _HOME + (_MOUSE_ON if mouse else ""))
    sys.stdout.flush()


def leave() -> None:
    """Give the buffer back and show the cursor. The inverse of :func:`enter`.

    Mouse reporting is turned off whether or not it was turned on: a terminal left reporting
    clicks into a shell prompt is a terminal the reader has to close, and the sequence costs
    nothing when it was never enabled.
    """
    sys.stdout.write(_MOUSE_OFF + SHOW_CURSOR + _ALT_OFF)
    sys.stdout.flush()


def capture_keys():
    """Put the terminal in cbreak mode so keys arrive one at a time, and return what it was.

    ``IXON`` is cleared with it. Ctrl-S and Ctrl-Q are flow control to the line discipline, which
    would swallow them before any of this saw them, and a reader who presses Ctrl-S out of habit
    would find a frozen screen and nothing to press to fix it. Interrupts are left enabled, so
    Ctrl-C stops the program the way it does everywhere else.

    Returns:
        The saved terminal attributes, to be handed back to :func:`release_keys`, or ``None``
        when there is no terminal to configure — off a tty, and on Windows.
    """
    if termios is None or tty is None:
        return None
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        mode = termios.tcgetattr(fd)
        mode[0] &= ~termios.IXON
        termios.tcsetattr(fd, termios.TCSANOW, mode)
        return saved
    except (AttributeError, ValueError, OSError, termios.error):
        return None


def release_keys(saved) -> None:
    """Put the terminal back the way :func:`capture_keys` found it. ``None`` is a no-op."""
    if saved is None or termios is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
    except (AttributeError, ValueError, OSError, termios.error):  # pragma: no cover
        pass


@contextmanager
def screen(*, mouse: bool = False) -> Iterator[None]:
    """Hold the alternate screen in cbreak mode for the body of the ``with``.

    The restore runs under ``finally``, in the reverse order of the setup, so an exception inside
    the interface still leaves a usable terminal — the traceback then prints onto the normal
    buffer, where it can be read and scrolled.

    Args:
        mouse: Ask for click reporting; see :func:`enter`.
    """
    saved = capture_keys()
    enter(mouse=mouse)
    try:
        yield
    finally:
        release_keys(saved)
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


def decode_csi(sequence: str) -> str:
    """What a CSI sequence means, as a key name. Pure — everything after ``ESC [``.

    Args:
        sequence: The parameter bytes and the final byte, e.g. ``"A"`` for Up or
            ``"<0;42;7M"`` for a left click at column 42, row 7.

    Returns:
        A key name, a click token, or ``""`` for a sequence nothing here reacts to — a mouse
        release, a wheel tilt, a function key. Swallowed rather than handed on, because a
        sequence delivered as stray letters would type into whatever field is open.

    Note:
        A click comes back as ``"click:<row>:<column>"``, **zero-based**, because what a caller
        does with it is index the line it drew there. :func:`click_at` parses it, so no caller
        has to know that.
    """
    if not sequence:
        return ""
    final, params = sequence[-1], sequence[:-1]
    if params.startswith("<") and final in ("M", "m"):
        try:
            button, column, row = (int(part) for part in params[1:].split(";"))
        except ValueError:
            return ""
        if button & _WHEEL:
            # The wheel reports as a button press with the wheel bit set; up is 0, down is 1.
            # Scrolling a list is what every reader will try, and it costs one line to honour.
            return "down" if button & _BUTTON else "up"
        if button & _MOTION or final != "M" or (button & _BUTTON) != 0:
            # Motion, a release, or a button that is not the left one. A release always follows a
            # press, so acting on both would open everything twice.
            return ""
        return f"click:{max(0, row - 1)}:{max(0, column - 1)}"
    if final == "~":
        # Home, Insert, Delete, a function key: parameterised, and none of them mean anything
        # here. An arrow with a modifier (``ESC [ 1 ; 2 A``) still arrives as its arrow.
        return ""
    return _CSI.get(final, "")


def click_at(key: str) -> tuple[int, int] | None:
    """The ``(row, column)`` of a click token, or ``None`` if that is not what this key is.

    Zero-based, so the row indexes the list of lines the caller last drew.
    """
    if not key.startswith("click:"):
        return None
    try:
        _tag, row, column = key.split(":")
        return int(row), int(column)
    except ValueError:  # pragma: no cover - only reachable from a hand-made token
        return None


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
        # ``ESC [ 1 ~``, or a mouse report, cannot leave its tail behind to be read as typing.
        sequence = ""
        for _ in range(_CSI_MAX):
            piece = _read_char(fd)
            if not piece:
                break
            sequence += piece
            if "@" <= piece <= "~":
                break
        return decode_csi(sequence)
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
