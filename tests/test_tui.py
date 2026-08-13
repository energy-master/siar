# Vixen Intelligence c.2026
"""The ``--tui`` frame.

Two things are worth pinning and neither is cosmetic. Every line of a frame must be exactly as
wide as every other, because the frame is redrawn by moving the cursor up over itself and a line
that is one character too long wraps, shifts the count, and leaves torn copies of the panel up the
screen. And the frame must never be taller than the terminal, for the same reason: a frame that
scrolls has moved the anchor it is about to redraw from.

Rendered directly rather than through a terminal — the display's callbacks and its layout are
ordinary functions, and driving them needs no pty.
"""
from __future__ import annotations

import re
import shutil
import time

import pytest

from siarapp.cli.format import set_colour, visible_len
from siarapp.cli.tui import TuiDisplay


class _Result:
    """The parts of a ``FileResult`` the display reads."""

    def __init__(self, rel_path, status="scanned", count=0, duration_sec=10.0, elapsed_sec=1.0,
                 shapes=None, phases=None, error=""):
        self.rel_path = rel_path
        self.status = status
        self.count = count
        self.duration_sec = duration_sec
        self.elapsed_sec = elapsed_sec
        self.shapes = shapes or {}
        self.phases = phases or {}
        self.error = error
        self.lane = 0


@pytest.fixture(autouse=True, params=[False, True], ids=["plain", "colour"])
def colour(request):
    """Every layout test runs twice: colour must change how a frame looks, never its geometry.

    A row's padding is computed from visible columns, and an escape sequence occupies none — get
    that wrong and a coloured frame is a character wider than an uncoloured one, which tears the
    panel on the next repaint.
    """
    set_colour(request.param)
    yield request.param
    set_colour(None)


@pytest.fixture
def size(monkeypatch):
    """Fix the terminal size, since the frame is drawn to whatever the terminal says it is."""
    def _set(columns: int, lines: int) -> None:
        monkeypatch.setattr(shutil, "get_terminal_size",
                            lambda fallback=(100, 24): os_terminal_size(columns, lines))
    return _set


def os_terminal_size(columns, lines):
    return shutil.os.terminal_size((columns, lines))


def _plain(text):
    """One frame line with its colour taken back off, for asserting on what it says."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _text(frame):
    """A whole frame as plain text."""
    return "\n".join(_plain(line) for line in frame)


def _busy(display, *, workers=4, files=12):
    """A display part-way through a run, with something in every section."""
    display.on_corpus(files, 600.0, workers)
    for lane in range(workers):
        display.on_start(lane + 1, files, f"station-a/2026-07-0{lane + 1}.wav", 50.0, lane)
    display.on_result(1, files, _Result(
        "station-a/2026-07-01.wav", count=3, shapes={"tonal": 2, "click": 1},
        phases={"decode": 0.2, "fft": 1.0, "scan": 8.0, "write": 0.1, "thumbnail": 0.3},
    ))
    display.on_result(2, files, _Result("station-a/2026-07-02.wav", status="error",
                                        error="could not decode"))
    display.close()
    return display


@pytest.mark.parametrize("columns,lines", [(40, 24), (80, 24), (118, 40), (100, 14),
                                           (60, 30), (200, 50), (300, 60)])
def test_every_line_of_a_frame_is_the_same_width(size, columns, lines):
    size(columns, lines)
    display = _busy(TuiDisplay("all_structures_sensitive"), workers=8)
    frame = display._render(time.time())
    widths = {visible_len(line) for line in frame}
    assert len(widths) == 1, f"ragged frame at {columns}x{lines}: {sorted(widths)}"
    # The frame is the window: it takes the whole width, not a share of it.
    assert widths.pop() == columns


@pytest.mark.parametrize("lines", [10, 14, 24, 40, 60])
def test_a_frame_fills_the_terminal_exactly(size, lines):
    """Neither shorter nor taller: short leaves a dead half-screen, tall scrolls and tears."""
    size(100, lines)
    display = _busy(TuiDisplay("all_structures"), workers=8)
    assert len(display._render(time.time())) == lines


def test_the_completions_stretch_to_reach_the_bottom(size):
    """A tall window is filled with run, not with blank rows above the closing rule."""
    size(100, 60)
    display = TuiDisplay("all_structures")
    display.on_corpus(200, 6000.0, 2)
    for i in range(55):
        display.on_result(i + 1, 200, _Result(f"station-a/{i:04d}.wav", count=i,
                                              shapes={"tonal": 1},
                                              phases={"fft": 0.5, "scan": 4.0}))
    display.close()

    frame = display._render(time.time())
    assert len(frame) == 60
    blank = sum(1 for line in frame if _plain(line).strip("│ ") == "")
    assert blank == 0, f"{blank} empty rows in a 60-line frame with 55 completions to show"
    # Newest first: the run that just landed is at the top of the section, not buried under history.
    text = _text(frame)
    assert text.index("station-a/0054.wav") < text.index("station-a/0040.wav")


def test_the_bar_and_the_totals_survive_a_terminal_too_short_for_the_rest(size):
    size(100, 10)
    display = _busy(TuiDisplay("all_structures"), workers=16)
    frame = display._render(time.time())
    assert "0/12 files" not in _plain(frame[1]), "the bar should have moved"
    # The speed rides on the bar line itself; the line under it is what is left to do.
    assert "realtime" in _plain(frame[1])
    assert "left" in _plain(frame[2])
    # The last line closes the box however much had to be dropped above it.
    assert _plain(frame[-1]).startswith("╰")


def test_a_wide_frame_puts_stages_and_structures_side_by_side(size):
    size(118, 40)
    display = _busy(TuiDisplay("all_structures"))
    frame = _text(display._render(time.time()))
    assert "time by stage" in frame and "structures found" in frame
    assert "┬" in frame, "the two columns should be divided"


def test_a_narrow_frame_stacks_them_instead(size):
    size(72, 40)
    display = _busy(TuiDisplay("all_structures"))
    frame = _text(display._render(time.time()))
    assert "time by stage" in frame and "structures found" in frame
    assert "┬" not in frame


def test_stages_are_listed_worst_first(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    stages = [_plain(line).split()[0] for line in display._stage_lines()]
    assert stages[0] == "scan"
    assert stages == ["scan", "fft", "thumbnail", "decode", "write"]


def test_what_was_found_is_counted_as_results_land(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    assert display._structures == 3
    assert display._shapes == {"tonal": 2, "click": 1}
    assert display._by_status == {"scanned": 1, "error": 1}


def test_a_finished_run_is_held_on_screen_with_its_metrics(size):
    """The panel earns its keep at the end: the closing table, in the frame, until dismissed.

    `hold` blocks on a keypress, and returns at once here because pytest's stdin is not a terminal —
    which is the same path a cron job takes, and it must not wait forever for a key nobody will
    press.
    """
    size(110, 30)
    display = _busy(TuiDisplay("all_structures"))
    display._hidden = True  # `_busy` closed it; `hold` only draws when it owns a screen
    display.hold([["recordings", "12", ""], ["  scan", "8.0 s", "89%"], ["realtime", "6.00x", ""]])

    frame = display._render(time.time())
    text = _text(frame)
    assert "performance" in text, "the closing metrics replace the live stage block"
    assert re.search(r"realtime\s+6\.00x", text), "the metric rows are laid out as a table"
    assert "Ctrl-Q" in text, "a held panel has to say how to leave it"
    assert "finished in" in _plain(frame[0])
    # The lanes are all idle by now and their rows would say nothing, so that section is gone —
    # the title still names the worker count, which is a fact about the run.
    assert "─ workers ─" not in text
    assert len(frame) == 30
    display._hidden = False
    display.close()


def test_a_held_panel_stops_its_clock(size):
    size(100, 26)
    display = _busy(TuiDisplay("all_structures"))
    display._hidden = True
    display.hold([["recordings", "12", ""]])
    display._hidden = False
    display.close()

    first = _plain(display._render(time.time() + 3600)[0])
    second = _plain(display._render(time.time() + 7200)[0])
    # An hour later the run still took as long as it took.
    assert first == second


def test_a_failure_is_reprinted_on_the_way_out(size, capsys):
    size(100, 40)
    display = TuiDisplay("all_structures")
    display.on_corpus(2, 20.0, 1)
    display.on_result(1, 2, _Result("bad.wav", status="error", error="could not decode"))
    display.note("mixed sample rates in this folder")
    display.close()

    err = capsys.readouterr().err
    # The frame is wiped on the way out, so anything only ever drawn inside it would be lost.
    assert "error: bad.wav: could not decode" in err
    assert "warning: mixed sample rates in this folder" in err


def test_closing_a_display_that_never_started_is_harmless(capsys):
    TuiDisplay("all_structures").close()
    assert capsys.readouterr().err == ""
