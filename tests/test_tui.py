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


# -- picking a finished recording, and opening its picture ------------------------------------


def test_the_completions_carry_the_recording_they_name(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))

    rows = display._recent_lines(90, 10)

    assert [path for _line, path in rows] == ["station-a/2026-07-02.wav",
                                              "station-a/2026-07-01.wav"]


def test_a_click_lands_on_the_recording_drawn_on_that_line(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    frame = display._render(time.time())

    # Every line the render mapped is a completion, and it is the one drawn there.
    assert display._clicks, "the completions are the one thing on this screen to click"
    for line, rel_path in display._clicks.items():
        assert rel_path in _plain(frame[line])

    line = min(display._clicks)
    display._handle(f"click:{line}:20")
    assert display._showing == display._clicks[line]


def test_a_click_on_anything_else_opens_nothing(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    display._render(time.time())

    display._handle("click:0:5")          # the top rule

    assert display._showing == ""


def test_the_arrow_keys_pick_a_recording_and_enter_opens_it(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    display._render(time.time())

    display._handle("down")
    assert display._picked == "station-a/2026-07-02.wav", "newest first, so down starts at the top"
    display._handle("down")
    assert display._picked == "station-a/2026-07-01.wav"
    display._handle("down")
    assert display._picked == "station-a/2026-07-01.wav", "the list ends rather than wrapping"

    display._handle("enter")
    assert display._showing == "station-a/2026-07-01.wav"


def test_the_pick_follows_its_recording_as_new_ones_land(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    display._handle("down")
    picked = display._picked

    display.on_result(3, 12, _Result("station-a/2026-07-03.wav", count=1))

    assert display._picked == picked, "a pick is a recording, not a position in a list"
    display._handle("down")
    assert display._picked != picked, "and it still moves from where that recording now is"


def test_enter_with_nothing_picked_opens_the_newest(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))

    display._handle("enter")

    assert display._showing == "station-a/2026-07-02.wav"


def test_q_closes_the_picture_before_it_closes_the_panel(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    display._finished = True
    display._handle("enter")

    display._handle("q")
    assert display._showing == "" and not display._quit.is_set()

    display._handle("q")
    assert display._quit.is_set(), "the second one dismisses a finished run"


def test_q_does_nothing_while_the_run_is_going(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))

    display._handle("q")

    assert not display._quit.is_set(), "the way to stop a scan is Ctrl-C, and only that"


def test_the_open_picture_moves_between_recordings(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))
    display._handle("enter")                      # opens the newest
    assert display._showing == "station-a/2026-07-02.wav"

    display._handle("right")
    assert display._showing == "station-a/2026-07-01.wav", "right is further down the list"
    display._handle("left")
    assert display._showing == "station-a/2026-07-02.wav"
    display._handle("left")
    assert display._showing == "station-a/2026-07-02.wav", "the list ends rather than wrapping"


def test_the_picture_is_read_once_per_recording_and_redrawn_when_it_has_to_be(size, monkeypatch):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures", out="/scan"))
    reads, draws = [], []

    class _Loaded:
        def band(self):
            return (100.0, 200.0)

    def fake_load(out_root, rel_path, **kwargs):
        reads.append((out_root, rel_path))
        return _Loaded()

    def fake_render(picture, width, height, *, position="", band=None, depth=None):
        draws.append((position, band))
        return [f"picture of {position}"]

    monkeypatch.setattr("siarapp.cli.tui.load_picture", fake_load)
    monkeypatch.setattr("siarapp.cli.tui.render_picture", fake_render)

    display._handle("enter")
    display._render(time.time())                   # the completions list, for the position
    assert display._frame() == ["picture of 1 of 2"]
    assert reads == [("/scan", "station-a/2026-07-02.wav")]

    display._frame()
    assert len(reads) == 1 and len(draws) == 1, "a still panel re-reads nothing"

    display._handle("z")
    display._frame()
    assert len(reads) == 1, "zooming is a redraw of what was already read"
    assert draws[-1][1] == (100.0, 200.0)

    display._handle("right")
    display._frame()
    assert reads[-1] == ("/scan", "station-a/2026-07-01.wav")


def test_the_panel_says_that_a_finished_recording_can_be_opened(size):
    size(100, 40)
    display = _busy(TuiDisplay("all_structures"))

    text = _text(display._render(time.time()))

    assert "just finished" in text
    assert "click" in text, "a screen that only works if you already knew it does is not one"
