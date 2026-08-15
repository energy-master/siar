# Vixen Intelligence c.2026
"""Reading a terminal: escape sequences in, key names out.

The parsing is the part worth pinning, and it is pure — a CSI sequence is a string, and what it
means is a function of that string. So none of this needs a pty.

The rule it enforces is that anything unrecognised is **swallowed**. A terminal says a great deal
that this program has no opinion about — function keys, mouse releases, wheel tilts, bracketed
paste — and a sequence handed on as stray letters would type into whatever field was open, or
open a picture nobody clicked on.
"""
from __future__ import annotations

import pytest

from siarapp.cli.screen import click_at, decode_csi, draw, size
from siarapp.cli.format import set_colour, visible_len


@pytest.mark.parametrize("sequence,expected", [
    ("A", "up"),
    ("B", "down"),
    ("C", "right"),
    ("D", "left"),
    ("H", "home"),
    ("F", "end"),
    ("1;2A", "up"),          # an arrow with a modifier is still that arrow
])
def test_the_keys_an_interface_moves_with(sequence, expected):
    assert decode_csi(sequence) == expected


@pytest.mark.parametrize("sequence", ["1~", "3~", "15~", "", "Z", "?1049h", "garbage"])
def test_everything_else_is_swallowed(sequence):
    assert decode_csi(sequence) == ""


def test_a_left_click_comes_back_as_the_cell_it_landed_on():
    # SGR mouse: button 0, column 42, row 7 — one-based, as terminals report them.
    assert decode_csi("<0;42;7M") == "click:6:41"
    assert click_at("click:6:41") == (6, 41), "zero-based, so it indexes the line that was drawn"


def test_a_release_is_not_a_second_click():
    assert decode_csi("<0;42;7m") == "", "every press is followed by one, and would open twice"


@pytest.mark.parametrize("sequence", ["<1;5;5M", "<2;5;5M", "<32;5;5M", "<35;5;5M"])
def test_other_buttons_and_mouse_motion_do_nothing(sequence):
    assert decode_csi(sequence) == ""


def test_the_wheel_scrolls_the_list_it_is_over():
    assert decode_csi("<64;10;10M") == "up"
    assert decode_csi("<65;10;10M") == "down"


def test_a_mouse_report_that_will_not_parse_is_ignored():
    assert decode_csi("<0;not;7M") == ""
    assert decode_csi("<0;7M") == ""


def test_click_at_says_no_to_anything_that_is_not_a_click():
    assert click_at("up") is None
    assert click_at("q") is None


def test_the_size_is_clamped_to_something_drawable():
    columns, rows = size(default=(10, 2))

    assert columns >= 60 and rows >= 12, "a frame is not laid out twice for a two-row window"


def test_draw_pads_every_row_and_leaves_no_trailing_newline(capsys):
    set_colour(False)
    try:
        draw(["one", "two"], width=10, height=4)
    finally:
        set_colour(None)

    written = capsys.readouterr().out
    assert written.startswith("\033[H"), "a frame is painted from the top left, not appended"
    assert not written.endswith("\n"), "a newline on the bottom line scrolls the frame away"
    body = [line for line in written.split("\r\n")]
    assert len(body) == 4, "every row of the window is written, blank ones included"
    assert visible_len(body[0].replace("\033[H", "").replace("\033[K", "")) == 10
