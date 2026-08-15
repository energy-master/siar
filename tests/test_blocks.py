# Vixen Intelligence c.2026
"""Drawing a picture into character cells.

The one rule that matters is arithmetic: a cell holds two pixels, so a picture of ``h`` rows is
``ceil(h / 2)`` lines and every line is exactly as many cells as the picture is wide. A frame that
got either wrong would tear the panel it is drawn in — and a colour code counts for no columns at
all, which is exactly the mistake ``len()`` makes.

The three depths are tested for what distinguishes them rather than for their exact escapes: that
truecolour states a pixel's own RGB, that the palette states an index, and that neither says
anything at all when the reader asked for no colour.
"""
from __future__ import annotations

import numpy as np
import pytest

from siarapp.cli.blocks import (
    ASCII_RAMP,
    HALF_BLOCK,
    PALETTE,
    PLAIN,
    TRUECOLOUR,
    colour_depth,
    render_blocks,
    resample,
)
from siarapp.cli.format import set_colour, visible_len


@pytest.fixture(autouse=True)
def colour():
    """Colour decisions are cached in the formatter; leave the environment's answer behind."""
    yield
    set_colour(None)


def _image(rows: int, cols: int, value: int = 128) -> np.ndarray:
    return np.full((rows, cols, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("depth", [TRUECOLOUR, PALETTE, PLAIN])
@pytest.mark.parametrize("rows,cols", [(2, 10), (8, 40), (1, 4), (7, 13)])
def test_a_picture_is_half_as_many_lines_as_it_is_pixels_tall(depth, rows, cols):
    lines = render_blocks(_image(rows, cols), depth=depth)

    assert len(lines) == (rows + 1) // 2
    assert all(visible_len(line) == cols for line in lines)


def test_an_odd_row_count_is_padded_rather_than_dropped():
    lines = render_blocks(_image(5, 6), depth=PLAIN)

    assert len(lines) == 3, "the last row of pixels has to survive into a cell of its own"


def test_an_empty_picture_draws_nothing():
    assert render_blocks(np.zeros((0, 10, 3), dtype=np.uint8), depth=PLAIN) == []
    assert render_blocks(np.zeros((4, 0, 3), dtype=np.uint8), depth=PLAIN) == []


def test_truecolour_states_the_pixels_own_colour():
    image = np.zeros((2, 1, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    image[1, 0] = (40, 50, 60)

    line = render_blocks(image, depth=TRUECOLOUR)[0]

    assert "\033[38;2;10;20;30m" in line, "the upper pixel is the foreground"
    assert "\033[48;2;40;50;60m" in line, "the lower pixel is the background"
    assert HALF_BLOCK in line


def test_a_run_of_one_colour_is_stated_once():
    wide = render_blocks(_image(2, 40), depth=TRUECOLOUR)[0]

    assert wide.count("\033[38;2;") == 1, "an unchanged colour is not restated per cell"
    assert wide.count(HALF_BLOCK) == 40


def test_the_palette_depth_uses_indices_not_rgb():
    line = render_blocks(_image(2, 4, value=200), depth=PALETTE)[0]

    assert "\033[38;5;" in line and "\033[48;5;" in line
    assert "38;2;" not in line


def test_without_colour_the_picture_becomes_an_intensity_ramp():
    dark = render_blocks(_image(2, 3, value=0), depth=PLAIN)[0]
    light = render_blocks(_image(2, 3, value=255), depth=PLAIN)[0]

    assert dark == ASCII_RAMP[0] * 3
    assert light == ASCII_RAMP[-1] * 3
    assert "\033[" not in dark, "a picture in a log file must not carry escapes"


def test_the_depth_follows_the_terminal_and_the_readers_wishes():
    set_colour(True)
    assert colour_depth({"COLORTERM": "truecolor"}) == TRUECOLOUR
    assert colour_depth({"COLORTERM": "24bit"}) == TRUECOLOUR
    assert colour_depth({}) == PALETTE

    set_colour(False)
    assert colour_depth({"COLORTERM": "truecolor"}) == PLAIN, "NO_COLOR beats any capability"


def test_resample_hits_the_size_asked_for_whichever_way_it_has_to_go():
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6, 1)

    assert resample(image, 2, 3).shape == (2, 3, 1)
    assert resample(image, 9, 11).shape == (9, 11, 1)
    assert resample(image, 4, 6) is image, "a picture already the right size is not copied"


def test_resample_keeps_a_thin_bright_line_rather_than_averaging_it_away():
    image = np.zeros((16, 4, 1), dtype=np.uint8)
    image[8] = 255  # one bright row: a whistle, on a picture that has to shrink

    shrunk = resample(image, 8, 4)

    assert shrunk.max() == 255, "averaging would turn a one-pixel structure into a grey smudge"


def test_resample_of_an_empty_picture_is_a_blank_one_of_the_size_asked_for():
    blank = resample(np.zeros((0, 0, 3), dtype=np.uint8), 3, 5)

    assert blank.shape == (3, 5, 3) and blank.max() == 0
