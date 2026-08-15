# Vixen Intelligence c.2026
"""A picture on a terminal: two pixels to a cell.

A spectrogram is the answer to "what did it find, and is it right" — and on the machine that ran
the scan there is usually no browser, no X server and no way to open a PNG. What there is, is a
terminal with 24-bit colour, and a character cell that can hold two pixels: ``▀`` painted with a
foreground for its top half and a background for its bottom half. An 80x24 window is then a
80x48 image, and a full-screen one on a modern terminal is around 200x100 — small, and enough to
see a sweep, a click train and whether the boxes are on top of them.

Three depths, because the machines this runs on are not all the same terminal:

* **Truecolour** (``$COLORTERM`` says so) — the pixel's own RGB, and the picture is the picture.
* **256 colours** — the 6x6x6 cube plus the greys, which viridis survives surprisingly well.
* **No colour at all** (``NO_COLOR``, ``TERM=dumb``, a pipe) — an intensity ramp in ASCII. It is
  not a spectrogram any more, but the boxes and the loud parts are still where they are.

Nothing here knows what a spectrogram is. It takes an ``(rows, cols, 3)`` array of bytes and
returns lines, so the same function draws the picture, the boxes drawn onto it, and anything else
this CLI ever needs to put on a terminal. It lives beside the other terminal modules rather than
under :mod:`siarapp.viz` for that reason: what it produces is a frame, not an image file.
"""
from __future__ import annotations

import os

import numpy as np

from siarapp.cli.format import RESET, colour_enabled

__all__ = [
    "ASCII_RAMP",
    "HALF_BLOCK",
    "colour_depth",
    "render_blocks",
    "resample",
]

#: The character that carries two pixels: its foreground is the upper pixel, its background the
#: lower one. Every terminal font that draws box characters at all draws this one.
HALF_BLOCK = "▀"

#: Intensity, dark to light, when there is no colour to be had. Ten steps is as much as a reader
#: can tell apart in a proportional-looking ramp, and more of them just looks like noise.
ASCII_RAMP = " .:-=+*#%@"

#: What each depth is called. Not an enum: it is three strings that go straight into a docstring
#: and a test, and an enum would be a class to import in order to compare to a constant.
TRUECOLOUR, PALETTE, PLAIN = "truecolour", "palette", "plain"


def colour_depth(env: dict | None = None) -> str:
    """How much colour this terminal can be asked for.

    Args:
        env: Environment to read. Defaults to the real one — passed in by tests, which should not
            have to mutate the process to ask what a machine would do.

    Returns:
        :data:`TRUECOLOUR`, :data:`PALETTE` or :data:`PLAIN`.

    Note:
        ``$COLORTERM`` is the only reliable signal for 24-bit colour — there is no terminfo
        capability every emulator agrees on — and a terminal that does not set it is assumed to
        have the 256-colour palette, which has been near-universal for twenty years. Both fall
        back to nothing at all the moment :func:`~siarapp.cli.format.colour_enabled` says the
        output is not a terminal or the reader asked for no colour.
    """
    environ = os.environ if env is None else env
    if not colour_enabled():
        return PLAIN
    if str(environ.get("COLORTERM", "")).lower() in ("truecolor", "24bit"):
        return TRUECOLOUR
    return PALETTE


def resample(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resize of an ``(h, w, ...)`` array to exactly ``rows`` x ``cols``.

    Nearest rather than an average, and deliberately: this picture is read for *structure* — the
    thin bright line of a whistle, the vertical spike of a click — and averaging a 1200-column
    preview down to 180 cells is how a click stops being visible at all. A dropped column is
    honest about being a sample; a smeared one is not.

    Args:
        image: The source, at least two dimensions.
        rows: Rows wanted.
        cols: Columns wanted.

    Returns:
        The resized array. Returns the input untouched when it is already that size, and an empty
        array of the requested shape when the source has no pixels.
    """
    rows, cols = max(1, int(rows)), max(1, int(cols))
    height, width = int(image.shape[0]), int(image.shape[1])
    if height == rows and width == cols:
        return image
    if height == 0 or width == 0:
        return np.zeros((rows, cols) + image.shape[2:], dtype=image.dtype)
    y = np.minimum((np.arange(rows) * height) // rows, height - 1)
    x = np.minimum((np.arange(cols) * width) // cols, width - 1)
    return image[np.ix_(y, x)]


def _palette_index(rgb: np.ndarray) -> np.ndarray:
    """An ``(..., 3)`` array of bytes as xterm-256 colour indices.

    The 6x6x6 cube for colour, and the 24-step grey ramp for anything close to neutral — grey is
    where the cube is coarsest, and a spectrogram's quiet background is exactly there.
    """
    value = rgb.astype(np.int16)
    r, g, b = value[..., 0], value[..., 1], value[..., 2]
    cube = 16 + 36 * ((r * 5 + 127) // 255) + 6 * ((g * 5 + 127) // 255) + ((b * 5 + 127) // 255)
    grey_level = (r + g + b) // 3
    grey = 232 + np.clip((grey_level - 8) // 10, 0, 23)
    spread = value.max(axis=-1) - value.min(axis=-1)
    return np.where(spread <= 8, grey, cube).astype(np.int16)


def _rows_of_pairs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an image into the upper and lower pixel of each character cell.

    An odd number of rows is padded with one black row rather than dropped: a picture that lost
    its last row of pixels would be a picture that quietly lied about where the top of the band
    was.
    """
    height = int(image.shape[0])
    if height % 2:
        image = np.concatenate([image, np.zeros((1,) + image.shape[1:], dtype=image.dtype)])
    return image[0::2], image[1::2]


def render_blocks(image: np.ndarray, *, depth: str | None = None) -> list[str]:
    """An ``(rows, cols, 3)`` byte image as terminal lines, two pixel rows per line.

    Args:
        image: ``uint8`` RGB. An odd row count is padded with one black row.
        depth: Force a depth (see :func:`colour_depth`); ``None`` asks the terminal.

    Returns:
        ``ceil(rows / 2)`` lines. Each one ends in a reset, so a picture can never leak its last
        colour into the rest of the frame.
    """
    picture = np.asarray(image, dtype=np.uint8)
    if picture.ndim != 3 or picture.shape[0] == 0 or picture.shape[1] == 0:
        return []
    mode = depth or colour_depth()
    top, bottom = _rows_of_pairs(picture)

    if mode == PLAIN:
        # Both halves of the cell average into one character: there is only one glyph to spend.
        levels = (top.astype(np.int16).mean(axis=2) + bottom.astype(np.int16).mean(axis=2)) / 2.0
        steps = np.clip((levels / 255.0 * (len(ASCII_RAMP) - 1)).round(), 0,
                        len(ASCII_RAMP) - 1).astype(np.int16)
        return ["".join(ASCII_RAMP[step] for step in row) for row in steps]

    lines = []
    if mode == TRUECOLOUR:
        for upper, lower in zip(top, bottom):
            parts = []
            last = None
            for (r1, g1, b1), (r2, g2, b2) in zip(upper, lower):
                cell = (r1, g1, b1, r2, g2, b2)
                # Only re-state the colour when it changes. A picture of a quiet recording is
                # mostly one shade, and the escape codes are otherwise four fifths of the bytes
                # written to the terminal on every repaint.
                if cell != last:
                    parts.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m")
                    last = cell
                parts.append(HALF_BLOCK)
            lines.append("".join(parts) + RESET)
        return lines

    upper_idx, lower_idx = _palette_index(top), _palette_index(bottom)
    for upper, lower in zip(upper_idx, lower_idx):
        parts = []
        last = None
        for fg, bg in zip(upper, lower):
            if (fg, bg) != last:
                parts.append(f"\033[38;5;{fg}m\033[48;5;{bg}m")
                last = (fg, bg)
            parts.append(HALF_BLOCK)
        lines.append("".join(parts) + RESET)
    return lines
