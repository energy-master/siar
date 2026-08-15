# Vixen Intelligence c.2026
"""One finished recording, on the screen, with its boxes on it.

A run writes a spectrogram and a set of structures for every recording, and until now the only
way to look at either was to finish the scan, carry the folder somewhere with a browser, and open
it. That is the wrong order. The question a survey asks at recording forty of ten thousand is
*"is this model finding the right thing"*, and the answer is one picture — the one the run has
already computed everything for.

So the run panel opens it in place. Pick a finished recording, and this draws its spectrogram in
the terminal with every structure outlined in the same colour the web viewer would use, over the
same reduction the lane thumbnail and the remote preview are drawn from
(:func:`siarapp.serve.preview.read_preview`). Not a fourth picture of the same recording: the
same one, at the size of the window it is being looked at in.

The picture is **read back out of the output folder**, never held from the scan. The folder is
complete after every recording — audio, sidecar and thumbnail — so everything needed is on the
disk by the time a row can be picked, and a run does not pay a byte of memory for a panel nobody
may open. A recording too short to picture, or one that failed, is a sentence in the frame rather
than a missing panel.
"""
from __future__ import annotations

import json
import os

import numpy as np

from siarapp.cli.blocks import render_blocks, resample
from siarapp.cli.format import (
    BOLD,
    DIM,
    RED,
    YELLOW,
    clock,
    duration,
    fit,
    fit_path,
    paint,
    visible_len,
)
from siarapp.io.output import SIDECAR_SUFFIX
from siarapp.viz.colormap import shape_colour, viridis_lut

__all__ = ["PANEL_CHROME", "Picture", "artefact_paths", "load_picture", "panel_size",
           "render_picture"]

#: Lines of the panel that are not picture: the name, two rules, the facts, the time ruler, the
#: legend and the keys. Named once because :func:`panel` sizes the preview it reads against it and
#: :func:`render_picture` lays it out against it, and a picture read one row taller than the frame
#: has room for is a row of a recording nobody ever sees.
PANEL_CHROME = 7

#: Columns kept on the left for frequency labels — ``"96.0k"`` and a space.
_GUTTER = 7

#: Frequencies labelled down the side, as a fraction of Nyquist. The ends and the quarters: enough
#: to read a band off the picture, few enough that a 20-row panel is not all labels.
_FREQ_TICKS = (1.0, 0.75, 0.5, 0.25, 0.0)

#: Times labelled along the bottom, as a fraction of the recording.
_TIME_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Smallest picture worth drawing, in character cells. Below this the frame says so instead: a
#: spectrogram eight cells tall is four pixels of frequency, which cannot show anything.
_MIN_COLS = 24
_MIN_ROWS = 4

#: Widest and tallest picture asked of the preview reader, in *pixels*. A terminal that reports
#: 400 columns is real (a wall display, a tiling window manager), and asking for a preview that
#: size costs seconds; past this the picture is resampled up instead, which is free.
_MAX_PIXEL_COLS = 400
_MAX_PIXEL_ROWS = 256

#: Room left above and below the structures when the picture zooms to them, as a fraction of the
#: band they span. A box drawn against the top edge of a picture reads as one that ran off it.
_BAND_PADDING = 0.35

#: A zoom is only offered when the structures fit inside this fraction of the full range.
#: Zooming from 0-48 kHz to 0-40 kHz would move every landmark on the picture and reveal nothing.
_BAND_WORTH_IT = 0.6


class Picture:
    """One recording reduced to a picture, and the boxes that were found in it.

    Attributes:
        rel_path: The recording's path relative to the scanned folder — what the run panel calls
            it, and what the output folder is keyed by.
        audio_path: Where the copy in the output folder is.
        cells: ``(rows, cols)`` ``uint8`` intensity, **row 0 is the lowest frequency**, or
            ``None`` when there is no picture to draw.
        duration_sec: Length of the recording, so a column maps to a time.
        nyquist_hz: Half its sample rate, so a row maps to a frequency.
        boxes: The structures from the sidecar, each with ``tmin``/``tmax``/``fmin``/``fmax`` and
            a ``shape``.
        error: Why there is no picture, when there is none.
    """

    __slots__ = ("rel_path", "audio_path", "cells", "duration_sec", "nyquist_hz", "boxes",
                 "error")

    def __init__(self, rel_path: str, audio_path: str = "", cells=None,
                 duration_sec: float = 0.0, nyquist_hz: float = 0.0,
                 boxes: list[dict] | None = None, error: str = "") -> None:
        self.rel_path = rel_path
        self.audio_path = audio_path
        self.cells = cells
        self.duration_sec = float(duration_sec)
        self.nyquist_hz = float(nyquist_hz)
        self.boxes = list(boxes or [])
        self.error = error

    @property
    def drawable(self) -> bool:
        """Whether there is an image to put on the screen."""
        return self.cells is not None and self.cells.size > 0

    def shapes(self) -> list[tuple[str, int]]:
        """Structures by shape, most numerous first — the legend under the picture."""
        counts: dict[str, int] = {}
        for box in self.boxes:
            shape = str(box.get("shape") or "structure")
            counts[shape] = counts.get(shape, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def band(self) -> tuple[float, float] | None:
        """The frequency range the structures occupy, with room around it — or ``None``.

        A band-limited model draws its boxes into a twentieth of a picture that runs to Nyquist,
        and four pixels of box is not an answer to "is it finding the right thing". This is what
        the panel zooms to: the span of everything found, padded by
        :data:`_BAND_PADDING` of its own height so the boxes are not against the edges, and
        clamped to what the recording actually has.

        Returns:
            ``(low Hz, high Hz)``, or ``None`` when there is nothing found, no Nyquist to clamp
            against, or the structures already cover most of the picture — in which case a zoom
            would move everything and show nothing new.
        """
        if not self.boxes or self.nyquist_hz <= 0:
            return None
        try:
            low = min(float(box.get("fmin", 0.0)) for box in self.boxes)
            high = max(float(box.get("fmax", 0.0)) for box in self.boxes)
        except (TypeError, ValueError):
            return None
        if high <= low:
            return None
        pad = (high - low) * _BAND_PADDING
        low, high = max(0.0, low - pad), min(self.nyquist_hz, high + pad)
        if (high - low) >= self.nyquist_hz * _BAND_WORTH_IT:
            return None
        return low, high


def artefact_paths(out_root: str, rel_path: str) -> tuple[str, str]:
    """Where one recording's audio and structures sidecar sit in an output folder.

    The sidecar rule is :mod:`siarapp.io.output`'s — the audio basename with its extension
    replaced — and is followed here rather than guessed, because a reader that guessed would
    silently show a picture with no boxes on it whenever the two disagreed.

    Args:
        out_root: The output folder.
        rel_path: The recording's path relative to the scanned folder, POSIX separators.

    Returns:
        ``(audio path, sidecar path)``. Neither is checked for existence.
    """
    audio = os.path.join(out_root, *str(rel_path).split("/"))
    return audio, os.path.splitext(audio)[0] + SIDECAR_SUFFIX


def _read_boxes(path: str) -> list[dict]:
    """The structures out of a sidecar, or none at all when it cannot be read.

    A missing sidecar is the normal state of a recording that was skipped or failed, and an
    unreadable one is a recording being written while it is being looked at. Both are a picture
    with no boxes, never an error in front of the picture.
    """
    try:
        with open(path, "rb") as handle:
            document = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError):
        return []
    boxes = document.get("structures")
    return [box for box in boxes if isinstance(box, dict)] if isinstance(boxes, list) else []


def load_picture(out_root: str, rel_path: str, *, cols: int, rows: int,
                 channel: str = "mix") -> Picture:
    """Read one recording back out of the output folder and reduce it to ``rows`` x ``cols``.

    Args:
        out_root: The output folder the run is writing.
        rel_path: Which recording, relative to the scanned folder.
        cols: Character columns the picture will occupy — one column of pixels each.
        rows: Character rows. Each holds two pixel rows, so twice this many are asked for.
        channel: The run's channel selection, so the picture shows what was scanned.

    Returns:
        A :class:`Picture`. One that could not be read carries the reason in ``error`` rather
        than raising: a panel opened over a recording that has just been deleted should say so,
        not take the run's display down with it.
    """
    audio, sidecar = artefact_paths(out_root, rel_path)
    if not os.path.isfile(audio):
        return Picture(rel_path, audio, error=f"no recording at {audio}")

    # Imported here, not at the top: this module is loaded by the run panel, and the preview
    # reader pulls in soundfile and the whole thumbnail pipeline for a panel that may never be
    # opened.
    from siarapp.serve.preview import read_preview

    pixels_wide = max(_MIN_COLS, min(int(cols), _MAX_PIXEL_COLS))
    pixels_high = max(_MIN_ROWS * 2, min(int(rows) * 2, _MAX_PIXEL_ROWS))
    preview = read_preview(audio, width=pixels_wide, height=pixels_high, channel=channel)
    if preview is None:
        return Picture(rel_path, audio, boxes=_read_boxes(sidecar),
                       error="this recording is too short to picture, or cannot be read")
    return Picture(
        rel_path,
        audio,
        cells=preview.cells,
        duration_sec=preview.duration_sec,
        nyquist_hz=preview.nyquist_hz,
        boxes=_read_boxes(sidecar),
    )


def _view(picture: Picture, band: tuple[float, float] | None) -> tuple[float, float]:
    """The frequency range on screen: the whole recording, or the band asked for."""
    if band is None or picture.nyquist_hz <= 0:
        return 0.0, picture.nyquist_hz
    low, high = float(band[0]), float(band[1])
    if high <= low:
        return 0.0, picture.nyquist_hz
    return max(0.0, low), min(picture.nyquist_hz, high)


def _image(picture: Picture, cols: int, rows: int,
           view: tuple[float, float]) -> np.ndarray:
    """The picture as an ``(rows * 2, cols, 3)`` viridis raster, low frequency at the bottom.

    Resampled to exactly the cells available rather than letterboxed. A spectrogram has no
    aspect ratio worth preserving — its axes are seconds and hertz, and both are labelled.

    Args:
        picture: What was loaded.
        cols: Columns of pixels, and
        rows: character rows — two pixel rows each.
        view: ``(low Hz, high Hz)`` to show. Cropped *before* resampling, so zooming into a band
            spends every row it has on that band rather than magnifying the rows it kept.
    """
    cells = picture.cells
    low, high = view
    if picture.nyquist_hz > 0 and (low > 0 or high < picture.nyquist_hz):
        # Row 0 of a preview is DC, the last row is Nyquist, and they are evenly spaced.
        total = int(cells.shape[0])
        first = int(np.clip(round(low / picture.nyquist_hz * total), 0, total - 1))
        last = int(np.clip(round(high / picture.nyquist_hz * total), first + 1, total))
        cells = cells[first:last]
    cells = resample(cells, rows * 2, cols)
    # Row 0 of a preview is the lowest frequency; a picture is drawn the other way up.
    return viridis_lut[np.flipud(cells)]


def _draw_boxes(image: np.ndarray, picture: Picture,
                view: tuple[float, float]) -> np.ndarray:
    """Outline every structure on the raster, in the shape's own colour.

    The mapping is the web viewer's, line for line (``local_web/viewer.js``): x from time over
    the recording's duration, y from frequency over the range on screen, and the top edge of a box
    is ``fmax`` because the picture has the lowest frequency at the bottom. A box in the terminal
    and the same box in the browser are then the same rectangle over the same picture.

    A box outside the range on screen is clamped to its edge rather than dropped: a zoomed picture
    that silently stopped drawing half of what was found would be the most misleading thing on
    this screen.
    """
    height, width = image.shape[0], image.shape[1]
    seconds = picture.duration_sec
    low, high = view
    span = high - low
    if not seconds or span <= 0 or not picture.boxes:
        return image

    out = image.copy()
    for box in picture.boxes:
        try:
            tmin, tmax = float(box.get("tmin", 0.0)), float(box.get("tmax", 0.0))
            fmin, fmax = float(box.get("fmin", 0.0)), float(box.get("fmax", 0.0))
        except (TypeError, ValueError):
            continue
        x0 = int(np.clip(tmin / seconds * width, 0, width - 1))
        x1 = int(np.clip(tmax / seconds * width, 0, width - 1))
        # Frequency counts up the picture, so fmax is the *top* row and fmin the bottom one.
        y0 = int(np.clip((1.0 - (fmax - low) / span) * height, 0, height - 1))
        y1 = int(np.clip((1.0 - (fmin - low) / span) * height, 0, height - 1))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        # A box thinner than a pixel still has to be visible: a click is a millisecond long and
        # would otherwise be drawn as nothing at all on a picture of an hour.
        x1 = min(width - 1, max(x1, x0 + 1))
        y1 = min(height - 1, max(y1, y0 + 1))
        colour = np.asarray(shape_colour(box.get("shape")), dtype=np.uint8)
        out[y0, x0:x1 + 1] = colour
        out[y1, x0:x1 + 1] = colour
        out[y0:y1 + 1, x0] = colour
        out[y0:y1 + 1, x1] = colour
    return out


def _frequency_gutter(rows: int, view: tuple[float, float]) -> list[str]:
    """The left-hand column of frequency labels, one entry per picture row."""
    labels = [" " * _GUTTER for _ in range(rows)]
    low, high = view
    if rows < 2 or high <= low:
        return labels
    for fraction in _FREQ_TICKS:
        row = int(round((1.0 - fraction) * (rows - 1)))
        hz = low + (high - low) * fraction
        text = f"{hz / 1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"
        labels[row] = paint(f"{text:>{_GUTTER - 1}} ", DIM)
    return labels


def _time_ruler(cols: int, seconds: float) -> str:
    """The time axis under the picture: a handful of stamps at fixed fractions of the recording."""
    if cols <= 0 or seconds <= 0:
        return ""
    line = [" "] * cols
    for fraction in _TIME_TICKS:
        stamp = clock(seconds * fraction)
        start = int(round(fraction * (cols - 1)))
        # The last stamp is pulled inside the right edge rather than being allowed to run off it.
        start = min(max(0, start - (len(stamp) if fraction == 1.0 else 0)), max(0, cols - len(stamp)))
        for offset, char in enumerate(stamp):
            if start + offset < cols:
                line[start + offset] = char
    return paint("".join(line), DIM)


def _legend(picture: Picture, width: int) -> str:
    """Structures by shape, each in the colour its boxes are drawn in."""
    shapes = picture.shapes()
    if not shapes:
        return paint("  no structures in this recording", DIM)
    parts = []
    for shape, count in shapes:
        red, green, blue = shape_colour(shape)
        swatch = paint("█", f"\033[38;2;{red};{green};{blue}m")
        parts.append(f"{swatch} {shape} {count:,}")
    return fit("  " + "   ".join(parts), width)


def render_picture(picture: Picture, width: int, height: int, *, position: str = "",
                   band: tuple[float, float] | None = None,
                   depth: str | None = None) -> list[str]:
    """The whole panel: the recording's name, its spectrogram, the axes, and the legend.

    Pure, and given a size rather than asking the terminal for one, so a frame can be rendered and
    asserted on in a test with no terminal anywhere near it.

    Args:
        picture: What :func:`load_picture` returned.
        width: Columns available.
        height: Rows available.
        position: ``"3 of 12"`` — where this recording sits in the list it was opened from.
        band: Frequency range to draw, or ``None`` for the whole recording. See
            :meth:`Picture.band`.
        depth: Force a colour depth (see :func:`siarapp.cli.blocks.colour_depth`).

    Returns:
        At most ``height`` lines, each at most ``width`` visible columns.
    """
    view = _view(picture, band)
    head = f" {paint(fit_path(picture.rel_path, max(10, width - 30)), BOLD)}"
    if position:
        head += paint(f"   {position}", DIM)
    lines = [fit(head, width), paint("─" * width, DIM)]

    facts = []
    if picture.duration_sec:
        facts.append(duration(picture.duration_sec))
    if view[1] > view[0]:
        facts.append(f"{view[0] / 1000:.1f}–{view[1] / 1000:.1f} kHz"
                     + (paint("  zoomed to what was found", BOLD) if band else ""))
    facts.append(f"{len(picture.boxes):,} structure{'' if len(picture.boxes) == 1 else 's'}")
    keys = " ←/→ previous/next recording   q close"
    if picture.band() is not None:
        keys += "   z " + ("whole recording" if band else "zoom to the structures")
    footer = [
        paint("─" * width, DIM),
        _legend(picture, width),
        paint(keys, DIM),
    ]

    cols = max(0, width - _GUTTER)
    rows = height - len(lines) - len(footer) - 2  # two lines for the facts and the time ruler
    if picture.error or not picture.drawable:
        message = picture.error or "no picture for this recording"
        body = [paint(f"  {message}", YELLOW if not picture.error else RED)]
        if picture.boxes:
            body.append(paint(f"  its sidecar still holds {len(picture.boxes):,} structures", DIM))
        body += [""] * max(0, height - len(lines) - len(footer) - len(body))
        return [fit(line, width) for line in (lines + body + footer)[:height]]

    if cols < _MIN_COLS or rows < _MIN_ROWS:
        body = [paint("  the window is too small to draw a spectrogram in", YELLOW)]
        body += [""] * max(0, height - len(lines) - len(footer) - len(body))
        return [fit(line, width) for line in (lines + body + footer)[:height]]

    rows = max(_MIN_ROWS, rows)
    image = _draw_boxes(_image(picture, cols, rows, view), picture, view)
    gutter = _frequency_gutter(rows, view)
    picture_lines = render_blocks(image, depth=depth)
    lines.append(fit(paint("  " + "  ·  ".join(facts), DIM), width))
    lines += [label + row for label, row in zip(gutter, picture_lines)]
    lines.append(" " * _GUTTER + _time_ruler(cols, picture.duration_sec))

    # Padded to the full height before the footer so the legend and the keys sit on the bottom
    # two lines of the window rather than floating under a short picture.
    lines += [""] * max(0, height - len(lines) - len(footer))
    frame = lines + footer
    # The picture's own lines are already exactly `cols` cells wide and carry a colour code per
    # cell; `fit` would walk every escape in them for nothing, so only the text lines are fitted.
    return [line if visible_len(line) == width else fit(line, width) for line in frame[:height]]


def panel_size(width: int, height: int) -> tuple[int, int]:
    """The picture's own size inside a window that big — ``(cols, rows)``.

    What a display asks :func:`load_picture` for, so the recording is never read finer than the
    terminal can show or coarser than it could have been. Here rather than at the caller because
    it is the same arithmetic :func:`render_picture` lays the frame out with, and the two
    disagreeing would cost a row of the recording nobody ever sees.
    """
    return max(1, int(width) - _GUTTER), max(_MIN_ROWS, int(height) - PANEL_CHROME)
