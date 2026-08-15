# Vixen Intelligence c.2026
"""The picture behind a finished recording.

What has to be right here is the *mapping*. A box is four numbers in seconds and hertz, and a
picture is a grid of pixels; if the two are put together even slightly wrong, the panel draws a
rectangle over the wrong part of the recording and says the model found something it did not.
So the tests below check where a box actually lands, at both ends of both axes, and that the
zoomed view moves it correctly rather than merely differently.

The rest is the rule the whole per-file loop follows: a recording that cannot be pictured is a
sentence in the frame, never an exception. A deleted file, a sidecar being written while it is
read, a window too small to draw in — each is a panel that still opens.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from siarapp.cli import picture as pic
from siarapp.cli.blocks import PLAIN
from siarapp.cli.format import set_colour, visible_len
from siarapp.io.output import SIDECAR_SUFFIX
from siarapp.viz.colormap import shape_colour

RATE = 16000
SECONDS = 4.0


@pytest.fixture(autouse=True)
def plain_colour():
    """Rendered without colour, so a frame's geometry is what the assertions see."""
    set_colour(False)
    yield
    set_colour(None)


def _wav(path, *, seconds: float = SECONDS, rate: int = RATE):
    t = np.arange(int(rate * seconds)) / rate
    signal = 0.4 * np.sin(2 * np.pi * 2000.0 * t) + 0.01 * np.sin(2 * np.pi * 6000.0 * t)
    sf.write(str(path), signal.astype("float32"), rate)
    return path


def _sidecar(path, boxes):
    document = {"format": "siar-scanner-structures-v1", "count": len(boxes), "structures": boxes}
    path.write_text(json.dumps(document))
    return path


@pytest.fixture
def folder(tmp_path):
    """An output folder with one recording and one structure in the middle of it."""
    (tmp_path / "station-a").mkdir()
    _wav(tmp_path / "station-a" / "clip.wav")
    _sidecar(tmp_path / "station-a" / ("clip" + SIDECAR_SUFFIX), [
        {"tmin": 1.0, "tmax": 2.0, "fmin": 1800.0, "fmax": 2200.0, "shape": "tonal"},
    ])
    return tmp_path


def test_the_sidecar_is_found_by_the_rule_the_writer_uses(tmp_path):
    audio, sidecar = pic.artefact_paths(str(tmp_path), "station-a/clip.wav")

    assert audio == str(tmp_path / "station-a" / "clip.wav")
    assert sidecar == str(tmp_path / "station-a" / ("clip" + SIDECAR_SUFFIX))


def test_a_recording_loads_with_its_structures(folder):
    loaded = pic.load_picture(str(folder), "station-a/clip.wav", cols=80, rows=20)

    assert loaded.drawable and loaded.error == ""
    assert loaded.duration_sec == pytest.approx(SECONDS, abs=0.05)
    assert loaded.nyquist_hz == pytest.approx(RATE / 2)
    assert [box["shape"] for box in loaded.boxes] == ["tonal"]
    assert loaded.shapes() == [("tonal", 1)]


def test_a_recording_that_is_not_there_is_a_sentence_not_an_exception(folder):
    missing = pic.load_picture(str(folder), "station-a/gone.wav", cols=80, rows=20)

    assert not missing.drawable
    assert "no recording at" in missing.error


def test_a_recording_too_short_to_picture_still_carries_its_structures(tmp_path):
    _wav(tmp_path / "tiny.wav", seconds=0.001)
    _sidecar(tmp_path / ("tiny" + SIDECAR_SUFFIX), [{"tmin": 0, "tmax": 0.1, "fmin": 0,
                                                     "fmax": 100, "shape": "click"}])

    loaded = pic.load_picture(str(tmp_path), "tiny.wav", cols=80, rows=20)

    assert not loaded.drawable and "too short" in loaded.error
    assert len(loaded.boxes) == 1


def test_a_missing_or_broken_sidecar_is_a_picture_with_no_boxes(tmp_path):
    _wav(tmp_path / "clip.wav")
    loaded = pic.load_picture(str(tmp_path), "clip.wav", cols=60, rows=12)
    assert loaded.drawable and loaded.boxes == []

    (tmp_path / ("clip" + SIDECAR_SUFFIX)).write_text("{ half a file")
    assert pic.load_picture(str(tmp_path), "clip.wav", cols=60, rows=12).boxes == []


# -- where a box lands ------------------------------------------------------------------------


def _picture(boxes, *, duration=10.0, nyquist=1000.0, rows=40, cols=40):
    cells = np.zeros((rows, cols), dtype=np.uint8)
    return pic.Picture("x.wav", "x.wav", cells=cells, duration_sec=duration,
                       nyquist_hz=nyquist, boxes=boxes)


def _drawn(picture, cols, rows, band=None):
    view = pic._view(picture, band)
    return pic._draw_boxes(pic._image(picture, cols, rows, view), picture, view)


def _painted(image, shape="tonal"):
    """Where a box of this shape was actually drawn, as ``(row, column)`` pairs.

    Matched on the shape's colour rather than on "not black": viridis paints its own background,
    so every pixel of a spectrogram is lit and a test that looked for any colour at all would
    pass whether or not a box had ever been drawn.
    """
    colour = np.asarray(shape_colour(shape), dtype=np.uint8)
    return np.argwhere((image == colour).all(axis=2))


def test_a_box_is_drawn_where_its_seconds_and_hertz_say_it_is():
    # The middle of the recording, and the top half of the band.
    picture = _picture([{"tmin": 5.0, "tmax": 7.5, "fmin": 500.0, "fmax": 1000.0,
                         "shape": "tonal"}])

    image = _drawn(picture, cols=40, rows=10)   # 20 pixel rows

    painted = _painted(image)
    rows, columns = painted[:, 0], painted[:, 1]
    assert rows.min() == 0, "fmax at Nyquist is the top row of the picture"
    assert rows.max() == pytest.approx(10, abs=1), "fmin at half of Nyquist is the middle"
    assert columns.min() == pytest.approx(20, abs=1), "tmin at half the recording"
    assert columns.max() == pytest.approx(30, abs=1)


def test_the_lowest_frequency_is_the_bottom_of_the_picture():
    picture = _picture([{"tmin": 0.0, "tmax": 10.0, "fmin": 0.0, "fmax": 100.0,
                         "shape": "click"}])

    image = _drawn(picture, cols=20, rows=10)

    painted = _painted(image, "click")
    assert painted[:, 0].max() == 19, "a box down to 0 Hz reaches the last pixel row"


def test_a_box_thinner_than_a_pixel_is_still_drawn():
    # A click: five milliseconds of a ten-second recording, on a forty-column picture.
    picture = _picture([{"tmin": 5.0, "tmax": 5.005, "fmin": 400.0, "fmax": 600.0,
                         "shape": "click"}])

    image = _drawn(picture, cols=40, rows=10)

    assert len(_painted(image, "click")), \
        "a structure the run found must not vanish because it is narrow"


def test_every_shape_keeps_the_colour_the_web_viewer_gives_it():
    picture = _picture([{"tmin": 1.0, "tmax": 2.0, "fmin": 100.0, "fmax": 200.0,
                         "shape": "sweep"}])

    assert len(_painted(_drawn(picture, cols=40, rows=10), "sweep"))


def test_an_unreadable_box_is_skipped_and_the_others_are_drawn():
    picture = _picture([
        {"tmin": "not a number", "tmax": 2.0, "fmin": 0.0, "fmax": 100.0, "shape": "blob"},
        {"tmin": 1.0, "tmax": 2.0, "fmin": 100.0, "fmax": 200.0, "shape": "sweep"},
    ])

    assert len(_painted(_drawn(picture, cols=40, rows=10), "sweep"))


# -- the zoom ---------------------------------------------------------------------------------


def test_the_band_is_what_was_found_with_room_around_it():
    picture = _picture([{"tmin": 0, "tmax": 1, "fmin": 400.0, "fmax": 500.0, "shape": "tonal"}])

    low, high = picture.band()

    assert low < 400.0 and high > 500.0, "boxes against the edge read as boxes that ran off it"
    assert low >= 0.0 and high <= picture.nyquist_hz


def test_there_is_no_zoom_when_the_structures_already_fill_the_picture():
    wide = _picture([{"tmin": 0, "tmax": 1, "fmin": 10.0, "fmax": 900.0, "shape": "tonal"}])

    assert wide.band() is None, "a zoom that moves everything and reveals nothing is not offered"


def test_there_is_no_zoom_without_structures():
    assert _picture([]).band() is None


def test_zooming_spends_the_whole_picture_on_the_band():
    picture = _picture([{"tmin": 0.0, "tmax": 10.0, "fmin": 400.0, "fmax": 500.0,
                         "shape": "tonal"}])

    full = _painted(_drawn(picture, 40, 10))[:, 0]
    zoomed = _painted(_drawn(picture, 40, 10, band=picture.band()))[:, 0]

    assert (full.max() - full.min()) < (zoomed.max() - zoomed.min())


# -- the panel --------------------------------------------------------------------------------


SIZES = [(120, 40), (100, 30), (80, 24), (60, 14), (200, 60)]


@pytest.mark.parametrize("width,height", SIZES)
def test_the_panel_fills_the_window_exactly(width, height, folder):
    cols, rows = pic.panel_size(width, height)
    loaded = pic.load_picture(str(folder), "station-a/clip.wav", cols=cols, rows=rows)
    frame = pic.render_picture(loaded, width, height, depth=PLAIN)

    assert len(frame) <= height
    assert all(visible_len(line) == width for line in frame)


def test_the_panel_names_the_recording_and_what_was_found(folder):
    cols, rows = pic.panel_size(120, 30)
    loaded = pic.load_picture(str(folder), "station-a/clip.wav", cols=cols, rows=rows)

    frame = "\n".join(pic.render_picture(loaded, 120, 30, position="1 of 8", depth=PLAIN))

    assert "station-a/clip.wav" in frame
    assert "1 of 8" in frame
    assert "1 structure" in frame
    assert "tonal 1" in frame, "the legend says what shape, in the colour it was drawn"
    assert "8.0k" in frame, "the frequency axis is labelled"


def test_a_window_too_small_for_a_spectrogram_says_so_rather_than_drawing_one(folder):
    loaded = pic.load_picture(str(folder), "station-a/clip.wav", cols=20, rows=4)

    frame = "\n".join(pic.render_picture(loaded, 40, 9, depth=PLAIN))

    assert "the window is too small" in frame


def test_a_panel_for_a_recording_that_failed_still_has_its_keys(folder):
    frame = "\n".join(pic.render_picture(
        pic.Picture("gone.wav", error="no recording at gone.wav"), 90, 20, depth=PLAIN))

    assert "no recording at" in frame
    assert "q close" in frame, "the way out has to be on every frame this panel draws"


def test_the_zoom_is_only_offered_when_there_is_one(folder):
    cols, rows = pic.panel_size(120, 30)
    loaded = pic.load_picture(str(folder), "station-a/clip.wav", cols=cols, rows=rows)

    offered = "\n".join(pic.render_picture(loaded, 120, 30, depth=PLAIN))
    assert "z zoom to the structures" in offered

    loaded.boxes = []
    assert "z zoom" not in "\n".join(pic.render_picture(loaded, 120, 30, depth=PLAIN))
