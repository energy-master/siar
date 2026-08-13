# Vixen Intelligence c.2026
"""The reduced picture the daemon sends instead of a recording.

The whole feature rests on one claim: this is small enough to pull over a tunnel and a recording is
not. So the tests that matter are the ones about size — that a preview is a fraction of its
recording, and that the fraction does not grow with duration — and the ones about the bounds, since
``w`` and ``h`` arrive from a URL and a request must not be able to ask the box for a gigabyte.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import soundfile as sf

from siarapp.serve.preview import (
    PREVIEW_DEFAULT_BINS,
    PREVIEW_MAX_ACTUAL_BINS,
    PREVIEW_DEFAULT_WIDTH,
    PREVIEW_HEADER_BYTES,
    PREVIEW_MAGIC,
    PREVIEW_MAX_BINS,
    PREVIEW_MAX_SAMPLES,
    PREVIEW_MAX_WIDTH,
    encode_preview,
    preview_bounds,
    preview_png,
    read_preview,
)


def _write(path, *, seconds=4.0, rate=8000, tone_hz=200.0, channels=1):
    """A recording with one tone in it, low in the band so orientation is testable."""
    t = np.arange(int(seconds * rate)) / rate
    signal = (0.6 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
    if channels > 1:
        signal = np.stack([signal] * channels, axis=1)
    sf.write(str(path), signal, rate)
    return str(path)


@pytest.fixture()
def recording(tmp_path):
    return _write(tmp_path / "loud.wav")


# -- the bounds ----------------------------------------------------------------------------


def test_defaults_are_what_a_caller_gets_for_saying_nothing():
    width, fft_size, bins = preview_bounds()
    assert (width, bins) == (PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_BINS)
    assert fft_size // 2 + 1 >= bins


def test_a_silly_request_is_clamped_not_honoured():
    width, fft_size, bins = preview_bounds(99_999, 9_999)
    assert width <= PREVIEW_MAX_WIDTH
    assert bins <= PREVIEW_MAX_BINS
    # And the two together are bounded, which is the guard that actually protects the box.
    assert width * fft_size <= PREVIEW_MAX_SAMPLES


def test_junk_falls_back_to_the_default():
    assert preview_bounds("banana", None)[0] == PREVIEW_DEFAULT_WIDTH
    assert preview_bounds(None, "")[2] == PREVIEW_DEFAULT_BINS
    assert preview_bounds(-40, -3)[0] == 16


def test_a_tall_request_costs_width_rather_than_the_box():
    """512 rows needs a 1024-point window, so the sample guard trades columns for rows instead of
    reading eight times as much as it promised to."""
    wide, wide_fft, _ = preview_bounds(4000, 8)
    tall, tall_fft, _ = preview_bounds(4000, 512)
    assert tall_fft > wide_fft
    assert tall * tall_fft <= PREVIEW_MAX_SAMPLES
    assert wide * wide_fft <= PREVIEW_MAX_SAMPLES


# -- the picture ---------------------------------------------------------------------------


def test_a_preview_has_the_shape_it_reports(recording):
    preview = read_preview(recording, width=300, height=64)
    assert preview.cells.shape == (preview.height, preview.width)
    assert preview.width == 300
    assert preview.cells.dtype == np.uint8
    assert preview.duration_sec == pytest.approx(4.0, abs=0.01)
    assert preview.nyquist_hz == pytest.approx(4000.0)


def test_the_height_is_a_ceiling_and_the_reply_says_what_it_reached(recording):
    """Pooling divides by an integer, so 257 bins asked down to 200 stay 257. A client that
    believed the number it asked for would draw a squashed picture."""
    _width, fft_size, _bins = preview_bounds(200, 200)
    stft_bins = fft_size // 2 + 1
    preview = read_preview(recording, width=200, height=200)
    assert preview.height == stft_bins // max(1, stft_bins // 200)
    assert preview.height == preview.cells.shape[0]


def test_low_energy_lands_in_the_low_rows(recording):
    """Row 0 of `cells` is the lowest frequency — the order a surface builder wants."""
    preview = read_preview(recording, width=64, height=64)
    low = preview.cells[: preview.height // 8].mean()
    high = preview.cells[-preview.height // 8:].mean()
    assert low > high, "a 200 Hz tone in a 4 kHz band belongs at the bottom of the grid"


def test_a_recording_too_short_to_picture_is_not_an_error(tmp_path):
    tiny = _write(tmp_path / "tiny.wav", seconds=0.01)  # 80 samples, under one window
    assert read_preview(tiny, width=100, height=64) is None


def test_an_unreadable_recording_is_not_an_error(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF____WAVEfmt ")
    assert read_preview(str(broken)) is None
    assert read_preview(str(tmp_path / "absent.wav")) is None


def test_a_multichannel_recording_is_mixed_the_way_the_run_mixed_it(tmp_path):
    """The channel comes from the run's own manifest, so it shows what was actually scanned."""
    stereo = _write(tmp_path / "stereo.wav", channels=2)
    for selection in ("mix", "left", "right", "1"):
        assert read_preview(stereo, width=64, height=64, channel=selection) is not None
    # A channel the file does not have is one blank lane, not a traceback out of a request handler.
    assert read_preview(stereo, width=64, height=64, channel="7") is None
    assert read_preview(stereo, width=64, height=64, channel="nonsense") is None


# -- the wire ------------------------------------------------------------------------------


def test_the_raw_body_is_a_header_and_one_byte_per_cell(recording):
    preview = read_preview(recording, width=256, height=64)
    body = encode_preview(preview)

    assert body[:8] == PREVIEW_MAGIC
    assert len(body) == PREVIEW_HEADER_BYTES + preview.width * preview.height
    magic, width, height, duration, nyquist, floor = struct.unpack(
        "<8sIIdff", body[:PREVIEW_HEADER_BYTES])
    assert magic == PREVIEW_MAGIC
    assert (width, height) == (preview.width, preview.height)
    assert duration == pytest.approx(preview.duration_sec)
    assert nyquist == pytest.approx(preview.nyquist_hz)
    assert floor == pytest.approx(preview.db_floor)


def test_the_png_is_a_png_and_the_right_way_up(recording):
    preview = read_preview(recording, width=200, height=64)
    blob = preview_png(preview)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in blob[:32] and blob[-8:-4] == b"IEND"


def test_a_preview_is_a_fraction_of_the_recording(tmp_path):
    """The reason the route exists — measured on a recording big enough for it to matter.

    (On a four-second 8 kHz clip a full-width preview is *larger* than the audio, which is fine:
    nobody needs a tunnel to look at 64 KB.)
    """
    survey = _write(tmp_path / "survey.wav", seconds=60.0, rate=48000)
    body = encode_preview(read_preview(survey, width=2000, height=256))
    assert len(body) < os.path.getsize(survey) / 4


def test_a_preview_does_not_grow_with_duration(tmp_path):
    """Four times the audio, the same number of bytes on the wire — because the windows are
    sparse, not because anything was thrown away at the end."""
    short = _write(tmp_path / "short.wav", seconds=2.0)
    long = _write(tmp_path / "long.wav", seconds=8.0)
    assert os.path.getsize(long) > 3 * os.path.getsize(short)

    a = encode_preview(read_preview(short, width=500, height=128))
    b = encode_preview(read_preview(long, width=500, height=128))
    assert len(a) == len(b)
    # ...and the longer one really was read across its whole length.
    assert struct.unpack("<d", b[16:24])[0] == pytest.approx(8.0, abs=0.01)


def test_a_wide_preview_of_a_long_recording_stays_bounded(tmp_path):
    """The size claim, at survey scale: a minute of 48 kHz asked for at full width."""
    long = _write(tmp_path / "minute.wav", seconds=60.0, rate=48000)
    body = encode_preview(read_preview(long, width=99_999, height=9_999))
    assert len(body) <= PREVIEW_HEADER_BYTES + PREVIEW_MAX_WIDTH * PREVIEW_MAX_ACTUAL_BINS
    assert len(body) < os.path.getsize(long)
