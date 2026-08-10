# Vixen Intelligence c.2026
"""The 200x64 spectrogram thumbnail the app shows on every lane.

Without one, a folder opened in IDent Dynamics is a column of bare bars: the user clicks a lane,
waits for a decode, and only then finds out whether that recording was worth opening. With one,
the strip reads as a waterfall and the interesting files are visible at a glance. Since
siar-app has already decoded every file to scan it, rendering the thumbnail costs one extra
sparse transform per recording — a few milliseconds against seconds of scanning.

A faithful port of the app's ``js/thumbnail.js``, by way of ``brahma/viz/thumbnail.py``:

* a *sparse* STFT — :data:`THUMB_FRAMES` hann-windowed :data:`FFT_SIZE`-point transforms placed
  evenly across the signal, not a fully-overlapped one (200 transforms, not ~110k);
* dB normalisation against the file's own peak with a :data:`DB_FLOOR` floor;
* average-pooling the frequency axis down to :data:`THUMB_BINS`;
* the app's polynomial viridis (see :mod:`siarapp.viz.colormap`);
* y flipped so the lowest frequency sits at the bottom.

Three implementations of this picture now exist — browser, brahma box, and here. They are
supposed to be indistinguishable, and a lane that disagrees with its neighbours is worse than a
blank one. Change any of them, change all three.
"""
from __future__ import annotations

import numpy as np

from siarapp.viz.colormap import viridis_lut
from siarapp.viz.png import encode_png

__all__ = [
    "DB_FLOOR",
    "FFT_SIZE",
    "THUMB_BINS",
    "THUMB_FRAMES",
    "render_rgb",
    "thumbnail_png",
]

#: Analysis frames across the signal — the thumbnail's width in pixels.
THUMB_FRAMES = 200

#: Frequency bins after average-pooling — the thumbnail's height.
THUMB_BINS = 64

#: FFT size. Small, so each of the 200 transforms is nearly free.
FFT_SIZE = 256

#: dB floor for normalisation.
DB_FLOOR = -90.0

#: Guard against ``log10(0)``. The app's ``dsp/scale.js`` uses the same constant.
_EPSILON = 1e-20


def _sparse_magnitude_grid(signal: np.ndarray, fft_size: int, num_frames: int) -> np.ndarray:
    """Place ``num_frames`` windows evenly across ``signal`` and transform each one.

    The hop here is far *larger* than the window, which is why this does not go through
    :mod:`siarapp.dsp.stft` — that requires ``hop <= fft_size``, as does the browser's.

    Args:
        signal: Mono signal.
        fft_size: Power-of-two window length.
        num_frames: Desired frame count. A signal exactly ``fft_size`` long yields one frame,
            not ``num_frames`` identical ones.

    Returns:
        A ``(frames, fft_size // 2 + 1)`` ``float32`` array of linear magnitudes.
    """
    window = np.hanning(fft_size)  # n-1 denominator, matching the app's `hann`
    max_start = max(0, signal.shape[0] - fft_size)
    frames = 1 if max_start == 0 else num_frames

    starts = np.zeros(frames, dtype=np.int64)
    if frames > 1:
        t = np.arange(frames, dtype=np.float64) / (frames - 1)
        starts = np.floor(t * max_start).astype(np.int64)

    idx = starts[:, None] + np.arange(fft_size, dtype=np.int64)[None, :]
    block = signal[idx].astype(np.float64) * window[None, :]
    return np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)


def _normalise_db(grid: np.ndarray, db_floor: float = DB_FLOOR) -> np.ndarray:
    """Map linear magnitudes to ``[0, 1]`` on a dB scale relative to the grid's peak.

    ``20 * log10(m / peak)`` clamped to ``[db_floor, 0]`` and stretched onto ``[0, 1]`` — the
    ``scale: 'db'`` branch of the app's ``normalise``. An all-zero grid normalises to zeros
    rather than dividing by zero, so a silent recording gives a flat dark lane instead of NaNs.
    """
    peak = float(grid.max()) if grid.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    db = 20.0 * np.log10(grid.astype(np.float64) / peak + _EPSILON)
    return np.clip((db - db_floor) / -db_floor, 0.0, 1.0).astype(np.float32)


def render_rgb(signal: np.ndarray) -> np.ndarray | None:
    """The thumbnail as an ``(out_bins, frames, 3)`` ``uint8`` image.

    Args:
        signal: Mono signal.

    Returns:
        The raster, or ``None`` when the signal is shorter than one analysis window — a
        recording too short to picture, which is a fact rather than an error.
    """
    sig = np.asarray(signal)
    if sig.shape[0] < FFT_SIZE:
        return None

    normalised = _normalise_db(_sparse_magnitude_grid(sig, FFT_SIZE, THUMB_FRAMES))
    frames, stft_bins = normalised.shape

    # Average-pool the frequency axis. Remainder bins (129 does not divide by 64) are dropped
    # from the high end — they land in the spectrogram's top sliver, rarely diagnostic.
    bin_factor = max(1, stft_bins // THUMB_BINS)
    out_bins = stft_bins // bin_factor
    pooled = (
        normalised[:, : out_bins * bin_factor]
        .reshape(frames, out_bins, bin_factor)
        .mean(axis=2)
    )

    lut_idx = np.clip(np.floor(pooled * 255.0 + 0.5), 0, 255).astype(np.uint8)
    rgb = viridis_lut[lut_idx]  # (frames, out_bins, 3)
    # Transpose to (bins, frames), then flip so the lowest frequency is the bottom row.
    return np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1])


def thumbnail_png(signal: np.ndarray) -> bytes | None:
    """Render a mono signal straight to PNG bytes, or ``None`` if it is too short."""
    rgb = render_rgb(signal)
    return None if rgb is None else encode_png(rgb)
