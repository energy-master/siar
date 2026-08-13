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

The four steps are public — :func:`sparse_frame_starts`, :func:`normalise_db`, :func:`pool_bins`,
:func:`colourise` — because :mod:`siarapp.serve.preview` draws the same picture at a larger size
for a recording it is streaming to a browser, and a preview that disagreed with the lane strip
above it would be a fourth implementation to keep in step. :func:`sparse_frame_starts` is the one
that matters there: knowing *where* the windows fall without holding the signal is what lets a
remote viewer seek to each one instead of decoding forty minutes of audio.
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
    "colourise",
    "hann_window",
    "normalise_db",
    "pool_bins",
    "render_rgb",
    "sparse_frame_starts",
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


def sparse_frame_starts(n_samples: int, fft_size: int, num_frames: int) -> np.ndarray:
    """Where the ``num_frames`` windows begin, spread evenly across ``n_samples``.

    Public, and separate from the transform, because it is the whole of what makes this picture
    affordable on a long recording: a caller holding a file rather than a signal — the remote
    viewer in :mod:`siarapp.serve.preview` — can seek to each of these offsets and read one window,
    instead of decoding forty minutes of audio to draw half a megabyte of it.

    Args:
        n_samples: Length of the signal in samples.
        fft_size: Power-of-two window length.
        num_frames: Desired frame count.

    Returns:
        A ``(frames,)`` ``int64`` array of sample offsets. Length is ``1`` when the signal is
        exactly one window long — that yields one frame, not ``num_frames`` identical ones — and
        empty when it is shorter than a window.
    """
    if n_samples < fft_size:
        return np.zeros(0, dtype=np.int64)
    max_start = max(0, n_samples - fft_size)
    frames = 1 if max_start == 0 else max(1, int(num_frames))
    if frames == 1:
        return np.zeros(1, dtype=np.int64)
    t = np.arange(frames, dtype=np.float64) / (frames - 1)
    return np.floor(t * max_start).astype(np.int64)


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
    starts = sparse_frame_starts(signal.shape[0], fft_size, num_frames)
    idx = starts[:, None] + np.arange(fft_size, dtype=np.int64)[None, :]
    block = signal[idx].astype(np.float64) * hann_window(fft_size)[None, :]
    return np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)


def hann_window(fft_size: int) -> np.ndarray:
    """The window every one of these transforms is taken through.

    ``np.hanning``'s ``n-1`` denominator, which is what the app's ``hann`` uses — named here so
    the remote viewer's per-window read cannot drift onto a different one.
    """
    return np.hanning(fft_size)


def normalise_db(grid: np.ndarray, db_floor: float = DB_FLOOR) -> np.ndarray:
    """Map linear magnitudes to ``[0, 1]`` on a dB scale relative to the grid's peak.

    ``20 * log10(m / peak)`` clamped to ``[db_floor, 0]`` and stretched onto ``[0, 1]`` — the
    ``scale: 'db'`` branch of the app's ``normalise``. An all-zero grid normalises to zeros
    rather than dividing by zero, so a silent recording gives a flat dark lane instead of NaNs.

    Args:
        grid: ``(frames, bins)`` linear magnitudes.
        db_floor: Decibels below the peak that map to 0.

    Returns:
        A ``float32`` array of the same shape, in ``[0, 1]``.
    """
    peak = float(grid.max()) if grid.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    db = 20.0 * np.log10(grid.astype(np.float64) / peak + _EPSILON)
    return np.clip((db - db_floor) / -db_floor, 0.0, 1.0).astype(np.float32)


def pool_bins(grid: np.ndarray, target_bins: int) -> np.ndarray:
    """Average-pool the frequency axis of a ``(frames, bins)`` grid down towards ``target_bins``.

    The factor is an integer, so the result is *at least* ``target_bins`` wide and rarely exactly
    it — 129 bins asked down to 64 gives 64, but asked down to 100 gives 129. Every caller has to
    read the height it actually got rather than the one it asked for.

    Remainder bins are dropped from the high end: they land in the spectrogram's top sliver, and
    are rarely what anybody is looking at.

    Args:
        grid: ``(frames, bins)`` values.
        target_bins: The bin count to aim at, as a ceiling on the pooling factor.

    Returns:
        A ``(frames, out_bins)`` array of the same dtype family, with
        ``out_bins = bins // max(1, bins // target_bins)``.
    """
    frames, bins = grid.shape
    factor = max(1, bins // max(1, int(target_bins)))
    out_bins = max(1, bins // factor)
    return grid[:, : out_bins * factor].reshape(frames, out_bins, factor).mean(axis=2)


def colourise(pooled: np.ndarray) -> np.ndarray:
    """A normalised ``(frames, bins)`` grid as an ``(bins, frames, 3)`` viridis raster.

    Shared with the remote viewer's preview so that a lane's thumbnail and the picture drawn when
    that lane is opened are the same picture at two sizes, not two renderings that nearly agree.
    """
    lut_idx = np.clip(np.floor(pooled * 255.0 + 0.5), 0, 255).astype(np.uint8)
    rgb = viridis_lut[lut_idx]  # (frames, out_bins, 3)
    # Transpose to (bins, frames), then flip so the lowest frequency is the bottom row.
    return np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1])


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
    normalised = normalise_db(_sparse_magnitude_grid(sig, FFT_SIZE, THUMB_FRAMES))
    return colourise(pool_bins(normalised, THUMB_BINS))


def thumbnail_png(signal: np.ndarray) -> bytes | None:
    """Render a mono signal straight to PNG bytes, or ``None`` if it is too short."""
    rgb = render_rgb(signal)
    return None if rgb is None else encode_png(rgb)
