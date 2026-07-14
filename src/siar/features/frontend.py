# Vixen Intelligence c.2026
"""Waveform -> grid. The whole signal frontend, in torch.

One public entry point, :func:`build_grid`, which turns a mono waveform into the
``(frames, n_bins)`` ``float32`` picture a detector consumes:

    resample -> torch.stft -> power -> band (pooled-linear | mel) -> sqrt -> log1p

Two design notes worth keeping in mind:

**Why torch and not a hand-rolled numpy STFT.** ``torch.stft`` is batched, well-tested, and
runs on whatever device the model is on. SIAR has no bit-parity obligation to any other
codebase, so there is nothing to be gained by owning an FFT.

**Why ``log1p`` of amplitude and not dB.** dB needs a floor (``max(x, 1e-10)``) and that floor
becomes a hyperparameter that silently clips exactly the quiet structure an anomaly detector is
supposed to notice. ``log1p`` is defined at zero, compresses the loud end just as well, and has
no knob. Both feature modes end on the same scale, so a model can be compared across them.

The other thing this module owns is :func:`bin_edges_hz` — the inverse map from a grid row back
to a frequency in Hz. Box extraction depends on it entirely: get it wrong and every detection
is reported at the wrong pitch.
"""
from __future__ import annotations

import math

import numpy as np

from siar.features.spec import FeatureSpec

__all__ = ["bin_edges_hz", "build_grid", "mel_filterbank", "resample", "row_support_hz"]


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to mels (HTK convention)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    """Convert mels to Hz (HTK convention)."""
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def bin_edges_hz(spec: FeatureSpec) -> np.ndarray:
    """A monotone frequency axis for the grid — one edge between each pair of rows.

    This is the **rendering** axis: what to label the Y axis of a spectrogram with. It tiles
    ``[fmin_hz, fmax_hz]`` into ``n_bins`` non-overlapping intervals.

    It is *not* the right thing to measure a detection with when ``mode == "log_mel"``, because
    mel filters overlap and a monotone edge array cannot represent overlapping bands — row
    ``i``'s interval here is only the *rising half* of filter ``i``. Use :func:`row_support_hz`
    for boxes.

    Args:
        spec: The feature spec.

    Returns:
        A ``(n_bins + 1,)`` ``float64`` array, strictly ascending, spanning ``fmin_hz`` to
        ``fmax_hz``.

    Raises:
        ValueError: If ``spec.mode`` is unrecognised.
    """
    if spec.mode == "pooled_linear":
        # Mean-pooling groups the linear FFT bins with np.array_split, which distributes the
        # remainder across the first groups. Replicate that split exactly, then read off the
        # Hz edge of each group's first bin.
        hz_per_bin = spec.sample_rate / spec.fft_size
        lo = int(math.floor(spec.fmin_hz / hz_per_bin))
        hi = min(int(math.ceil(spec.fmax_hz / hz_per_bin)), spec.fft_size // 2 + 1)
        groups = np.array_split(np.arange(lo, hi), spec.n_bins)
        edges = [float(g[0]) * hz_per_bin for g in groups]
        edges.append(float(hi) * hz_per_bin)
        return np.asarray(edges, dtype=np.float64)

    if spec.mode == "log_mel":
        m = np.linspace(
            _hz_to_mel(spec.fmin_hz), _hz_to_mel(spec.fmax_hz), spec.n_bins + 2, dtype=np.float64
        )
        f = _mel_to_hz(m)  # (n_bins + 2,) — filter i spans f[i]..f[i+2], peaking at f[i+1]
        # Rows tile on the filter *centres*: row i owns [f[i+1], f[i+2]) as its share of the
        # axis, with the first edge pulled down to fmin so the axis starts where the band does.
        edges = np.empty(spec.n_bins + 1, dtype=np.float64)
        edges[0] = f[0]
        edges[1:] = f[2 : spec.n_bins + 2]
        return edges

    raise ValueError(f"unknown feature mode {spec.mode!r}")


def row_support_hz(spec: FeatureSpec) -> np.ndarray:
    """The frequency range each grid row actually responds to.

    This is the map **box extraction** uses. A detection occupying rows ``m0..m1`` inclusive is
    reported as ``f_low = support[m0, 0]``, ``f_high = support[m1, 1]``.

    Unlike :func:`bin_edges_hz`, consecutive rows here may **overlap** — which is the truth for
    a mel filterbank, where filter ``i`` has support ``[f[i], f[i+2]]`` and shares half of that
    with each of its neighbours. Reporting only the non-overlapping share would understate every
    detection's bandwidth: a 6 kHz tone would come back labelled 5573–5926 Hz, a band that does
    not contain it. Taking the full support instead errs *wide*, which is the right direction
    for a bounding box — it should contain the event, not bisect it.

    For ``pooled_linear`` the pooling groups genuinely do not overlap, so this returns the same
    intervals as :func:`bin_edges_hz`.

    Args:
        spec: The feature spec.

    Returns:
        A ``(n_bins, 2)`` ``float64`` array of ``[low_hz, high_hz]`` per row. Both columns are
        ascending; ``support[i, 1] >= support[i, 0]``.

    Raises:
        ValueError: If ``spec.mode`` is unrecognised.
    """
    if spec.mode == "pooled_linear":
        edges = bin_edges_hz(spec)
        return np.stack([edges[:-1], edges[1:]], axis=1)

    if spec.mode == "log_mel":
        m = np.linspace(
            _hz_to_mel(spec.fmin_hz), _hz_to_mel(spec.fmax_hz), spec.n_bins + 2, dtype=np.float64
        )
        f = _mel_to_hz(m)
        return np.stack([f[: spec.n_bins], f[2 : spec.n_bins + 2]], axis=1)

    raise ValueError(f"unknown feature mode {spec.mode!r}")


def mel_filterbank(spec: FeatureSpec) -> np.ndarray:
    """Build a triangular mel filterbank over the spec's band.

    Slaney-style area normalisation is deliberately *not* applied: it scales each filter by its
    bandwidth, which makes high-frequency filters quieter and biases reconstruction error toward
    the bottom of the spectrum. SIAR whitens per-bin downstream anyway
    (:mod:`siar.detect.normalise`), so an unnormalised bank keeps the two concerns separate.

    Args:
        spec: The feature spec. Must have ``mode == "log_mel"``.

    Returns:
        A ``(n_bins, fft_size // 2 + 1)`` ``float32`` matrix, to be applied to the power
        spectrum as ``power @ fb.T``.
    """
    n_fft_bins = spec.fft_size // 2 + 1
    fft_freqs = np.linspace(0.0, spec.sample_rate / 2.0, n_fft_bins, dtype=np.float64)

    m = np.linspace(
        _hz_to_mel(spec.fmin_hz), _hz_to_mel(spec.fmax_hz), spec.n_bins + 2, dtype=np.float64
    )
    f = _mel_to_hz(m)

    fb = np.zeros((spec.n_bins, n_fft_bins), dtype=np.float64)
    for i in range(spec.n_bins):
        lo, mid, hi = f[i], f[i + 1], f[i + 2]
        if mid > lo:
            rising = (fft_freqs - lo) / (mid - lo)
            fb[i] = np.maximum(0.0, np.minimum(1.0, rising))
        if hi > mid:
            falling = (hi - fft_freqs) / (hi - mid)
            fb[i] = np.minimum(fb[i], np.maximum(0.0, falling))
        fb[i][(fft_freqs < lo) | (fft_freqs > hi)] = 0.0
    return fb.astype(np.float32)


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a mono signal, if needed.

    Uses polyphase filtering (``scipy.signal.resample_poly``), which is exact for the rational
    ratios that real sample rates give (48000/44100 = 160/147) and does not assume the signal is
    periodic the way an FFT resampler does.

    Args:
        samples: 1-D ``float32`` mono signal.
        sr_in: The signal's current sample rate.
        sr_out: The desired sample rate.

    Returns:
        A 1-D ``float32`` signal at ``sr_out``. The input array is returned unchanged if the
        rates already match.
    """
    if sr_in == sr_out:
        return samples.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return resample_poly(samples, up, down).astype(np.float32)


def build_grid(samples: np.ndarray, sample_rate: int, spec: FeatureSpec) -> np.ndarray:
    """Turn a mono waveform into the grid a detector sees.

    Args:
        samples: 1-D mono signal, any sample rate.
        sample_rate: The signal's sample rate in Hz.
        spec: The feature spec. The signal is resampled to ``spec.sample_rate`` first.

    Returns:
        A ``(frames, n_bins)`` ``float32`` array, where
        ``frames = 1 + (n - fft_size) // hop_size`` (zero if the signal is shorter than one
        frame). Values are ``log1p`` of a band amplitude, so they are non-negative.

    Raises:
        ValueError: If ``spec.mode`` is unrecognised.
    """
    import torch

    sig = resample(np.asarray(samples, dtype=np.float32).ravel(), sample_rate, spec.sample_rate)
    if sig.size < spec.fft_size:
        return np.zeros((0, spec.n_bins), dtype=np.float32)

    x = torch.from_numpy(np.ascontiguousarray(sig))
    win = torch.hann_window(spec.fft_size, dtype=torch.float32)
    # center=False: no reflection padding, so every frame contains only real signal and frame f
    # starts at sample f * hop. Padding would invent structure at the file edges, and an anomaly
    # detector would dutifully flag it.
    z = torch.stft(
        x,
        n_fft=spec.fft_size,
        hop_length=spec.hop_size,
        win_length=spec.fft_size,
        window=win,
        center=False,
        return_complex=True,
    )
    power = (z.real**2 + z.imag**2).transpose(0, 1)  # (frames, fft_bins)

    if spec.mode == "pooled_linear":
        hz_per_bin = spec.sample_rate / spec.fft_size
        lo = int(math.floor(spec.fmin_hz / hz_per_bin))
        hi = min(int(math.ceil(spec.fmax_hz / hz_per_bin)), power.shape[1])
        band = power[:, lo:hi]
        groups = np.array_split(np.arange(band.shape[1]), spec.n_bins)
        pooled = torch.stack([band[:, g[0] : g[-1] + 1].mean(dim=1) for g in groups], dim=1)
    elif spec.mode == "log_mel":
        fb = torch.from_numpy(mel_filterbank(spec))  # (n_bins, fft_bins)
        pooled = power @ fb.T
    else:
        raise ValueError(f"unknown feature mode {spec.mode!r}")

    # Power -> amplitude -> log1p. sqrt first so both modes land on the same amplitude scale,
    # and so the log compresses a 2x loudness change the same way wherever it happens.
    grid = torch.log1p(torch.sqrt(torch.clamp(pooled, min=0.0)))
    return grid.numpy().astype(np.float32, copy=False)
