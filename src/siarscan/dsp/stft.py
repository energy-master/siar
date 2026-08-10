# Vixen Intelligence c.2026
"""Short-time Fourier transform of a real mono signal.

A port of the web app's ``js/dsp/stft.js``: the signal is cut into overlapping frames of
``fft_size`` samples, each multiplied by the window, transformed, and reduced to magnitudes.
Two details are load-bearing and both are inherited rather than chosen:

* **Frame count is** ``floor((n - fft) / hop) + 1``, and zero when the signal is shorter than
  one window. No padding, no centring, no partial final frame. A scanner reports a box at
  ``frame * hop / sample_rate``; if the frame grid here started half a window earlier than the
  browser's — which is what ``center=True`` in most STFT libraries would do — every box this
  CLI wrote would land in the wrong place when opened in the app.
* **Magnitudes are linear**, ``|rfft|``, frame-major, ``float32``. The scanners take their own
  logs; which log, and against what reference, is part of what each one does.

The transform runs in blocks rather than one batch. The obvious vectorisation — gather every
frame into an ``(frames, fft_size)`` array and call ``rfft`` once — allocates a float64 block
of ``frames * fft * 8`` bytes, which for ten minutes at 96 kHz is about 1.8 GB *on top of* the
grid it is filling. Blocking caps that at :data:`_BLOCK_FRAMES` frames and costs nothing
measurable in time.
"""
from __future__ import annotations

import numpy as np

from siarscan.dsp.windows import by_name

__all__ = ["StftResult", "stft", "frame_count", "grid_bytes"]

#: Frames transformed per batch. 512 frames of a 4096-point FFT is a 16 MB working block —
#: large enough that the per-call overhead of ``rfft`` disappears, small enough to ignore.
_BLOCK_FRAMES = 512


class StftResult:
    """The transform's output.

    Attributes:
        magnitudes: ``(frames, bins)`` ``float32`` linear magnitudes, frame-major.
        frames: Number of analysis frames.
        bins: ``fft_size // 2 + 1`` — DC through Nyquist inclusive.
        fft_size: FFT size used.
        hop_size: Hop in samples between frames.
        window_name: The window applied.
    """

    __slots__ = ("magnitudes", "frames", "bins", "fft_size", "hop_size", "window_name")

    def __init__(self, magnitudes, frames, bins, fft_size, hop_size, window_name):
        self.magnitudes = magnitudes
        self.frames = int(frames)
        self.bins = int(bins)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.window_name = str(window_name)


def frame_count(n_samples: int, fft_size: int, hop_size: int) -> int:
    """Frames a signal of ``n_samples`` yields — the browser's formula, exactly.

    Args:
        n_samples: Signal length in samples.
        fft_size: FFT size.
        hop_size: Hop in samples.

    Returns:
        The frame count; ``0`` when the signal is shorter than one window.
    """
    if n_samples < fft_size:
        return 0
    return (n_samples - fft_size) // hop_size + 1


def grid_bytes(n_samples: int, fft_size: int, hop_size: int) -> int:
    """Bytes the magnitude grid for this signal will occupy.

    The CLI prints this before starting a long file, because "it went quiet for four minutes
    and then the machine swapped" is the failure mode of an STFT sized by accident.
    """
    return frame_count(n_samples, fft_size, hop_size) * (fft_size // 2 + 1) * 4


def stft(
    signal: np.ndarray,
    *,
    fft_size: int,
    hop_size: int,
    window_name: str = "hann",
) -> StftResult:
    """Transform a real mono signal.

    Args:
        signal: 1-D real signal (any float dtype; converted to float64 for the transform).
        fft_size: Power-of-two FFT size, e.g. 1024.
        hop_size: Hop in samples, ``1 <= hop_size <= fft_size``.
        window_name: See :mod:`siarscan.dsp.windows`.

    Returns:
        The :class:`StftResult`. A signal shorter than one window yields ``frames == 0`` and an
        empty ``(0, bins)`` grid rather than an error — one too-short recording in a folder is
        a fact about the folder, not a reason to stop.

    Raises:
        ValueError: If ``fft_size`` is not a power of two, ``hop_size`` is out of range, or the
            signal is not 1-D.
    """
    if fft_size < 2 or (fft_size & (fft_size - 1)) != 0:
        raise ValueError(f"fft_size {fft_size} is not a power of two")
    if hop_size < 1 or hop_size > fft_size:
        raise ValueError(f"hop_size {hop_size} out of range [1, {fft_size}]")

    sig = np.asarray(signal)
    if sig.ndim != 1:
        raise ValueError(f"expected a 1-D mono signal, got shape {sig.shape}")

    window = by_name(window_name)(fft_size).astype(np.float64)
    bins = fft_size // 2 + 1
    frames = frame_count(sig.shape[0], fft_size, hop_size)

    magnitudes = np.empty((frames, bins), dtype=np.float32)
    if frames == 0:
        return StftResult(magnitudes, 0, bins, fft_size, hop_size, window_name)

    sig64 = sig.astype(np.float64, copy=False)
    taps = np.arange(fft_size, dtype=np.int64)
    for start in range(0, frames, _BLOCK_FRAMES):
        stop = min(start + _BLOCK_FRAMES, frames)
        offsets = (np.arange(start, stop, dtype=np.int64) * hop_size)[:, None]
        block = sig64[offsets + taps[None, :]] * window[None, :]
        magnitudes[start:stop] = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)

    return StftResult(magnitudes, frames, bins, fft_size, hop_size, window_name)
