# Vixen Intelligence c.2026
"""Window functions for short-time Fourier analysis.

A port of the web app's ``js/dsp/windows.js``, coefficient for coefficient. It matters that
this is a port and not "numpy's equivalent window": every generator here divides by ``n - 1``
(a *symmetric* window), where ``scipy.signal.get_window`` defaults to the periodic ``n``
divisor. The difference is one sample of taper, which shifts every magnitude in the grid by a
fraction of a dB — invisible on a picture, but enough to move a scanner's box by a bin when a
cell sits on its threshold.

``numpy.hanning`` and ``numpy.hamming`` happen to use the same ``n - 1`` convention, so those
two are thin wrappers; blackman is spelled out because numpy's uses the classic 0.42/0.5/0.08
coefficients under a different name and the equivalence is worth being explicit about rather
than assumed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["WINDOWS", "hann", "hamming", "blackman", "rectangular", "by_name"]

_TAU = 2.0 * np.pi


def hann(n: int) -> np.ndarray:
    """Symmetric Hann window of length ``n`` (``float32``, values in ``[0, 1]``)."""
    i = np.arange(n, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(_TAU * i / (n - 1)))).astype(np.float32)


def hamming(n: int) -> np.ndarray:
    """Symmetric Hamming window of length ``n``."""
    i = np.arange(n, dtype=np.float64)
    return (0.54 - 0.46 * np.cos(_TAU * i / (n - 1))).astype(np.float32)


def blackman(n: int) -> np.ndarray:
    """Symmetric Blackman window of length ``n`` (textbook alpha = 0.16)."""
    x = _TAU * np.arange(n, dtype=np.float64) / (n - 1)
    return (0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2.0 * x)).astype(np.float32)


def rectangular(n: int) -> np.ndarray:
    """Rectangular (boxcar) window of length ``n`` — all ones."""
    return np.ones(n, dtype=np.float32)


#: Every window the CLI accepts, by the name the web app uses for it.
WINDOWS: dict[str, Callable[[int], np.ndarray]] = {
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
    "rectangular": rectangular,
}


def by_name(name: str) -> Callable[[int], np.ndarray]:
    """Look up a window generator.

    Args:
        name: One of :data:`WINDOWS`.

    Returns:
        The generator.

    Raises:
        ValueError: If ``name`` is not a known window. The message lists the valid ones, since
            this is nearly always a typo on a command line.
    """
    try:
        return WINDOWS[name]
    except KeyError:
        raise ValueError(
            f"unknown window {name!r}; choose from {', '.join(sorted(WINDOWS))}"
        ) from None
