# Vixen Intelligence c.2026
"""The viridis look-up table the web app draws with.

Not matplotlib's viridis. The app's WebGL colour map (``js/webgl/colormap.js``) evaluates Matt
Zucker's 6th-degree polynomial fit (Shadertoy WlfXRN, public domain), and matplotlib's sampled
table differs from it by a couple of least-significant bits per channel. That is invisible in
isolation and obvious in a strip: a lane rendered here sitting next to a lane rendered by the
browser, in the same column, faintly the wrong colour.

Copied from ``brahma/viz/thumbnail.py``, which copied it from the app. All three must agree.
"""
from __future__ import annotations

import numpy as np

__all__ = ["VIRIDIS_COEFFS", "build_lut", "viridis_lut"]

#: Matt Zucker's polynomial coefficients, degree 0 first, one ``(r, g, b)`` triple per degree.
VIRIDIS_COEFFS = (
    (0.2777273272234177, 0.005407344544966578, 0.3340998053353061),
    (0.1050930431085774, 1.404613529898575, 1.384590162594685),
    (-0.3308618287255563, 0.214847559468213, 0.09509516302823659),
    (-4.634230498983486, -5.799100973351585, -19.33244095627987),
    (6.228269936347081, 14.17993336680509, 56.69055260068105),
    (4.776384997670288, -13.74514537774601, -65.35303263337234),
    (-5.435455855934631, 4.645852612178535, 26.3124352495832),
)


def build_lut() -> np.ndarray:
    """Build the 256-entry RGB table, matching the browser's ``buildLut('viridis')``.

    Returns:
        A ``(256, 3)`` ``uint8`` array. Each channel is ``round(clamp01(poly(t)) * 255)`` at
        ``t = i / 255``, evaluated by Horner's method exactly as the app's ``evalPoly`` does.
    """
    t = np.arange(256, dtype=np.float64) / 255.0
    c = np.asarray(VIRIDIS_COEFFS, dtype=np.float64)  # (7, 3)
    acc = np.empty((256, 3), dtype=np.float64)
    for k in range(3):
        v = np.full(256, c[6, k])
        for deg in range(5, -1, -1):
            v = c[deg, k] + t * v
        acc[:, k] = v
    np.clip(acc, 0.0, 1.0, out=acc)
    # JS Math.round is half-away-from-zero; numpy's rint is half-to-even. Values here are
    # non-negative, so floor(x + 0.5) reproduces the browser's rounding exactly.
    return np.floor(acc * 255.0 + 0.5).astype(np.uint8)


#: The table, built once at import.
viridis_lut = build_lut()
