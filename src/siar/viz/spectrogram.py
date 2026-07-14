# Vixen Intelligence c.2026
"""Rendering a grid to a PNG the dashboard can draw boxes on.

The important property, and the reason this is not just "make a picture":

    **one PNG pixel == one grid cell.**

The image is exactly ``frames`` wide and ``n_bins`` tall, with no resampling, no axes, no
margins. The browser then scales it with CSS. That means a detection at grid cell
``(frame_lo, bin_lo)`` maps to an image pixel by simple proportion, with no interpolation and no
accumulated drift — the JS never has to know anything about FFT sizes or mel filterbanks to put a
box in the right place. Render the image "nicely", with padding or a fitted aspect ratio, and
that guarantee is gone and the boxes creep.

Row 0 of a PNG is the top of the image, and the top of a spectrogram is the *highest* frequency,
so the grid is flipped vertically on the way out.
"""
from __future__ import annotations

import numpy as np

from siar.viz.colormap import apply_colormap
from siar.viz.png import encode_png

__all__ = ["render_error_map", "render_spectrogram"]


def render_spectrogram(grid: np.ndarray, *, colormap: str = "viridis") -> bytes:
    """Render a feature grid as a PNG.

    Args:
        grid: ``(frames, n_bins)`` ``float32`` — the picture the model saw.
        colormap: A name from :data:`siar.viz.colormap.COLORMAPS`.

    Returns:
        PNG bytes, ``frames`` px wide and ``n_bins`` px tall.

    Raises:
        ValueError: If ``grid`` is not 2-D.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"grid must be 2-D (frames, n_bins), got shape {g.shape}")
    if g.size == 0:
        return encode_png(np.zeros((1, 1, 3), dtype=np.uint8))

    # (frames, bins) -> (bins, frames) so time runs left-to-right, then flip so high frequency
    # is at the top.
    image = np.flipud(g.T)
    return encode_png(apply_colormap(image, name=colormap))


def render_error_map(z_map: np.ndarray, *, threshold: float | None = None) -> bytes:
    """Render a z-map as a PNG, for the dashboard's "why did it fire here?" overlay.

    Scaled in log space. A z-map's dynamic range is enormous — a real detection can sit four
    orders of magnitude above the threshold — so a linear scale renders as a black rectangle with
    one white dot, which tells the viewer nothing about the structure around the event.

    Args:
        z_map: ``(frames, n_bins)`` per-bin z-scores.
        threshold: If given, everything below it is floored, so the image shows only what the
            detector actually considered.

    Returns:
        PNG bytes, ``frames`` px wide and ``n_bins`` px tall.
    """
    z = np.asarray(z_map, dtype=np.float64)
    if z.ndim != 2 or z.size == 0:
        return encode_png(np.zeros((1, 1, 3), dtype=np.uint8))

    floor = 0.0 if threshold is None else float(threshold)
    shown = np.log1p(np.maximum(z - floor, 0.0))
    image = np.flipud(shown.T)
    return encode_png(apply_colormap(image, name="magma", vmin=0.0))
