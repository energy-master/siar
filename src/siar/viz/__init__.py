# Vixen Intelligence c.2026
"""Rendering grids and error maps to PNGs, with no image-library dependency."""
from __future__ import annotations

from siar.viz.colormap import COLORMAPS, apply_colormap, lut
from siar.viz.png import encode_png
from siar.viz.spectrogram import render_error_map, render_spectrogram

__all__ = [
    "COLORMAPS",
    "apply_colormap",
    "encode_png",
    "lut",
    "render_error_map",
    "render_spectrogram",
]
