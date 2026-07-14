# Vixen Intelligence c.2026
"""Colour maps as 256-entry lookup tables.

Two, and they are chosen rather than decorative:

``viridis`` for the spectrogram. Perceptually uniform (equal steps in value look like equal steps
in colour) and legible to the colour-blind — which "jet" and its relatives are emphatically not.
A non-uniform map invents structure that is not in the data, and on a spectrogram people then
point at the artefact.

``magma`` for the error map, so the two layers are instantly distinguishable when toggled in the
dashboard.

Built by interpolating the maps' published anchor stops; visually indistinguishable from
matplotlib's tables, at zero dependency.
"""
from __future__ import annotations

import numpy as np

__all__ = ["COLORMAPS", "apply_colormap", "lut"]

_VIRIDIS_STOPS = (
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
)

_MAGMA_STOPS = (
    (0.001, 0.000, 0.014), (0.113, 0.065, 0.277), (0.302, 0.071, 0.485),
    (0.482, 0.146, 0.531), (0.665, 0.215, 0.500), (0.844, 0.298, 0.421),
    (0.956, 0.469, 0.372), (0.992, 0.652, 0.472), (0.996, 0.827, 0.622),
    (0.987, 0.991, 0.749),
)

_STOPS = {"viridis": _VIRIDIS_STOPS, "magma": _MAGMA_STOPS}

#: Available colour-map names.
COLORMAPS: tuple[str, ...] = tuple(_STOPS)


def lut(name: str = "viridis") -> np.ndarray:
    """Build a 256-entry RGB lookup table.

    Args:
        name: One of :data:`COLORMAPS`.

    Returns:
        A ``(256, 3)`` ``uint8`` array, dark at index 0.

    Raises:
        ValueError: If ``name`` is not a known colour map.
    """
    try:
        stops = np.asarray(_STOPS[name], dtype=np.float64)
    except KeyError:
        raise ValueError(f"unknown colormap {name!r}; choose from {COLORMAPS}") from None

    src = np.linspace(0.0, 1.0, stops.shape[0])
    dst = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(dst, src, stops[:, c]) for c in range(3)]
    return (np.stack(channels, axis=1) * 255.0 + 0.5).astype(np.uint8)


def apply_colormap(
    values: np.ndarray,
    *,
    name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map a 2-D array of scalars to RGB.

    Args:
        values: ``(height, width)`` of numbers.
        name: One of :data:`COLORMAPS`.
        vmin: Value mapped to the bottom of the map. Defaults to the 1st percentile.
        vmax: Value mapped to the top. Defaults to the 99th percentile.

            Percentiles rather than min/max because a single hot pixel — which is exactly what an
            anomaly *is* — would otherwise compress everything else into the bottom of the range
            and the spectrogram would render as a black rectangle with one bright dot.

    Returns:
        ``(height, width, 3)`` ``uint8``.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros((*v.shape, 3), dtype=np.uint8)

    lo = float(np.percentile(v, 1.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(v, 99.0)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-9

    norm = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    return lut(name)[idx]
