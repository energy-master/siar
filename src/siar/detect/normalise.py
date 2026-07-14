# Vixen Intelligence c.2026
"""Turning a raw error map into a comparable one.

Reconstruction error is not comparable across frequency. A busy, high-variance band reconstructs
worse than a quiet one *in the normal case*, so a raw error map is dominated by wherever the
corpus happens to be loud — and every box lands in the same part of the spectrum regardless of
what is actually anomalous. The fix is to whiten **per bin**: learn, on data the model did not
train on, what error is *typical for that bin*, and report each pixel as a robust z-score
against its own bin's baseline.

The failure mode this module has to defend against is division by a collapsed scale. In a
near-silent band the MAD of the error can be ~1e-9, and a pixel a hair above the median then
scores z = 500,000. That pixel is not anomalous, it is just in a quiet bin — but it will
outrank every real detection and win the top slot in the dashboard. So the per-bin MAD is
floored against the *global* MAD (:data:`MIN_MAD_FRACTION`): a bin is never allowed to claim it
is arbitrarily more certain than the map as a whole.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MIN_MAD_FRACTION", "Baseline", "fit_baseline"]

#: A per-bin MAD may not be smaller than this fraction of the global MAD. Without a floor, a
#: quiet bin's near-zero scale turns trivial error into an enormous z-score. See the module
#: docstring — this single constant is the difference between a usable ranking and a garbage one.
MIN_MAD_FRACTION = 0.05

#: Scaling so that a z of 1.0 is one "robust sigma". MAD * 1.4826 estimates sigma for Gaussian
#: data; error distributions are heavier-tailed than that, which is exactly why the threshold is
#: fitted empirically (:mod:`siar.detect.threshold`) rather than assumed.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class Baseline:
    """What "normal error" looks like, per frequency bin.

    Fitted on held-out calibration recordings and frozen onto the trained model, so inference on
    a new folder reproduces exactly the same normalisation.

    Attributes:
        median: ``(n_bins,)`` ``float64`` — the typical error in each bin.
        scale: ``(n_bins,)`` ``float64`` — the robust spread of error in each bin, already
            floored and sigma-scaled. Guaranteed strictly positive.
    """

    median: np.ndarray
    scale: np.ndarray

    @property
    def n_bins(self) -> int:
        """Number of frequency bins this baseline covers."""
        return int(self.median.shape[0])

    def apply(self, error_map: np.ndarray) -> np.ndarray:
        """Convert a raw error map to a per-bin robust z-map.

        Args:
            error_map: ``(frames, n_bins)`` raw reconstruction error.

        Returns:
            A ``(frames, n_bins)`` ``float32`` z-map. Zero means "exactly as well reconstructed
            as this bin usually is"; positive means worse than usual.

        Raises:
            ValueError: If the map's bin count does not match the baseline's.
        """
        e = np.asarray(error_map, dtype=np.float64)
        if e.ndim != 2 or e.shape[1] != self.n_bins:
            raise ValueError(
                f"error map has {e.shape[1] if e.ndim == 2 else '?'} bins, "
                f"baseline has {self.n_bins}"
            )
        return ((e - self.median) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        """Return the baseline as JSON-safe lists."""
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, obj: dict) -> "Baseline":
        """Rebuild a baseline from :meth:`to_dict` output.

        Args:
            obj: Dict with ``median`` and ``scale`` lists.

        Returns:
            The :class:`Baseline`.
        """
        return cls(
            median=np.asarray(obj["median"], dtype=np.float64),
            scale=np.asarray(obj["scale"], dtype=np.float64),
        )


def fit_baseline(error_maps: list[np.ndarray]) -> Baseline:
    """Learn the per-bin error baseline from calibration recordings.

    Args:
        error_maps: Raw ``(frames, n_bins)`` error maps, from recordings the detector did **not**
            train on. Using training recordings would understate normal error (the model has
            already fitted them) and the threshold would then fire constantly on new audio.

    Returns:
        The fitted :class:`Baseline`.

    Raises:
        ValueError: If no non-empty error maps are given, or they disagree on bin count.
    """
    maps = [np.asarray(m, dtype=np.float64) for m in error_maps if m.size and m.shape[0]]
    if not maps:
        raise ValueError("cannot fit a baseline: no non-empty error maps")
    n_bins = maps[0].shape[1]
    if any(m.shape[1] != n_bins for m in maps):
        raise ValueError("error maps disagree on bin count")

    stacked = np.concatenate(maps, axis=0)  # (total_frames, n_bins)

    median = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - median), axis=0)

    # Floor each bin's spread against the whole map's spread, so a quiet bin cannot manufacture
    # enormous z-scores out of numerical noise.
    global_mad = float(np.median(np.abs(stacked - np.median(stacked))))
    floor = max(global_mad * MIN_MAD_FRACTION, 1e-12)
    scale = np.maximum(mad, floor) * _MAD_TO_SIGMA

    return Baseline(median=median, scale=scale)
