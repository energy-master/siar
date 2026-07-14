# Vixen Intelligence c.2026
"""Choosing the line between "normal" and "anomalous", without labels.

There is no ground truth to tune against, so a threshold here is not a discovered fact — it is a
**budget the user sets**: how much of your data are you willing to have flagged? That framing is
the honest one, and it is what ``contamination`` expresses.

Three ways to spend that budget:

``quantile``  Take the ``1 - contamination`` quantile of the calibration z-scores. Assumption-
              free and always available. Its weakness is the far tail: at contamination 1e-4 on
              a million calibration pixels, the threshold rests on ~100 samples and is noisy.

``robust_z``  A fixed z, e.g. 5.0. Interpretable, but it assumes the tail is roughly Gaussian.
              Reconstruction errors are emphatically not — they are heavy-tailed and skewed — so
              this is a sanity baseline, not a recommendation.

``evt``       Peaks-Over-Threshold. Fit a Generalised Pareto to the *exceedances* above a high
              quantile and extrapolate. This is the statistically right tool for "how large is a
              once-in-100,000-pixels value", because it models the tail's shape instead of
              reading it off a handful of order statistics. It is the default.

All three are fitted on **calibration** recordings — held out from both training and validation,
so the reported false-positive budget is not optimistic.
"""
from __future__ import annotations

import numpy as np

__all__ = ["THRESHOLD_METHODS", "fit_threshold"]

#: The methods :func:`fit_threshold` understands.
THRESHOLD_METHODS = ("evt", "quantile", "robust_z")

#: Quantile above which exceedances are taken to be "the tail" for the EVT fit. 0.98 leaves
#: enough points to fit two parameters while staying high enough that the GPD is a good model.
_EVT_ANCHOR = 0.98


def fit_threshold(
    z_maps: list[np.ndarray],
    *,
    method: str = "evt",
    contamination: float = 1e-3,
    max_samples: int = 5_000_000,
    seed: int = 0,
) -> float:
    """Choose the z above which a pixel counts as anomalous.

    Args:
        z_maps: Per-bin z-maps of the **calibration** recordings
            (:meth:`siar.detect.normalise.Baseline.apply`).
        method: One of :data:`THRESHOLD_METHODS`.
        contamination: The fraction of pixels you are willing to have flagged on data assumed to
            be normal. Smaller means fewer, higher-confidence detections. For ``robust_z`` this
            is ignored and the threshold is a fixed z of 5.0.
        max_samples: Subsample above this many pixels — the quantiles are indistinguishable and
            it keeps the fit fast on a large corpus.
        seed: Seed for that subsample.

    Returns:
        The threshold, as a z-score.

    Raises:
        ValueError: If ``method`` is unknown, ``contamination`` is out of range, or no
            calibration data is given.
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {THRESHOLD_METHODS}")
    if not 0.0 < contamination < 0.5:
        raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

    flat = np.concatenate([np.asarray(z, dtype=np.float64).ravel() for z in z_maps if z.size])
    if flat.size == 0:
        raise ValueError("cannot fit a threshold: no calibration data")

    if flat.size > max_samples:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.size, max_samples, replace=False)]

    if method == "robust_z":
        return 5.0

    if method == "quantile":
        return float(np.quantile(flat, 1.0 - contamination))

    return _evt_threshold(flat, contamination)


def _evt_threshold(z: np.ndarray, contamination: float) -> float:
    """Peaks-Over-Threshold threshold via a Generalised Pareto fit.

    Fits a GPD to the exceedances above the :data:`_EVT_ANCHOR` quantile and inverts its survival
    function for the target exceedance probability. Falls back to the empirical quantile if the
    fit fails or there are too few exceedances to fit two parameters meaningfully.

    Args:
        z: 1-D calibration z-scores.
        contamination: Target fraction of pixels above the returned threshold.

    Returns:
        The threshold, as a z-score.
    """
    from scipy.stats import genpareto

    anchor = float(np.quantile(z, _EVT_ANCHOR))
    exceedances = z[z > anchor] - anchor
    if exceedances.size < 100:
        return float(np.quantile(z, 1.0 - contamination))

    # If the target is inside the empirical range we already have the data to answer directly;
    # EVT only earns its keep when extrapolating past it.
    tail_rate = exceedances.size / z.size
    if contamination >= tail_rate:
        return float(np.quantile(z, 1.0 - contamination))

    try:
        shape, _loc, scale = genpareto.fit(exceedances, floc=0.0)
    except Exception:  # scipy raises a variety of things on a bad fit
        return float(np.quantile(z, 1.0 - contamination))

    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return float(np.quantile(z, 1.0 - contamination))

    # P(Z > anchor + y) = tail_rate * (1 + shape*y/scale)^(-1/shape); solve for y at
    # P = contamination.
    ratio = tail_rate / contamination
    if abs(shape) < 1e-8:
        y = scale * np.log(ratio)
    else:
        y = (scale / shape) * (ratio**shape - 1.0)

    thr = anchor + float(y)
    if not np.isfinite(thr):
        return float(np.quantile(z, 1.0 - contamination))
    # A GPD with a heavy tail can extrapolate absurdly far; never return a threshold beyond the
    # point where nothing could ever fire.
    return float(min(thr, float(z.max()) * 10.0))
