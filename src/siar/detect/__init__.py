# Vixen Intelligence c.2026
"""Error map -> normalised z-map -> threshold -> boxes."""
from __future__ import annotations

from siar.detect.boxes import BoxSpec, Detection, extract_boxes
from siar.detect.normalise import Baseline, fit_baseline
from siar.detect.threshold import THRESHOLD_METHODS, fit_threshold

__all__ = [
    "THRESHOLD_METHODS",
    "Baseline",
    "BoxSpec",
    "Detection",
    "extract_boxes",
    "fit_baseline",
    "fit_threshold",
]
