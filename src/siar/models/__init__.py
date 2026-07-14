# Vixen Intelligence c.2026
"""Anomaly solutions.

Every detector SIAR knows about is imported here, for its registration side effect. This import
list *is* the plugin manifest — there is no scanning, no entry points, and no way for a detector
to be registered that is not visible on this page.
"""
from __future__ import annotations

from siar.models import conv_ae as _conv_ae  # noqa: F401  (registers "conv_ae")
from siar.models.base import Detector
from siar.models.registry import (
    DETECTORS,
    detector_for_format,
    get_detector,
    list_detectors,
    register_detector,
)

__all__ = [
    "DETECTORS",
    "Detector",
    "detector_for_format",
    "get_detector",
    "list_detectors",
    "register_detector",
]
