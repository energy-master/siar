# Vixen Intelligence c.2026
"""Audio -> grid -> patches.

Only :mod:`siar.features.spec` is re-exported here, so importing this package stays cheap.
:mod:`siar.features.frontend` pulls in torch and is imported explicitly by the code that needs
to actually build a grid.
"""
from __future__ import annotations

from siar.features.spec import FEATURE_MODES, FeatureSpec

__all__ = ["FEATURE_MODES", "FeatureSpec"]
