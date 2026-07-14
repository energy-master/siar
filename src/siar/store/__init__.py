# Vixen Intelligence c.2026
"""The local results database."""
from __future__ import annotations

from siar.store.db import Store, new_uid, utcnow
from siar.store.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "Store", "new_uid", "utcnow"]
