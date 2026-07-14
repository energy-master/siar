# Vixen Intelligence c.2026
"""Training a detector, and the model bundle that results."""
from __future__ import annotations

from siar.train.fit import TrainResult, default_spec, train_from_folder
from siar.train.model import MODEL_FORMAT, TrainedModel

__all__ = ["MODEL_FORMAT", "TrainResult", "TrainedModel", "default_spec", "train_from_folder"]
