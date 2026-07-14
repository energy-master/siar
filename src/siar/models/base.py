# Vixen Intelligence c.2026
"""The detector seam.

Everything SIAR does downstream of a model — thresholding, box extraction, storage, JSON export,
the dashboard — depends on exactly one thing:

    a detector turns a ``(frames, n_bins)`` grid into a ``(frames, n_bins)`` per-pixel error map

That is the whole contract. A convolutional autoencoder satisfies it by reconstructing the grid
and squaring the residual, but so would a VAE, a masked autoencoder, a memory-bank kNN over
patch embeddings, or a plain PCA reconstruction. Any of those can be dropped in without touching
a line outside ``siar/models/``.

The error map must be **per-pixel**, not per-patch. Collapsing a patch's error to a single
number is the difference between "something odd happens around 12.4 s" and "a 4–6 kHz chirp at
12.3–12.8 s", and it is why SIAR can draw boxes at all.

A detector must also survive a round trip through JSON. Weights are base64'd inside the
descriptor rather than pickled, and loaded with ``torch.load(..., weights_only=True)``, so
opening a model someone sent you can never execute their code.
"""
from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from siar.features.spec import FeatureSpec

__all__ = ["Detector", "TrainProgress"]

#: Called once per epoch during training: ``(epoch, total_epochs, train_loss, val_loss)``.
#: Used for CLI progress, run logs, and Optuna's pruner.
TrainProgress = Any


@runtime_checkable
class Detector(Protocol):
    """What every SIAR anomaly solution must implement.

    Attributes:
        format: The versioned identifier written into the model JSON, e.g.
            ``"siar-conv-ae-v1"``. Bump the version whenever the on-disk shape changes.
        spec: The :class:`~siar.features.spec.FeatureSpec` this detector was fitted on. Carried
            so inference can rebuild the exact grid it was trained against.
    """

    format: ClassVar[str]
    spec: FeatureSpec

    @staticmethod
    def suggest(trial: Any, spec: FeatureSpec) -> dict:
        """Sample this detector's hyperparameters from an Optuna trial.

        Args:
            trial: An ``optuna.Trial``.
            spec: The feature spec already sampled for this trial — the architecture may need it
                (a conv net's depth is constrained by the patch size, for instance).

        Returns:
            A config dict accepted by :meth:`fit`.
        """
        ...

    @classmethod
    def fit(
        cls,
        grids: list[np.ndarray],
        spec: FeatureSpec,
        config: dict,
        *,
        val_grids: list[np.ndarray] | None = None,
        progress: TrainProgress | None = None,
    ) -> "Detector":
        """Learn "normal" from a corpus of grids.

        Args:
            grids: Training grids, each ``(frames, n_bins)`` ``float32``.
            spec: The feature spec the grids were built with.
            config: Hyperparameters, as returned by :meth:`suggest`.
            val_grids: Optional held-out grids, for per-epoch validation loss and pruning.
            progress: Optional per-epoch callback.

        Returns:
            The fitted detector.

        Raises:
            ValueError: If the grids yield no usable training patches.
        """
        ...

    def error_map(self, grid: np.ndarray) -> np.ndarray:
        """Score one grid, per pixel.

        Args:
            grid: ``(frames, n_bins)`` ``float32``, built with :attr:`spec`.

        Returns:
            A ``(frames, n_bins)`` ``float32`` map of reconstruction error — same shape as the
            input, higher meaning less like the training data. Raw error, **not** normalised:
            per-bin whitening and thresholding are :mod:`siar.detect`'s job, not the model's.
        """
        ...

    def to_json(self) -> dict:
        """Serialise to a JSON-safe descriptor, weights included (base64)."""
        ...

    @classmethod
    def from_json(cls, obj: dict) -> "Detector":
        """Rebuild a detector from :meth:`to_json` output.

        Args:
            obj: The descriptor.

        Returns:
            The restored detector.

        Raises:
            ValueError: If the descriptor is malformed or its format is not recognised.
        """
        ...
