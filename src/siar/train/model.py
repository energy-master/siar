# Vixen Intelligence c.2026
"""``TrainedModel`` — everything needed to detect, in one object.

A fitted detector on its own is not enough to find anything. To turn a new recording into boxes
you also need the feature recipe that built its grid, the per-bin baseline that makes its error
comparable, the threshold, and the box-extraction settings. All five travel together or the model
is not reproducible.

That bundle is what ``siar train`` saves and what ``siar run`` loads, and it is what makes the
central promise — *train once, re-run on any folder later* — actually hold. Serialised as one
JSON document (``siar-model-v1``); weights are base64 inside it, never pickled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from siar.detect.boxes import BoxSpec, Detection, extract_boxes
from siar.detect.normalise import Baseline
from siar.features.frontend import build_grid
from siar.features.spec import FeatureSpec
from siar.models.registry import detector_for_format

__all__ = ["MODEL_FORMAT", "TrainedModel"]

MODEL_FORMAT = "siar-model-v1"


@dataclass(slots=True)
class TrainedModel:
    """A detector, plus everything required to apply it to new audio.

    Attributes:
        name: Human-readable name.
        detector: The fitted detector (see :class:`siar.models.base.Detector`).
        baseline: Per-bin error baseline, fitted on held-out calibration recordings.
        threshold: The z above which a pixel counts as anomalous.
        box_spec: Box-extraction settings.
        threshold_method: How ``threshold`` was chosen (``evt`` / ``quantile`` / ``robust_z``).
        contamination: The false-positive budget the threshold was fitted to.
        provenance: Free-form record of how this model came to be — corpus, split sizes, losses.
    """

    name: str
    detector: object
    baseline: Baseline
    threshold: float
    box_spec: BoxSpec = field(default_factory=BoxSpec)
    threshold_method: str = "evt"
    contamination: float = 1e-3
    provenance: dict = field(default_factory=dict)

    @property
    def spec(self) -> FeatureSpec:
        """The feature recipe this model was trained on."""
        return self.detector.spec  # type: ignore[attr-defined]

    def detect_grid(self, grid: np.ndarray) -> list[Detection]:
        """Find anomalies in an already-built grid.

        Args:
            grid: ``(frames, n_bins)`` ``float32``, built with :attr:`spec`.

        Returns:
            Detections, strongest first.
        """
        if grid.shape[0] == 0:
            return []
        z = self.baseline.apply(self.detector.error_map(grid))  # type: ignore[attr-defined]
        return extract_boxes(z, self.spec, self.threshold, self.box_spec)

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[Detection]:
        """Find anomalies in a waveform.

        Args:
            samples: 1-D mono signal.
            sample_rate: Its sample rate in Hz. Resampled to the model's rate automatically.

        Returns:
            Detections, strongest first.
        """
        return self.detect_grid(build_grid(samples, sample_rate, self.spec))

    def grid_and_boxes(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, list[Detection]]:
        """Both the grid and its detections, so a caller can render one and overlay the other.

        Args:
            samples: 1-D mono signal.
            sample_rate: Its sample rate in Hz.

        Returns:
            ``(grid, detections)``.
        """
        grid = build_grid(samples, sample_rate, self.spec)
        return grid, self.detect_grid(grid)

    # --- serialisation ------------------------------------------------------

    def to_json(self) -> dict:
        """Serialise the whole bundle to one JSON-safe document."""
        return {
            "format": MODEL_FORMAT,
            "name": self.name,
            "detector": self.detector.to_json(),  # type: ignore[attr-defined]
            "baseline": self.baseline.to_dict(),
            "threshold": float(self.threshold),
            "threshold_method": self.threshold_method,
            "contamination": float(self.contamination),
            "box_spec": self.box_spec.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "TrainedModel":
        """Rebuild a model from :meth:`to_json` output.

        The detector class is looked up by its ``format`` string, so a model file does not need
        to be told which code wrote it.

        Args:
            obj: The document.

        Returns:
            The restored :class:`TrainedModel`.

        Raises:
            ValueError: If the document is not a ``siar-model-v1``, or its detector format is not
                registered.
        """
        if obj.get("format") != MODEL_FORMAT:
            raise ValueError(f"expected format {MODEL_FORMAT!r}, got {obj.get('format')!r}")

        det_obj = obj["detector"]
        try:
            det_cls = detector_for_format(det_obj["format"])
        except KeyError as exc:
            raise ValueError(str(exc)) from None

        return cls(
            name=str(obj.get("name", "model")),
            detector=det_cls.from_json(det_obj),
            baseline=Baseline.from_dict(obj["baseline"]),
            threshold=float(obj["threshold"]),
            box_spec=BoxSpec.from_dict(obj.get("box_spec", {})),
            threshold_method=str(obj.get("threshold_method", "evt")),
            contamination=float(obj.get("contamination", 1e-3)),
            provenance=dict(obj.get("provenance", {})),
        )
