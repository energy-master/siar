# Vixen Intelligence c.2026
"""SIAR — Signal Intelligence and Reconnaissance.

An unsupervised acoustic anomaly-detection library. Point it at a folder of audio, run an
optimisation, and get back a trained detector that draws boxes around the parts of a
spectrogram that do not look like the rest of the corpus.

The pipeline, in one line:

    audio -> STFT -> band grid -> patches -> detector -> per-pixel error map -> boxes

Every stage is a separate module with a stated array contract, so a new anomaly solution
only has to satisfy :class:`siar.models.base.Detector` (grid in, per-pixel error map out) to
inherit thresholding, box extraction, storage, export and the dashboard for free.

This package is import-cheap: ``import siar`` pulls in no torch, no optuna and no audio
decoding. Those are imported lazily, inside the functions that need them.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
