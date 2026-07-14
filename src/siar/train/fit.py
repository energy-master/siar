# Vixen Intelligence c.2026
"""Training a detector on a folder of audio.

The glue between everything else: scan a folder, split it, build grids, fit a detector on the
training files, fit the error baseline and threshold on the **held-out calibration** files, and
hand back a :class:`~siar.train.model.TrainedModel` ready to be saved and re-run.

The split is the part to be careful about, and the reason it is here rather than left to the
caller. Fitting the threshold on recordings the network trained on measures error the network has
already minimised — it looks far too small, the threshold comes out far too low, and the detector
then fires on everything the moment it meets a file it has not seen. Calibration files are held
out for exactly this reason, and a corpus too small to hold any out is warned about, loudly.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from siar.data.audio import load_audio
from siar.data.dataset import AudioFile, Split, scan_folder, split_files
from siar.detect.boxes import BoxSpec
from siar.detect.normalise import fit_baseline
from siar.detect.threshold import fit_threshold
from siar.features.frontend import build_grid
from siar.features.spec import FeatureSpec
from siar.models.registry import get_detector
from siar.train.model import TrainedModel

__all__ = ["TrainResult", "default_spec", "train_from_folder"]

#: Called with a human-readable progress line.
Reporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class TrainResult:
    """A trained model plus what it cost to make.

    Attributes:
        model: The fitted :class:`~siar.train.model.TrainedModel`.
        split: Which files went where.
        train_loss: Final training reconstruction loss.
        n_train_patches: How many patches the detector saw.
    """

    model: TrainedModel
    split: Split
    train_loss: float
    n_train_patches: int


def default_spec(
    sample_rate: int,
    *,
    fmin_hz: float = 0.0,
    fmax_hz: float | None = None,
    fft_size: int = 1024,
    n_bins: int = 64,
    patch_frames: int = 16,
) -> FeatureSpec:
    """A sensible feature spec for a corpus at a given sample rate.

    Args:
        sample_rate: The rate every recording will be resampled to.
        fmin_hz: Low edge of the analysed band.
        fmax_hz: High edge. Defaults to Nyquist.
        fft_size: FFT size.
        n_bins: Grid height.
        patch_frames: Patch width in frames.

    Returns:
        A validated :class:`~siar.features.spec.FeatureSpec` with a quarter-FFT hop.
    """
    return FeatureSpec(
        mode="pooled_linear",
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=fft_size // 4,
        window="hann",
        fmin_hz=fmin_hz,
        fmax_hz=sample_rate / 2.0 if fmax_hz is None else fmax_hz,
        n_bins=n_bins,
        patch_frames=patch_frames,
        stride_frames=max(1, patch_frames // 2),
    )


def _grids(files: tuple[AudioFile, ...], spec: FeatureSpec, report: Reporter) -> list[np.ndarray]:
    """Decode and featurise a set of files, skipping any that yield no frames."""
    out: list[np.ndarray] = []
    for f in files:
        rec = load_audio(f.path)
        grid = build_grid(rec.samples, rec.sample_rate, spec)
        if grid.shape[0] == 0:
            report(f"  skipped {f.name}: too short for one {spec.fft_size}-sample frame")
            continue
        out.append(grid)
    return out


def train_from_folder(
    folder: str,
    *,
    detector: str = "conv_ae",
    spec: FeatureSpec | None = None,
    config: dict | None = None,
    box_spec: BoxSpec | None = None,
    threshold_method: str = "evt",
    contamination: float = 1e-3,
    val_fraction: float = 0.15,
    calib_fraction: float = 0.15,
    seed: int = 0,
    name: str | None = None,
    report: Reporter = lambda _line: None,
) -> TrainResult:
    """Train an anomaly detector on a folder of audio.

    Args:
        folder: The corpus. Assumed to be **predominantly normal** — SIAR learns what is usual
            here and flags what is not, so anything common in this folder will not be detected.
        detector: A registered detector's name (``siar detectors``).
        spec: The feature spec. Defaults to :func:`default_spec` at the corpus's dominant rate.
        config: Detector hyperparameters. Defaults to the detector's own ``default_config``.
        box_spec: Box-extraction settings.
        threshold_method: ``evt``, ``quantile`` or ``robust_z``.
        contamination: The fraction of pixels you accept being flagged on normal data.
        val_fraction: Files held out for validation loss.
        calib_fraction: Files held out to fit the baseline and threshold.
        seed: Seed for the split and for weight init.
        name: Model name. Defaults to the folder's name.
        report: Called with progress lines.

    Returns:
        The :class:`TrainResult`.

    Raises:
        FileNotFoundError: If ``folder`` is not a directory.
        ValueError: If the corpus has no readable audio, or yields no usable patches.
    """
    scan = scan_folder(folder)
    if not scan.files:
        raise ValueError(f"no readable audio in {folder}")

    if spec is None:
        spec = default_spec(int(scan.dominant_sample_rate or 48_000))
    if len(scan.sample_rates) > 1:
        report(
            f"  note: {len(scan.sample_rates)} sample rates present; resampling everything "
            f"to {spec.sample_rate} Hz"
        )

    split = split_files(
        scan.files, val_fraction=val_fraction, calib_fraction=calib_fraction, seed=seed
    )
    if len(scan.files) < 3:
        report(
            "  WARNING: fewer than 3 files, so validation and calibration reuse the training "
            "audio. The threshold is NOT held out and the false-positive rate it implies is "
            "optimistic. Use more recordings."
        )
    report(
        f"  split: {len(split.train)} train / {len(split.val)} val / {len(split.calib)} calib"
    )

    report("  building grids...")
    train_grids = _grids(split.train, spec, report)
    val_grids = _grids(split.val, spec, report)
    calib_grids = _grids(split.calib, spec, report)
    if not train_grids:
        raise ValueError("no usable training grids (recordings too short?)")
    if not calib_grids:
        raise ValueError("no usable calibration grids — cannot fit a threshold")

    det_cls = get_detector(detector)
    cfg = dict(config) if config else det_cls.default_config(spec)

    losses: list[float] = []

    def on_epoch(epoch: int, total: int, train_loss: float, val_loss: float | None) -> None:
        losses.append(train_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == total:
            val = f"  val {val_loss:.5f}" if val_loss is not None else ""
            report(f"  epoch {epoch:3d}/{total}  loss {train_loss:.5f}{val}")

    report(f"  training {detector}...")
    fitted = det_cls.fit(
        train_grids, spec, cfg, val_grids=val_grids or None, progress=on_epoch, seed=seed
    )
    report(f"  trained: {fitted.n_params:,} parameters")

    # The baseline and threshold come from calibration files ONLY. See the module docstring.
    report("  calibrating on held-out files...")
    calib_errors = [fitted.error_map(g) for g in calib_grids]
    baseline = fit_baseline(calib_errors)
    calib_z = [baseline.apply(e) for e in calib_errors]
    threshold = fit_threshold(
        calib_z, method=threshold_method, contamination=contamination, seed=seed
    )
    report(f"  threshold: z = {threshold:.2f} ({threshold_method}, contamination {contamination})")

    n_patches = sum(
        max(1, (g.shape[0] - spec.patch_frames) // spec.stride_frames + 1) for g in train_grids
    )
    model = TrainedModel(
        name=name or scan.root.rstrip("/").rsplit("/", 1)[-1],
        detector=fitted,
        baseline=baseline,
        threshold=float(threshold),
        box_spec=box_spec or BoxSpec(),
        threshold_method=threshold_method,
        contamination=contamination,
        provenance={
            "corpus": scan.root,
            "n_files": scan.n_files,
            "total_seconds": round(scan.total_duration, 1),
            "n_train_files": len(split.train),
            "n_val_files": len(split.val),
            "n_calib_files": len(split.calib),
            "n_train_patches": int(n_patches),
            "final_train_loss": round(losses[-1], 6) if losses else None,
            "config": cfg,
            "seed": seed,
        },
    )
    return TrainResult(
        model=model,
        split=split,
        train_loss=losses[-1] if losses else float("nan"),
        n_train_patches=int(n_patches),
    )
