# Vixen Intelligence c.2026
"""Tutorial 01 — Your first detector.

The runnable companion to ``01-first-detector.md``. Builds a synthetic corpus where the ground
truth is known by construction, trains an autoencoder on the normal files only, and checks that
it draws a box around the chirp we planted in the one anomalous file.

Run it:

    python tutorials/01_first_detector.py

Everything here is deliberately explicit — no CLI, no database, no dashboard. Just the pipeline.
"""
from __future__ import annotations

import numpy as np

from siar.detect import extract_boxes, fit_baseline, fit_threshold
from siar.features.frontend import build_grid
from siar.features.spec import FeatureSpec
from siar.models import get_detector

SAMPLE_RATE = 16_000
DURATION_S = 4

# The needle. The detector is never told any of this.
CHIRP_T0, CHIRP_T1 = 2.0, 3.0
CHIRP_F0, CHIRP_F1 = 3000.0, 3500.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise — a stand-in for a natural background.

    Args:
        n: Length in samples.
        rng: Source of randomness.

    Returns:
        A 1-D ``float32`` signal.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    k = np.arange(spectrum.size)
    k[0] = 1  # leave DC alone rather than dividing by zero
    return np.fft.irfft(spectrum / np.sqrt(k), n).astype(np.float32)


def normal_file(rng: np.random.Generator) -> np.ndarray:
    """A recording with nothing in it but background."""
    return 0.05 * pink_noise(SAMPLE_RATE * DURATION_S, rng)


def anomalous_file(rng: np.random.Generator) -> np.ndarray:
    """Background, plus a chirp planted at a known time and frequency."""
    x = normal_file(rng)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE  # one second of chirp
    rate = (CHIRP_F1 - CHIRP_F0) / 1.0
    phase = 2.0 * np.pi * (CHIRP_F0 * t + 0.5 * rate * t**2)
    start = int(CHIRP_T0 * SAMPLE_RATE)
    x[start : start + SAMPLE_RATE] += (0.10 * np.sin(phase)).astype(np.float32)
    return x


def main() -> None:
    """Run the tutorial end to end."""
    rng = np.random.default_rng(0)

    # Step 1 — what the model will look at.
    spec = FeatureSpec(
        mode="pooled_linear",
        sample_rate=SAMPLE_RATE,
        fft_size=512,
        hop_size=128,
        window="hann",
        fmin_hz=0.0,
        fmax_hz=8000.0,
        n_bins=32,
        patch_frames=16,
        stride_frames=8,
    )

    # Step 0 — the corpus. train and calib are both normal; calib is simply held out.
    train_grids = [build_grid(normal_file(rng), SAMPLE_RATE, spec) for _ in range(12)]
    calib_grids = [build_grid(normal_file(rng), SAMPLE_RATE, spec) for _ in range(4)]
    test_grid = build_grid(anomalous_file(rng), SAMPLE_RATE, spec)

    print(f"grid: {train_grids[0].shape[0]} frames x {spec.n_bins} bins "
          f"({spec.delta_t * 1000:.0f} ms per frame)")

    # Step 2 — learn "normal" from the training files only.
    conv_ae = get_detector("conv_ae")
    config = conv_ae.default_config(spec)
    print(f"config: {config['depth']} stages, {config['latent_channels']} latent channels")

    detector = conv_ae.fit(
        train_grids,
        spec,
        config,
        progress=lambda e, total, loss, _v: (
            print(f"  epoch {e:2d}/{total}  loss {loss:.5f}") if e % 5 == 0 else None
        ),
    )
    print(f"trained: {detector.n_params:,} parameters")

    # Steps 3 & 4 — per-pixel error, made comparable across frequency using held-out files.
    baseline = fit_baseline([detector.error_map(g) for g in calib_grids])
    calib_z = [baseline.apply(detector.error_map(g)) for g in calib_grids]

    # Step 5 — the threshold is a budget: 0.1% of pixels may be flagged on normal data.
    threshold = fit_threshold(calib_z, method="evt", contamination=1e-3)
    print(f"threshold: z = {threshold:.1f}")

    # Step 6 — boxes.
    false_positives = sum(len(extract_boxes(z, spec, threshold)) for z in calib_z)
    boxes = extract_boxes(baseline.apply(detector.error_map(test_grid)), spec, threshold)

    # Step 7 — did it find the thing we hid?
    print()
    print(f"PLANTED : t = {CHIRP_T0:.2f} - {CHIRP_T1:.2f} s    "
          f"f = {CHIRP_F0:.0f} - {CHIRP_F1:.0f} Hz")
    print()
    for box in boxes:
        overlaps = (
            box.t_start < CHIRP_T1 + 0.1
            and box.t_end > CHIRP_T0 - 0.1
            and box.f_low < CHIRP_F1 + 200
            and box.f_high > CHIRP_F0 - 200
        )
        print(
            f"DETECTED: t = {box.t_start:5.2f} - {box.t_end:5.2f} s    "
            f"f = {box.f_low:5.0f} - {box.f_high:5.0f} Hz    "
            f"score = {box.score:,.0f}"
            f"{'   <== the planted chirp' if overlaps else ''}"
        )
    print()
    print(f"false positives on {len(calib_grids)} clean held-out files: {false_positives}")


if __name__ == "__main__":
    main()
