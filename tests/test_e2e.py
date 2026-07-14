# Vixen Intelligence c.2026
"""The self-proving end-to-end test.

There is no labelled ground truth for an unsupervised detector, so we manufacture some: train on
pink noise, then plant a chirp at a known time and frequency and require SIAR to box it — and to
box nothing at all on clean held-out audio.

This is the test that says the product works. Everything else says a component works.
"""
from __future__ import annotations

import numpy as np
import pytest

from siar.detect import extract_boxes, fit_baseline, fit_threshold
from siar.features.frontend import build_grid
from siar.features.spec import FeatureSpec
from siar.models import get_detector
from siar.models.conv_ae import max_latent_channels, validate_config

SR = 16_000
CHIRP_T0, CHIRP_T1 = 2.0, 3.0
CHIRP_F0, CHIRP_F1 = 3000.0, 3500.0


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return FeatureSpec(
        mode="pooled_linear",
        sample_rate=SR,
        fft_size=512,
        hop_size=128,
        window="hann",
        fmin_hz=0.0,
        fmax_hz=8000.0,
        n_bins=32,
        patch_frames=16,
        stride_frames=8,
    )


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    k = np.arange(spectrum.size)
    k[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(k), n).astype(np.float32)


def _normal(rng: np.random.Generator) -> np.ndarray:
    return 0.05 * _pink(SR * 4, rng)


def _anomalous(rng: np.random.Generator) -> np.ndarray:
    x = _normal(rng)
    t = np.arange(SR) / SR
    phase = 2 * np.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) * t**2)
    x[int(CHIRP_T0 * SR) : int(CHIRP_T1 * SR)] += (0.10 * np.sin(phase)).astype(np.float32)
    return x


@pytest.fixture(scope="module")
def trained(spec):
    """Train once; several tests read the result."""
    rng = np.random.default_rng(0)
    train = [build_grid(_normal(rng), SR, spec) for _ in range(12)]
    calib = [build_grid(_normal(rng), SR, spec) for _ in range(4)]
    test = build_grid(_anomalous(rng), SR, spec)

    conv_ae = get_detector("conv_ae")
    detector = conv_ae.fit(train, spec, conv_ae.default_config(spec), seed=0)

    baseline = fit_baseline([detector.error_map(g) for g in calib])
    calib_z = [baseline.apply(detector.error_map(g)) for g in calib]
    threshold = fit_threshold(calib_z, method="evt", contamination=1e-3)
    return detector, baseline, threshold, calib_z, test


def test_finds_the_planted_chirp(spec, trained):
    detector, baseline, threshold, _calib_z, test = trained
    boxes = extract_boxes(baseline.apply(detector.error_map(test)), spec, threshold)
    assert boxes, "no detections at all on a file with an obvious chirp in it"

    def overlaps(box) -> bool:
        return (
            box.t_start < CHIRP_T1 + 0.15
            and box.t_end > CHIRP_T0 - 0.15
            and box.f_low < CHIRP_F1 + 250
            and box.f_high > CHIRP_F0 - 250
        )

    assert overlaps(boxes[0]), (
        f"the strongest detection ({boxes[0].t_start:.2f}-{boxes[0].t_end:.2f}s, "
        f"{boxes[0].f_low:.0f}-{boxes[0].f_high:.0f}Hz) is not the planted chirp"
    )


def test_the_box_excludes_most_of_the_spectrum(spec, trained):
    """A detection spanning the whole spectrum is not a localisation.

    Guards the regression where a globally-connected bottleneck smeared reconstruction error
    across every frequency, giving boxes of 250-7750 Hz around a 500 Hz event.
    """
    detector, baseline, threshold, _calib_z, test = trained
    box = extract_boxes(baseline.apply(detector.error_map(test)), spec, threshold)[0]
    bandwidth = box.f_high - box.f_low
    full_band = spec.fmax_hz - spec.fmin_hz
    assert bandwidth < 0.5 * full_band, (
        f"box spans {bandwidth:.0f} Hz of a {full_band:.0f} Hz spectrum — not localised"
    )


def test_no_false_positives_on_clean_audio(spec, trained):
    _detector, _baseline, threshold, calib_z, _test = trained
    fp = sum(len(extract_boxes(z, spec, threshold)) for z in calib_z)
    assert fp == 0, f"{fp} detections on audio known to be normal"


def test_error_map_keeps_the_grid_shape(spec, trained):
    detector, _baseline, _threshold, _calib_z, test = trained
    emap = detector.error_map(test)
    assert emap.shape == test.shape
    assert emap.dtype == np.float32
    assert np.all(emap >= 0.0)


def test_model_round_trips_through_json_and_scores_identically(spec, trained):
    """A saved model must reproduce its own error map exactly, or re-running it is meaningless."""
    detector, _baseline, _threshold, _calib_z, test = trained
    conv_ae = get_detector("conv_ae")

    restored = conv_ae.from_json(detector.to_json())
    assert restored.spec == detector.spec
    assert np.allclose(restored.error_map(test), detector.error_map(test), atol=1e-5)


def test_an_identity_capable_autoencoder_is_refused(spec):
    """The anti-degeneracy guard. Without it, HPO selects a detector that finds nothing."""
    conv_ae = get_detector("conv_ae")
    config = conv_ae.default_config(spec)
    config["latent_channels"] = max_latent_channels(spec, config["depth"]) + 1

    with pytest.raises(ValueError, match="identity map"):
        validate_config(spec, config)
    with pytest.raises(ValueError, match="identity map"):
        conv_ae.fit([np.zeros((100, spec.n_bins), dtype=np.float32)], spec, config)
