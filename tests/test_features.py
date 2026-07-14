# Vixen Intelligence c.2026
"""Feature-layer invariants: grid geometry, the frequency inverse-map, patch round-trips."""
from __future__ import annotations

import numpy as np
import pytest

from siar.features.frontend import bin_edges_hz, build_grid, row_support_hz
from siar.features.patches import overlap_add, patchify
from siar.features.spec import FeatureSpec

SR = 16_000


def make_spec(**overrides) -> FeatureSpec:
    """A valid spec, with any field overridden."""
    base = dict(
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
    base.update(overrides)
    return FeatureSpec(**base)


def tone(freq: float, seconds: float = 2.0) -> np.ndarray:
    """A pure tone."""
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("mode", ["pooled_linear", "log_mel"])
def test_grid_shape_matches_frame_formula(mode):
    spec = make_spec(mode=mode)
    x = tone(1000.0, seconds=2.0)
    grid = build_grid(x, SR, spec)
    expected = 1 + (x.size - spec.fft_size) // spec.hop_size
    assert grid.shape == (expected, spec.n_bins)
    assert grid.dtype == np.float32


def test_grid_is_empty_when_signal_shorter_than_one_frame():
    spec = make_spec()
    grid = build_grid(np.zeros(spec.fft_size - 1, dtype=np.float32), SR, spec)
    assert grid.shape == (0, spec.n_bins)


@pytest.mark.parametrize("mode", ["pooled_linear", "log_mel"])
def test_bin_edges_are_monotonic_and_span_the_band(mode):
    spec = make_spec(mode=mode)
    edges = bin_edges_hz(spec)
    assert edges.shape == (spec.n_bins + 1,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(spec.fmin_hz, abs=1.0)
    assert edges[-1] == pytest.approx(spec.fmax_hz, abs=1.0)


@pytest.mark.parametrize("mode", ["pooled_linear", "log_mel"])
@pytest.mark.parametrize("freq", [1000.0, 3000.0, 6000.0])
def test_row_support_contains_the_tone_that_excites_the_row(mode, freq):
    """The band a detection is reported in must actually contain the sound that caused it.

    This is the regression test for a real bug: taking only the rising half of each overlapping
    mel filter reported a 6 kHz tone as 5573-5926 Hz — a band that does not contain 6 kHz.
    """
    spec = make_spec(mode=mode)
    grid = build_grid(tone(freq), SR, spec)
    row = int(grid.mean(axis=0).argmax())
    low, high = row_support_hz(spec)[row]
    assert low <= freq <= high, f"row {row} reported as {low:.0f}-{high:.0f} Hz, tone is {freq}"


@pytest.mark.parametrize("stride", [1, 3, 5, 8, 16])
def test_overlap_add_returns_a_constant_unchanged(stride):
    """Constant error in must be constant error out — at every stride, including the padded tail.

    If this fails, the error map has a systematic ripple and box extraction will find regularly
    spaced phantom detections.
    """
    spec = make_spec(stride_frames=stride)
    grid = build_grid(tone(1000.0, seconds=1.7), SR, spec)
    patches, starts, frames = patchify(grid, spec.patch_frames, spec.stride_frames)
    errors = np.full((len(starts), spec.patch_frames, spec.n_bins), 7.0, dtype=np.float32)
    emap = overlap_add(errors, starts, frames)
    assert emap.shape == (frames, spec.n_bins)
    assert np.allclose(emap, 7.0, atol=1e-4)


def test_patches_cover_every_frame():
    """No frame may be left uncovered — an anomaly in the final half-second must be visible."""
    spec = make_spec(stride_frames=5)
    grid = build_grid(tone(1000.0, seconds=1.7), SR, spec)
    _patches, starts, frames = patchify(grid, spec.patch_frames, spec.stride_frames)
    covered = np.zeros(frames, dtype=bool)
    for s in starts:
        covered[s : s + spec.patch_frames] = True
    assert covered.all()
    assert starts[-1] == frames - spec.patch_frames


def test_short_grid_is_padded_not_dropped():
    spec = make_spec()
    grid = np.ones((3, spec.n_bins), dtype=np.float32)
    patches, starts, frames = patchify(grid, spec.patch_frames, spec.stride_frames)
    assert frames == spec.patch_frames
    assert patches.shape == (len(starts), spec.patch_frames, spec.n_bins)


def test_spec_rejects_a_patch_it_cannot_halve_twice():
    with pytest.raises(ValueError, match="multiple of 4"):
        make_spec(n_bins=30)
    with pytest.raises(ValueError, match="multiple of 4"):
        make_spec(patch_frames=15)


def test_spec_rejects_fmax_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        make_spec(fmax_hz=9000.0)


def test_spec_round_trips_through_json():
    spec = make_spec(mode="log_mel")
    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    assert FeatureSpec.from_dict(spec.to_dict()).cache_key() == spec.cache_key()
