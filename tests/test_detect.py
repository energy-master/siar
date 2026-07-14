# Vixen Intelligence c.2026
"""Detection-layer invariants: normalisation, thresholds, and box geometry.

The box tests build a z-map by hand, so the expected answer is known exactly. If
:func:`siar.detect.boxes.extract_boxes` cannot recover two rectangles that were drawn into an
array, nothing downstream of it is worth trusting.
"""
from __future__ import annotations

import numpy as np
import pytest

from siar.detect.boxes import BoxSpec, extract_boxes
from siar.detect.normalise import fit_baseline
from siar.detect.threshold import fit_threshold
from siar.features.frontend import row_support_hz
from siar.features.spec import FeatureSpec

SR = 16_000


def make_spec(**overrides) -> FeatureSpec:
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


def test_extract_boxes_recovers_two_hand_drawn_rectangles():
    """The single most important test in the repo."""
    spec = make_spec()
    z = np.zeros((200, spec.n_bins), dtype=np.float32)
    z[20:40, 4:10] = 50.0  # rectangle A
    z[120:150, 20:28] = 80.0  # rectangle B, stronger

    boxes = extract_boxes(z, spec, threshold=10.0)
    assert len(boxes) == 2

    strongest, weaker = boxes  # sorted by score, descending
    assert strongest.frame_lo == 120 and strongest.frame_hi == 149
    assert strongest.bin_lo == 20 and strongest.bin_hi == 27
    assert weaker.frame_lo == 20 and weaker.frame_hi == 39
    assert weaker.bin_lo == 4 and weaker.bin_hi == 9

    # ...and the conversion back to seconds and Hz is exact.
    support = row_support_hz(spec)
    assert strongest.t_start == pytest.approx(120 * spec.delta_t)
    assert strongest.t_end == pytest.approx(150 * spec.delta_t)
    assert strongest.f_low == pytest.approx(support[20][0])
    assert strongest.f_high == pytest.approx(support[27][1])


def test_a_component_whose_bbox_encloses_another_is_still_measured_correctly():
    """Regression: find_objects indexes slices by label, so the label id is index + 1.

    Recovering it with ``labels[sl].max()`` instead picks whichever label has the highest id
    inside the bounding box — which, when a big component's box encloses a smaller one, is the
    wrong component. The large detection was silently discarded.
    """
    spec = make_spec()
    z = np.zeros((120, spec.n_bins), dtype=np.float32)
    z[10:100, 2:30] = 40.0  # a big component...
    z[50:56, 14:18] = 0.0  # ...with a hole in it
    z[51:55, 15:17] = 90.0  # ...containing a separate, stronger component

    boxes = extract_boxes(z, spec, threshold=10.0, box_spec=BoxSpec(peak_fraction=0.0))
    assert len(boxes) >= 1
    # The big one must survive — it is the bug's victim.
    assert any(b.frame_hi - b.frame_lo > 50 for b in boxes)


def test_empty_and_all_quiet_maps_yield_no_boxes():
    spec = make_spec()
    assert extract_boxes(np.zeros((0, spec.n_bins), dtype=np.float32), spec, 1.0) == []
    assert extract_boxes(np.zeros((50, spec.n_bins), dtype=np.float32), spec, 1.0) == []


def test_ragged_component_is_rejected_by_the_fill_filter():
    """A diagonal streak has a huge bounding box and almost no area in it. It is noise."""
    spec = make_spec()
    z = np.zeros((100, spec.n_bins), dtype=np.float32)
    for i in range(30):
        z[i, i] = 60.0  # a one-pixel-wide diagonal

    boxes = extract_boxes(z, spec, threshold=10.0, box_spec=BoxSpec(min_fill=0.25))
    assert boxes == []


def test_peak_fraction_gate_tightens_a_box_around_its_core():
    """A strong core inside a broad weak halo should be boxed to the core, not the halo."""
    spec = make_spec()
    z = np.zeros((100, spec.n_bins), dtype=np.float32)
    z[30:70, 2:30] = 20.0  # halo, just above threshold
    z[45:55, 14:18] = 5000.0  # the actual event

    loose = extract_boxes(z, spec, 10.0, BoxSpec(peak_fraction=0.0))[0]
    tight = extract_boxes(z, spec, 10.0, BoxSpec(peak_fraction=0.1))[0]

    assert loose.bin_hi - loose.bin_lo > tight.bin_hi - tight.bin_lo
    assert (tight.bin_lo, tight.bin_hi) == (14, 17)
    assert (tight.frame_lo, tight.frame_hi) == (45, 54)


def test_baseline_floors_the_scale_of_a_silent_bin():
    """A near-zero-variance bin must not be able to manufacture enormous z-scores."""
    rng = np.random.default_rng(0)
    error = rng.random((500, 8)).astype(np.float32)
    error[:, 3] = 1e-9  # a bin with essentially no variance

    baseline = fit_baseline([error])
    assert np.all(baseline.scale > 0)

    probe = error.copy()
    probe[10, 3] = 1e-8  # a trivial absolute bump in the silent bin
    z = baseline.apply(probe)
    assert abs(z[10, 3]) < 10.0, "a quiet bin produced an absurd z-score"


def test_baseline_round_trips_through_json():
    rng = np.random.default_rng(0)
    baseline = fit_baseline([rng.random((200, 6)).astype(np.float32)])
    from siar.detect.normalise import Baseline

    restored = Baseline.from_dict(baseline.to_dict())
    assert np.allclose(restored.median, baseline.median)
    assert np.allclose(restored.scale, baseline.scale)


@pytest.mark.parametrize("method", ["evt", "quantile", "robust_z"])
def test_threshold_methods_return_a_finite_value_above_the_bulk(method):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2000, 16)).astype(np.float32)
    thr = fit_threshold([z], method=method, contamination=1e-2)
    assert np.isfinite(thr)
    assert thr > float(np.median(z))


def test_threshold_rejects_a_nonsense_contamination():
    z = [np.zeros((10, 4), dtype=np.float32)]
    with pytest.raises(ValueError, match="contamination"):
        fit_threshold(z, contamination=0.9)
