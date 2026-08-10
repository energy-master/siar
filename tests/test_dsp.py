# Vixen Intelligence c.2026
"""The DSP port has to agree with the browser, so these tests pin the ways it could quietly not.

None of this needs node. The two things that actually go wrong in an STFT port — the window's
symmetric-vs-periodic convention, and where frame zero starts — are both checkable against
first principles, and a direct O(n^2) DFT is a reference no FFT implementation can be wrong
about in the same way. ``tests/test_parity.py`` does the real browser comparison when a node
harness is available; this file is what runs everywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from siarscan.dsp.stft import frame_count, grid_bytes, stft
from siarscan.dsp.windows import blackman, by_name, hamming, hann, rectangular


def test_hann_is_symmetric_not_periodic():
    """The app divides by n-1, so the window touches zero at BOTH ends.

    numpy's ``hanning`` agrees; ``scipy.signal.get_window('hann', n)`` does not, and a periodic
    window would shift every magnitude by a fraction of a dB. That is invisible on a picture and
    enough to move a box by a bin when a cell sits on a scanner's threshold.
    """
    w = hann(8)
    assert w[0] == pytest.approx(0.0)
    assert w[-1] == pytest.approx(0.0)
    assert w[4] == pytest.approx(w[3], abs=0.2)
    np.testing.assert_allclose(w, np.hanning(8), atol=1e-6)


def test_other_windows_match_numpy():
    np.testing.assert_allclose(hamming(16), np.hamming(16), atol=1e-6)
    np.testing.assert_allclose(blackman(16), np.blackman(16), atol=1e-6)
    np.testing.assert_array_equal(rectangular(5), np.ones(5, dtype=np.float32))


def test_by_name_rejects_a_typo_with_the_choices():
    with pytest.raises(ValueError, match="hann"):
        by_name("hanning")


def test_frame_count_matches_the_browser_formula():
    """floor((n - fft) / hop) + 1, and zero below one window. No padding, no centring."""
    assert frame_count(1024, 1024, 256) == 1
    assert frame_count(1023, 1024, 256) == 0
    assert frame_count(1024 + 256, 1024, 256) == 2
    assert frame_count(96000, 1024, 256) == (96000 - 1024) // 256 + 1


def test_short_signal_yields_an_empty_grid_not_an_error():
    """One too-short recording in a folder is a fact about the folder, not a reason to stop."""
    result = stft(np.zeros(100, dtype=np.float32), fft_size=256, hop_size=64)
    assert result.frames == 0
    assert result.magnitudes.shape == (0, 129)


def test_first_frame_starts_at_sample_zero():
    """A scanner reports boxes at frame * hop / rate. If frame 0 were centred half a window
    earlier, every box this CLI wrote would land in the wrong place in the app."""
    signal = np.zeros(2048, dtype=np.float32)
    signal[:256] = 1.0  # energy only in the very first window
    result = stft(signal, fft_size=256, hop_size=256, window_name="rectangular")
    assert result.magnitudes[0].max() > 100.0
    assert result.magnitudes[1].max() == pytest.approx(0.0)


def test_magnitudes_match_a_direct_dft():
    """One windowed frame against an O(n^2) DFT — the reference an FFT cannot be subtly wrong
    against, and the check that the magnitudes are |X| and not power or dB."""
    n = 64
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(n).astype(np.float32)

    got = stft(signal, fft_size=n, hop_size=n, window_name="hann").magnitudes[0]

    windowed = signal.astype(np.float64) * np.hanning(n)
    k = np.arange(n // 2 + 1)[:, None]
    t = np.arange(n)[None, :]
    want = np.abs((windowed[None, :] * np.exp(-2j * np.pi * k * t / n)).sum(axis=1))

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_blocking_does_not_change_the_answer():
    """The transform runs in blocks to bound peak memory; crossing a block boundary must be
    invisible. Checked by comparing a signal long enough to span several blocks against the
    same frames computed one at a time."""
    # import_module, not `import siarscan.dsp.stft as ...`: the package re-exports the
    # function under that name, so attribute lookup would hand back the callable.
    from importlib import import_module

    stft_mod = import_module("siarscan.dsp.stft")

    rng = np.random.default_rng(11)
    signal = rng.standard_normal(64 * 1200 + 256).astype(np.float32)
    blocked = stft(signal, fft_size=256, hop_size=64).magnitudes

    original = stft_mod._BLOCK_FRAMES
    try:
        stft_mod._BLOCK_FRAMES = 7  # a block size that divides nothing evenly
        awkward = stft(signal, fft_size=256, hop_size=64).magnitudes
    finally:
        stft_mod._BLOCK_FRAMES = original

    np.testing.assert_array_equal(blocked, awkward)


def test_grid_bytes_is_the_number_the_warning_prints():
    """225,000 frames of 513 float32 bins is the ten-minute-at-96kHz case the CLI warns about."""
    assert grid_bytes(96000 * 600, 1024, 256) == frame_count(96000 * 600, 1024, 256) * 513 * 4


@pytest.mark.parametrize("fft_size", [0, 3, 100, 1000])
def test_non_power_of_two_is_refused(fft_size):
    with pytest.raises(ValueError, match="power of two"):
        stft(np.zeros(4096, dtype=np.float32), fft_size=fft_size, hop_size=1)


@pytest.mark.parametrize("hop", [0, -1, 513])
def test_hop_out_of_range_is_refused(hop):
    with pytest.raises(ValueError, match="hop_size"):
        stft(np.zeros(4096, dtype=np.float32), fft_size=512, hop_size=hop)
