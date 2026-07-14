# Vixen Intelligence c.2026
"""Tutorial 02 — build the demo corpus for the train / run / dashboard walkthrough.

This script only writes the audio. The tutorial itself is three shell commands — the point is to
use the actual product, not to drive the library from Python (tutorial 01 does that).

    python tutorials/02_train_run_dashboard.py

    siar train siar-demo/normal --name harbour-baseline
    siar run <model-uid> siar-demo/check
    siar dash --open

Two folders are written:

    siar-demo/normal/   20 files of pink noise — what "normal" is going to mean
    siar-demo/check/     4 files of pink noise, which should come back clean
                         1 file with a chirp planted at a known time and frequency
                         1 file with two clicks planted at known times

The anomalies are planted so that the answer is knowable. The detector is told nothing about them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION_S = 4
ROOT = Path("siar-demo")

# What we plant, and where. Compare these against what `siar run` reports.
CHIRP = {"t0": 2.0, "t1": 3.0, "f0": 3000.0, "f1": 3500.0}
CLICKS = {"times": (1.5, 2.5), "freq": 5500.0}


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise — a stand-in for a natural background."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    k = np.arange(spectrum.size)
    k[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(k), n).astype(np.float32)


def normal(rng: np.random.Generator) -> np.ndarray:
    """A recording with nothing in it but background."""
    return 0.05 * pink_noise(SAMPLE_RATE * DURATION_S, rng)


def with_chirp(rng: np.random.Generator) -> np.ndarray:
    """Background plus a frequency sweep."""
    x = normal(rng)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    rate = CHIRP["f1"] - CHIRP["f0"]
    phase = 2 * np.pi * (CHIRP["f0"] * t + 0.5 * rate * t**2)
    start = int(CHIRP["t0"] * SAMPLE_RATE)
    x[start : start + SAMPLE_RATE] += (0.10 * np.sin(phase)).astype(np.float32)
    return x


def with_clicks(rng: np.random.Generator) -> np.ndarray:
    """Background plus two short broadband-ish transients."""
    x = normal(rng)
    width = 400
    envelope = np.hanning(width)
    tone = np.sin(2 * np.pi * CLICKS["freq"] * np.arange(width) / SAMPLE_RATE)
    for when in CLICKS["times"]:
        at = int(when * SAMPLE_RATE)
        x[at : at + width] += (0.35 * envelope * tone).astype(np.float32)
    return x


def main() -> None:
    """Write the demo corpus."""
    rng = np.random.default_rng(7)
    normal_dir = ROOT / "normal"
    check_dir = ROOT / "check"
    normal_dir.mkdir(parents=True, exist_ok=True)
    check_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        sf.write(normal_dir / f"normal_{i:02d}.wav", normal(rng), SAMPLE_RATE)
    for i in range(4):
        sf.write(check_dir / f"quiet_{i:02d}.wav", normal(rng), SAMPLE_RATE)
    sf.write(check_dir / "chirp_at_2s.wav", with_chirp(rng), SAMPLE_RATE)
    sf.write(check_dir / "clicks_at_1.5_2.5s.wav", with_clicks(rng), SAMPLE_RATE)

    print(f"wrote {normal_dir}  20 files of pink noise")
    print(f"wrote {check_dir}   4 clean + 2 with planted events")
    print()
    print("planted, for you to check the answer against:")
    print(f"  chirp_at_2s          t = {CHIRP['t0']}-{CHIRP['t1']} s, "
          f"f = {CHIRP['f0']:.0f}-{CHIRP['f1']:.0f} Hz")
    print(f"  clicks_at_1.5_2.5s   t = {CLICKS['times'][0]} s and {CLICKS['times'][1]} s, "
          f"f ~ {CLICKS['freq']:.0f} Hz")
    print()
    print("now:")
    print(f"  siar train {normal_dir} --name harbour-baseline")
    print("  siar run <model-uid> " + str(check_dir))
    print("  siar dash --open")


if __name__ == "__main__":
    main()
