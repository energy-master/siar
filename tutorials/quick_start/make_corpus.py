# Vixen Intelligence c.2026
"""Quick start — build a sample corpus you can train on, score, and view in the dashboard.

This writes audio and nothing else. Training and scoring are the CLI's job, and the point of the
quick start is to use the CLI:

    python tutorials/quick_start/make_corpus.py
    siar train tutorials/quick_start/corpus/normal --name machine-room
    siar run <model-uid> tutorials/quick_start/corpus/check
    siar dash --open

The corpus is a **machine room**: a steady electrical hum over a broadband floor. That is what
"normal" is going to mean, and it is deliberately more structured than plain noise — a detector
that only works on white noise is not telling you anything.

    corpus/normal/   24 files, background only          <- train on this
    corpus/check/     6 files, background only          <- must come back clean
                      1 file with a chirp    (a sweep — something moving)
                      1 file with 3 impacts  (transients — something struck)
                      1 file with a squeal   (a steady tone — a bearing going)

Every anomaly is planted at a time and frequency we choose, so the answer is knowable and you can
check SIAR's boxes against the truth. That is the whole reason the demo audio is synthetic: in
unsupervised detection there is no ground truth, so we manufacture some. See ``--verify``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ["main"]

SAMPLE_RATE = 16_000
DURATION_S = 5
ROOT = Path(__file__).parent / "corpus"

#: The hum that defines "normal": a mains fundamental and its harmonics.
HUM_HZ = (100.0, 200.0, 300.0)

#: What we plant, and where. Compare these against what `siar run` reports.
CHIRP = {"t0": 3.0, "t1": 4.0, "f0": 2000.0, "f1": 2800.0}
IMPACTS = {"times": (1.0, 2.5, 4.0), "freq": 6000.0}
SQUEAL = {"t0": 1.5, "t1": 3.5, "freq": 5000.0}


def background(rng: np.random.Generator) -> np.ndarray:
    """A recording of the machine room with nothing wrong in it.

    Pink (1/f) noise for the room, plus a steady hum with a little wobble on each harmonic so that
    twenty-four files are alike but not identical — a corpus of byte-identical files teaches the
    autoencoder to memorise one waveform, not to model a distribution.

    Args:
        rng: Source of randomness.

    Returns:
        float32 mono waveform, ``SAMPLE_RATE * DURATION_S`` samples.
    """
    n = SAMPLE_RATE * DURATION_S
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    k = np.arange(spectrum.size)
    k[0] = 1
    x = 0.04 * np.fft.irfft(spectrum / np.sqrt(k), n)

    t = np.arange(n) / SAMPLE_RATE
    for i, hz in enumerate(HUM_HZ):
        amplitude = 0.06 / (i + 1) * rng.uniform(0.9, 1.1)
        drift = rng.uniform(-0.5, 0.5)
        x += amplitude * np.sin(2 * np.pi * (hz + drift) * t + rng.uniform(0, 2 * np.pi))

    return x.astype(np.float32)


def with_chirp(rng: np.random.Generator) -> np.ndarray:
    """Background plus a frequency sweep — something moving through the room."""
    x = background(rng)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    rate = CHIRP["f1"] - CHIRP["f0"]
    phase = 2 * np.pi * (CHIRP["f0"] * t + 0.5 * rate * t**2)
    start = int(CHIRP["t0"] * SAMPLE_RATE)
    x[start : start + SAMPLE_RATE] += (0.15 * np.sin(phase)).astype(np.float32)
    return x


def with_impacts(rng: np.random.Generator) -> np.ndarray:
    """Background plus three short transients — something being struck."""
    x = background(rng)
    width = 400
    envelope = np.hanning(width)
    tone = np.sin(2 * np.pi * IMPACTS["freq"] * np.arange(width) / SAMPLE_RATE)
    for when in IMPACTS["times"]:
        at = int(when * SAMPLE_RATE)
        x[at : at + width] += (0.35 * envelope * tone).astype(np.float32)
    return x


def with_squeal(rng: np.random.Generator) -> np.ndarray:
    """Background plus a sustained tone — a bearing on its way out."""
    x = background(rng)
    start = int(SQUEAL["t0"] * SAMPLE_RATE)
    stop = int(SQUEAL["t1"] * SAMPLE_RATE)
    n = stop - start
    t = np.arange(n) / SAMPLE_RATE
    # Fade in and out, or the edges are a bigger anomaly than the tone itself.
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.2)
    x[start:stop] += (0.07 * fade * np.sin(2 * np.pi * SQUEAL["freq"] * t)).astype(np.float32)
    return x


def planted() -> list[tuple[str, str, str]]:
    """The ground truth, as ``(file, time, frequency)`` rows.

    Returns:
        One row per planted event, for printing and for checking the boxes against.
    """
    return [
        ("chirp", f"{CHIRP['t0']:.1f}–{CHIRP['t1']:.1f} s",
         f"{CHIRP['f0']:.0f}–{CHIRP['f1']:.0f} Hz (sweeping)"),
        ("impacts", ", ".join(f"{s:.1f} s" for s in IMPACTS["times"]),
         f"~{IMPACTS['freq']:.0f} Hz (broadband)"),
        ("squeal", f"{SQUEAL['t0']:.1f}–{SQUEAL['t1']:.1f} s",
         f"{SQUEAL['freq']:.0f} Hz (steady)"),
    ]


def main() -> None:
    """Write the corpus and print the commands that follow."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=11, help="rng seed (default: 11)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    normal_dir = ROOT / "normal"
    check_dir = ROOT / "check"
    normal_dir.mkdir(parents=True, exist_ok=True)
    check_dir.mkdir(parents=True, exist_ok=True)

    for i in range(24):
        sf.write(normal_dir / f"normal_{i:02d}.wav", background(rng), SAMPLE_RATE)
    for i in range(6):
        sf.write(check_dir / f"quiet_{i:02d}.wav", background(rng), SAMPLE_RATE)
    sf.write(check_dir / "chirp.wav", with_chirp(rng), SAMPLE_RATE)
    sf.write(check_dir / "impacts.wav", with_impacts(rng), SAMPLE_RATE)
    sf.write(check_dir / "squeal.wav", with_squeal(rng), SAMPLE_RATE)

    print(f"wrote {normal_dir}  24 files, background only  (train on this)")
    print(f"wrote {check_dir}   6 clean + 3 with planted events  (score this)")
    print()
    print("planted, for you to check SIAR's boxes against:")
    for name, when, where in planted():
        print(f"  {name:<8}  {when:<22}  {where}")
    print()
    print("now, in order:")
    print(f"  siar train {normal_dir} --name machine-room")
    print(f"  siar run <model-uid> {check_dir}")
    print("  siar dash --open")


if __name__ == "__main__":
    main()
