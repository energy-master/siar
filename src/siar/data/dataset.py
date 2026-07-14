# Vixen Intelligence c.2026
"""Discovering a corpus on disk, and splitting it.

Two jobs:

**Discovery** — walk a folder, read audio headers (never full decodes), and report what is
there. :func:`scan_folder` is deliberately loud about heterogeneity: a corpus with three
different sample rates in it is usually a mistake the user wants to know about *before* a
one-hour optimisation, not after.

**Splitting** — :func:`split_files` divides the corpus into train / validation / calibration.
The split is at **file** level, never frame level. Frames from the same recording share a noise
floor, a microphone and often the same events; splitting by frame leaks all of that across the
boundary and every model then looks excellent on validation and fails on new recordings. This is
the single easiest way to build an anomaly detector that is silently worthless, so the split is
not configurable down to frames — only whole files move.
"""
from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

from siar.data.audio import read_info

__all__ = ["AUDIO_EXTENSIONS", "AudioFile", "CorpusScan", "Split", "scan_folder", "split_files"]

#: Extensions we attempt to read. libsndfile handles all of these.
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".w64")


@dataclass(frozen=True, slots=True)
class AudioFile:
    """One readable recording found on disk.

    Attributes:
        path: Absolute path.
        name: Base name without extension.
        sample_rate: Sample rate in Hz.
        channels: Channel count in the source file.
        n_samples: Length in samples (per channel).
        duration: Length in seconds.
        size_bytes: File size on disk.
    """

    path: str
    name: str
    sample_rate: int
    channels: int
    n_samples: int
    duration: float
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CorpusScan:
    """The result of walking a folder.

    Attributes:
        root: The folder that was scanned.
        files: Every readable recording, sorted by path.
        unreadable: ``(path, reason)`` for every file that looked like audio but would not open.
    """

    root: str
    files: tuple[AudioFile, ...]
    unreadable: tuple[tuple[str, str], ...]

    @property
    def n_files(self) -> int:
        """Number of readable recordings."""
        return len(self.files)

    @property
    def total_duration(self) -> float:
        """Total audio duration in seconds."""
        return sum(f.duration for f in self.files)

    @property
    def sample_rates(self) -> dict[int, int]:
        """Map of sample rate -> how many files have it."""
        return dict(Counter(f.sample_rate for f in self.files))

    @property
    def dominant_sample_rate(self) -> int | None:
        """The most common sample rate, or ``None`` for an empty corpus.

        This is the sensible default for ``FeatureSpec.sample_rate``: resampling the minority
        costs least.
        """
        if not self.files:
            return None
        return Counter(f.sample_rate for f in self.files).most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class Split:
    """A file-level partition of a corpus.

    Attributes:
        train: Files the detector learns "normal" from.
        val: Held-out files used to score a trial during optimisation.
        calib: Held-out files used to fit the detection threshold. Kept separate from ``val``
            so the threshold is not tuned on the same data that chose the model — otherwise the
            reported false-positive rate is optimistic and the user finds out in production.
    """

    train: tuple[AudioFile, ...]
    val: tuple[AudioFile, ...]
    calib: tuple[AudioFile, ...]


def scan_folder(root: str, *, recursive: bool = True) -> CorpusScan:
    """Walk a folder and read the header of every audio file in it.

    Args:
        root: Folder to scan.
        recursive: Descend into subdirectories.

    Returns:
        The :class:`CorpusScan`.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")

    candidates: list[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(AUDIO_EXTENSIONS):
                    candidates.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            full = os.path.join(root, fn)
            if os.path.isfile(full) and fn.lower().endswith(AUDIO_EXTENSIONS):
                candidates.append(full)

    files: list[AudioFile] = []
    unreadable: list[tuple[str, str]] = []
    for path in sorted(candidates):
        try:
            sr, ch, n, dur = read_info(path)
        except RuntimeError as exc:
            unreadable.append((os.path.abspath(path), str(exc)))
            continue
        files.append(
            AudioFile(
                path=os.path.abspath(path),
                name=os.path.splitext(os.path.basename(path))[0],
                sample_rate=sr,
                channels=ch,
                n_samples=n,
                duration=dur,
                size_bytes=os.path.getsize(path),
            )
        )

    return CorpusScan(os.path.abspath(root), tuple(files), tuple(unreadable))


def split_files(
    files: tuple[AudioFile, ...],
    *,
    val_fraction: float = 0.15,
    calib_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """Partition a corpus into train / val / calib, by file.

    Every split gets at least one file whenever the corpus is large enough to allow it (three
    files); below that, ``val`` and ``calib`` fall back to reusing ``train``, and the caller is
    responsible for warning that the resulting threshold and objective are not held out.

    Args:
        files: The corpus.
        val_fraction: Fraction of files held out to score optimisation trials.
        calib_fraction: Fraction of files held out to fit the detection threshold.
        seed: Seed for the shuffle, so a split is reproducible from the run record.

    Returns:
        The :class:`Split`.

    Raises:
        ValueError: If the fractions are not in ``[0, 1)`` or leave no training data.
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= calib_fraction < 1.0:
        raise ValueError("val_fraction and calib_fraction must each be in [0, 1)")
    if val_fraction + calib_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + calib_fraction must be < 1, got "
            f"{val_fraction} + {calib_fraction}"
        )
    if not files:
        raise ValueError("cannot split an empty corpus")

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        return Split(train=tuple(shuffled), val=tuple(shuffled), calib=tuple(shuffled))

    n_val = max(1, round(n * val_fraction))
    n_calib = max(1, round(n * calib_fraction))
    if n_val + n_calib >= n:  # pathological fractions on a tiny corpus
        n_val = n_calib = 1

    val = tuple(shuffled[:n_val])
    calib = tuple(shuffled[n_val : n_val + n_calib])
    train = tuple(shuffled[n_val + n_calib :])
    return Split(train=train, val=val, calib=calib)
