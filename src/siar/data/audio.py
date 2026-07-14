# Vixen Intelligence c.2026
"""Loading recordings into mono ``float32`` signals.

A thin wrapper over ``soundfile`` (libsndfile): read the file, mix any multi-channel source down
to mono by averaging, hand back an :class:`AudioRecording`. Everything above this layer assumes
mono ``float32``, so this is the only place that has to know about channel layouts and integer
sample formats.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = ["AudioRecording", "load_audio", "read_info"]


@dataclass(frozen=True, slots=True)
class AudioRecording:
    """A decoded mono signal plus its metadata.

    Attributes:
        name: The file's base name without extension.
        path: The path it was loaded from.
        samples: The mono signal, 1-D ``float32``.
        sample_rate: Sample rate in Hz.
        channels: Channel count of the *source* file, before the mono mixdown.
    """

    name: str
    path: str
    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def n_samples(self) -> int:
        """Length of the mono signal in samples."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_samples / float(self.sample_rate)


def load_audio(path: str) -> AudioRecording:
    """Decode an audio file to a mono ``float32`` signal.

    Args:
        path: Path to a file libsndfile can read (WAV, FLAC, OGG, AIFF, ...).

    Returns:
        The decoded :class:`AudioRecording`.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    import soundfile as sf

    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises a variety of types
        raise RuntimeError(f"could not decode {path}: {exc}") from exc

    channels = int(data.shape[1])
    samples = data.mean(axis=1).astype(np.float32) if channels > 1 else data[:, 0]
    name = os.path.splitext(os.path.basename(path))[0]
    return AudioRecording(
        name=name,
        path=os.path.abspath(path),
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=int(sample_rate),
        channels=channels,
    )


def read_info(path: str) -> tuple[int, int, int, float]:
    """Read an audio file's header without decoding it.

    Scanning a corpus of several GB should not cost a full decode, so ``siar scan`` uses this.

    Args:
        path: Path to the audio file.

    Returns:
        ``(sample_rate, channels, n_frames, duration_seconds)``.

    Raises:
        RuntimeError: If the header cannot be read.
    """
    import soundfile as sf

    try:
        info = sf.info(path)
    except Exception as exc:
        raise RuntimeError(f"could not read header of {path}: {exc}") from exc
    return int(info.samplerate), int(info.channels), int(info.frames), float(info.duration)
