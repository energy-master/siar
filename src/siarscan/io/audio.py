# Vixen Intelligence c.2026
"""Reading recordings off disk and reducing them to one mono signal.

Two things here differ from the browser and both are deliberate improvements the CLI can make
because it is not a browser:

* **No resampling.** ``decodeAudioData`` resamples everything to the tab's ``AudioContext``
  rate — typically 48 kHz — which throws away the top half of a 96 kHz sonar recording before
  a scanner ever sees it. ``soundfile`` reads the file's native rate, so a harbour-porpoise
  click at 130 kHz survives. The bin-to-Hz mapping is derived from that native rate, so boxes
  still land where the app draws them.
* **Header-only probing.** :func:`probe` reads duration, rate and channel count without
  decoding, so a run can plan (and warn about mixed sample rates) across a 10,000-file corpus
  in a second.

Channel selection matches ``js/audio/channel.js`` exactly: ``mix`` averages every channel,
``left`` / ``right`` take one, and a mono file ignores the request.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

__all__ = [
    "AUDIO_EXTENSIONS",
    "AudioInfo",
    "Recording",
    "find_recordings",
    "is_audio",
    "load_mono",
    "probe",
    "to_mono",
]

#: Extensions the walker treats as recordings. WAV and FLAC are what the app itself accepts;
#: the rest are read if soundfile can, because a folder is rarely purely one format.
AUDIO_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif", ".ogg", ".w64")

#: Filesystem debris that looks like audio to a suffix test. macOS writes an AppleDouble
#: ``._name.wav`` beside every file on a non-HFS volume; it is a 4 KB resource fork, not audio,
#: and a run that tries to decode a few thousand of them logs a few thousand errors.
_JUNK_PREFIXES = ("._",)


class AudioInfo:
    """What a file's header says, without decoding it.

    Attributes:
        path: Absolute path to the recording.
        sample_rate: Native sample rate in Hz.
        frames: Sample count per channel.
        channels: Channel count.
        format: soundfile's format string (``"WAV"``, ``"FLAC"``, ...).
    """

    __slots__ = ("path", "sample_rate", "frames", "channels", "format")

    def __init__(self, path: str, sample_rate: float, frames: int, channels: int, fmt: str):
        self.path = path
        self.sample_rate = float(sample_rate)
        self.frames = int(frames)
        self.channels = int(channels)
        self.format = str(fmt)

    @property
    def duration_sec(self) -> float:
        """Length in seconds."""
        return self.frames / self.sample_rate if self.sample_rate else 0.0


class Recording:
    """A decoded mono recording.

    Attributes:
        path: Where it came from.
        samples: 1-D ``float32`` mono signal.
        sample_rate: Native sample rate in Hz.
        channels: How many channels the source had (before mixing).
    """

    __slots__ = ("path", "samples", "sample_rate", "channels")

    def __init__(self, path: str, samples: np.ndarray, sample_rate: float, channels: int):
        self.path = path
        self.samples = samples
        self.sample_rate = float(sample_rate)
        self.channels = int(channels)

    @property
    def duration_sec(self) -> float:
        """Length in seconds."""
        return self.samples.shape[0] / self.sample_rate if self.sample_rate else 0.0


def is_audio(name: str) -> bool:
    """True when a filename looks like a recording we should try to read."""
    base = os.path.basename(name)
    if base.startswith(_JUNK_PREFIXES):
        return False
    return base.lower().endswith(AUDIO_EXTENSIONS)


def find_recordings(root: str, *, recursive: bool = True) -> list[str]:
    """Every recording under ``root``, sorted by relative path.

    Sorted, not in ``os.walk`` order: the output manifest, the console progress and the app's
    lane order should agree between runs and between machines, and directory order does not.

    Args:
        root: Folder to walk.
        recursive: Descend into subdirectories.

    Returns:
        Absolute paths.
    """
    root = os.path.abspath(root)
    if not recursive:
        try:
            names = os.listdir(root)
        except OSError:
            return []
        return sorted(os.path.join(root, n) for n in names if is_audio(n))

    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip our own output scaffolding and hidden trees (.thumbs, .git, ...).
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for n in filenames:
            if is_audio(n):
                found.append(os.path.join(dirpath, n))
    return sorted(found, key=lambda p: os.path.relpath(p, root))


def probe(path: str) -> AudioInfo | None:
    """Read a recording's header. Returns ``None`` if it cannot be read at all."""
    import soundfile as sf

    try:
        info = sf.info(path)
    except Exception:  # noqa: BLE001 - a corrupt header is a per-file outcome, not a crash
        return None
    return AudioInfo(os.path.abspath(path), info.samplerate, info.frames, info.channels,
                     getattr(info, "format", ""))


def to_mono(data: np.ndarray, selection: str = "mix") -> np.ndarray:
    """Reduce a decoded ``(frames, channels)`` (or 1-D) block to one channel.

    Args:
        data: Samples as soundfile returns them.
        selection: ``"mix"`` (average every channel), ``"left"``, ``"right"``, or ``"N"`` for a
            zero-based channel index.

    Returns:
        A 1-D ``float32`` signal.

    Raises:
        ValueError: If ``selection`` is neither a known name nor a channel index.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)

    channels = arr.shape[1]
    if channels == 1 or selection == "left":
        return np.ascontiguousarray(arr[:, 0], dtype=np.float32)
    if selection == "right":
        return np.ascontiguousarray(arr[:, min(1, channels - 1)], dtype=np.float32)
    if selection == "mix":
        return arr.mean(axis=1).astype(np.float32)
    if selection.isdigit():
        idx = int(selection)
        if idx >= channels:
            raise ValueError(f"channel {idx} requested but the file has {channels}")
        return np.ascontiguousarray(arr[:, idx], dtype=np.float32)
    raise ValueError(f"unknown channel selection {selection!r} (mix, left, right, or an index)")


def load_mono(path: str, *, channel: str = "mix") -> Recording:
    """Decode a recording to a mono signal at its native sample rate.

    Args:
        path: The recording.
        channel: See :func:`to_mono`.

    Returns:
        The :class:`Recording`.

    Raises:
        OSError: If the file cannot be read or decoded. Callers in a per-file loop should catch
            this and carry on; the run manifest records the failure.
    """
    import soundfile as sf

    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - soundfile raises several unrelated types
        raise OSError(f"could not decode {path}: {e}") from e
    return Recording(os.path.abspath(path), to_mono(data, channel), rate, data.shape[1])


def mixed_sample_rates(infos: Iterable[AudioInfo]) -> list[float]:
    """The distinct sample rates in a corpus, sorted.

    More than one is nearly always an accident, and much cheaper to hear about before a scan
    than after: every band-limited scanner's Hz-to-bin mapping depends on the rate, so a folder
    at two rates is really two folders.
    """
    return sorted({i.sample_rate for i in infos if i is not None})
