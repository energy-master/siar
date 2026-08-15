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
import time
from typing import Callable, Iterable

import numpy as np

__all__ = [
    "AUDIO_EXTENSIONS",
    "COUNT_BUDGET_SEC",
    "COUNT_LIMIT",
    "AudioInfo",
    "Recording",
    "count_recordings",
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

#: Frames read per block by :func:`load_mono`. A million frames is 4 MB of mono ``float32`` and,
#: at 96 kHz, about eleven seconds of audio: fine enough that the decode bar moves smoothly on
#: the longest recording in a survey, coarse enough that the per-read overhead is invisible
#: against the decoding itself.
_DECODE_BLOCK_FRAMES = 1 << 20

#: Recordings :func:`count_recordings` counts before it stops and says "and more". Far past any
#: figure a reader takes off a form field, far short of walking a survey drive to the end to put
#: an exact number on a line whose only job is to say "yes, there is a corpus here".
COUNT_LIMIT = 10_000

#: Seconds :func:`count_recordings` will walk for before it gives up. A folder of clips is counted
#: in milliseconds and a survey drive in a second or two; a home directory with a decade of
#: everything in it, or a mount that has gone away, is not counted at all — and the difference
#: must not be an interface that never opens.
COUNT_BUDGET_SEC = 4.0

#: How often a walk in progress reports in, in seconds. Slow enough to cost nothing on a local
#: folder, fast enough that a spinner driven by it looks alive rather than stuck.
_COUNT_TICK_SEC = 0.1


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
        root: Folder to walk, or a single recording — which comes back as a corpus of one, so
            that everything above this can be pointed at one file without knowing it is not a
            folder.
        recursive: Descend into subdirectories.

    Returns:
        Absolute paths.
    """
    root = os.path.abspath(root)
    if os.path.isfile(root):
        return [root] if is_audio(root) else []
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


def count_recordings(root: str, *, limit: int = COUNT_LIMIT,
                     budget_sec: float = COUNT_BUDGET_SEC,
                     on_progress: Callable[[int], None] | None = None) -> tuple[int, bool]:
    """How many recordings are under ``root``, giving up rather than taking forever.

    :func:`find_recordings` walks to the end because a run has to — it is about to read every
    file it finds. Answering "is there a corpus here" for a line on a form is a different
    question, asked about paths nobody has vouched for: a working directory, a mount point, a
    home directory. So this walk is bounded twice, by the count and by the clock, and it says
    which of the two stopped it by refusing to call the answer complete.

    Args:
        root: Folder to walk, or a single recording.
        limit: Stop once this many have been found.
        budget_sec: Stop after this long. Always applies; a caller that wants the true figure
            wants :func:`find_recordings`.
        on_progress: Called with the count so far, about every
            :data:`_COUNT_TICK_SEC`, so a caller can say what is happening while it happens. A
            walk that finishes inside one tick never calls it, which is why a spinner driven
            from here does not flash on a folder of six clips.

    Returns:
        ``(count, complete)``. ``complete`` is False when a bound stopped the walk, in which case
        the count is a floor and not a total — "12,000+", never "12,000".
    """
    root = os.path.abspath(root)
    if os.path.isfile(root):
        return (1, True) if is_audio(root) else (0, True)

    started = time.monotonic()
    deadline = started + max(0.0, float(budget_sec))
    next_tick = started + _COUNT_TICK_SEC
    count = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        now = time.monotonic()
        if now >= deadline:
            return count, False
        if on_progress is not None and now >= next_tick:
            next_tick = now + _COUNT_TICK_SEC
            on_progress(count)
        # The same trees find_recordings skips, for the same reasons — a count that disagreed
        # with the walk the run does would be a form field that promises files nothing reads.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if is_audio(name):
                count += 1
                if count >= limit:
                    return count, False
    return count, True


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


def load_mono(path: str, *, channel: str = "mix",
              on_progress: Callable[[int, int], None] | None = None) -> Recording:
    """Decode a recording to a mono signal at its native sample rate.

    Read a block at a time rather than in one call, for two reasons that matter on the long
    recordings this is slowest on. It can *say how far through it is* — the header gives the
    frame count before a byte is decoded, so ``decode`` is one of the two stages of a recording
    that reports real progress instead of an estimate against elapsed time. And it never holds
    the interleaved signal and the mono signal at once: a one-hour stereo recording at 96 kHz is
    2.6 GiB decoded, and mixing it down from a whole-file read needs both that and the 1.3 GiB
    result at the same moment.

    Args:
        path: The recording.
        channel: See :func:`to_mono`.
        on_progress: Called ``(frames_done, frames_total)`` as each block lands, with
            ``frames_total`` of ``0`` when the header would not say how long the file is.

    Returns:
        The :class:`Recording`.

    Raises:
        OSError: If the file cannot be read or decoded. Callers in a per-file loop should catch
            this and carry on; the run manifest records the failure.
        ValueError: If ``channel`` names no channel this file has — the user's mistake, the same
            for every recording in the folder, and not something to bury in a per-file error.
    """
    import soundfile as sf

    try:
        handle = sf.SoundFile(path)
    except Exception as e:  # noqa: BLE001 - soundfile raises several unrelated types
        raise OSError(f"could not decode {path}: {e}") from e

    with handle:
        rate, channels, frames = handle.samplerate, handle.channels, int(handle.frames)
        try:
            samples = _read_mono(handle, channel, frames, on_progress)
        except ValueError:
            raise  # to_mono's: a bad --channel, which is not this file being unreadable
        except Exception as e:  # noqa: BLE001 - a truncated or corrupt file, mid-decode
            raise OSError(f"could not decode {path}: {e}") from e
    return Recording(os.path.abspath(path), samples, rate, channels)


def _read_mono(handle, selection: str, frames: int, on_progress) -> np.ndarray:
    """Read an open recording block by block into one mono signal.

    Args:
        handle: An open ``soundfile.SoundFile``.
        selection: See :func:`to_mono`.
        frames: What the header says the length is, or ``0`` when it will not say. A known
            length is written into one array allocated up front; an unknown one is collected and
            joined, which costs a second copy and is why the header is preferred.
        on_progress: See :func:`load_mono`.

    Returns:
        The 1-D ``float32`` signal, truncated to what was actually there if the file ends early.
    """
    samples = np.empty(frames, dtype=np.float32) if frames > 0 else None
    parts: list[np.ndarray] = []
    done = 0
    for block in handle.blocks(blocksize=_DECODE_BLOCK_FRAMES, dtype="float32", always_2d=True):
        mono = to_mono(block, selection)
        if samples is not None:
            # A file whose header overstates its length is truncated, not fatal: take what fits.
            end = min(frames, done + mono.shape[0])
            samples[done:end] = mono[: end - done]
            done = end
        else:
            parts.append(mono)
            done += mono.shape[0]
        if on_progress is not None:
            on_progress(done, frames)
        if samples is not None and done >= frames:
            break
    if samples is None:
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return samples if done == frames else samples[:done]


def mixed_sample_rates(infos: Iterable[AudioInfo]) -> list[float]:
    """The distinct sample rates in a corpus, sorted.

    More than one is nearly always an accident, and much cheaper to hear about before a scan
    than after: every band-limited scanner's Hz-to-bin mapping depends on the rate, so a folder
    at two rates is really two folders.
    """
    return sorted({i.sample_rate for i in infos if i is not None})
