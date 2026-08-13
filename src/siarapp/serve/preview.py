# Vixen Intelligence c.2026
"""One recording, drawn small enough to send over a tunnel.

The analysis grid a scan builds is **four times the size of the audio it came from** — a
forty-minute recording at 96 kHz is 440 MB of WAV and 1.76 GB of ``float32`` magnitudes — so
shipping it to a browser would be worse than shipping the recording. What a reader actually needs
is a picture at the size it will be drawn: a couple of thousand columns, a couple of hundred rows,
dB-normalised and quantised to a byte each. That is 150 KB at the default size and 2 MB at the
largest this will draw — **whatever the recording's length** — and it is what this module produces.

Two things make it cheap, and both matter more than the arithmetic:

* **The windows are sparse.** :func:`siarapp.viz.thumbnail.sparse_frame_starts` spreads ``width``
  windows evenly across the recording instead of hopping through it, so the cost is the number of
  columns asked for, not the duration.
* **The file is seeked, not decoded.** Each window is one ``seek`` and one ``read`` of ``fft``
  frames, so drawing 2000 columns of a forty-minute recording reads about 2 M samples out of 230 M.
  :func:`siarapp.io.audio.load_mono` is never called here — decoding 921 MB to draw 0.5 MB of it is
  the thing this module exists to avoid.

The reduction itself is :mod:`siarapp.viz.thumbnail`'s, step for step. That is deliberate: the lane
strip is 200x64 of exactly this picture, and a preview computed a second way would be a fourth
implementation of an image three are already supposed to keep in step.

**The height asked for is a ceiling, not a promise.** Frequency pooling divides the bin count by an
integer, so 257 bins asked down to 200 stay 257. Every reply states the height it actually produced
and a client that assumes otherwise draws a squashed picture.
"""
from __future__ import annotations

import struct

import numpy as np

from siarapp.io.audio import to_mono
from siarapp.viz.png import encode_png
from siarapp.viz.thumbnail import (
    DB_FLOOR,
    colourise,
    hann_window,
    normalise_db,
    pool_bins,
    sparse_frame_starts,
)

__all__ = [
    "PREVIEW_DEFAULT_BINS",
    "PREVIEW_DEFAULT_WIDTH",
    "PREVIEW_HEADER_BYTES",
    "PREVIEW_MAGIC",
    "PREVIEW_MAX_ACTUAL_BINS",
    "PREVIEW_MAX_BINS",
    "PREVIEW_MAX_SAMPLES",
    "PREVIEW_MAX_WIDTH",
    "PREVIEW_MIN_FFT",
    "Preview",
    "actual_bins",
    "encode_preview",
    "preview_bounds",
    "preview_png",
    "read_preview",
]

#: Columns and rows a request may ask for. Wide enough for a 4K window, bounded so a URL cannot
#: ask the box for a gigabyte of transforms.
PREVIEW_MAX_WIDTH = 4000
PREVIEW_MAX_BINS = 512

#: Rows a reply may actually contain, which is one more than can be asked for.
#:
#: An rfft of a power-of-two window gives ``2^k + 1`` bins — the Nyquist bin sits on top of a power
#: of two — and the pooling factor is an integer, so 513 bins asked down to 512 stay 513. Rather
#: than trim a real bin to make a round number, the reply reports the height it reached and this is
#: the honest ceiling for a size bound.
PREVIEW_MAX_ACTUAL_BINS = PREVIEW_MAX_BINS + 1

#: What a caller that says nothing gets: a picture for a normal window at a normal zoom.
PREVIEW_DEFAULT_WIDTH = 1200
PREVIEW_DEFAULT_BINS = 128

#: Samples one preview may read, across every window together. The ceiling that keeps a request
#: honest: 4000 columns of a 2048-point window is 8 M samples, which is a tenth of a second of work
#: and a few megabytes of reads however long the recording is.
PREVIEW_MAX_SAMPLES = 8_000_000

#: Smallest transform used, whatever height was asked for. Below this the frequency resolution is
#: too coarse to show a tonal, and the saving is nothing.
PREVIEW_MIN_FFT = 512

#: The raw wire format: magic, then a fixed 32-byte header, then ``width * height`` bytes.
PREVIEW_MAGIC = b"SIARPRV1"
PREVIEW_HEADER_BYTES = 32


class Preview:
    """A reduced spectrogram, ready to encode.

    Attributes:
        cells: ``(height, width)`` ``uint8``, **row 0 is the lowest frequency** — the natural order
            for a caller building a surface from it. :func:`preview_png` flips it, because a
            picture is drawn the other way up.
        width: Columns, which is frames.
        height: Rows, which is frequency bins. What pooling *achieved*, not what was asked for.
        duration_sec: The recording's length, so a column maps to a time.
        nyquist_hz: Half the sample rate, so a row maps to a frequency.
        db_floor: Decibels below the recording's own peak that map to 0.
    """

    __slots__ = ("cells", "width", "height", "duration_sec", "nyquist_hz", "db_floor")

    def __init__(self, cells: np.ndarray, *, duration_sec: float, nyquist_hz: float,
                 db_floor: float) -> None:
        self.cells = cells
        self.height, self.width = int(cells.shape[0]), int(cells.shape[1])
        self.duration_sec = float(duration_sec)
        self.nyquist_hz = float(nyquist_hz)
        self.db_floor = float(db_floor)

    def header(self) -> dict:
        """The five numbers a reader needs to put axes on the picture."""
        return {
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 6),
            "nyquist_hz": round(self.nyquist_hz, 3),
            "db_floor": round(self.db_floor, 2),
        }


def preview_bounds(width: int | None = None, height: int | None = None) -> tuple[int, int, int]:
    """Turn a request into what will actually be computed.

    Pure, so a caller — or a test — can ask what a request costs before paying for it.

    Args:
        width: Columns asked for, or ``None`` for the default.
        height: Rows asked for, or ``None`` for the default. A ceiling: see the module docstring.

    Returns:
        ``(width, fft_size, bin_target)``. ``width`` is clamped, and reduced further when the
        window needed for the requested height would read more than :data:`PREVIEW_MAX_SAMPLES`.
        ``fft_size`` is derived from the height and never taken from the wire.
    """
    columns = _clamp(width, PREVIEW_DEFAULT_WIDTH, 16, PREVIEW_MAX_WIDTH)
    bins = _clamp(height, PREVIEW_DEFAULT_BINS, 8, PREVIEW_MAX_BINS)

    # The smallest power of two whose rfft gives at least the rows asked for. Small windows are
    # cheap, so there is no reason to transform more finely than the picture can show.
    fft_size = PREVIEW_MIN_FFT
    while fft_size // 2 + 1 < bins and fft_size < 1 << 16:
        fft_size *= 2

    if columns * fft_size > PREVIEW_MAX_SAMPLES:
        columns = max(16, PREVIEW_MAX_SAMPLES // fft_size)
    return columns, fft_size, bins


def actual_bins(fft_size: int, bin_target: int) -> int:
    """How many rows :func:`siarapp.viz.thumbnail.pool_bins` will really produce.

    Pooling divides by an integer, so this is at least ``bin_target`` and can be one more than it —
    the Nyquist bin on top of a power of two. Exposed so a caller can size a buffer, or a bound,
    without computing a picture first.
    """
    stft_bins = fft_size // 2 + 1
    return stft_bins // max(1, stft_bins // max(1, int(bin_target)))


def read_preview(path: str, *, width: int | None = None, height: int | None = None,
                 db_floor: float = DB_FLOOR, channel: str = "mix") -> Preview | None:
    """Read ``width`` windows out of a recording and reduce them to a picture.

    Args:
        path: The recording, as it sits in the output folder.
        width: Columns wanted. Clamped by :func:`preview_bounds`.
        height: Rows wanted, as a ceiling.
        db_floor: Decibels below the peak that map to 0.
        channel: ``"mix"``, ``"left"``, ``"right"`` or an index — the run's own channel choice, so
            the preview shows what was actually scanned.

    Returns:
        The :class:`Preview`, or ``None`` when the recording is shorter than one window, or cannot
        be read at all. A recording that cannot be pictured is a fact about the recording, not an
        error — the same rule the per-file scan loop follows.
    """
    import soundfile as sf

    columns, fft_size, bins = preview_bounds(width, height)
    try:
        with sf.SoundFile(path) as fh:
            starts = sparse_frame_starts(int(fh.frames), fft_size, columns)
            if starts.size == 0:
                return None
            sample_rate = float(fh.samplerate)
            duration = float(fh.frames) / sample_rate if sample_rate else 0.0
            window = hann_window(fft_size)
            grid = np.zeros((starts.size, fft_size // 2 + 1), dtype=np.float32)
            for row, start in enumerate(starts):
                fh.seek(int(start))
                block = fh.read(fft_size, dtype="float32", always_2d=True)
                if block.shape[0] < fft_size:
                    # A short tail can only happen if the file shrank under us mid-read; pad it
                    # rather than abandon a picture that is otherwise finished.
                    block = np.pad(block, ((0, fft_size - block.shape[0]), (0, 0)))
                mono = to_mono(block, channel).astype(np.float64) * window
                grid[row] = np.abs(np.fft.rfft(mono))
    except Exception:  # noqa: BLE001 - an unreadable recording is one blank lane, never a 500
        return None

    pooled = pool_bins(normalise_db(grid, db_floor), bins)
    # (frames, bins) -> (bins, frames), row 0 = DC = lowest frequency.
    cells = np.clip(np.rint(pooled.T * 255.0), 0, 255).astype(np.uint8)
    return Preview(np.ascontiguousarray(cells), duration_sec=duration,
                   nyquist_hz=sample_rate / 2.0, db_floor=db_floor)


def preview_png(preview: Preview) -> bytes:
    """The preview as a viridis PNG, drawn the way the lane strip draws it.

    Goes through :func:`siarapp.viz.thumbnail.colourise`, so a lane's 200x64 thumbnail and the
    picture drawn when that lane is opened are the same image at two sizes.
    """
    # `colourise` wants (frames, bins) and flips so the lowest frequency lands at the bottom.
    return encode_png(colourise(preview.cells.T.astype(np.float32) / 255.0))


def encode_preview(preview: Preview) -> bytes:
    """The preview as bytes: the magic, a 32-byte header, then one byte per cell.

    Self-describing on purpose. The same five numbers also travel as response headers, but a body
    that carries them means a reader needs one fetch and no header access — which is what a
    cross-origin caller would otherwise have to be granted.

    Returns:
        ``PREVIEW_HEADER_BYTES + width * height`` bytes.
    """
    head = struct.pack(
        "<8sIIdff",
        PREVIEW_MAGIC,
        preview.width,
        preview.height,
        preview.duration_sec,
        preview.nyquist_hz,
        preview.db_floor,
    )
    assert len(head) == PREVIEW_HEADER_BYTES  # noqa: S101 - a wire format worth asserting once
    return head + preview.cells.tobytes()


def _clamp(value, default: int, low: int, high: int) -> int:
    """One integer from the wire: absent or unparseable becomes the default, then clamp."""
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
