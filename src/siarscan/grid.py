# Vixen Intelligence c.2026
"""The seam between this open package and a closed scanning algorithm.

Everything an algorithm ever sees is a :class:`FrameGrid`; everything it ever returns is a list
of :class:`Region`. That is the whole contract, and keeping it this narrow is what lets the
algorithms ship obfuscated: the audio, the decode, the file walk and the output writing all
happen out here in readable code, and the part that is worth protecting is handed a matrix.

The grid is field-for-field the evaluation context the browser scanners read — in the web app
every one of them destructures ``{magnitudes, frames, bins, sampleRate, fftSize, hopSize}`` and
nothing else — so a bundle scores a CLI-built grid and a browser-built one identically. It is
deliberately the same shape as the IDent Dynamics SDK's ``identdynamics.FrameGrid``: that
contract returns a per-frame score curve (``score(grid)``), this one returns boxes
(``scan(grid)``), and that single difference is the reason this module exists rather than a
dependency on the SDK.

A conforming algorithm package exposes one top-level factory::

    def algorithm(manifest=None) -> ScannerAlgorithm

:class:`ScannerAlgorithm` is a base class an algorithm *may* subclass; the loader only
duck-types, so an obfuscated package need not import anything from here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "REGION_FIELDS",
    "FrameGrid",
    "Region",
    "ScannerAlgorithm",
    "ScannerError",
    "normalise_regions",
]


class ScannerError(RuntimeError):
    """An algorithm failed to load, or broke its side of the contract."""


#: The fields of a region that survive into a sidecar, in the order the web app's
#: ``js/folder-run.js`` ``STRUCT_KEEP`` lists them. An algorithm may return more (per-run
#: diagnostics: fill, trend, cohesion, membership, frame indices) and :func:`normalise_regions`
#: keeps them under ``extra``, but these nine are the ones the overlay, the structure list and
#: the score-analysis panel actually read, and the only ones a consumer may rely on.
REGION_FIELDS = (
    "tmin",       # seconds from the start of the recording
    "tmax",
    "fmin",       # Hz
    "fmax",
    "peakHz",     # frequency of the region's brightest cell
    "cells",      # active cells in the region — its "mass"
    "peakSigma",  # brightest cell's contrast above its own local background
    "confidence", # [0, 1], anchored to the algorithm's own bar, not to the file's maximum
    "shape",      # 'sweep' | 'tonal' | 'click' | 'click_train' | 'patch' | 'blob' | ...
)


class FrameGrid:
    """The Fourier grid handed to an algorithm — the only input it sees.

    Attributes:
        magnitudes: ``float32`` array shaped ``(frames, bins)``: ``|X|`` per frame per rfft bin,
            linear (**not** dB — the scanners take their own logs, and the choice of log scaling
            is part of what each one does).
        frames: Number of STFT frames (rows).
        bins: Number of rfft bins (columns), ``fft // 2 + 1``.
        sample_rate: Audio sample rate in Hz.
        fft: FFT size used for the STFT.
        hop: Hop size in samples between successive frames.
        window: Window name (``"hann"``, ``"hamming"``, ...).
    """

    __slots__ = ("magnitudes", "frames", "bins", "sample_rate", "fft", "hop", "window")

    def __init__(
        self,
        magnitudes: Any,
        frames: int,
        bins: int,
        sample_rate: float,
        fft: int,
        hop: int,
        window: str = "hann",
    ) -> None:
        self.magnitudes = np.asarray(magnitudes, dtype=np.float32).reshape(int(frames), int(bins))
        self.frames = int(frames)
        self.bins = int(bins)
        self.sample_rate = float(sample_rate)
        self.fft = int(fft)
        self.hop = int(hop)
        self.window = str(window)

    @property
    def seconds_per_frame(self) -> float:
        """Seconds advanced per frame — ``hop / sample_rate``."""
        return self.hop / self.sample_rate if self.sample_rate else 0.0

    @property
    def hz_per_bin(self) -> float:
        """Hz spanned by one rfft bin — ``sample_rate / fft``."""
        return self.sample_rate / self.fft if self.fft else 0.0

    @property
    def duration_sec(self) -> float:
        """Seconds of audio the grid covers, window included."""
        if not self.sample_rate:
            return 0.0
        return (max(0, self.frames - 1) * self.hop + self.fft) / self.sample_rate

    def __repr__(self) -> str:
        return (
            f"FrameGrid(frames={self.frames}, bins={self.bins}, "
            f"sample_rate={self.sample_rate:g}, fft={self.fft}, hop={self.hop}, "
            f"window={self.window!r})"
        )


class Region:
    """One boxed structure: a time span, a frequency band, and what the scanner made of it.

    Algorithms are free to return plain dicts instead — :func:`normalise_regions` accepts both.
    This class exists so a port can be written against something typed.

    Attributes:
        tmin: Start time in seconds.
        tmax: End time in seconds.
        fmin: Low edge in Hz.
        fmax: High edge in Hz.
        peak_hz: Frequency of the brightest cell in the region.
        cells: Active cell count.
        peak_sigma: Brightest cell's contrast above its own local background.
        confidence: ``[0, 1]``.
        shape: The scanner's descriptive label for the structure.
        extra: Anything else the scanner wants to say about this box.
    """

    __slots__ = (
        "tmin", "tmax", "fmin", "fmax", "peak_hz", "cells",
        "peak_sigma", "confidence", "shape", "extra",
    )

    def __init__(
        self,
        tmin: float,
        tmax: float,
        fmin: float,
        fmax: float,
        *,
        peak_hz: float = 0.0,
        cells: int = 0,
        peak_sigma: float = 0.0,
        confidence: float = 0.0,
        shape: str = "structure",
        extra: dict | None = None,
    ) -> None:
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.peak_hz = float(peak_hz)
        self.cells = int(cells)
        self.peak_sigma = float(peak_sigma)
        self.confidence = float(confidence)
        self.shape = str(shape)
        self.extra = dict(extra or {})

    def to_dict(self) -> dict:
        """The region as the camelCase dict a sidecar stores (see :data:`REGION_FIELDS`)."""
        out = {
            "tmin": round(self.tmin, 4),
            "tmax": round(self.tmax, 4),
            "fmin": round(self.fmin, 1),
            "fmax": round(self.fmax, 1),
            "peakHz": round(self.peak_hz, 1),
            "cells": self.cells,
            "peakSigma": round(self.peak_sigma, 4),
            "confidence": round(self.confidence, 4),
            "shape": self.shape,
        }
        out.update(self.extra)
        return out


class ScannerAlgorithm:
    """Base class a scanning algorithm *may* subclass (the loader only duck-types).

    Subclassing buys the guards in :meth:`run`; implement :meth:`scan` and set the metadata.
    The build pipeline obfuscates the subclass, so this base stays in the open package while the
    method does not.

    Attributes:
        slug: Stable identifier, matching the registry (``"all_structures"``).
        version: The algorithm's own version string.
        shapes: The shape labels this algorithm can emit, for the CLI listing.
        expects: ``{"fft": 1024, "hop": 256, "window": "hann", "sample_rate": 96000}``, or
            ``{}`` when the algorithm is grid-agnostic. Whatever is set here becomes the CLI's
            default STFT for this algorithm, so a scanner tuned to a 1 ms click gets its short
            window without the user having to know.
        params_schema: ``{name: {"type", "default", "min", "max", "help"}}`` — what ``--param``
            may set. Drives ``siar-scanner algorithms --params``.
    """

    slug: str = "algorithm"
    version: str = "0"
    shapes: tuple[str, ...] = ()
    expects: dict = {}
    params_schema: dict = {}

    def configure(self, **params: Any) -> None:
        """Apply run-time parameters. Default: ignore them."""

    def scan(self, grid: FrameGrid) -> list:
        """Return the structures found in ``grid``, as :class:`Region` or dicts."""
        raise NotImplementedError

    def run(self, grid: FrameGrid) -> list[dict]:
        """Scan ``grid`` and return normalised region dicts."""
        return run_scan(self, grid)


def run_scan(algorithm: Any, grid: FrameGrid) -> list[dict]:
    """Scan a grid through an algorithm, validating the contract both ways.

    Args:
        algorithm: Anything with a ``scan(grid)`` method.
        grid: The :class:`FrameGrid` to scan.

    Returns:
        Region dicts, strongest-looking first is *not* guaranteed — order is the algorithm's.

    Raises:
        ScannerError: If scanning raised, or returned something that is not a list of regions.
    """
    try:
        regions = algorithm.scan(grid)
    except Exception as e:  # noqa: BLE001 - the algorithm is closed; surface it as ours
        raise ScannerError(f"scanning failed: {e}") from e
    if regions is None:
        return []
    if not isinstance(regions, (list, tuple)):
        raise ScannerError(
            f"algorithm returned {type(regions).__name__}, expected a list of regions"
        )
    return normalise_regions(regions)


def normalise_regions(regions: Any) -> list[dict]:
    """Coerce whatever an algorithm returned into sidecar-ready dicts.

    Accepts :class:`Region` instances, dicts using either the camelCase sidecar names or the
    snake_case attribute names, or anything exposing the attributes. Unknown keys are kept:
    a scanner's own diagnostics are worth writing down even though nothing reads them yet.

    Args:
        regions: The algorithm's return value.

    Returns:
        A list of dicts, each carrying at least :data:`REGION_FIELDS`.

    Raises:
        ScannerError: If a region is missing a time or frequency bound, or carries a
            non-finite one — a box that cannot be drawn is a bug worth stopping for, not
            something to silently drop.
    """
    out: list[dict] = []
    for i, r in enumerate(regions):
        if isinstance(r, Region):
            out.append(r.to_dict())
            continue
        d = dict(r) if isinstance(r, dict) else _from_attrs(r)
        rec = {
            "tmin": _num(d, i, "tmin"),
            "tmax": _num(d, i, "tmax"),
            "fmin": _num(d, i, "fmin"),
            "fmax": _num(d, i, "fmax"),
            "peakHz": _num(d, i, "peakHz", "peak_hz", default=0.0),
            "cells": int(_num(d, i, "cells", default=0)),
            "peakSigma": _num(d, i, "peakSigma", "peak_sigma", default=0.0),
            "confidence": _num(d, i, "confidence", default=0.0),
            "shape": str(d.get("shape", "structure")),
        }
        for k, v in d.items():
            if k not in rec and k not in ("peak_hz", "peak_sigma"):
                rec[k] = v
        out.append(rec)
    return out


def _from_attrs(obj: Any) -> dict:
    """Pull the known region fields off an arbitrary object."""
    keys = ("tmin", "tmax", "fmin", "fmax", "peakHz", "peak_hz", "cells",
            "peakSigma", "peak_sigma", "confidence", "shape")
    return {k: getattr(obj, k) for k in keys if hasattr(obj, k)}


def _num(d: dict, index: int, *names: str, default: float | None = None) -> float:
    """Read the first present key as a finite float, or raise / fall back to ``default``."""
    for n in names:
        if n in d and d[n] is not None:
            v = float(d[n])
            if v != v or v in (float("inf"), float("-inf")):
                raise ScannerError(f"region {index} has a non-finite {n}")
            return v
    if default is not None:
        return default
    raise ScannerError(f"region {index} is missing {names[0]!r}")
