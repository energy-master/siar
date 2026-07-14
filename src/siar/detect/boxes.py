# Vixen Intelligence c.2026
"""From a z-map to boxes.

This is where SIAR's output actually gets made: a ``(frames, n_bins)`` z-map goes in, and a list
of time-frequency rectangles comes out, each with a start and end time in seconds, a low and high
frequency in Hz, and a score.

The pipeline, and why each step is there:

1. **Threshold** the z-map into a binary mask.
2. **Open**, then **close** (morphology). Opening deletes isolated hot pixels — statistical
   noise, not events. Closing then re-joins an event that the threshold fragmented (a tonal that
   flickers, a click smeared over two frames).
3. **Dilate and label.** A real event can still be broken into several components. Dilating with
   a small rectangle before labelling lets neighbouring fragments merge into one component — but
   the box is then measured on the *original* mask inside that component, so the dilation joins
   without inflating.
4. **Filter.** Drop components that are too small, too brief, too narrow — or too *ragged*.
5. **Convert** grid coordinates to seconds and Hz.
6. **Merge** boxes that overlap heavily, and cap the count.

Step 4's raggedness test (``min_fill``) is the one most people leave out, and it is the one that
decides whether the dashboard shows a handful of real detections or a screenful of confetti. A
component that snakes diagonally across the map has an enormous bounding box and almost no area
inside it; its fill ratio is near zero. It is nearly always noise, and its box — being huge —
would otherwise outrank and visually swamp every genuine detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from siar.features.frontend import row_support_hz
from siar.features.spec import FeatureSpec

__all__ = ["BoxSpec", "Detection", "extract_boxes"]


@dataclass(frozen=True, slots=True)
class BoxSpec:
    """The knobs of box extraction.

    Stored on the trained model, so re-running it on a new folder reproduces the same boxes.

    Attributes:
        open_iters: Binary-opening iterations. Removes speckle. 0 disables.
        close_iters: Binary-closing iterations. Re-joins fragmented events. 0 disables.
        bridge_frames: Width, in frames, of the dilation element used to join neighbouring
            fragments before labelling.
        bridge_bins: Height, in bins, of that element.
        min_frames: Reject detections shorter than this many frames.
        min_bins: Reject detections narrower than this many bins.
        min_area: Reject detections with fewer than this many pixels above threshold.
        min_fill: Reject detections whose above-threshold pixels fill less than this fraction of
            their own bounding box. See the module docstring — this is the anti-confetti filter.
        peak_fraction: Having *found* a component, keep only the pixels within this fraction of
            its own peak z when measuring its extent. This is a dynamic-range gate, and without
            it the boxes are useless.

            The threshold answers "is anything here?" — it is set from clean calibration data and
            is deliberately sensitive. But a real anomaly can exceed it by four orders of
            magnitude, and its error halo then also clears it, so the component floods outward
            and the box swells to the whole spectrum. The event is detected and its extent is
            meaningless. Gating relative to the component's own peak (0.01 = keep within 20 dB of
            it) draws the box around the event instead of around its shadow.
        merge_iou: Merge two boxes whose intersection-over-union exceeds this.
        max_boxes: Keep at most this many boxes per recording, highest score first. A safety
            valve: a badly-configured model must not be able to write a million rows.
    """

    open_iters: int = 1
    close_iters: int = 1
    bridge_frames: int = 3
    bridge_bins: int = 2
    min_frames: int = 2
    min_bins: int = 2
    min_area: int = 6
    min_fill: float = 0.25
    peak_fraction: float = 0.1
    merge_iou: float = 0.4
    max_boxes: int = 500

    def to_dict(self) -> dict:
        """Return the spec as a plain JSON-safe dict."""
        return {
            "open_iters": self.open_iters,
            "close_iters": self.close_iters,
            "bridge_frames": self.bridge_frames,
            "bridge_bins": self.bridge_bins,
            "min_frames": self.min_frames,
            "min_bins": self.min_bins,
            "min_area": self.min_area,
            "min_fill": self.min_fill,
            "peak_fraction": self.peak_fraction,
            "merge_iou": self.merge_iou,
            "max_boxes": self.max_boxes,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "BoxSpec":
        """Rebuild a :class:`BoxSpec` from :meth:`to_dict` output."""
        fields = cls.__dataclass_fields__
        return cls(**{k: obj[k] for k in fields if k in obj})


@dataclass(frozen=True, slots=True)
class Detection:
    """One anomaly: a rectangle in time and frequency.

    Attributes:
        t_start: Start time in seconds.
        t_end: End time in seconds.
        f_low: Low frequency edge in Hz.
        f_high: High frequency edge in Hz.
        score: The detection's strength — the 95th percentile of the z-scores inside it. A
            percentile rather than the max, so one hot pixel cannot promote a weak detection;
            and rather than the mean, so a large mostly-marginal region cannot dilute a genuinely
            strong core.
        peak_score: The single highest z inside it.
        area: Above-threshold pixels in the component.
        fill: ``area`` divided by the bounding box's pixel count.
        frame_lo: First grid frame of the box, inclusive.
        frame_hi: Last grid frame, inclusive.
        bin_lo: Lowest grid bin, inclusive.
        bin_hi: Highest grid bin, inclusive.
    """

    t_start: float
    t_end: float
    f_low: float
    f_high: float
    score: float
    peak_score: float
    area: int
    fill: float
    frame_lo: int
    frame_hi: int
    bin_lo: int
    bin_hi: int

    def to_dict(self) -> dict:
        """Return the detection as a plain JSON-safe dict."""
        return {
            "t_start": round(self.t_start, 4),
            "t_end": round(self.t_end, 4),
            "f_low": round(self.f_low, 1),
            "f_high": round(self.f_high, 1),
            "score": round(self.score, 3),
            "peak_score": round(self.peak_score, 3),
            "area": self.area,
            "fill": round(self.fill, 3),
            "frame_lo": self.frame_lo,
            "frame_hi": self.frame_hi,
            "bin_lo": self.bin_lo,
            "bin_hi": self.bin_hi,
        }


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two boxes, in grid coordinates."""
    t0 = max(a.frame_lo, b.frame_lo)
    t1 = min(a.frame_hi, b.frame_hi)
    f0 = max(a.bin_lo, b.bin_lo)
    f1 = min(a.bin_hi, b.bin_hi)
    if t1 < t0 or f1 < f0:
        return 0.0
    inter = (t1 - t0 + 1) * (f1 - f0 + 1)
    area_a = (a.frame_hi - a.frame_lo + 1) * (a.bin_hi - a.bin_lo + 1)
    area_b = (b.frame_hi - b.frame_lo + 1) * (b.bin_hi - b.bin_lo + 1)
    return inter / float(area_a + area_b - inter)


def _merge(a: Detection, b: Detection, spec: FeatureSpec, support: np.ndarray) -> Detection:
    """Union two overlapping boxes, keeping the stronger one's scores."""
    frame_lo = min(a.frame_lo, b.frame_lo)
    frame_hi = max(a.frame_hi, b.frame_hi)
    bin_lo = min(a.bin_lo, b.bin_lo)
    bin_hi = max(a.bin_hi, b.bin_hi)
    strong = a if a.score >= b.score else b
    box_px = (frame_hi - frame_lo + 1) * (bin_hi - bin_lo + 1)
    area = a.area + b.area
    return Detection(
        t_start=frame_lo * spec.delta_t,
        t_end=(frame_hi + 1) * spec.delta_t,
        f_low=float(support[bin_lo][0]),
        f_high=float(support[bin_hi][1]),
        score=strong.score,
        peak_score=max(a.peak_score, b.peak_score),
        area=area,
        fill=min(1.0, area / float(box_px)),
        frame_lo=frame_lo,
        frame_hi=frame_hi,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
    )


def extract_boxes(
    z_map: np.ndarray,
    spec: FeatureSpec,
    threshold: float,
    box_spec: BoxSpec | None = None,
) -> list[Detection]:
    """Turn a z-map into a list of time-frequency detections.

    Args:
        z_map: ``(frames, n_bins)`` per-bin z-scores
            (:meth:`siar.detect.normalise.Baseline.apply`).
        spec: The feature spec — supplies the time and frequency axes.
        threshold: The z above which a pixel is anomalous
            (:func:`siar.detect.threshold.fit_threshold`).
        box_spec: Extraction settings. Defaults to :class:`BoxSpec`'s defaults.

    Returns:
        Detections, strongest first, at most ``box_spec.max_boxes`` of them.

    Raises:
        ValueError: If ``z_map`` is not 2-D or its bin count disagrees with ``spec``.
    """
    from scipy import ndimage as ndi

    z = np.asarray(z_map, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_map must be 2-D (frames, n_bins), got shape {z.shape}")
    if z.shape[1] != spec.n_bins:
        raise ValueError(f"z_map has {z.shape[1]} bins, spec says {spec.n_bins}")
    if z.shape[0] == 0:
        return []

    bs = box_spec or BoxSpec()
    support = row_support_hz(spec)

    mask = z > threshold
    if not mask.any():
        return []

    cross = np.ones((3, 3), dtype=bool)
    if bs.open_iters > 0:
        mask = ndi.binary_opening(mask, structure=cross, iterations=bs.open_iters)
    if bs.close_iters > 0:
        mask = ndi.binary_closing(mask, structure=cross, iterations=bs.close_iters)
    if not mask.any():
        return []

    # Dilate only to decide *what belongs together*; measure on `mask`, never on `joined`.
    bridge = np.ones((max(1, bs.bridge_frames), max(1, bs.bridge_bins)), dtype=bool)
    joined = ndi.binary_dilation(mask, structure=bridge)
    labels, n = ndi.label(joined, structure=cross)
    if n == 0:
        return []

    found: list[Detection] = []
    # find_objects returns one slice per label, in label order: index i is label i + 1. The
    # label id must come from the index, NOT from labels[sl].max() — a component's bounding box
    # routinely contains pixels of a *different*, higher-numbered component, and taking the max
    # then measures the wrong one and silently discards the real detection.
    for index, sl in enumerate(ndi.find_objects(labels)):
        if sl is None:
            continue
        label_id = index + 1
        core = mask[sl] & (labels[sl] == label_id)
        if not core.any():
            continue

        # Dynamic-range gate: measure the event, not its halo. See BoxSpec.peak_fraction.
        z_local = z[sl]
        peak = float(z_local[core].max())
        if bs.peak_fraction > 0.0:
            gate = max(threshold, peak * bs.peak_fraction)
            gated = core & (z_local >= gate)
            if gated.any():
                core = gated

        area = int(core.sum())
        if area < bs.min_area:
            continue

        rows = np.flatnonzero(core.any(axis=1))
        cols = np.flatnonzero(core.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        frame_lo = int(sl[0].start + rows[0])
        frame_hi = int(sl[0].start + rows[-1])
        bin_lo = int(sl[1].start + cols[0])
        bin_hi = int(sl[1].start + cols[-1])

        n_frames = frame_hi - frame_lo + 1
        n_bins_box = bin_hi - bin_lo + 1
        if n_frames < bs.min_frames or n_bins_box < bs.min_bins:
            continue

        fill = area / float(n_frames * n_bins_box)
        if fill < bs.min_fill:
            continue

        zs = z[sl][core]
        found.append(
            Detection(
                # A frame's energy spans [f, f+1) of the hop grid, so the box runs from the
                # start of its first frame to the end of its last.
                t_start=frame_lo * spec.delta_t,
                t_end=(frame_hi + 1) * spec.delta_t,
                f_low=float(support[bin_lo][0]),
                f_high=float(support[bin_hi][1]),
                score=float(np.percentile(zs, 95)),
                peak_score=float(zs.max()),
                area=area,
                fill=float(fill),
                frame_lo=frame_lo,
                frame_hi=frame_hi,
                bin_lo=bin_lo,
                bin_hi=bin_hi,
            )
        )

    found.sort(key=lambda d: d.score, reverse=True)

    merged: list[Detection] = []
    for det in found:
        for i, kept in enumerate(merged):
            if _iou(det, kept) >= bs.merge_iou:
                merged[i] = _merge(kept, det, spec, support)
                break
        else:
            merged.append(det)

    merged.sort(key=lambda d: d.score, reverse=True)
    return merged[: bs.max_boxes]
