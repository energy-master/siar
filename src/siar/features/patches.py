# Vixen Intelligence c.2026
"""Cutting a grid into patches, and putting per-pixel errors back.

Patches span the **full height** of the grid (all ``n_bins``) and stride only in time. Two-
dimensional localisation does *not* come from a 2-D patch lattice — it comes from keeping the
autoencoder's reconstruction error at **per-pixel** resolution and reassembling it into a full
error map (:func:`overlap_add`). That is the whole trick, and it means the patch layer stays
one-dimensional and simple.

The pair of functions here are exact inverses in the following sense: if every patch's error is
a constant ``c``, :func:`overlap_add` returns a grid of exactly ``c`` everywhere — including in
the overlap regions and the edge-padded tail. That property is what the round-trip test pins
down, and it is easy to break by mishandling the final partial patch.
"""
from __future__ import annotations

import numpy as np

__all__ = ["overlap_add", "patch_starts", "patchify"]


def patch_starts(frames: int, patch_frames: int, stride_frames: int) -> list[int]:
    """Frame indices at which each patch starts.

    The last start is always ``frames - patch_frames``, appended if the stride happened to skip
    over it. Without that, a stride that does not divide the grid length leaves a tail of frames
    that no patch covers — and an anomaly living in the final half-second of every recording
    would be invisible.

    Args:
        frames: Number of frames in the (possibly padded) grid. Must be >= ``patch_frames``.
        patch_frames: Patch width in frames.
        stride_frames: Hop between patches, in frames.

    Returns:
        Ascending patch start indices, with no duplicates.
    """
    starts = list(range(0, frames - patch_frames + 1, max(1, stride_frames)))
    if not starts:
        starts = [0]
    if starts[-1] != frames - patch_frames:
        starts.append(frames - patch_frames)
    return starts


def patchify(
    grid: np.ndarray, patch_frames: int, stride_frames: int
) -> tuple[np.ndarray, list[int], int]:
    """Cut a grid into overlapping full-height patches.

    A grid shorter than one patch is edge-padded (its last frame repeated) up to ``patch_frames``
    rather than dropped — a two-second recording in a corpus of one-minute ones should still be
    scoreable.

    Args:
        grid: ``(frames, n_bins)`` ``float32``.
        patch_frames: Patch width in frames.
        stride_frames: Hop between patches, in frames.

    Returns:
        A tuple of:
          * ``patches``: ``(n_patches, patch_frames, n_bins)`` ``float32``;
          * ``starts``: the frame index each patch begins at;
          * ``padded_frames``: the frame count after any edge padding — what
            :func:`overlap_add` must be given so the error map lines up.

    Raises:
        ValueError: If ``grid`` is not 2-D or is empty.
    """
    g = np.asarray(grid, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError(f"grid must be 2-D (frames, n_bins), got shape {g.shape}")
    if g.shape[0] == 0:
        raise ValueError("grid has no frames")

    if g.shape[0] < patch_frames:
        pad = patch_frames - g.shape[0]
        g = np.pad(g, ((0, pad), (0, 0)), mode="edge")

    frames = g.shape[0]
    starts = patch_starts(frames, patch_frames, stride_frames)
    # A strided view, so this costs nothing until the caller batches it into torch.
    windows = np.lib.stride_tricks.sliding_window_view(g, patch_frames, axis=0)
    # sliding_window_view gives (frames - pf + 1, n_bins, pf); pick our starts and restore
    # (n_patches, pf, n_bins).
    patches = np.ascontiguousarray(windows[starts].transpose(0, 2, 1))
    return patches, starts, frames


def overlap_add(
    patch_errors: np.ndarray, starts: list[int], frames: int, taper: bool = True
) -> np.ndarray:
    """Reassemble per-patch, per-pixel errors into one error map over the whole grid.

    Where patches overlap, the errors are **averaged**, not summed — otherwise the middle of a
    recording (covered by many patches) would score higher than its edges purely as an artefact
    of the lattice.

    Args:
        patch_errors: ``(n_patches, patch_frames, n_bins)`` ``float32`` — the per-pixel
            reconstruction error of each patch.
        starts: The frame index each patch began at, from :func:`patchify`.
        frames: The padded frame count from :func:`patchify`.
        taper: Weight each patch by a Hann taper along time when averaging. A patch reconstructs
            its own edges worse than its middle (the convolutions run out of context there), so
            without this the error map grows a faint stripe every ``stride_frames`` and box
            extraction finds regularly-spaced phantom detections. On by default; there is no
            good reason to turn it off outside of tests.

    Returns:
        A ``(frames, n_bins)`` ``float32`` error map.

    Raises:
        ValueError: If the shapes are inconsistent with ``starts``.
    """
    e = np.asarray(patch_errors, dtype=np.float64)
    if e.ndim != 3:
        raise ValueError(f"patch_errors must be (n, patch_frames, n_bins), got {e.shape}")
    if e.shape[0] != len(starts):
        raise ValueError(f"got {e.shape[0]} patches but {len(starts)} starts")

    n_patches, patch_frames, n_bins = e.shape
    if taper and patch_frames > 1:
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(patch_frames) / (patch_frames - 1)))
        # A pure Hann hits exactly zero at both ends, which would leave the first and last frame
        # of the whole grid with zero weight and a 0/0 average. Lift it off the floor.
        w = np.maximum(w, 1e-3)
    else:
        w = np.ones(patch_frames, dtype=np.float64)
    weights = w[:, None]  # (patch_frames, 1), broadcast across bins

    acc = np.zeros((frames, n_bins), dtype=np.float64)
    cnt = np.zeros((frames, n_bins), dtype=np.float64)
    for i, s in enumerate(starts):
        acc[s : s + patch_frames] += e[i] * weights
        cnt[s : s + patch_frames] += weights

    return (acc / np.maximum(cnt, 1e-12)).astype(np.float32)
