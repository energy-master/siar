# Vixen Intelligence c.2026
"""A minimal PNG encoder, built on stdlib ``zlib`` alone.

SIAR needs to write exactly one kind of image — an 8-bit RGB raster with no transparency, no
interlacing and no palette. That is about forty lines of PNG, and writing them here means the
library does not drag in Pillow (or, worse, matplotlib) just to save a picture.

The format: an 8-byte signature, then length-prefixed, CRC-suffixed chunks. ``IHDR`` (dimensions
and colour type), ``IDAT`` (zlib-compressed scanlines, each prefixed with a filter byte), ``IEND``.
Filter type 0 ("None") on every scanline — spectrograms are noisy, so the fancier predictors buy
almost nothing and cost clarity.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["encode_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPE_RGB = 2
_BIT_DEPTH = 8


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """Build one length-prefixed, CRC-suffixed PNG chunk.

    Args:
        tag: The 4-byte chunk type, e.g. ``b"IHDR"``.
        payload: The chunk's data.

    Returns:
        The encoded chunk.
    """
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def encode_png(rgb: np.ndarray, *, compress_level: int = 6) -> bytes:
    """Encode an RGB raster as a PNG.

    Args:
        rgb: ``(height, width, 3)`` ``uint8``. Row 0 is the **top** of the image.
        compress_level: zlib level, 0-9.

    Returns:
        The PNG file's bytes.

    Raises:
        ValueError: If ``rgb`` is not a ``(h, w, 3)`` uint8 array.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (height, width, 3), got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {arr.dtype}")

    height, width, _ = arr.shape

    # Every scanline is prefixed with its filter byte (0 = None). Build the whole raster in one
    # allocation rather than concatenating per row.
    raw = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = arr.reshape(height, width * 3)

    ihdr = struct.pack(">IIBBBBB", width, height, _BIT_DEPTH, _COLOR_TYPE_RGB, 0, 0, 0)
    idat = zlib.compress(raw.tobytes(), compress_level)

    return _SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
