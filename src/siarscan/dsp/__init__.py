# Vixen Intelligence c.2026
"""Signal processing: the window functions and the STFT that build a scanner's input grid.

Ports of the web app's ``js/dsp/{windows,stft}.js``. See those modules' docstrings for why
"port" rather than "equivalent" is the operative word.
"""
from __future__ import annotations

from siarscan.dsp.stft import StftResult, frame_count, grid_bytes, stft
from siarscan.dsp.windows import WINDOWS, by_name

__all__ = ["StftResult", "stft", "frame_count", "grid_bytes", "WINDOWS", "by_name"]
