# Vixen Intelligence c.2026
"""siar-scanner — run the IDent Dynamics structure scanners over a folder of recordings.

Point it at a root folder of WAV or FLAC, choose one of the scanning algorithms your IDent
Dynamics account can see, and it writes an output folder that drags straight into the web app:
every recording, every box it found, and a spectrogram thumbnail for every lane.

```python
from siarscan import Client, RunOptions, load_remote, run_folder

client = Client("https://goident.ai", token=...)
handle = load_remote(client, "all_structures")
manifest = run_folder(handle, "~/survey-audio", "~/survey-scan", RunOptions(resume=True))
print(manifest["structures"], "structures in", manifest["files"], "files")
```

The algorithms are not in this package. They download as obfuscated bundles and are cached
under ``~/.siar-scanner/algorithms/``; see :mod:`siarscan.loader`. What lives here is
everything on the open side of that line — the decode, the transform, the output folder — and
:mod:`siarscan.grid` is the seam between the two.
"""
from __future__ import annotations

__version__ = "0.1.0"

from siarscan.api import ApiError, AuthError, Client, client_from_credentials
from siarscan.grid import FrameGrid, Region, ScannerAlgorithm, ScannerError
from siarscan.io.output import OutputFolder, sidecar_document
from siarscan.loader import AlgorithmHandle, load_cached, load_local, load_remote
from siarscan.runner import FileResult, RunOptions, run_folder, scan_one

__all__ = [
    "__version__",
    "ApiError",
    "AuthError",
    "Client",
    "client_from_credentials",
    "FrameGrid",
    "Region",
    "ScannerAlgorithm",
    "ScannerError",
    "AlgorithmHandle",
    "load_cached",
    "load_local",
    "load_remote",
    "OutputFolder",
    "sidecar_document",
    "FileResult",
    "RunOptions",
    "run_folder",
    "scan_one",
]
