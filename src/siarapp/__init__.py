# Vixen Intelligence c.2026
"""siar-app — run the IDent Dynamics structure scanners over a folder of recordings.

Point it at a root folder of WAV or FLAC, choose one of the scanning algorithms your IDent
Dynamics account can see, and it writes an output folder that drags straight into the web app:
every recording, every box it found, and a spectrogram thumbnail for every lane.

```python
from siarapp import Client, RunOptions, load_remote, run_folder

client = Client("https://goident.ai", token=...)
handle = load_remote(client, "all_structures")
manifest = run_folder(handle, "~/survey-audio", "~/survey-scan", RunOptions(resume=True))
print(manifest["structures"], "structures in", manifest["files"], "files")
```

``RunOptions(workers=N)`` scans N recordings at once, in worker processes started by spawn — so
a script that uses it must guard its entry point with ``if __name__ == "__main__":``, the same
rule every :mod:`multiprocessing` program follows. The CLI's own entry point already does.

The algorithms are not in this package. They download as obfuscated bundles and are cached
under ``~/.siar-app/algorithms/``; see :mod:`siarapp.loader`. What lives here is
everything on the open side of that line — the decode, the transform, the output folder — and
:mod:`siarapp.grid` is the seam between the two.
"""
from __future__ import annotations

__version__ = "0.2.0"

from siarapp.api import ApiError, AuthError, Client, client_from_credentials
from siarapp.grid import FrameGrid, Region, ScannerAlgorithm, ScannerError
from siarapp.io.output import OutputFolder, sidecar_document
from siarapp.loader import AlgorithmHandle, load_cached, load_local, load_remote
from siarapp.runner import FileResult, RunOptions, run_folder, scan_one

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
