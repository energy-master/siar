# Vixen Intelligence c.2026
"""Serving a scan's output folder over HTTP, read-only, so it need not be copied.

A survey is increasingly scanned on a fast headless box, which leaves the output folder in the
wrong place: it holds a copy of every recording, so a real corpus is hundreds of gigabytes and
moving it to a laptop costs more time than the remote machine saved. What somebody actually wants
to look at — the boxes, the lane previews, the per-file counts, how far a running scan has got — is
a rounding error beside the audio, and the audio is the part nobody needs until they open one lane.

So this subpackage answers questions about a folder in place, over an ordinary ssh tunnel:

* :mod:`siarapp.serve.folder` reads the folder. The run manifest is the index — it is already
  recursive, already sorted, already rewritten after every recording — and it is also the allowlist
  that makes a path the run never wrote *unknown* rather than merely refused.
* :mod:`siarapp.serve.preview` draws one recording small enough to send: a dB-normalised, uint8,
  time-decimated picture of its spectrogram, computed by seeking to a few thousand windows instead
  of decoding forty minutes of audio.
* :mod:`siarapp.serve.http` puts the two behind a handler that answers ``GET``, ``HEAD`` and
  ``OPTIONS`` and nothing else.

Nothing here writes to the folder, and nothing here opens a file for writing. That is not a policy
enforced by a check somewhere — it is the absence of any code that could.

Deliberately not re-exported from :mod:`siarapp` itself: importing :mod:`http.server` on every
``siar-app version`` would cost every user something to benefit the few who serve.
"""
from __future__ import annotations

from siarapp.serve.folder import ARTEFACT_KINDS, FolderError, PathRefused, ServedFolder, check_rel
from siarapp.serve.http import (
    DEFAULT_PORT,
    ServeOptions,
    build_server,
    local_host_hint,
    mint_token,
    page_url,
    serve_folder,
)
from siarapp.serve.preview import Preview, preview_bounds, read_preview

__all__ = [
    "ARTEFACT_KINDS",
    "DEFAULT_PORT",
    "FolderError",
    "PathRefused",
    "Preview",
    "ServeOptions",
    "ServedFolder",
    "build_server",
    "check_rel",
    "local_host_hint",
    "mint_token",
    "page_url",
    "preview_bounds",
    "read_preview",
    "serve_folder",
]
