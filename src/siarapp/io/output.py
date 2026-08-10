# Vixen Intelligence c.2026
"""Writing the output folder — the whole point of the CLI.

A scan that leaves its answers in a terminal has not finished. What finishes it is a folder the
user can drag into IDent Dynamics and immediately work in: every recording, every box, every
lane preview, laid out so the app's existing folder-open path finds them without being told.

```
<out>/
  <relpath>/recording.wav               the source, copied (or hardlinked with --link)
  <relpath>/recording.structures.json   the boxes
  <relpath>/recording.png               the 200x64 lane thumbnail
  siar-app-run.json                 what was run, over what, with what, and what came back
```

Three decisions are load-bearing:

* **The audio is copied.** The app needs the actual bytes to render a surface when a lane is
  clicked, and a folder that references audio somewhere else is not a folder you can hand to a
  colleague. ``--link`` hardlinks instead when the input and output share a filesystem, which
  costs nothing and keeps the folder self-contained for as long as the source exists.
* **The sidecar is named ``<audio-basename>.structures.json``.** The app pairs sidecars to audio
  by "longest audio basename that prefixes the sidecar basename", so this pairs with
  ``recording.wav`` and could not be confused with ``recording_2.wav``.
* **The relative layout is preserved.** A survey folder organised by station or by date stays
  organised; flattening it would collide names from different subfolders, silently, at whatever
  scale the corpus happens to be.

Everything here is atomic-write-then-rename, because the failure that matters at 10,000 files is
an interrupted run leaving a truncated JSON that the app then refuses to parse.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from siarapp.io.performance import PERFORMANCE_NAME

__all__ = [
    "RUN_MANIFEST_NAME",
    "SIDECAR_FORMAT",
    "SIDECAR_SUFFIX",
    "OutputFolder",
    "sidecar_document",
]

#: The suffix appended to an audio basename to name its boxes.
SIDECAR_SUFFIX = ".structures.json"

#: Written at the root of every output folder.
RUN_MANIFEST_NAME = "siar-app-run.json"

#: Stamped into every sidecar so a future reader can tell what it is holding.
#:
#: Deliberately still says "scanner" after the rename to siar-app. It is a wire-format version,
#: not a product name: every sidecar ever written carries it, the web app matches on it, and
#: changing it would mean a v2 that nothing gains from.
SIDECAR_FORMAT = "siar-scanner-structures-v1"

#: The family a model belongs to when its bundle does not say — which is every bundle built
#: before families existed, all of which are structure scanners.
DEFAULT_FAMILY = "structure_scanners"


def sidecar_document(
    *,
    filename: str,
    duration_sec: float,
    algorithm: str,
    structures: list[dict],
    algorithm_version: str = "",
    family: str = DEFAULT_FAMILY,
    params: dict | None = None,
    stft: dict | None = None,
    source_path: str = "",
) -> dict:
    """Build one recording's structures sidecar.

    The ``structures`` key is what the web app sniffs on to tell this document from a labels or
    decisions sidecar, so it is present even when empty — a file that was scanned and found
    nothing is a different statement from a file that was never scanned, and the app should be
    able to tell them apart.

    Args:
        filename: The audio file's name, as it sits beside this sidecar.
        duration_sec: Recording length.
        algorithm: The slug that produced the boxes. Becomes the model name the app files them
            under, so the Structures panel's per-model toggles read as scanner names.
        structures: The region dicts (see :data:`siarapp.grid.REGION_FIELDS`).
        algorithm_version: The algorithm's own version.
        family: The model family, from the bundle's manifest.
        params: The run-time parameters it was configured with.
        stft: ``{"fft", "hop", "window", "sample_rate"}`` the grid was built at.
        source_path: Where the recording was read from, for provenance.

    Returns:
        The document, ready for :meth:`OutputFolder.write_sidecar`.
    """
    return {
        "format": SIDECAR_FORMAT,
        "filename": filename,
        "duration_sec": round(float(duration_sec), 4),
        "algorithm": algorithm,
        "algorithm_version": algorithm_version,
        # What KIND of model produced this, so a reader can route the document without
        # inferring the answer from which keys happen to be present. A structure scanner and a
        # click detector both emit boxes; only the family says which panel they belong in.
        "family": family or DEFAULT_FAMILY,
        "params": params or {},
        "stft": stft or {},
        "source_path": source_path,
        "count": len(structures),
        "structures": structures,
    }


class OutputFolder:
    """The destination directory, and the four kinds of thing written into it.

    Args:
        root: Output directory. Created if absent.
        source_root: The scanned folder, so relative layout can be preserved.
        link: Hardlink the audio instead of copying it, falling back to a copy across
            filesystems.
    """

    def __init__(self, root: str, source_root: str, *, link: bool = False) -> None:
        self.root = os.path.abspath(root)
        self.source_root = os.path.abspath(source_root)
        self.link = link
        os.makedirs(self.root, exist_ok=True)

    # -- paths ------------------------------------------------------------------------

    def rel_path(self, source_file: str) -> str:
        """The source file's path relative to the scanned folder, in POSIX form.

        POSIX separators because this string goes into a JSON manifest that may be read on
        another OS, and a Windows-flavoured key would not match there.
        """
        rel = os.path.relpath(os.path.abspath(source_file), self.source_root)
        return rel.replace(os.sep, "/")

    def audio_path(self, source_file: str) -> str:
        """Where this recording's copy lives in the output folder."""
        return os.path.join(self.root, *self.rel_path(source_file).split("/"))

    def sidecar_path(self, source_file: str) -> str:
        """Where this recording's structures sidecar lives."""
        stem = os.path.splitext(self.audio_path(source_file))[0]
        return stem + SIDECAR_SUFFIX

    def thumbnail_path(self, source_file: str) -> str:
        """Where this recording's lane thumbnail lives."""
        return os.path.splitext(self.audio_path(source_file))[0] + ".png"

    # -- writes -----------------------------------------------------------------------

    def place_audio(self, source_file: str) -> str:
        """Put the recording in the output folder, by hardlink or copy.

        Skips the work when a file of the same size is already there, which is what makes
        ``--resume`` cheap on a corpus that is mostly done.

        Args:
            source_file: The source recording.

        Returns:
            The destination path.

        Raises:
            OSError: If the copy fails — out of disk, most likely, and worth stopping the run
                for rather than producing a folder that is quietly missing recordings.
        """
        dest = self.audio_path(source_file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(source_file):
            return dest
        if self.link:
            try:
                if os.path.exists(dest):
                    os.unlink(dest)
                os.link(source_file, dest)
                return dest
            except OSError:
                pass  # cross-device, or a filesystem without hardlinks — copy instead
        tmp = dest + f".tmp-{os.getpid()}"
        shutil.copyfile(source_file, tmp)
        os.replace(tmp, dest)
        return dest

    def write_sidecar(self, source_file: str, document: dict) -> str:
        """Write one recording's structures sidecar. Returns the path."""
        path = self.sidecar_path(source_file)
        blob = json.dumps(document, separators=(",", ":"), default=json_default)
        _write_atomic(path, blob.encode("utf-8"))
        return path

    def write_thumbnail(self, source_file: str, png: bytes | None) -> str | None:
        """Write one recording's lane thumbnail, or nothing when it was too short to render."""
        if not png:
            return None
        path = self.thumbnail_path(source_file)
        _write_atomic(path, png)
        return path

    def write_manifest(self, manifest: dict) -> str:
        """Write the run manifest at the folder root. Returns the path."""
        path = os.path.join(self.root, RUN_MANIFEST_NAME)
        blob = json.dumps(manifest, indent=2, default=json_default) + "\n"
        _write_atomic(path, blob.encode("utf-8"))
        return path

    def write_performance(self, document: dict) -> str:
        """Write the performance report at the folder root. Returns the path."""
        path = os.path.join(self.root, PERFORMANCE_NAME)
        blob = json.dumps(document, indent=2, default=json_default) + "\n"
        _write_atomic(path, blob.encode("utf-8"))
        return path

    def already_done(self, source_file: str) -> bool:
        """True when this recording's sidecar and audio are both already in place.

        What ``--resume`` consults. Deliberately does not check the thumbnail: a recording too
        short to render one would otherwise be re-scanned on every resumed run, forever.
        """
        return os.path.isfile(self.sidecar_path(source_file)) and os.path.isfile(
            self.audio_path(source_file)
        )


def _write_atomic(path: str, payload: bytes) -> None:
    """Write bytes via a temporary file and a rename, so a reader never sees a partial one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def json_default(value: Any) -> Any:
    """Serialise numpy scalars, which is what a port's region dict is full of."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")
