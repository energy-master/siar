# Vixen Intelligence c.2026
"""Where SIAR keeps its things.

One workspace directory, ``~/.siar`` by default, overridable with ``$SIAR_HOME`` (handy for
tests, and for keeping a project's results next to the project). Inside it:

    siar.db          the results database — datasets, studies, trials, models, detections
    cache/<key>/     feature cache, one directory per FeatureSpec, one .npy per recording
    runs/<uid>/      per-run outputs: spectrogram PNGs, exported JSON
    optuna/<uid>/    Optuna's own journal storage, kept away from siar.db (see siar.store.db)

Nothing here creates directories as a side effect of import; call :func:`ensure_workspace`.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "cache_dir",
    "db_path",
    "ensure_workspace",
    "optuna_dir",
    "run_dir",
    "workspace",
]


def workspace() -> Path:
    """The SIAR workspace root.

    Returns:
        ``$SIAR_HOME`` if set, else ``~/.siar``.
    """
    env = os.environ.get("SIAR_HOME")
    return Path(env).expanduser() if env else Path.home() / ".siar"


def db_path() -> Path:
    """Path to the results database."""
    return workspace() / "siar.db"


def cache_dir(cache_key: str) -> Path:
    """Feature-cache directory for one :class:`~siar.features.spec.FeatureSpec`.

    Args:
        cache_key: The spec's :meth:`~siar.features.spec.FeatureSpec.cache_key`.

    Returns:
        The directory holding that spec's cached grids.
    """
    return workspace() / "cache" / cache_key


def run_dir(uid: str) -> Path:
    """Output directory for one run (PNGs, exports).

    Args:
        uid: The run or inference uid.

    Returns:
        The directory for that run's artefacts.
    """
    return workspace() / "runs" / uid


def optuna_dir(uid: str) -> Path:
    """Directory holding one study's Optuna journal storage.

    Args:
        uid: The study uid.

    Returns:
        The directory for that study's Optuna state.
    """
    return workspace() / "optuna" / uid


def ensure_workspace() -> Path:
    """Create the workspace root and its subdirectories if they do not exist.

    Returns:
        The workspace root.
    """
    root = workspace()
    for sub in ("", "cache", "runs", "optuna"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
