# Vixen Intelligence c.2026
"""The workspace at ``~/.siar-app`` — credentials, the algorithm cache, run history.

One directory, three things in it, and a clear rule about which of them may be deleted: all of
them. Credentials cost a re-login, the cache costs a re-download, the history is a convenience.
Nothing here is the product of a scan — that goes in the output folder the user names.

```
~/.siar-app/
  credentials.json           the bearer token, mode 0600
  algorithms/<slug>/<platform>/   unpacked bundles, one tree per build
  runs.json                  the last RUN_HISTORY_MAX runs, for `siar-app runs`
```

``$SIAR_APP_HOME`` moves the lot, which is what a shared or containerised install needs.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any

__all__ = [
    "DEFAULT_BASE_URL",
    "RUN_HISTORY_MAX",
    "SUPPORTED_PYTHON",
    "algorithm_cache_dir",
    "clear_credentials",
    "default_platform_tag",
    "home",
    "imported_dir",
    "legacy_env",
    "libc_flavour",
    "load_credentials",
    "platform_compatible",
    "python_supported",
    "read_json",
    "recent_cost",
    "record_run",
    "run_history",
    "save_credentials",
    "supported_python_text",
    "write_json",
]

#: Where the CLI talks to unless ``--server`` or ``$SIAR_APP_URL`` says otherwise.
DEFAULT_BASE_URL = "https://goident.ai"

#: Runs kept in the local history. Enough to answer "what did I run last week", far short of
#: turning a convenience file into a database.
RUN_HISTORY_MAX = 200


#: The workspace this tool used when it was called siar-scanner, and the environment-variable
#: prefix that went with it. Both are still honoured: a machine that was logged in before the
#: rename must not silently lose its credentials, its algorithm cache and its licence
#: acceptance over a change of product name.
_LEGACY_DIR = ".siar-scanner"
_LEGACY_ENV_PREFIX = "SIAR_SCANNER_"


def legacy_env(name: str) -> str | None:
    """Read ``SIAR_APP_<name>``, falling back to the pre-rename ``SIAR_SCANNER_<name>``."""
    return os.environ.get(f"SIAR_APP_{name}") or os.environ.get(f"{_LEGACY_ENV_PREFIX}{name}")


def home() -> str:
    """The workspace directory, created if absent.

    A ``~/.siar-scanner`` left over from before the rename is **moved** to the new name the
    first time this runs, rather than being read in place. One workspace beats two: reading the
    old path forever would mean every later feature has to know about both, and copying would
    leave a stale token behind on disk.

    Returns:
        Absolute path — ``$SIAR_APP_HOME`` (or the legacy ``$SIAR_SCANNER_HOME``) if set, else
        ``~/.siar-app``.
    """
    override = legacy_env("HOME")
    if override:
        os.makedirs(override, mode=0o700, exist_ok=True)
        return os.path.abspath(override)

    path = os.path.join(os.path.expanduser("~"), ".siar-app")
    legacy = os.path.join(os.path.expanduser("~"), _LEGACY_DIR)
    if not os.path.isdir(path) and os.path.isdir(legacy):
        try:
            os.rename(legacy, path)
            print(f"Moved {legacy} to {path} (siar-scanner is now siar-app).", file=sys.stderr)
        except OSError:
            # A failed move is not a reason to fail the command — fall back to the old
            # directory and carry on, rather than silently starting from an empty workspace.
            os.makedirs(legacy, mode=0o700, exist_ok=True)
            return os.path.abspath(legacy)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return os.path.abspath(path)


def algorithm_cache_dir(slug: str, platform_tag: str) -> str:
    """Where one algorithm build is unpacked. Created if absent."""
    path = os.path.join(home(), "algorithms", _safe(slug), _safe(platform_tag))
    os.makedirs(path, exist_ok=True)
    return path


def imported_dir() -> str:
    """Where models imported from another machine live. Created if absent.

    Beside the algorithm cache and deliberately not inside it: that directory is a cache, and
    everything in it can be fetched again from the server. This one cannot — an imported model
    exists on the machine that built it and on the ones somebody carried it to, and clearing a
    cache must never be the thing that loses it.

    Returns:
        ``$SIAR_APP_HOME/models``, absolute.
    """
    path = os.path.join(home(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def _safe(name: str) -> str:
    """A filesystem-safe slug — a registry name should never escape the cache directory."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name)) or "unnamed"


# -- small JSON helpers ------------------------------------------------------------------


def read_json(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` if it is missing or unreadable."""
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: str, payload: Any, *, mode: int | None = None) -> None:
    """Write JSON atomically, optionally chmod-ing the result.

    Atomically because a half-written ``credentials.json`` locks the user out of their own CLI
    with a parse error, and the fix ("delete this file") is not discoverable.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


# -- credentials -------------------------------------------------------------------------


def _credentials_path() -> str:
    return os.path.join(home(), "credentials.json")


def load_credentials() -> dict:
    """The saved login, or ``{}``.

    ``$SIAR_APP_TOKEN`` and ``$SIAR_APP_URL`` override the file, so a CI job or a
    container can run without ``siar-app login`` ever having been called.

    Returns:
        ``{"base_url": ..., "token": ..., "username": ...}`` — possibly partial.
    """
    creds = dict(read_json(_credentials_path(), {}) or {})
    token = legacy_env("TOKEN")
    if token:
        creds["token"] = token
    url = legacy_env("URL")
    if url:
        creds["base_url"] = url
    creds.setdefault("base_url", DEFAULT_BASE_URL)
    return creds


def save_credentials(base_url: str, token: str, username: str = "") -> str:
    """Persist a token at mode 0600.

    Args:
        base_url: The IDent Dynamics install the token is for.
        token: The bearer token.
        username: Who it belongs to, for the ``login`` confirmation and ``runs`` listing.

    Returns:
        The path written.
    """
    path = _credentials_path()
    write_json(
        path,
        {"base_url": base_url.rstrip("/"), "token": token, "username": username},
        mode=0o600,
    )
    return path


def clear_credentials() -> bool:
    """Delete the saved token. Returns True if there was one."""
    try:
        os.unlink(_credentials_path())
        return True
    except OSError:
        return False


# -- platform tags -----------------------------------------------------------------------


#: The Python this package's algorithm bundles are built for. An obfuscated bundle's bytecode is
#: marshalled by the interpreter that built it, so this is an equality, not a floor — which is
#: why ``pyproject.toml`` pins ``>=3.13,<3.14`` rather than ``>=3.13``, and why the recommended
#: install is ``uv tool install``, which fetches this interpreter instead of using whichever one
#: the machine happens to have.
SUPPORTED_PYTHON = (3, 13)


def default_platform_tag() -> str:
    """This machine's build tag, e.g. ``"linux-x86_64-cp313"``.

    An obfuscated bundle's runtime is pinned to OS + CPU architecture + Python minor version, so
    a build is only ever runnable on machines reporting this same string. Computed identically
    to the IDent Dynamics SDK's ``default_platform_tag``
    (``identdynamics/client.py``) and to the publisher's build script
    (``siar-lib-algorithms/tools/build_and_publish.py``, ``host_platform_tag``) — if these three
    ever disagree, downloads silently stop matching. Change one, change all three.

    Note:
        macOS reports ``darwin`` here, never ``macos`` — ``platform.system()`` is ``Darwin``. The
        tag on Apple silicon is ``darwin-arm64-cp313``.

    Returns:
        ``"<system>-<machine>-cp<major><minor>"``, lowercased.
    """
    system = (platform.system() or "any").lower()
    machine = (platform.machine() or "any").lower()
    return f"{system}-{machine}-cp{sys.version_info[0]}{sys.version_info[1]}"


def python_supported() -> bool:
    """Whether this interpreter is the one the published bundles are built for."""
    return sys.version_info[:2] == SUPPORTED_PYTHON


def supported_python_text() -> str:
    """``"3.13"`` — the Python the bundles are built for, for use in messages."""
    return f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}"


def libc_flavour() -> str:
    """``"glibc"``, ``"musl"`` or ``""`` off Linux.

    The platform tag cannot carry this: ``platform.system()`` says ``linux`` on Alpine exactly as
    it does on Debian, so an Alpine machine matches — and downloads — a bundle whose runtime is
    linked against glibc. That fails at import with a relocation error naming a symbol the reader
    has no reason to recognise. Detecting it here is what turns that into a sentence.
    """
    if platform.system().lower() != "linux":
        return ""
    try:
        name = (platform.libc_ver()[0] or "").lower()
    except OSError:
        name = ""
    if "glibc" in name or name == "libc":
        return "glibc"
    if name:
        return name
    # libc_ver() reads the executable and comes back empty on musl, so ask the filesystem.
    import glob

    return "musl" if glob.glob("/lib/ld-musl-*.so.1") else "glibc"


def platform_compatible(build_tag: str | None, local_tag: str | None = None) -> bool:
    """Whether a build tagged ``build_tag`` can run on ``local_tag`` (default: this machine).

    An empty or ``"any"`` tag is treated as portable — pure-Python bundles exist and refusing
    them for lack of a tag would be wrong.
    """
    build = (build_tag or "").strip().lower()
    if not build or build == "any":
        return True
    return build == (local_tag or default_platform_tag()).strip().lower()


# -- run history -------------------------------------------------------------------------


def _runs_path() -> str:
    return os.path.join(home(), "runs.json")


def run_history() -> list[dict]:
    """The recorded runs, newest first."""
    rows = read_json(_runs_path(), []) or []
    return rows if isinstance(rows, list) else []


def record_run(entry: dict) -> None:
    """Prepend one run to the history, trimming to :data:`RUN_HISTORY_MAX`.

    Best-effort: a workspace on a read-only or full filesystem must not fail a scan that has
    already written its real output.
    """
    try:
        rows = run_history()
        rows.insert(0, entry)
        write_json(_runs_path(), rows[:RUN_HISTORY_MAX])
    except OSError:
        pass


def recent_cost(algorithm: str) -> tuple[float, dict]:
    """What this algorithm cost per second of audio, last time it ran here, and how it divided.

    This is what a live display needs before it has learned anything of its own. Without it the
    first round of a run draws no progress at all — every bar waits on the first completed
    recording, and on a corpus of hour-long files that is an hour of a screen that says only
    "still going". The same machine running the same algorithm again is a good enough guess to
    draw with, and it is replaced by this run's own measurement the moment one file lands.

    Matched on the algorithm alone. The recording lengths do not matter — the figure is per
    second of audio — and the grid rarely does, because ``--fft`` is overridden on a small
    minority of runs and a wrong guess costs a bar that corrects itself within one file.

    Args:
        algorithm: The slug being run.

    Returns:
        ``(wall seconds per second of audio, per-stage seconds)``. ``(0.0, {})`` when nothing
        usable was recorded — a first run on this machine, or a history written before the
        figures were kept. The stage split is what lets a bar know that a recording is nine
        tenths ``scan``, which is the difference between a lane bar that means something and one
        that is full before the scan starts.
    """
    for row in run_history():
        if not isinstance(row, dict) or row.get("algorithm") != algorithm:
            continue
        audio = float(row.get("audio_sec") or 0.0)
        worker = float(row.get("worker_sec") or 0.0)
        if audio > 0 and worker > 0:
            phases = row.get("phases")
            return worker / audio, dict(phases) if isinstance(phases, dict) else {}
    return 0.0, {}
