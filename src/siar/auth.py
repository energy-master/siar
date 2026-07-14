# Vixen Intelligence c.2026
"""API-key gate for the CLI.

Not real security — the source is open Python — but a barrier that stops casual use of a
distributed binary without an issued key.

The key is set once with ``siar activate <key>``, which writes a salted SHA-256 hash to
``~/.siar/.api_key``.  Every subsequent command reads ``$SIAR_API_KEY`` from the environment and
compares it against the stored hash.  If either piece is missing, the CLI refuses to run.

The hash file is dot-prefixed so it does not show up in casual ``ls`` of the workspace.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

from siar import paths

__all__ = ["activate", "require_key"]

_FILE = ".api_key"
_SEP = ":"


def _key_path() -> Path:
    """Path to the stored key hash."""
    return paths.workspace() / _FILE


def _hash(salt: str, key: str) -> str:
    """Salted SHA-256 of *key*."""
    return hashlib.sha256(f"{salt}{key}".encode("utf-8")).hexdigest()


def activate(key: str) -> Path:
    """Store a salted hash of *key* so future commands can verify it.

    Args:
        key: The API key to register.  Whitespace is stripped.

    Returns:
        The path the hash was written to.

    Raises:
        ValueError: If *key* is empty after stripping.
    """
    key = key.strip()
    if not key:
        raise ValueError("key must not be empty")

    paths.ensure_workspace()
    salt = secrets.token_hex(16)
    digest = _hash(salt, key)
    path = _key_path()
    path.write_text(f"{salt}{_SEP}{digest}", encoding="utf-8")
    path.chmod(0o600)
    return path


def require_key() -> None:
    """Verify that ``$SIAR_API_KEY`` matches the stored hash.

    Raises:
        SystemExit: If no key is stored, or the env var is missing, or the key is wrong.
    """
    import os
    import sys

    path = _key_path()
    if not path.is_file():
        print("siar: not activated — run `siar activate <key>` first", file=sys.stderr)
        raise SystemExit(1)

    env_key = os.environ.get("SIAR_API_KEY", "")
    if not env_key:
        print(
            "siar: $SIAR_API_KEY is not set — export it in your shell to use SIAR",
            file=sys.stderr,
        )
        raise SystemExit(1)

    stored = path.read_text(encoding="utf-8").strip()
    if _SEP not in stored:
        print("siar: corrupt key file — re-run `siar activate <key>`", file=sys.stderr)
        raise SystemExit(1)

    salt, expected = stored.split(_SEP, 1)
    actual = _hash(salt, env_key)

    if not hmac.compare_digest(actual, expected):
        print("siar: invalid API key", file=sys.stderr)
        raise SystemExit(1)
