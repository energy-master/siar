# Vixen Intelligence c.2026
"""The licence this CLI is offered under, and the record that it was accepted.

The terms are shown once, on the first command that does any work, and the acceptance is
recorded in the workspace (``~/.siar-app/license.json``). Delete that file and the terms
are shown again.

Two decisions worth stating, because both would otherwise look like oversights:

* **The gate is on the CLI, not on the algorithms.** This package is MIT. The scanning
  algorithms it downloads are not — they are proprietary bundles under whatever terms your
  IDent Dynamics installation grants, and accepting this says nothing about those.
* **A shell that cannot prompt is not silently accepted for.** A CI job or a container gets a
  message telling it to set ``$SIAR_APP_ACCEPT_LICENSE`` or run ``siar-app license
  --accept`` once, rather than having agreement inferred from its inability to answer.

``LICENSE_VERSION`` exists so that changing the terms re-prompts. Bump it whenever
:data:`LICENSE_TEXT` changes in substance.
"""
from __future__ import annotations

import datetime
import os
import sys

from siarapp.config import home, legacy_env, read_json, write_json

__all__ = [
    "LICENSE_NAME",
    "LICENSE_TEXT",
    "LICENSE_VERSION",
    "accept_license",
    "license_accepted",
    "license_path",
    "require_license",
]

#: Which licence, for the record file and the prompt.
LICENSE_NAME = "MIT"

#: Bump when the terms change in substance, so every user is asked again.
LICENSE_VERSION = 1

#: The year the copyright notice in the licence starts from.
_COPYRIGHT_YEAR = 2026

LICENSE_TEXT = f"""MIT License

Copyright (c) {_COPYRIGHT_YEAR} Vixen Intelligence

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

#: The one line that stops this being mistaken for a grant over the algorithms.
SCOPE_NOTE = (
    "This covers the siar-app command line only. The scanning algorithms it downloads "
    "are proprietary and are licensed separately by the IDent Dynamics installation you sign "
    "in to."
)


def license_path() -> str:
    """Where acceptance is recorded."""
    return os.path.join(home(), "license.json")


def license_accepted() -> bool:
    """True when the current licence version has been accepted on this machine."""
    if legacy_env("ACCEPT_LICENSE"):
        return True
    record = read_json(license_path(), default={}) or {}
    return (
        record.get("license") == LICENSE_NAME
        and int(record.get("version") or 0) >= LICENSE_VERSION
    )


def accept_license() -> str:
    """Record acceptance and return the path it was written to."""
    path = license_path()
    write_json(path, {
        "license": LICENSE_NAME,
        "version": LICENSE_VERSION,
        "accepted_at": datetime.datetime.now(datetime.timezone.utc)
                               .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    })
    return path


def require_license(*, stream=None) -> bool:
    """Show the terms and ask, unless they have already been accepted.

    Args:
        stream: Where to write. Defaults to stderr, so a piped ``--json`` command that happens
            to be the first one ever run still emits clean JSON on stdout.

    Returns:
        True to continue, False to stop. Declining is not an error — it is an answer — so the
        caller exits 1 without a traceback.
    """
    if license_accepted():
        return True

    out = stream if stream is not None else sys.stderr
    print(file=out)
    print(LICENSE_TEXT, file=out)
    print(file=out)
    print(SCOPE_NOTE, file=out)
    print(file=out)

    try:
        answer = input(f"Accept the {LICENSE_NAME} licence? [y/N]: ").strip().lower()
    except EOFError:
        print(
            "This shell cannot prompt, so the licence cannot be accepted interactively.\n"
            "  Accept it once in a terminal:   siar-app license --accept\n"
            "  Or for a script or container:   export SIAR_APP_ACCEPT_LICENSE=1",
            file=out,
        )
        return False

    if answer not in ("y", "yes"):
        print("Not accepted — nothing was run.", file=out)
        return False

    path = accept_license()
    print(f"Accepted. Recorded in {path}.", file=out)
    return True
