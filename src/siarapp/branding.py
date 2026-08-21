# Vixen Intelligence c.2026
"""What the CLI says about itself: the product banner and the copyright line.

Printed on **stderr**, never stdout. That is not a detail — ``algorithms --json``,
``installed --json`` and ``runs --json`` exist to be piped into something, and a banner on
stdout would make every one of them emit invalid JSON. Anyone reading a terminal sees stderr
just as plainly, and anyone reading a pipe gets what they asked for.

The year is taken from the clock rather than baked in, so a build that ships in January does
not spend a year claiming the wrong one.
"""
from __future__ import annotations

import datetime
import sys

from siarapp.config import legacy_env

__all__ = [
    "HOME_URL",
    "PRODUCT",
    "TAGLINE",
    "banner",
    "copyright_line",
    "print_banner",
]

#: The product family this CLI belongs to.
PRODUCT = "SIaR"

#: What the initials stand for. Spelled out because nobody guesses it.
TAGLINE = "Signal Intelligence and Reconnaissance"

#: Where the app it talks to lives.
HOME_URL = "goident.ai"

#: The rights holder.
OWNER = "Vixen Intelligence"

# ANSI, used only when stderr is a terminal. Dim rather than colour: this appears above every
# command, and a banner that shouts is a banner people suppress.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def copyright_line(year: int | None = None) -> str:
    """``© Vixen Intelligence <year>``, with the year from the clock unless given."""
    return f"© {OWNER} {year or datetime.date.today().year}"


def banner(version: str, *, colour: bool = False) -> str:
    """The two-line product banner.

    Args:
        version: The package version, shown so a bug report carries it without being asked.
        colour: Emit ANSI dim/bold. Callers pass ``stderr.isatty()``.

    Returns:
        Two lines, no trailing newline.
    """
    if colour:
        first = f"{_BOLD}{PRODUCT}{_RESET}{_DIM} · {TAGLINE} · {HOME_URL}{_RESET}"
        second = f"{_DIM}siar-app {version} · {copyright_line()}{_RESET}"
    else:
        first = f"{PRODUCT} · {TAGLINE} · {HOME_URL}"
        second = f"siar-app {version} · {copyright_line()}"
    return f"{first}\n{second}"


def print_banner(version: str, *, stream=None) -> None:
    """Print the banner to stderr, unless it has been suppressed.

    ``$SIAR_APP_NO_BANNER`` turns it off for a script that logs stderr and would rather not
    log two lines per invocation. There is no way to turn it off permanently, and it is not
    printed twice in one process.
    """
    if legacy_env("NO_BANNER"):
        return
    out = stream if stream is not None else sys.stderr
    print(banner(version, colour=bool(getattr(out, "isatty", lambda: False)())), file=out)
