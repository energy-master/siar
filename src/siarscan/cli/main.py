# Vixen Intelligence c.2026
"""The ``siar-scanner`` command line.

This module builds the argparse tree and dispatches; the work is in
:mod:`siarscan.cli.commands`. The console script ``siar-scanner`` (see ``pyproject.toml``) calls
:func:`main`.

Subcommands:

* ``siar-scanner version``    — the package version and this machine's build tag.
* ``siar-scanner login``      — sign in to IDent Dynamics and cache a token.
* ``siar-scanner logout``     — forget the cached token.
* ``siar-scanner whoami``     — who the cached token belongs to.
* ``siar-scanner algorithms`` — the scanning algorithms your account can download.
* ``siar-scanner scan``       — summarise a folder from headers alone.
* ``siar-scanner run``        — scan a folder and build an output folder for the app.
* ``siar-scanner runs``       — what has been run from this machine.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from siarscan import __version__
from siarscan.cli import commands

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="siar-scanner",
        description="Run the IDent Dynamics structure scanners over a folder of recordings.",
        epilog="Start with: siar-scanner login, then siar-scanner algorithms.",
    )
    parser.add_argument("--version", action="version", version=f"siar-scanner {__version__}")
    parser.add_argument(
        "--server",
        metavar="URL",
        help="IDent Dynamics install to talk to (default: the one you logged in to)",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("version", help="print the package version and this machine's build tag")

    p_login = sub.add_parser(
        "login",
        help="sign in to IDent Dynamics and cache a token",
        description="Exchange your IDent Dynamics username (or email) and password for a "
        "bearer token, cached at ~/.siar-scanner/credentials.json. Set $SIAR_SCANNER_PASSWORD "
        "to avoid the prompt in a script.",
    )
    p_login.add_argument("login", nargs="?", metavar="USERNAME",
                         help="username or email (prompted for if omitted)")
    p_login.add_argument("--device", metavar="LABEL",
                         help="how this machine is labelled in your account's token list")

    sub.add_parser("logout", help="forget the cached token on this machine")
    sub.add_parser("whoami", help="show who the cached token belongs to")

    p_algos = sub.add_parser(
        "algorithms",
        help="list the scanning algorithms your account can download",
        description="The catalogue your IDent Dynamics super user has published. Each row's "
        "description is theirs, so it says what the algorithm is actually being used for.",
    )
    p_algos.add_argument("--params", action="store_true",
                         help="also print each algorithm's tunable parameters")
    p_algos.add_argument("--json", action="store_true", help="print the raw catalogue as JSON")

    p_scan = sub.add_parser(
        "scan",
        help="summarise a folder of recordings from headers alone",
        description="Reads headers only, so a multi-GB corpus is summarised in a second. Worth "
        "running before a long scan: it catches a folder at mixed sample rates.",
    )
    p_scan.add_argument("folder", metavar="FOLDER")
    p_scan.add_argument("--no-recursive", action="store_true",
                        help="only the top level of the folder")

    p_run = sub.add_parser(
        "run",
        help="scan a folder and build an output folder for the web app",
        description="Runs one algorithm over every recording under FOLDER and writes an output "
        "folder holding the audio, one structures sidecar per recording, and a spectrogram "
        "thumbnail per lane. Open that folder in IDent Dynamics to work through the results.",
    )
    p_run.add_argument("folder", metavar="FOLDER", help="root folder of WAV/FLAC recordings")
    p_run.add_argument("--algorithm", "-a", metavar="SLUG",
                       help="which algorithm (see `siar-scanner algorithms`)")
    p_run.add_argument("--out", "-o", required=True, metavar="DIR",
                       help="output folder to create")
    p_run.add_argument("--algorithm-path", metavar="DIR",
                       help="run an unobfuscated algorithm package straight off disk "
                            "(development; needs no login)")
    p_run.add_argument("--platform", metavar="TAG",
                       help="download the build for this platform tag instead of this machine's")
    p_run.add_argument("--refresh", action="store_true",
                       help="re-download the algorithm even if it is cached")

    grid = p_run.add_argument_group("analysis grid (defaults come from the algorithm)")
    grid.add_argument("--fft", type=int, metavar="N", help="FFT size, a power of two")
    grid.add_argument("--hop", type=int, metavar="N", help="hop in samples (default: fft/4)")
    grid.add_argument("--window", default="hann",
                      choices=["hann", "hamming", "blackman", "rectangular"])
    grid.add_argument("--channel", default="mix", metavar="SEL",
                      help="mix (default), left, right, or a channel index")

    tune = p_run.add_argument_group("algorithm parameters")
    tune.add_argument("--param", action="append", metavar="NAME=VALUE",
                      help="set one algorithm parameter; repeatable")
    tune.add_argument("--fmin", type=float, metavar="HZ", help="low edge of the band to scan")
    tune.add_argument("--fmax", type=float, metavar="HZ", help="high edge of the band to scan")

    out = p_run.add_argument_group("output")
    out.add_argument("--link", action="store_true",
                     help="hardlink the audio instead of copying it (same filesystem only)")
    out.add_argument("--resume", action="store_true",
                     help="skip recordings already written to the output folder")
    out.add_argument("--no-thumbnails", action="store_true",
                     help="skip the per-recording lane thumbnails")
    out.add_argument("--limit", type=int, metavar="N",
                     help="stop after N recordings (a trial run over a big corpus)")
    out.add_argument("--no-recursive", action="store_true",
                     help="only the top level of the folder")
    out.add_argument("--quiet", "-q", action="store_true", help="no per-file progress")

    p_runs = sub.add_parser("runs", help="list the scans run from this machine")
    p_runs.add_argument("--limit", type=int, default=20, metavar="N")
    p_runs.add_argument("--json", action="store_true")

    return parser


#: Subcommand name -> handler. A dict rather than a chain of ifs, so adding a command is one line.
_DISPATCH = {
    "version": commands.cmd_version,
    "login": commands.cmd_login,
    "logout": commands.cmd_logout,
    "whoami": commands.cmd_whoami,
    "algorithms": commands.cmd_algorithms,
    "scan": commands.cmd_scan,
    "run": commands.cmd_run,
    "runs": commands.cmd_runs,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 on a handled failure, 130 on Ctrl-C.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command or "")
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except KeyboardInterrupt:
        # A part-written output folder is still usable — `--resume` picks it up — so say that
        # rather than leave the user wondering whether they have to start again.
        print("\nInterrupted. Re-run with --resume to carry on where this stopped.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
