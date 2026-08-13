# Vixen Intelligence c.2026
"""The ``siar-app`` command line.

This module builds the argparse tree and dispatches; the work is in
:mod:`siarapp.cli.commands`. The console script ``siar-app`` (see ``pyproject.toml``) calls
:func:`main`.

Subcommands:

* ``siar-app version``    — the package version and this machine's build tag.
* ``siar-app license``    — show the licence, or accept it non-interactively.
* ``siar-app quick-start``— open the illustrated quickstart in a browser, offline.
* ``siar-app readme``     — open the full manual in a browser, offline.
* ``siar-app signup``     — create an IDent Dynamics account.
* ``siar-app login``      — sign in to IDent Dynamics and cache a token.
* ``siar-app logout``     — forget the cached token.
* ``siar-app whoami``     — who the cached token belongs to.
* ``siar-app algorithms`` — the scanning algorithms your account can download.
* ``siar-app installed``  — the algorithm bundles on this machine, and their versions.
* ``siar-app feedback``   — rate how well an algorithm performed, 0-9.
* ``siar-app scan``       — summarise a folder from headers alone.
* ``siar-app run``        — scan a folder and build an output folder for the app.
* ``siar-app runs``       — what has been run from this machine.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from siarapp import __version__
from siarapp.branding import PRODUCT, TAGLINE, print_banner
from siarapp.cli import commands
from siarapp.config import python_supported, supported_python_text
from siarapp.licensing import require_license

__all__ = ["build_parser", "main"]

#: Commands that run before the licence has been accepted.
#:
#: ``version`` and ``license`` are the two a user needs in order to answer the question the gate
#: is asking — what is this, and what am I agreeing to. Gating either would be a licence you
#: have to accept before you may read it. The two documentation commands join them for the
#: same reason: they are where somebody goes to find out what they have just installed, and a
#: prompt in front of the manual is a poor greeting.
_UNGATED = frozenset({"version", "license", "quick-start", "quickstart", "readme"})


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="siar-app",
        description=f"{PRODUCT} — {TAGLINE}. "
                    "Run the IDent Dynamics structure scanners over a folder of recordings.",
        epilog="New here? Run siar-app quick-start for the illustrated walkthrough, or "
               "siar-app readme for the manual. Otherwise: siar-app signup (or login), "
               "then siar-app algorithms.",
    )
    parser.add_argument("--version", action="version", version=f"siar-app {__version__}")
    # Accepted on BOTH sides of the subcommand. `siar-app login --server URL` is what
    # everyone types, and argparse rejects it if the option lives only on the top-level parser
    # — so the top-level copy uses its own dest and main() folds the two together. Sharing one
    # dest instead would let the subcommand's None clobber a value given before the subcommand.
    parser.add_argument(
        "--server",
        dest="global_server",
        metavar="URL",
        help="IDent Dynamics install to talk to (default: the one you logged in to)",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add_server(target: argparse.ArgumentParser) -> None:
        """Give a subcommand its own --server, so it works after the subcommand too."""
        target.add_argument("--server", metavar="URL",
                            help="IDent Dynamics install to talk to")

    sub.add_parser("version", help="print the package version and this machine's build tag")

    p_license = sub.add_parser(
        "license",
        help="show the licence, or accept it without a prompt",
        description="The terms this command line is offered under. They are shown once, on "
        "first use, and the acceptance is recorded in your workspace. This covers the CLI "
        "only — the scanning algorithms it downloads are proprietary and licensed separately.",
    )
    p_license.add_argument("--accept", action="store_true",
                           help="record acceptance and exit — for a script or a container")

    sub.add_parser(
        "quick-start",
        aliases=["quickstart"],
        help="open the illustrated quickstart in your browser",
        description="Open the thirteen-step quickstart — install, sign up, scan a folder, "
        "read the result — in whatever browser this machine has. It ships inside the "
        "package, so it needs no network at all.",
    )

    p_readme = sub.add_parser(
        "readme",
        help="open the full manual in your browser",
        description="Open the complete siar-app manual in your browser. It is rendered from "
        "the copy carried inside this install, so it matches the version you are running and "
        "needs no network.",
    )
    p_readme.add_argument("--text", action="store_true",
                          help="print it as Markdown instead of opening a browser")

    p_signup = sub.add_parser(
        "signup",
        help="create an IDent Dynamics account",
        description="Create an account on an IDent Dynamics install — the same self-service "
        "signup as the web form, without the browser. Every field is prompted for if not "
        "given. The new account must confirm its email before it can sign in, so this "
        "command ends by pointing you at that link rather than by logging you in.",
        epilog="Then: check your email, click the link, and run siar-app login.",
    )
    p_signup.add_argument("--email", metavar="ADDRESS",
                          help="where the verification link is sent (prompted for if omitted)")
    p_signup.add_argument("--username", metavar="NAME",
                          help="3-64 characters: letters, digits, and . _ -")
    p_signup.add_argument("--display-name", metavar="NAME",
                          help="how your name appears in the app (default: your username). "
                               "Pass an empty string to accept the default without a prompt")
    add_server(p_signup)

    p_login = sub.add_parser(
        "login",
        help="sign in to IDent Dynamics and cache a token",
        description="Exchange your IDent Dynamics username (or email) and password for a "
        "bearer token, cached at ~/.siar-app/credentials.json. Set $SIAR_APP_PASSWORD "
        "to avoid the prompt in a script.",
    )
    p_login.add_argument("login", nargs="?", metavar="USERNAME",
                         help="username or email (prompted for if omitted)")
    p_login.add_argument("--device", metavar="LABEL",
                         help="how this machine is labelled in your account's token list")
    add_server(p_login)

    sub.add_parser("logout", help="forget the cached token on this machine")
    add_server(sub.add_parser("whoami", help="show who the cached token belongs to"))

    p_algos = sub.add_parser(
        "algorithms",
        help="list the scanning algorithms your account can download",
        description="The catalogue your IDent Dynamics super user has published. Each row's "
        "description is theirs, so it says what the algorithm is actually being used for.",
    )
    p_algos.add_argument("--family", metavar="NAME",
                         help="only models in this family (name or title)")
    p_algos.add_argument("--params", action="store_true",
                         help="also print each algorithm's tunable parameters")
    p_algos.add_argument("--json", action="store_true", help="print the raw catalogue as JSON")
    add_server(p_algos)

    p_installed = sub.add_parser(
        "installed",
        help="list the algorithm bundles downloaded to this machine",
        description="What is cached under ~/.siar-app/algorithms, and at which version. "
        "Works offline — it reads each bundle's manifest and never imports one.",
    )
    p_installed.add_argument("--check", action="store_true",
                             help="also ask the server whether a newer version is published")
    p_installed.add_argument("--json", action="store_true", help="print the raw list as JSON")
    add_server(p_installed)

    p_feedback = sub.add_parser(
        "feedback",
        help="rate how well an algorithm performed, 0-9",
        description="Tell the people who publish these algorithms how one did on your "
        "recordings. The rating is filed against the version you have installed, since that "
        "is the build whose output you are judging, and one rating per person per build is "
        "kept — rating it again replaces your last answer.",
    )
    p_feedback.add_argument("slug", nargs="?", metavar="NAME",
                            help="which model (see `siar-app installed`)")
    p_feedback.add_argument("--score", "-s", type=int, metavar="0-9",
                            help="0 found nothing useful, 9 found what was there and "
                                 "little else (prompted for if omitted)")
    p_feedback.add_argument("--comment", "-m", metavar="TEXT",
                            help="a sentence on what it did well or badly")
    p_feedback.add_argument("--mine", action="store_true",
                            help="list the ratings you have given instead of adding one")
    add_server(p_feedback)

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
    p_run.add_argument("--algorithm", "-a", metavar="NAME",
                       help="which model (see `siar-app algorithms`)")
    p_run.add_argument("--out", "-o", required=True, metavar="DIR",
                       help="output folder to create")
    p_run.add_argument("--algorithm-path", metavar="DIR",
                       help="run an unobfuscated algorithm package straight off disk "
                            "(development; needs no login)")
    p_run.add_argument("--platform", metavar="TAG",
                       help="download the build for this platform tag instead of this machine's")
    p_run.add_argument("--refresh", action="store_true",
                       help="re-download the algorithm even if it is cached")
    add_server(p_run)

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
    out.add_argument("--parallel", nargs="?", type=int, const=0, default=1, metavar="N",
                     help="scan N recordings at once, one process each; bare --parallel uses "
                          "every core this machine's memory will hold")
    out.add_argument("--no-recursive", action="store_true",
                     help="only the top level of the folder")
    out.add_argument("--tui", action="store_true",
                     help="draw the whole run in one live panel: progress, where the time is "
                          "going, what is being found, and a row per worker")
    out.add_argument("--quiet", "-q", action="store_true", help="no per-file progress")

    p_runs = sub.add_parser("runs", help="list the scans run from this machine")
    p_runs.add_argument("--limit", type=int, default=20, metavar="N")
    p_runs.add_argument("--json", action="store_true")

    return parser


#: Subcommand name -> handler. A dict rather than a chain of ifs, so adding a command is one line.
_DISPATCH = {
    "version": commands.cmd_version,
    "license": commands.cmd_license,
    "quick-start": commands.cmd_quickstart,
    "quickstart": commands.cmd_quickstart,
    "readme": commands.cmd_readme,
    "signup": commands.cmd_signup,
    "login": commands.cmd_login,
    "logout": commands.cmd_logout,
    "whoami": commands.cmd_whoami,
    "algorithms": commands.cmd_algorithms,
    "installed": commands.cmd_installed,
    "feedback": commands.cmd_feedback,
    "scan": commands.cmd_scan,
    "run": commands.cmd_run,
    "runs": commands.cmd_runs,
}


def _warn_unsupported_python() -> None:
    """Say once, early, that this interpreter cannot load any algorithm.

    A warning rather than an exit: ``version`` and ``license`` still work, and a user who has
    hit this needs to be able to run ``siar-app version`` and paste the result to us. The hard
    failure belongs where it is real — in the loader, at the point an actual bundle is refused.
    """
    if python_supported():
        return
    have = f"{sys.version_info[0]}.{sys.version_info[1]}"
    print(
        f"warning: this is Python {have}; the scanning algorithms are built for "
        f"{supported_python_text()} and cannot be loaded here.\n"
        f"         uv tool install --python {supported_python_text()} siar-app",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 on a handled failure, 130 on Ctrl-C.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Fold the two --server spellings together, preferring the one nearest the subcommand.
    # Every command reads args.server, including the ones that never take one.
    args.server = getattr(args, "server", None) or getattr(args, "global_server", None)
    handler = _DISPATCH.get(args.command or "")
    if handler is None:
        parser.print_help()
        return 0

    # Banner first, so the licence prompt appears under the name of the thing asking.
    print_banner(__version__)
    _warn_unsupported_python()
    if args.command not in _UNGATED and not require_license():
        return 1
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
