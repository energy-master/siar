# Vixen Intelligence c.2026
"""The ``siar-app`` command line.

This module builds the argparse tree and dispatches; the work is in
:mod:`siarapp.cli.commands`. The console script ``siar-app`` (see ``pyproject.toml``) calls
:func:`main`.

Subcommands:

* ``siar-app`` (no command) and ``siar-app lib`` — the library: everything this machine can run,
  the bots and features behind each of it, and the form that starts a scan.
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
* ``siar-app published``  — the models other accounts published here that you may run.
* ``siar-app feedback``   — rate how well an algorithm performed, 0-9.
* ``siar-app scan``       — summarise a folder from headers alone.
* ``siar-app run``        — scan a folder and build an output folder for the app.
* ``siar-app runs``       — what has been run from this machine.
* ``siar-app serve``      — browse an output folder from a laptop, over an ssh tunnel.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from siarapp import __version__
from siarapp.branding import PRODUCT, TAGLINE, print_banner
from siarapp.cli import commands
from siarapp.config import python_supported, supported_python_text
from siarapp.licensing import require_license
from siarapp.runner import DEFAULT_MAX_BYTES

__all__ = ["build_parser", "byte_size", "main"]

#: Commands that run before the licence has been accepted.
#:
#: ``version`` and ``license`` are the two a user needs in order to answer the question the gate
#: is asking — what is this, and what am I agreeing to. Gating either would be a licence you
#: have to accept before you may read it. The two documentation commands join them for the
#: same reason: they are where somebody goes to find out what they have just installed, and a
#: prompt in front of the manual is a poor greeting.
_UNGATED = frozenset({"version", "license", "quick-start", "quickstart", "readme"})

#: ``siar-app serve``'s default port, spelled out here rather than imported from
#: :mod:`siarapp.serve.http` — building the parser must not drag :mod:`http.server` into every
#: invocation. :func:`siarapp.cli.commands.cmd_serve` imports the real constant and would fail its
#: own test if the two ever disagreed.
SERVE_PORT = 8420

#: Multipliers ``--max-size`` accepts. Powers of 1024 under the names a disk uses: someone
#: writing ``--max-size 550MB`` means the number their file manager showed them, and quibbling
#: about MB versus MiB at a ceiling this rough would buy nothing.
_SIZE_UNITS = {
    "": 1,
    "B": 1,
    "K": 1024, "KB": 1024, "KIB": 1024,
    "M": 1024**2, "MB": 1024**2, "MIB": 1024**2,
    "G": 1024**3, "GB": 1024**3, "GIB": 1024**3,
    "T": 1024**4, "TB": 1024**4, "TIB": 1024**4,
}


def byte_size(text: str) -> int:
    """Parse a size like ``550MB``, ``1.5G`` or ``4096`` into bytes.

    Args:
        text: The value as typed. A bare number is bytes; ``0`` means no ceiling.

    Returns:
        The size in bytes.

    Raises:
        argparse.ArgumentTypeError: If it is not a number, carries an unknown unit, or is
            negative — all three of which argparse turns into a usage error naming the flag.
    """
    raw = str(text).strip().replace(" ", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]*)", raw)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a size — write bytes, or a number with a unit like 550MB"
        )
    unit = match.group(2).upper()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(
            f"unknown size unit {match.group(2)!r} (KB, MB, GB, TB, or nothing for bytes)"
        )
    return int(float(match.group(1)) * _SIZE_UNITS[unit])


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
               "then siar-app algorithms. With no command at all it opens the library — "
               "the same screen as siar-app lib.",
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

    sub.add_parser(
        "lib",
        aliases=["library"],
        help="browse what this machine can run, and start a scan from it",
        description="The library: the algorithm bundles downloaded here and the models built "
        "here with siar-build, on one screen, with the bots and features behind each of them — "
        "then an input, an output and a worker count, and Enter to run. It is what `siar-app` "
        "with no command opens, and it needs a terminal.",
        epilog="Keys: ↑↓ select, tab pane, i input, o output, p parallel, enter edit or run, "
               "R reload, q quit.",
    )

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

    p_published = sub.add_parser(
        "published",
        help="list the models other people published to this install that you can run",
        description="Models somebody bred with siar-build on their own machine and published to "
        "this install. Not the same catalogue as `siar-app algorithms`: that one is the "
        "install's and everybody who can sign in sees all of it, while a published model is "
        "granted per account — you always see your own uploads, a super sees every one, and "
        "anybody else sees what has been released and granted to them. `run -a <name>` fetches "
        "one on demand; --get pulls it now, which is what to do before going somewhere with no "
        "link.",
    )
    p_published.add_argument("--get", metavar="NAME",
                             help="download one now instead of listing them")
    p_published.add_argument("--owner", metavar="USER",
                             help="whose upload, when two accounts published the same name")
    p_published.add_argument("--refresh", action="store_true",
                             help="fetch again even if it is already here — a society "
                                  "republishes as its leaderboard moves")
    p_published.add_argument("--remove", metavar="NAME",
                             help="delete one from this machine (it can be fetched again)")
    p_published.add_argument("--json", action="store_true", help="print the raw list as JSON")
    add_server(p_published)

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
    out.add_argument("--max-size", type=byte_size, default=DEFAULT_MAX_BYTES, metavar="SIZE",
                     help=f"skip recordings larger than this, so one enormous file cannot take "
                          f"the run's memory with it (default "
                          f"{DEFAULT_MAX_BYTES // 1024**2}MB; 0 for no ceiling). Accepts KB, MB, "
                          f"GB or a plain byte count")
    out.add_argument("--parallel", nargs="?", type=int, const=0, default=1, metavar="N",
                     help="scan N recordings at once, one process each; bare --parallel uses "
                          "every core this machine's memory will hold")
    out.add_argument("--no-recursive", action="store_true",
                     help="only the top level of the folder")
    out.add_argument("--tui", action="store_true",
                     help="draw the whole run in one live panel: progress, where the time is "
                          "going, what is being found, and a row per worker")
    out.add_argument("--quiet", "-q", action="store_true", help="no per-file progress")

    p_serve = sub.add_parser(
        "serve",
        help="browse an output folder in a browser, without copying it",
        description="Serve one output folder read-only over HTTP, so a scan that ran on a headless "
        "box can be looked at from a laptop through an ssh tunnel. Binds 127.0.0.1 and requires a "
        "token; there is no route that can change anything in the folder.",
        epilog="On your laptop: ssh -N -L 8420:localhost:8420 <user>@<this-host>, then open the "
               "URL this command prints.",
    )
    p_serve.add_argument("folder", nargs="?", metavar="DIR",
                         help="output folder to serve (default: the most recent `siar-app run`)")
    p_serve.add_argument("--port", type=int, default=SERVE_PORT, metavar="N",
                         help=f"port to listen on (default {SERVE_PORT}; 0 picks a free one)")
    p_serve.add_argument("--bind", default="127.0.0.1", metavar="ADDR",
                         help="address to listen on (default 127.0.0.1 — the ssh -L end)")
    p_serve.add_argument("--allow-remote", action="store_true",
                         help="permit a --bind other than loopback, over plain HTTP")
    p_serve.add_argument("--token", metavar="VALUE",
                         help="use this token instead of a fresh random one")
    p_serve.add_argument("--open", action="store_true",
                         help="also open the page in a browser on THIS machine")
    p_serve.add_argument("--no-audio", action="store_true",
                         help="refuse to serve the recordings themselves")
    p_serve.add_argument("--allow-origin", action="append", metavar="ORIGIN",
                         help="let this web origin read the daemon (repeatable; none by default)")
    p_serve.add_argument("--verbose", "-v", action="store_true",
                         help="log one line per request")

    p_export = sub.add_parser(
        "export",
        help="write a model built here to one file another machine can import",
        description="Put a model you built with siar-build — its package, its bots and the "
        "figures it was calibrated on — into a single .siarmodel file. Copy that file to "
        "another machine and `siar-app import` it there. Downloaded algorithms cannot be "
        "exported: they are licensed per machine, and that machine fetches its own. With no "
        "name, lists what could be exported.",
    )
    p_export.add_argument("model", nargs="?", metavar="NAME",
                          help="the model to export (its name in `siar-app lib`)")
    p_export.add_argument("--out", metavar="PATH",
                          help="file to write, or a folder to write it into "
                               "(default: <name>-<version>.siarmodel here)")

    p_import = sub.add_parser(
        "import",
        help="unpack a model bundle exported from another machine",
        description="Install a .siarmodel written by `siar-app export`. It lands in this "
        "machine's workspace, appears in `siar-app lib` with the bots and corpus it came off, "
        "and is runnable by name with `siar-app run -a <name>`. Nothing in the bundle is "
        "executed by this command. With no file, lists the models already imported here.",
    )
    p_import.add_argument("bundle", nargs="?", metavar="FILE", help="the .siarmodel to import")
    p_import.add_argument("--into", metavar="DIR",
                          help="unpack somewhere other than the workspace (not listed in lib)")
    p_import.add_argument("--inspect", action="store_true",
                          help="print what the bundle says about itself and import nothing")

    p_runs = sub.add_parser("runs", help="list the scans run from this machine")
    p_runs.add_argument("--limit", type=int, default=20, metavar="N")
    p_runs.add_argument("--json", action="store_true")

    return parser


#: Subcommand name -> handler. A dict rather than a chain of ifs, so adding a command is one line.
_DISPATCH = {
    "lib": commands.cmd_lib,
    "library": commands.cmd_lib,
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
    "published": commands.cmd_published,
    "export": commands.cmd_export,
    "import": commands.cmd_import,
    "feedback": commands.cmd_feedback,
    "scan": commands.cmd_scan,
    "run": commands.cmd_run,
    "runs": commands.cmd_runs,
    "serve": commands.cmd_serve,
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
        # No command. On a terminal that is somebody who has installed this and typed its name to
        # see what it does, and the honest answer to that is the library: what they have, what it
        # can do, and how to run it. Off a terminal — a pipe, a CI job, a Dockerfile — there is no
        # screen to draw, so the help is still what comes back.
        from siarapp.cli.screen import is_tty

        if not is_tty():
            parser.print_help()
            return 0
        handler = commands.cmd_lib

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
