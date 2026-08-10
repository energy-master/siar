# Vixen Intelligence c.2026
"""What each ``siar-scanner`` subcommand does.

One function per command, each taking the parsed ``argparse.Namespace`` and returning a process
exit code. Printing happens here and nowhere else — :mod:`siarscan.runner` and the modules under
it return values, so they can be tested without capturing stdout and reused without a terminal.

Every command returns rather than calls ``sys.exit``, and every expected failure is a message
plus a non-zero code rather than a traceback. A user whose token expired mid-survey should read
one sentence telling them to log in again, not forty lines of urllib internals.
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import time
from argparse import Namespace

from siarscan import __version__
from siarscan.api import ApiError, AuthError, Client, client_from_credentials
from siarscan.cli.table import render_table
from siarscan.config import (
    DEFAULT_BASE_URL,
    clear_credentials,
    default_platform_tag,
    load_credentials,
    record_run,
    run_history,
    save_credentials,
)
from siarscan.grid import ScannerError
from siarscan.io.audio import find_recordings, probe
from siarscan.loader import load_local, load_remote
from siarscan.runner import RunOptions, run_folder

__all__ = [
    "cmd_algorithms",
    "cmd_login",
    "cmd_logout",
    "cmd_run",
    "cmd_runs",
    "cmd_version",
    "cmd_whoami",
]


def _err(message: str) -> int:
    """Print an error to stderr and return the failure exit code."""
    print(f"error: {message}", file=sys.stderr)
    return 1


# -- version / auth ------------------------------------------------------------------------


def cmd_version(_args: Namespace) -> int:
    """Print the package version and this machine's build tag.

    The platform tag belongs here because it is the first thing to check when a download is
    refused, and asking a user to run a Python one-liner to find it is a poor start.
    """
    print(f"siar-scanner {__version__}")
    print(f"platform     {default_platform_tag()}")
    return 0


#: What to tell someone whose terminal cannot prompt — a CI job, a pipe, an agent shell.
_NO_TTY_HELP = (
    "cannot prompt for credentials: this shell has no interactive input.\n"
    "  Either run `siar-scanner login` in a normal terminal, or supply both without a prompt:\n"
    "      siar-scanner login <username> --server <url>   with $SIAR_SCANNER_PASSWORD set\n"
    "  Or skip logging in entirely by exporting a token you already have:\n"
    "      export SIAR_SCANNER_TOKEN=...  SIAR_SCANNER_URL=..."
)


def cmd_login(args: Namespace) -> int:
    """Exchange IDent Dynamics credentials for a bearer token and save it."""
    base_url = args.server or load_credentials().get("base_url") or DEFAULT_BASE_URL

    login_id = args.login
    if not login_id:
        try:
            login_id = input("IDent Dynamics username or email: ").strip()
        except EOFError:
            return _err(_NO_TTY_HELP)
    if not login_id:
        return _err("no username given")

    password = os.environ.get("SIAR_SCANNER_PASSWORD")
    if not password:
        try:
            password = getpass.getpass("Password: ")
        except (EOFError, getpass.GetPassWarning):
            return _err(_NO_TTY_HELP)
    if not password:
        return _err("no password given")

    client = Client(base_url)
    try:
        res = client.login(login_id, password, device=args.device or "")
    except AuthError as e:
        return _err(str(e))
    except ApiError as e:
        return _err(str(e))

    user = res.get("user") or {}
    username = str(user.get("username") or login_id)
    path = save_credentials(base_url, client.token or "", username)
    print(f"Signed in to {base_url} as {username}.")
    print(f"Token saved to {path} (delete it, or run `siar-scanner logout`, to sign out).")
    return 0


def cmd_logout(_args: Namespace) -> int:
    """Forget the saved token.

    Local only: it does not revoke the token server-side, which is done from the account page.
    Say so, rather than let a user believe a lost laptop has been dealt with.
    """
    if clear_credentials():
        print("Signed out on this machine.")
        print("The token itself is still valid — revoke it from your account page to kill it.")
    else:
        print("Not signed in.")
    return 0


def cmd_whoami(_args: Namespace) -> int:
    """Show who the saved credentials belong to, and where they point."""
    creds = load_credentials()
    if not creds.get("token"):
        print("Not signed in. Run `siar-scanner login`.")
        return 1
    print(f"user     {creds.get('username') or '(unknown)'}")
    print(f"server   {creds.get('base_url')}")
    print(f"platform {default_platform_tag()}")
    return 0


# -- the catalogue -------------------------------------------------------------------------


def cmd_algorithms(args: Namespace) -> int:
    """List the scanning algorithms this account can download.

    The description column is the one the super user wrote in the admin panel, which is the
    whole reason it is stored server-side rather than baked into a build: what an algorithm is
    for changes with what it has been used for, and re-obfuscating it to fix a sentence would be
    absurd.
    """
    try:
        client = client_from_credentials(args.server)
        rows = client.algorithms()
    except AuthError as e:
        return _err(str(e))
    except ApiError as e:
        return _err(str(e))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No scanning algorithms have been published for your account yet.")
        return 0

    local = default_platform_tag()
    table = [
        [
            r.get("slug", ""),
            ", ".join(r.get("shapes") or []) or "—",
            "yes" if r.get("runnable") else "no",
            (r.get("description") or r.get("title") or "").strip() or "—",
        ]
        for r in rows
    ]
    print(render_table(
        ["SLUG", "FINDS", "RUNS HERE", "DESCRIPTION"],
        table,
        flex=3,
    ))

    unrunnable = [r for r in rows if not r.get("runnable")]
    if unrunnable:
        print()
        print(
            f"{len(unrunnable)} of these have no build for {local}. An obfuscated bundle is "
            "pinned to OS, CPU architecture and Python minor version — ask for a build for "
            "this machine."
        )
    if args.params:
        _print_params(rows)
    return 0


def _print_params(rows: list[dict]) -> None:
    """Print each algorithm's tunable parameters under its slug."""
    for r in rows:
        schema = r.get("params_schema") or {}
        if not schema:
            continue
        print()
        print(f"{r.get('slug', '')} — parameters (--param name=value):")
        table = [
            [
                name,
                str(spec.get("type", "number")),
                str(spec.get("default", "")),
                str(spec.get("help", "")).strip(),
            ]
            for name, spec in sorted(schema.items())
        ]
        print(render_table(["NAME", "TYPE", "DEFAULT", "MEANING"], table, flex=3))


# -- the scan ------------------------------------------------------------------------------


def cmd_run(args: Namespace) -> int:
    """Scan a folder and build the output folder."""
    source = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(source):
        return _err(f"{args.folder} is not a folder")

    out_root = os.path.abspath(os.path.expanduser(args.out))
    if os.path.abspath(out_root) == source:
        return _err("--out must not be the folder being scanned")

    try:
        params = _parse_params(args.param or [])
    except ValueError as e:
        return _err(str(e))
    if args.fmin is not None:
        params["fmin"] = args.fmin
    if args.fmax is not None:
        params["fmax"] = args.fmax

    try:
        handle = _resolve_algorithm(args)
    except (AuthError, ApiError, ScannerError) as e:
        return _err(str(e))

    options = RunOptions(
        fft=args.fft,
        hop=args.hop,
        window=args.window,
        channel=args.channel,
        params=params,
        link=args.link,
        resume=args.resume,
        thumbnails=not args.no_thumbnails,
        limit=args.limit,
        recursive=not args.no_recursive,
    )

    print(f"algorithm  {handle.slug} ({handle.platform})")
    print(f"scanning   {source}")
    print(f"output     {out_root}")

    started = time.time()
    try:
        manifest = run_folder(
            handle, source, out_root, options,
            progress=None if args.quiet else _progress,
            warn=lambda m: print(f"warning: {m}", file=sys.stderr),
        )
    except FileNotFoundError as e:
        return _err(str(e))
    except (ValueError, ScannerError) as e:
        return _err(str(e))

    _print_summary(manifest, out_root)
    record_run({
        "at": manifest["started_at"],
        "algorithm": handle.slug,
        "source": source,
        "out": out_root,
        "files": manifest["files"],
        "structures": manifest["structures"],
        "elapsed_sec": round(time.time() - started, 2),
    })
    return 1 if manifest["by_status"].get("error") else 0


def _resolve_algorithm(args: Namespace):
    """Load the algorithm named on the command line, locally or from the server."""
    if args.algorithm_path:
        return load_local(os.path.expanduser(args.algorithm_path), args.algorithm or "")
    if not args.algorithm:
        raise ScannerError("give --algorithm SLUG (see `siar-scanner algorithms`)")
    client = client_from_credentials(args.server)
    return load_remote(client, args.algorithm, args.platform, refresh=args.refresh)


def _parse_params(pairs: list[str]) -> dict:
    """Parse ``--param name=value`` into a dict, converting numbers and booleans.

    Values are typed by what they look like, because a command line has no types and an
    algorithm that wanted a float should not receive the string ``"2.5"``.

    Raises:
        ValueError: On a pair with no ``=``.
    """
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--param must be name=value, got {pair!r}")
        name, _, raw = pair.partition("=")
        out[name.strip()] = _coerce(raw.strip())
    return out


def _coerce(raw: str):
    """Turn a command-line value into a bool, int, float, None, or the string itself."""
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _progress(done: int, total: int, result) -> None:
    """One line per recording, overwritten in place when the terminal allows it."""
    tail = {
        "scanned": f"{result.count} structure{'' if result.count == 1 else 's'}",
        "skipped": "skipped (already done)",
        "too_short": "too short to scan",
        "error": f"ERROR — {result.error}",
    }.get(result.status, result.status)
    line = f"[{done}/{total}] {result.rel_path}: {tail}"
    # Errors stay on screen; everything else may be overwritten by the next file.
    if result.status == "error" or not sys.stdout.isatty():
        print(line)
        return
    print(f"\r\033[K{line}", end="", flush=True)
    if done == total:
        print()


def _duration(seconds: float) -> str:
    """A duration a human reads at a glance.

    Hours for a survey, minutes for a session, seconds for a handful of clips. Always printing
    hours makes a twelve-second trial run report "0.00 h of audio", which reads like a failure.
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _print_summary(manifest: dict, out_root: str) -> None:
    """The closing report: what was found, and what to do with it."""
    by_status = manifest["by_status"]
    scanned = by_status.get("scanned", 0)
    print()

    # A fully-resumed run scans nothing, and reporting "0.0 s of audio / no structures found"
    # for it reads like a failure rather than like "there was nothing left to do".
    if scanned == 0 and by_status.get("skipped"):
        print(f"Nothing to do — all {by_status['skipped']} file(s) were already scanned.")
        print("Drop --resume, or delete the output folder, to scan them again.")
    else:
        print(
            f"{manifest['files']} file(s), "
            f"{_duration(manifest['audio_sec'])} of audio, "
            f"in {manifest['elapsed_sec']:.1f}s"
        )
        if manifest["shapes"]:
            found = ", ".join(f"{n} {shape}" for shape, n in manifest["shapes"].items())
            print(f"{manifest['structures']} structures: {found}")
        else:
            print("No structures found.")
        for status in ("skipped", "too_short", "error"):
            if by_status.get(status):
                print(f"{by_status[status]} file(s) {status.replace('_', ' ')}")
    print()
    print(f"Output folder: {out_root}")
    print("Open it in IDent Dynamics (Open folder) to see the boxes on the spectrogram.")


# -- history / probing -----------------------------------------------------------------------


def cmd_runs(args: Namespace) -> int:
    """List the scans run from this machine."""
    rows = run_history()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No runs recorded on this machine yet.")
        return 0
    table = [
        [
            r.get("at", ""),
            r.get("algorithm", ""),
            str(r.get("files", "")),
            str(r.get("structures", "")),
            r.get("out", ""),
        ]
        for r in rows[: args.limit]
    ]
    print(render_table(
        ["WHEN", "ALGORITHM", "FILES", "FOUND", "OUTPUT"],
        table,
        align=["<", "<", ">", ">", "<"],
        flex=4,
    ))
    return 0


def cmd_scan(args: Namespace) -> int:
    """Summarise a folder from headers alone — no decode, no algorithm, no login.

    Worth having before a long run: it answers "how much audio is this, and is it all at one
    sample rate" in about a second for a corpus that would take hours to scan.
    """
    root = os.path.abspath(os.path.expanduser(args.folder))
    files = find_recordings(root, recursive=not args.no_recursive)
    if not files:
        return _err(f"no recordings under {root}")

    infos = [probe(p) for p in files]
    good = [i for i in infos if i is not None]
    unreadable = len(infos) - len(good)
    total_sec = sum(i.duration_sec for i in good)
    rates: dict[float, int] = {}
    for i in good:
        rates[i.sample_rate] = rates.get(i.sample_rate, 0) + 1

    print(f"{len(files)} recording(s) under {root}")
    print(f"{_duration(total_sec)} of audio")
    if unreadable:
        print(f"{unreadable} file(s) could not be read")
    print()
    print(render_table(
        ["SAMPLE RATE", "FILES"],
        [[f"{rate:g} Hz", str(n)] for rate, n in sorted(rates.items())],
        align=["<", ">"],
    ))
    if len(rates) > 1:
        print()
        print(
            "More than one sample rate. Every frequency band maps to a different bin at each "
            "rate, so scanning this folder as one is really scanning several."
        )
    return 0
