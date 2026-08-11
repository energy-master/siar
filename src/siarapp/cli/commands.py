# Vixen Intelligence c.2026
"""What each ``siar-app`` subcommand does.

One function per command, each taking the parsed ``argparse.Namespace`` and returning a process
exit code. Printing happens here and nowhere else — :mod:`siarapp.runner` and the modules under
it return values, so they can be tested without capturing stdout and reused without a terminal.

Every command returns rather than calls ``sys.exit``, and every expected failure is a message
plus a non-zero code rather than a traceback. A user whose token expired mid-survey should read
one sentence telling them to log in again, not forty lines of urllib internals.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import sys
import time
from argparse import Namespace
from typing import Any

from siarapp import __version__
from siarapp.api import ApiError, AuthError, Client, client_from_credentials
from siarapp.branding import copyright_line
from siarapp.cli.table import render_table, terminal_width
from siarapp.config import (
    DEFAULT_BASE_URL,
    clear_credentials,
    default_platform_tag,
    libc_flavour,
    python_supported,
    supported_python_text,
    home,
    legacy_env,
    load_credentials,
    read_json,
    record_run,
    run_history,
    save_credentials,
)
from siarapp.licensing import (
    LICENSE_NAME,
    LICENSE_TEXT,
    SCOPE_NOTE,
    accept_license,
    license_accepted,
    license_path,
)
from siarapp.grid import ScannerError
from siarapp.io.audio import find_recordings, probe
from siarapp.loader import installed_algorithms, load_local, load_remote
from siarapp.runner import RunOptions, run_folder

__all__ = [
    "cmd_algorithms",
    "cmd_feedback",
    "cmd_installed",
    "cmd_license",
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
    print(f"siar-app {__version__}")
    print(f"platform     {default_platform_tag()}")
    py = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    if python_supported():
        print(f"python       {py}")
    else:
        print(f"python       {py}  (unsupported: algorithms are built for "
              f"{supported_python_text()})")
    libc = libc_flavour()
    if libc and libc != "glibc":
        print(f"libc         {libc}  (bundles are built against glibc)")
    print(f"licence      {LICENSE_NAME}" + ("" if license_accepted() else " (not yet accepted)"))
    print(copyright_line())
    return 0


def cmd_license(args: Namespace) -> int:
    """Show the licence, or record acceptance without a prompt.

    Printed on stdout here, unlike the first-run prompt: someone who typed ``license`` asked
    for the text and may well want to redirect it.
    """
    if args.accept:
        path = accept_license()
        print(f"{LICENSE_NAME} licence accepted. Recorded in {path}.")
        print("You will not be asked again on this machine.")
        return 0

    print(LICENSE_TEXT)
    print()
    print(SCOPE_NOTE)
    print()
    if license_accepted():
        record = read_json(license_path(), default={}) or {}
        when = record.get("accepted_at")
        if legacy_env("ACCEPT_LICENSE"):
            print("Accepted via $SIAR_APP_ACCEPT_LICENSE.")
        elif when:
            print(f"Accepted on this machine at {when} ({license_path()}).")
        else:
            print(f"Accepted on this machine ({license_path()}).")
    else:
        print("Not yet accepted. Run `siar-app license --accept`, or accept the prompt on "
              "the first command that does any work.")
    return 0


#: What to tell someone whose terminal cannot prompt — a CI job, a pipe, an agent shell.
_NO_TTY_HELP = (
    "cannot prompt for credentials: this shell has no interactive input.\n"
    "  Either run `siar-app login` in a normal terminal, or supply both without a prompt:\n"
    "      siar-app login <username> --server <url>   with $SIAR_APP_PASSWORD set\n"
    "  Or skip logging in entirely by exporting a token you already have:\n"
    "      export SIAR_APP_TOKEN=...  SIAR_APP_URL=..."
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

    password = legacy_env("PASSWORD")
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
    print(f"Token saved to {path} (delete it, or run `siar-app logout`, to sign out).")
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
        print("Not signed in. Run `siar-app login`.")
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
    if args.family:
        wanted = args.family.strip().lower()
        rows = [r for r in rows
                if wanted in (str(r.get("family", "")).lower(),
                              str(r.get("family_title", "")).lower())]
        if not rows:
            return _err(f"no models in a family called {args.family!r} — "
                        "`siar-app algorithms` lists the families you have")

    # `published` is only ever sent to a super user, and only on the rows nobody else can see
    # yet. Marking the slug is the whole point: a super building an algorithm sees it in this
    # list long before it is anyone else's, and without the mark they cannot tell which of
    # these their own users are actually being offered.
    #
    # Printed one table per FAMILY rather than one flat list with a family column. SIaR is not
    # only structure scanners any more, and the question a user brings to this listing is
    # "what have I got for clicks" long before it is "what have I got".
    headers = ["NAME", "FINDS", "RUNS HERE", "RATED", "WHAT IT IS"]
    for family, group in _by_family(rows):
        fixed = [
            [
                r.get("slug", "") + ("" if r.get("published", True) else " *"),
                _shapes(r.get("shapes")),
                "yes" if r.get("runnable") else "no",
                _rating(r.get("rating")),
            ]
            for r in group
        ]
        # Measure the four fixed columns and give the summary exactly what is left, so every
        # model occupies one row. Guessing a width instead is how the column ends up wrapping
        # on the terminals nobody tested on.
        used = sum(
            max([len(headers[c])] + [len(row[c]) for row in fixed])
            for c in range(4)
        ) + 2 * 4
        summary_width = max(20, terminal_width() - used)
        print()
        print(family)
        print(render_table(
            headers,
            [
                row + [_one_line(r.get("description") or r.get("title") or "", summary_width)]
                for row, r in zip(fixed, group)
            ],
            align=["<", "<", "<", ">", "<"],
            flex=4,
        ))

    unpublished = [r for r in rows if not r.get("published", True)]
    if unpublished:
        print()
        print(
            f"* {len(unpublished)} of these are unpublished — visible to you as a super user "
            "only. Tick them in Admin -> Scanner algorithms to offer them to everyone."
        )

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


def _shapes(shapes: Any, keep: int = 3) -> str:
    """What a model finds, abbreviated so the description gets the width instead.

    Five shape names run to 37 characters and push the one thing a reader is actually scanning
    for — what the model is — off the end of the line. The full list is one ``--json`` away.
    """
    names = [str(s) for s in (shapes or []) if str(s)]
    if not names:
        return "—"
    if len(names) <= keep:
        return ", ".join(names)
    return ", ".join(names[:keep]) + f" +{len(names) - keep}"


def _one_line(text: str, limit: int) -> str:
    """The first sentence of a description, on one line, within ``limit`` characters.

    Descriptions are written to be read once, in the admin panel, and they run to a paragraph:
    what the model does, when to reach for it, what it was measured at. Wrapped into a table
    cell that is ten rows deep, ten times over, the catalogue stops being a list you can scan
    at all — which is the one job this listing has.

    So the table shows the opening sentence and nothing else. The full text is still there for
    anyone who wants it: ``--json`` returns it verbatim, and the admin panel is where it is
    written. This is a display decision, not a truncation of the data.
    """
    flat = " ".join(str(text or "").split())
    if not flat:
        return "—"
    # First sentence, if there is one and it is not itself an essay. The period has to be
    # followed by a space so "0.5" and "e.g." do not end the sentence early.
    for stop in range(len(flat) - 1):
        if flat[stop] == "." and flat[stop + 1] == " ":
            candidate = flat[: stop + 1]
            if len(candidate) <= limit:
                return candidate
            break
    if len(flat) <= limit:
        return flat
    # Cut on a word boundary rather than mid-word, then mark that there is more.
    cut = flat[: max(1, limit - 1)].rsplit(" ", 1)[0]
    return f"{cut}…"


def _by_family(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group catalogue rows by family, keeping the order the server sent.

    The server orders by family already, so this preserves that rather than re-sorting: the
    display order of families is the super user's decision, not this client's.
    """
    groups: dict[str, list[dict]] = {}
    titles: dict[str, str] = {}
    for row in rows:
        key = str(row.get("family") or "structure_scanners")
        titles.setdefault(key, str(row.get("family_title") or key))
        groups.setdefault(key, []).append(row)
    return [(titles[key], group) for key, group in groups.items()]


def _rating(rating: Any) -> str:
    """A catalogue row's user rating, as ``6.7/9 (3)``.

    Blank rather than "0.0/9" when nobody has rated it: an unrated algorithm and a badly-rated
    one must not look alike, and on a scale where 0 is a real answer they otherwise would.
    """
    if not isinstance(rating, dict) or not rating.get("count"):
        return "—"
    return f"{rating.get('average', 0)}/9 ({rating['count']})"


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


# -- what is on this machine ------------------------------------------------------------------


def cmd_installed(args: Namespace) -> int:
    """List the algorithm bundles downloaded to this machine, and their versions.

    Offline by design. `algorithms` says what the server offers; this says what is actually
    here, which is the question that matters on a vessel with no link — and the two answers
    diverge the moment a build is republished, which is what ``--check`` exists to notice.
    """
    rows = installed_algorithms()

    latest: dict[str, str] = {}
    if args.check and rows:
        try:
            client = client_from_credentials(args.server)
            latest = {
                str(r.get("slug")): str(r.get("version") or "")
                for r in client.algorithms()
            }
        except (AuthError, ApiError) as e:
            # A failed check must not fail the command: the local answer is still correct, and
            # it is the answer someone offline asked for.
            print(f"warning: could not check the server ({e})", file=sys.stderr)
            args.check = False

    if args.json:
        if latest:
            for r in rows:
                r["server_version"] = latest.get(r["slug"], "")
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No algorithms downloaded on this machine yet.")
        print("`siar-app algorithms` lists what you can run; the first `run` fetches one.")
        return 0

    headers = ["NAME", "VERSION", "PLATFORM", "SIZE", "DOWNLOADED", "RUNS HERE"]
    align = ["<", "<", "<", ">", "<", "<"]
    table = [
        [
            r["slug"],
            r["version"],
            r["platform"],
            _human_bytes(r["bytes"]),
            _when(r["downloaded_at"]),
            "yes" if r["runnable"] else "no",
        ]
        for r in rows
    ]
    if args.check:
        headers.append("SERVER")
        align.append("<")
        for row, r in zip(table, rows):
            row.append(_version_status(r["version"], latest.get(r["slug"])))

    print(render_table(headers, table, align=align))
    print()
    total = sum(r["bytes"] for r in rows)
    print(f"{len(rows)} bundle(s), {_human_bytes(total)} in {os.path.join(home(), 'algorithms')}")
    if args.check and any(row[-1].endswith("available") for row in table):
        print("`siar-app run --refresh -a <name>` replaces a cached bundle with the newest.")
    return 0


def _version_status(local: str, server: str | None) -> str:
    """How a cached version compares with what the server now publishes."""
    if not server:
        # Either it was withdrawn, or this account can no longer see it. Both mean the same
        # thing to the person reading: nothing will replace what is cached here.
        return "not offered"
    if local == server:
        return "up to date"
    if _version_key(server) > _version_key(local):
        return f"{server} available"
    return f"ahead ({server} published)"


def _version_key(text: str) -> tuple:
    """Sortable key for a dotted version, so 1.10.0 beats 1.9.0 as it should."""
    parts = re.findall(r"\d+", str(text))
    return tuple(int(p) for p in parts[:4]) or (0,)


def _when(epoch: int) -> str:
    """A download timestamp, local time, to the minute."""
    if not epoch:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def _human_bytes(n: int | None) -> str:
    """Bytes at a human scale. KiB not kB: this is a file on a disk, not a link speed."""
    if not n:
        return "0 B"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


# -- feedback ---------------------------------------------------------------------------------


#: What each point on the scale means, printed above the prompt. A bare "0-9?" gets a shrug and
#: a 5; naming the ends gets an answer that means something when averaged.
_SCORE_HELP = (
    "  0-2  found nothing useful, or buried the recording in false boxes\n"
    "  3-5  usable, but needed a lot of sorting through\n"
    "  6-7  found what was there, with some noise\n"
    "  8-9  found what was there and little else"
)


def cmd_feedback(args: Namespace) -> int:
    """Rate an algorithm's performance, or list the ratings this account has given.

    The rating is filed against the version installed on this machine, not the newest one
    published: the user is reacting to output that a specific build produced, and rolling that
    onto a later build would credit it with a predecessor's reputation.
    """
    try:
        client = client_from_credentials(args.server)
    except AuthError as e:
        return _err(str(e))

    if args.mine:
        return _print_my_feedback(client)

    if not args.slug:
        return _err("which model? `siar-app feedback <name> --score 7` "
                    "(see `siar-app installed`)")

    score = args.score
    if score is None:
        score = _prompt_score(args.slug)
        if score is None:
            return 1
    if not 0 <= score <= 9:
        return _err(f"--score must be from 0 to 9, got {score}")

    # The installed build is the one that produced whatever the user is reacting to. Falling
    # back to the empty string lets the server file it against what it currently publishes.
    version, platform = "", default_platform_tag()
    for row in installed_algorithms():
        if row["slug"] == args.slug:
            version, platform = row["version"], row["platform"]
            break

    try:
        res = client.send_feedback(
            args.slug, score, version=version,
            comment=args.comment or "", platform=platform,
        )
    except (AuthError, ApiError) as e:
        return _err(str(e))

    stamped = res.get("version") or "the published build"
    print(f"Thanks — {args.slug} {stamped} rated {score}/9.")
    build = res.get("build") or {}
    if build.get("count", 0) > 1:
        print(f"That build now averages {build['average']}/9 from {build['count']} ratings.")
    whole = res.get("algorithm") or {}
    if whole.get("count", 0) > build.get("count", 0):
        print(f"Across every version: {whole['average']}/9 from {whole['count']}.")
    return 0


def _prompt_score(slug: str) -> int | None:
    """Ask for a score interactively. ``None`` if there is no tty or the answer is unusable."""
    print(f"How well did {slug} do on your recordings?")
    print(_SCORE_HELP)
    try:
        raw = input("score 0-9: ").strip()
    except EOFError:
        _err("no --score given and this shell cannot prompt for one")
        return None
    if not raw.isdigit() or not 0 <= int(raw) <= 9:
        _err(f"not a score from 0 to 9: {raw!r}")
        return None
    return int(raw)


def _print_my_feedback(client) -> int:
    """Print every rating this account has given."""
    try:
        rows = client.my_feedback()
    except (AuthError, ApiError) as e:
        return _err(str(e))
    if not rows:
        print("You have not rated any algorithms yet.")
        print("After a scan: `siar-app feedback <name> --score 0..9`")
        return 0
    print(render_table(
        ["WHEN", "MODEL", "VERSION", "SCORE", "COMMENT"],
        [
            [
                str(r.get("updated_at", ""))[:16],
                r.get("slug", ""),
                r.get("version", "") or "—",
                f"{r.get('score', '')}/9",
                (r.get("comment") or "").strip() or "—",
            ]
            for r in rows
        ],
        align=["<", "<", "<", ">", "<"],
        flex=4,
    ))
    return 0


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
        raise ScannerError("give --algorithm SLUG (see `siar-app algorithms`)")
    client = client_from_credentials(args.server)
    return load_remote(
        client, args.algorithm, args.platform, refresh=args.refresh,
        on_progress=None if args.quiet else _download_reporter(args.algorithm),
    )


#: Width of the download bar. Fixed rather than terminal-relative: the line it sits on is short,
#: and a bar that changes width between runs reads as a different thing happening.
_BAR_WIDTH = 24


def _download_reporter(slug: str):
    """Build the ``on_progress`` callback that draws a bundle download.

    A first run fetches a few hundred KiB before anything else is printed, and on a vessel link
    that is long enough to look like a hang. The bar is the whole fix.

    On a terminal it redraws one line in place. Off one — a log, a CI job, a pipe — redrawing is
    noise, so it announces the download once and reports the total when it lands. Nothing is
    drawn at all when the bundle is already cached: :func:`load_remote` only reports on a real
    transfer.
    """
    tty = sys.stdout.isatty()
    announced = False

    def report(done: int, total: int | None) -> None:
        nonlocal announced
        if not tty:
            if not announced:
                announced = True
                size = f" ({_human_bytes(total)})" if total else ""
                print(f"downloading {slug}{size}…")
            if total and done >= total:
                print(f"downloaded  {slug} — {_human_bytes(done)}")
            return
        if total:
            fraction = done / total
            line = (f"downloading {slug}  {_bar(fraction)} {fraction * 100:3.0f}%  "
                    f"{_human_bytes(done)} / {_human_bytes(total)}")
        else:
            # No Content-Length: report what has arrived and skip the bar rather than animate
            # a fraction we would have to invent.
            line = f"downloading {slug}  {_human_bytes(done)}"
        print(f"\r\033[K{line}", end="", flush=True)
        if total and done >= total:
            print()

    return report


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    """A fixed-width progress bar."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


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
        ["WHEN", "MODEL", "FILES", "FOUND", "OUTPUT"],
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
