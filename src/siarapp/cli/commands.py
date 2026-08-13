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
import threading
import time
import urllib.parse
from argparse import Namespace
from typing import Any

from siarapp import __version__, docs
from siarapp.api import ApiError, AuthError, Client, client_from_credentials
from siarapp.branding import copyright_line
from siarapp.cli.format import (
    BAR_CAP,
    BAR_WIDTH,
    HIDE_CURSOR,
    SHOW_CURSOR,
    bar,
    clip,
    clock,
    cost,
    duration,
    factor_text,
    fit_path,
    share,
)
from siarapp.cli.table import render_table, terminal_width
from siarapp.cli.tui import TuiDisplay
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
from siarapp.io.performance import PHASES, progress_block, realtime_factor
from siarapp.loader import installed_algorithms, load_local, load_remote
from siarapp.runner import RunOptions, run_folder

__all__ = [
    "cmd_algorithms",
    "cmd_feedback",
    "cmd_installed",
    "cmd_license",
    "cmd_login",
    "cmd_logout",
    "cmd_quickstart",
    "cmd_readme",
    "cmd_run",
    "cmd_runs",
    "cmd_signup",
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


def _show(path: str, what: str) -> int:
    """Open a local page in the browser, or say where it is when there is no browser."""
    if docs.open_path(path):
        print(f"Opened the {what} in your browser.")
        print(f"  {path}")
        return 0
    print(f"No browser to open here. The {what} is at:")
    print(f"  {path}")
    print("Copy that path into a browser on any machine — it needs no network.")
    return 0


def cmd_quickstart(_args: Namespace) -> int:
    """Open the illustrated quickstart, offline, in the user's browser."""
    try:
        return _show(docs.quickstart_path(), "quickstart")
    except docs.DocsError as e:
        return _err(str(e))


def cmd_readme(args: Namespace) -> int:
    """Open the full manual in the browser, or print it as Markdown.

    The README ships in the wheel's own metadata rather than as a second copy of the file,
    so this works from an installed tool with no network and no repository in sight.
    """
    try:
        if args.text:
            print(docs.readme_markdown())
            return 0
        return _show(docs.write_readme_page(), "manual")
    except docs.DocsError as e:
        return _err(str(e))


#: What to tell someone whose terminal cannot prompt — a CI job, a pipe, an agent shell.
_NO_TTY_HELP = (
    "cannot prompt for credentials: this shell has no interactive input.\n"
    "  Either run `siar-app login` in a normal terminal, or supply both without a prompt:\n"
    "      siar-app login <username> --server <url>   with $SIAR_APP_PASSWORD set\n"
    "  Or skip logging in entirely by exporting a token you already have:\n"
    "      export SIAR_APP_TOKEN=...  SIAR_APP_URL=..."
)

#: The same, for signup — where the escape hatch is flags rather than an existing token.
_NO_TTY_SIGNUP_HELP = (
    "cannot prompt for account details: this shell has no interactive input.\n"
    "  Either run `siar-app signup` in a normal terminal, or supply every field:\n"
    "      siar-app signup --email you@example.com --username you --display-name 'You'\n"
    "  with $SIAR_APP_PASSWORD set."
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
        message = str(e)
        if e.code == "unauthorized":
            # Only for wrong-credentials. A disabled or unverified account already *has* a
            # signup behind it, and telling its owner to make another would be bad advice.
            message += f"\n  No account on {base_url} yet? Create one: siar-app signup"
        return _err(message)
    except ApiError as e:
        return _err(str(e))

    user = res.get("user") or {}
    username = str(user.get("username") or login_id)
    path = save_credentials(base_url, client.token or "", username)
    print(f"Signed in to {base_url} as {username}.")
    print(f"Token saved to {path} (delete it, or run `siar-app logout`, to sign out).")
    return 0


#: Mirrors the server's own rule (``lib/signup.php``), so a typo costs a re-prompt rather than
#: a round trip. The server still validates — this is courtesy, not the gate.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")


def cmd_signup(args: Namespace) -> int:
    """Create an IDent Dynamics account, then point the user at the verification email.

    Deliberately does not sign in afterwards. The account is created unverified, so a login
    attempt would be refused until the emailed link is used; offering to log in here would
    only produce a confusing failure a few seconds after a success.
    """
    base_url = args.server or load_credentials().get("base_url") or DEFAULT_BASE_URL

    try:
        email = (args.email or input("Email: ")).strip()
        if not email:
            return _err("no email given")
        if "@" not in email:
            return _err(f"{email!r} does not look like an email address")

        username = (args.username or input("Username (3-64 chars, letters digits . _ -): ")).strip()
        if not username:
            return _err("no username given")
        if not _USERNAME_RE.match(username):
            return _err("username must be 3-64 characters: letters, digits, and . _ -")

        display_name = args.display_name
        if display_name is None:
            display_name = input("Display name (optional): ").strip()
    except EOFError:
        return _err(_NO_TTY_SIGNUP_HELP)

    password = legacy_env("PASSWORD")
    if not password:
        try:
            password = getpass.getpass("Password (8+ characters): ")
            confirm = getpass.getpass("Confirm password: ")
        except (EOFError, getpass.GetPassWarning):
            return _err(_NO_TTY_SIGNUP_HELP)
        if password != confirm:
            return _err("the two passwords don't match")
    if len(password) < 8:
        return _err("password must be at least 8 characters")

    client = Client(base_url)
    try:
        res = client.register(email, username, password, display_name=display_name or "")
    except ApiError as e:
        # AuthError too, which register.php returns for "signup is closed on this install" —
        # not a case where telling the user to log in would help, so it is not special-cased.
        return _err(str(e))

    user = res.get("user") or {}
    print(f"Account created on {base_url} as {user.get('username') or username}.")
    if res.get("verification_sent"):
        print(f"We've emailed a verification link to {user.get('email') or email}.")
        print("Click it, then run `siar-app login`.")
    else:
        print("The verification email could not be sent. Request another link from")
        print(f"  {base_url}/resend_verification.php?email={urllib.parse.quote(email)}")
        print("then run `siar-app login`.")
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
        workers=args.parallel,
    )

    print(f"algorithm  {handle.slug} ({handle.platform})")
    print(f"scanning   {source}")
    print(f"output     {out_root}")

    started = time.time()
    reporter = _reporter(args, handle.slug, source, out_root)
    failure = ""
    try:
        manifest = run_folder(
            handle, source, out_root, options,
            progress=None if args.quiet else reporter.on_result,
            on_start=None if args.quiet else reporter.on_start,
            on_corpus=None if args.quiet else reporter.on_corpus,
            on_idle=None if args.quiet else reporter.on_idle,
            # A display that owns the screen would draw over a warning printed from under it, so it
            # is offered the warning instead and decides where it goes.
            warn=getattr(reporter, "note", None) or _warn,
        )
        # A panel that wiped itself the moment the run ended would take the answer with it. The one
        # display that owns the screen keeps it, with the closing metrics in it, until dismissed.
        hold = getattr(reporter, "hold", None)
        if hold is not None:
            hold(_metric_rows(manifest))
    except (FileNotFoundError, ValueError, ScannerError) as e:
        # Held rather than printed here: an `except` body runs *before* the `finally`, so a
        # message printed from one lands in the middle of a live display that is about to be
        # wiped off the screen — taking the only explanation of the failure with it.
        failure = str(e)
    finally:
        # Before anything else prints — a summary, an error, a traceback — the display has to
        # stop owning the bottom of the screen, or it redraws over whatever comes next.
        reporter.close()

    if failure:
        return _err(failure)

    _print_summary(manifest, out_root)
    record_run({
        "at": manifest["started_at"],
        "algorithm": handle.slug,
        "source": source,
        "out": out_root,
        "files": manifest["files"],
        "structures": manifest["structures"],
        "elapsed_sec": round(time.time() - started, 2),
        # Recorded because it is the difference between two otherwise identical rows: the same
        # corpus and the same algorithm, four hours apart or forty minutes.
        "workers": int(manifest.get("workers", 1)),
    })
    return 1 if manifest["by_status"].get("error") else 0


def _warn(message: str) -> None:
    """A run-wide warning, on stderr so a piped stdout stays machine-readable."""
    print(f"warning: {message}", file=sys.stderr)


def _reporter(args: Namespace, slug: str, source: str = "", out_root: str = ""):
    """Choose the live display for this run.

    Three of them, because they draw genuinely different runs and neither can draw another's: one
    recording at a time is a line, several at a time is a row per worker, and ``--tui`` is the whole
    run in one panel. The choice is made here rather than by a flag inside one class that would
    spend half its code deciding which of the three it was.

    ``--tui`` needs a terminal: it takes over the screen and redraws in place. Off one — a log, a
    pipe, a CI job — it would emit a frame every quarter second, so it falls back to the display
    that degrades to one line per recording and says so once.

    Args:
        args: The parsed command line.
        slug: The algorithm being run, for the panel's title.
        source: Folder being scanned, and
        out_root: folder being written. Only ``--tui`` uses them, because it is the only display
            that hides the lines which announced them.
    """
    if args.tui:
        if sys.stdout.isatty():
            return TuiDisplay(slug, workers=args.parallel if args.parallel > 0 else 1,
                              source=source, out=out_root)
        _warn("--tui needs a terminal; reporting one line per recording instead")
    return WorkerPanel() if args.parallel != 1 else ScanReporter()


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
            line = (f"downloading {slug}  {bar(fraction)} {fraction * 100:3.0f}%  "
                    f"{_human_bytes(done)} / {_human_bytes(total)}")
        else:
            # No Content-Length: report what has arrived and skip the bar rather than animate
            # a fraction we would have to invent.
            line = f"downloading {slug}  {_human_bytes(done)}"
        print(f"\r\033[K{line}", end="", flush=True)
        if total and done >= total:
            print()

    return report


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


#: How often the live line redraws. Fast enough to read as a clock rather than a slideshow,
#: slow enough that a ten-hour survey does not spend its life writing to a terminal.
_TICK_SEC = 0.2


class ScanReporter:
    """The live per-recording line, and the estimate behind its bar.

    Nearly all of a recording's time — measured at 94% on a five-minute file — is spent inside
    the algorithm's ``scan``, which is one opaque call into an obfuscated bundle with nowhere to
    report from. So there is no true completion fraction to draw, and the two honest things to
    show are the ones here: elapsed time, which is a fact, and an estimate clearly presented as
    one.

    The estimate is throughput learned from this run: seconds of wall time per second of audio,
    averaged over the recordings already scanned, applied to the current recording's duration
    from the header probe. It needs no knowledge of the algorithm and costs nothing, and it is
    wrong in exactly the ways an average is — an unusually dense recording sits at the cap for a
    while. Nothing is drawn before the first recording finishes, because until then there is no
    throughput to speak of and a bar would be invention rather than estimate.

    Off a terminal — a log, a pipe, a CI job — no live line is drawn at all. Redrawing in place
    depends on the cursor going back, and without that every tick becomes another line.
    """

    def __init__(self) -> None:
        self._tty = sys.stdout.isatty()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Learned throughput: wall seconds spent per second of audio.
        self._audio_sec = 0.0
        self._wall_sec = 0.0
        self._current: tuple[int, int, str, float] | None = None
        self._started_at = 0.0

    # -- the callbacks run_folder wants ----------------------------------------------------

    def on_corpus(self, files: int, audio_sec: float, workers: int) -> None:
        """Nothing to do: this reporter's estimate is learned from the run, not from the corpus."""

    def on_idle(self, lane: int) -> None:
        """Nothing to do: a serial run has one lane and it is idle only when the run is over."""

    def on_start(self, index: int, total: int, rel_path: str, duration_sec: float,
                 lane: int = 0) -> None:
        """Begin reporting on a recording."""
        if not self._tty:
            return
        with self._lock:
            self._current = (index, total, rel_path, duration_sec)
            self._started_at = time.time()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def on_result(self, done: int, total: int, result) -> None:
        """Finish a recording: learn from it, then print its permanent line."""
        if result.status == "scanned" and result.duration_sec and result.elapsed_sec:
            self._audio_sec += float(result.duration_sec)
            self._wall_sec += float(result.elapsed_sec)

        tail = {
            "scanned": f"{result.count} structure{'' if result.count == 1 else 's'}",
            "skipped": "skipped (already done)",
            "too_short": "too short to scan",
            "error": f"ERROR — {result.error}",
        }.get(result.status, result.status)
        line = f"[{done}/{total}] {result.rel_path}: {tail}"

        with self._lock:
            self._current = None
            # Errors stay on screen; everything else may be overwritten by the next recording.
            if result.status == "error" or not self._tty:
                if self._tty:
                    print("\r\033[K", end="")
                print(line, flush=True)
            else:
                print(f"\r\033[K{line}", end="", flush=True)
                if done == total:
                    print()

    def close(self) -> None:
        """Stop the ticker. Safe to call when it never started."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # -- the live line ---------------------------------------------------------------------

    def _rate(self) -> float | None:
        """Wall seconds per second of audio, or ``None`` before anything has been scanned."""
        if self._audio_sec <= 0 or self._wall_sec <= 0:
            return None
        return self._wall_sec / self._audio_sec

    def _tick(self) -> None:
        while not self._stop.wait(_TICK_SEC):
            with self._lock:
                current = self._current
                if current is None:
                    continue
                elapsed = time.time() - self._started_at
                print(f"\r\033[K{self._render(current, elapsed)}", end="", flush=True)

    def _render(self, current: tuple[int, int, str, float], elapsed: float) -> str:
        index, total, rel_path, seconds = current
        head = f"[{index}/{total}] {rel_path}"
        rate = self._rate()
        if rate is None or seconds <= 0:
            # First recording of the run, or a header that would not read: elapsed alone. It
            # says the run is alive, which is the whole job until there is something to predict.
            return f"{head}  scanning… {elapsed:.0f}s"

        estimate = seconds * rate
        pct = min(BAR_CAP, 100.0 * elapsed / estimate) if estimate > 0 else 0.0
        filled = int(round(BAR_WIDTH * pct / 100.0))
        drawn = "█" * filled + "░" * (BAR_WIDTH - filled)
        return f"{head}  {drawn} {pct:2.0f}%  {elapsed:.0f}s of ~{estimate:.0f}s"


#: Lane bar width in the worker panel. Narrower than the download bar because there is one per
#: core and the recording's name is what the reader is actually looking for.
_LANE_BAR = 14


class WorkerPanel:
    """The live panel for a parallel run: one row per worker, redrawn in place.

    A serial run has one thing happening and reports it on one line. A run on sixteen cores has
    sixteen, each on a different recording of a different length, and interleaving their lines
    produces a screen nobody can read: the interesting question — *is anything stuck* — is
    answered by seeing all sixteen at once, not by watching them take turns.

    So it draws what ``htop`` draws. A header of run-wide totals, then one fixed row per worker
    holding the recording that worker is on, how long it has been on it, and the same
    learned-throughput estimate the serial reporter draws — and rows keep their position, so a
    lane that stops moving is obvious at a glance rather than being something you have to notice
    in a scrolling log.

    The rows are worker *lanes*, not process IDs. Exactly as many recordings are in flight as
    there are workers, so a lane is a real worker's worth of work; which operating-system process
    picks it up is the pool's business and changes nothing the reader can act on.

    Completions do not print. On a corpus that takes three weeks of audio to get through, one
    line per recording is a hundred thousand lines of scrollback that say "done" — the counters
    say it better. Errors do print, above the panel, because those are the lines somebody has to
    come back and read.

    Off a terminal — a log, a pipe, a CI job — the panel cannot redraw in place, so it degrades
    to one line per recording with the lane number on it, which is what a log wants anyway.
    """

    def __init__(self) -> None:
        self._tty = sys.stdout.isatty()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._workers = 1
        self._lanes: list[tuple | None] = []
        self._started = 0.0
        self._files_total = 0
        self._audio_total = 0.0
        self._files_done = 0
        self._files_worked = 0
        self._audio_done = 0.0
        self._audio_worked = 0.0
        # Durations from the headers, held from on_start to on_result: a skipped or unreadable
        # recording reports no duration of its own, and the progress bar still has to move past it.
        self._planned: dict[str, float] = {}
        # Learned per-file cost — wall seconds per second of audio *inside one worker* — which is
        # what a lane's own bar is drawn from. Distinct from the run's throughput, which is that
        # divided by the number of workers, and drawn in the header.
        self._scan_audio = 0.0
        self._scan_wall = 0.0
        self._structures = 0
        self._drawn = 0
        self._hidden = False

    # -- the callbacks run_folder wants ----------------------------------------------------

    def on_corpus(self, files: int, audio_sec: float, workers: int) -> None:
        """Learn the size of the run, and start drawing."""
        with self._lock:
            self._workers = max(1, workers)
            self._lanes = [None] * self._workers
            self._files_total = files
            self._audio_total = audio_sec
            self._started = time.time()
        if not self._tty:
            print(f"workers    {self._workers}")
            print(f"corpus     {files} recording(s), {duration(audio_sec)} of audio", flush=True)
            return
        sys.stdout.write(HIDE_CURSOR)
        self._hidden = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def on_start(self, index: int, total: int, rel_path: str, duration_sec: float,
                 lane: int = 0) -> None:
        """A worker has taken a recording."""
        with self._lock:
            self._planned[rel_path] = duration_sec
            self._ensure_lanes(lane)
            self._lanes[lane] = (index, rel_path, duration_sec, time.time())

    def on_idle(self, lane: int) -> None:
        """A worker has run out of work — the tail of the run."""
        with self._lock:
            self._ensure_lanes(lane)
            self._lanes[lane] = None

    def on_result(self, done: int, total: int, result) -> None:
        """A recording is finished: fold it into the totals."""
        with self._lock:
            seconds = self._planned.pop(result.rel_path, result.duration_sec)
            self._files_done = done
            self._files_total = max(self._files_total, total)
            self._audio_done += seconds
            if result.status != "skipped":
                self._files_worked += 1
                self._audio_worked += seconds
            if result.status == "scanned" and result.duration_sec and result.elapsed_sec:
                self._scan_audio += float(result.duration_sec)
                self._scan_wall += float(result.elapsed_sec)
            self._structures += result.count

            if result.status == "error":
                self._say(f"[{done}/{total}] w{result.lane + 1} {result.rel_path}: "
                          f"ERROR — {result.error}")
            elif not self._tty:
                tail = {
                    "scanned": f"{result.count} structure{'' if result.count == 1 else 's'}",
                    "skipped": "skipped (already done)",
                    "too_short": "too short to scan",
                }.get(result.status, result.status)
                self._say(f"[{done}/{total}] w{result.lane + 1} {result.rel_path}: {tail}")

    def close(self) -> None:
        """Stop drawing and give the terminal back. Safe to call when nothing ever started."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._hidden:
            with self._lock:
                self._erase()
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
            self._hidden = False

    # -- drawing ---------------------------------------------------------------------------

    def _ensure_lanes(self, lane: int) -> None:
        """Grow the lane list if a lane arrives before (or without) ``on_corpus``."""
        if lane >= len(self._lanes):
            self._lanes.extend([None] * (lane + 1 - len(self._lanes)))
            self._workers = len(self._lanes)

    def _say(self, line: str) -> None:
        """Print a line that stays on screen, above the panel. Caller holds the lock."""
        self._erase()
        print(line, flush=True)

    def _erase(self) -> None:
        """Take the panel off the screen. Caller holds the lock."""
        if self._drawn:
            sys.stdout.write(f"\033[{self._drawn}A\033[J")
            self._drawn = 0

    def _tick(self) -> None:
        while not self._stop.wait(_TICK_SEC):
            with self._lock:
                if not self._lanes:
                    continue
                self._draw(self._render(time.time()))

    def _draw(self, lines: list[str]) -> None:
        """Redraw the panel in place. Caller holds the lock."""
        out = [f"\033[{self._drawn}A"] if self._drawn else []
        out += [f"\r\033[K{line}\n" for line in lines]
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._drawn = len(lines)

    def _cost_rate(self) -> float:
        """Wall seconds one worker spends per second of audio, or ``0.0`` before it is known."""
        if self._scan_audio <= 0 or self._scan_wall <= 0:
            return 0.0
        return self._scan_wall / self._scan_audio

    def _render(self, now: float) -> list[str]:
        """The whole panel: a header of run totals, then one row per worker."""
        width = terminal_width()
        block = progress_block(
            started=self._started,
            files_total=self._files_total,
            files_done=self._files_done,
            files_worked=self._files_worked,
            audio_total_sec=self._audio_total,
            audio_done_sec=self._audio_done,
            audio_worked_sec=self._audio_worked,
            now=now,
        )
        elapsed = max(0.0, now - self._started)
        factor = self._audio_worked / elapsed if elapsed > 0 else 0.0
        eta = block["eta_sec"]
        # Two lines rather than one: everything below is one row per core, and on a sixteen-core
        # run the header is the only thing competing for width with the file names.
        lines = [
            clip(f"{bar(block['fraction'])} {block['fraction'] * 100:3.0f}%  "
                  f"{self._files_done}/{self._files_total} files  "
                  f"{duration(self._audio_done)} of {duration(self._audio_total)} audio", width),
            clip(f"{self._workers} worker{'' if self._workers == 1 else 's'}  ·  "
                  f"{factor:.1f}x realtime  ·  {clock(elapsed)} elapsed  ·  "
                  f"{'estimating' if eta is None else clock(eta) + ' left'}  ·  "
                  f"{self._structures} structure{'' if self._structures == 1 else 's'}", width),
        ]
        for lane, entry in enumerate(self._lanes):
            lines.append(clip(self._lane_line(lane, entry, now, width), width))
        return lines

    def _lane_line(self, lane: int, entry: tuple | None, now: float, width: int) -> str:
        """One worker's row."""
        label = f" {lane + 1:>2}  "
        if entry is None:
            return f"{label}{'·' * _LANE_BAR}    —      idle"
        _index, rel_path, seconds, started_at = entry
        elapsed = now - started_at
        rate = self._cost_rate()
        estimate = seconds * rate
        if estimate > 0:
            pct = min(BAR_CAP, 100.0 * elapsed / estimate)
            filled = int(round(_LANE_BAR * pct / 100.0))
            drawn = "█" * filled + "░" * (_LANE_BAR - filled)
            stat = f"{drawn} {pct:3.0f}%  {elapsed:4.0f}s"
        else:
            # First round of the run: no throughput learned yet, so elapsed alone. Same rule as
            # the serial line — an estimate drawn before anything has finished is invention.
            stat = f"{'░' * _LANE_BAR}   —   {elapsed:4.0f}s"
        return f"{label}{stat}  {fit_path(rel_path, width - len(label + stat) - 3)}"




def _metric_rows(manifest: dict) -> list[list[str]]:
    """The run's metrics, as table rows: how much was scanned, where the time went, what came out.

    Args:
        manifest: The finished run manifest from :func:`siarapp.runner.run_folder`.

    Returns:
        ``[label, value, share]`` triples in reading order. ``share`` is filled only on the phase
        rows, where a percentage of the total above them is the whole point; every other row leaves
        it empty rather than inventing a denominator.

        Outcome rows other than ``scanned`` appear only when they happened, so a clean run's table
        is not padded with three zeros.
    """
    by_status = manifest["by_status"]
    elapsed = float(manifest["elapsed_sec"])
    workers = int(manifest.get("workers", 1))
    # Audio actually scanned, not the corpus total: a resumed run must not take credit for hours
    # it skipped, and this is the same rule the performance JSON's realtime_factor follows.
    audio_sec = sum(
        float(row.get("duration_sec") or 0.0)
        for row in manifest["manifest"]
        if row.get("status") == "scanned"
    )

    rows = [["recordings", f"{manifest['files']:,}", ""]]
    for status, label in (
        ("scanned", "scanned"),
        ("skipped", "skipped (resume)"),
        ("too_short", "too short"),
        ("error", "errors"),
    ):
        if by_status.get(status):
            rows.append([label, f"{by_status[status]:,}", ""])

    rows.append(["audio scanned", duration(audio_sec), ""])
    # Same unit as the audio row above, deliberately: the two numbers are only worth putting in
    # one table if they can be compared by eye, and `_clock` would print a twelve-second trial as
    # "0:12" against "12.0 s" of audio.
    rows.append(["wall time", duration(elapsed), ""])
    rows.extend(_phase_rows(manifest.get("phases") or {}, elapsed, workers))
    # The number anyone planning a survey reads first: 3.0 means the scan ran three times faster
    # than the audio it scanned, so an hour of recording cost twenty minutes.
    rows.append(["realtime", factor_text(realtime_factor(audio_sec, elapsed)), ""])
    rows.append(["workers", str(workers), ""])
    if workers > 1:
        # Per worker, because the pooled figure above is the one that flatters the machine and
        # this is the one that says whether adding cores is still buying anything.
        rows.append(["realtime per worker",
                     factor_text(realtime_factor(audio_sec, elapsed * workers)), ""])
    rows.append(["structures", f"{manifest['structures']:,}", ""])
    return rows


def _phase_rows(phases: dict, elapsed: float, workers: int) -> list[list[str]]:
    """The wall-time breakdown: one indented row per stage, with its share of the total.

    Args:
        phases: ``{phase: seconds}`` from :func:`siarapp.io.performance.phase_totals`.
        elapsed: The run's wall time.
        workers: Recordings in flight at once.

    Returns:
        Rows to sit under ``wall time``, or an empty list when nothing was measured — a fully
        skipped run has no stages to account for.

    A serial run's stages add up to its wall time, so the remainder gets a row of its own:
    ``overhead`` is the header probe, the manifest rewrites and the pool start, and naming it is
    what stops a reader assuming the stages are the whole story. A ``--parallel`` run's stages were
    spent by several workers at once and add up to *more* than the wall clock, so they are totalled
    under their own heading instead and shared out against that — subtracting them from wall time
    there would print a negative overhead, which is worse than printing none.
    """
    measured = sum(float(v) for v in phases.values())
    if measured <= 0:
        return []

    rows: list[list[str]] = []
    if workers > 1:
        total = measured
        rows.append(["worker time", duration(measured), ""])
    else:
        total = max(elapsed, measured)

    for name in PHASES:
        if name in phases:
            rows.append([f"  {name}", cost(float(phases[name])), share(phases[name], total)])
    if workers == 1 and elapsed > measured:
        rows.append(["  overhead", cost(elapsed - measured), share(elapsed - measured, total)])
    return rows






def _print_summary(manifest: dict, out_root: str) -> None:
    """The closing report: what the run cost, what was found, and what to do with it."""
    by_status = manifest["by_status"]
    scanned = by_status.get("scanned", 0)
    print()

    # A fully-resumed run scans nothing, and reporting "0.0 s of audio / no structures found"
    # for it reads like a failure rather than like "there was nothing left to do".
    if scanned == 0 and by_status.get("skipped"):
        print(f"Nothing to do — all {by_status['skipped']} file(s) were already scanned.")
        print("Drop --resume, or delete the output folder, to scan them again.")
    else:
        print(render_table(["METRIC", "VALUE", "SHARE"], _metric_rows(manifest),
                           align=["<", ">", ">"]))
        print()
        if manifest["shapes"]:
            print(render_table(
                ["STRUCTURE", "FOUND"],
                [[shape, f"{n:,}"] for shape, n in manifest["shapes"].items()],
                align=["<", ">"],
            ))
        else:
            print("No structures found.")
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
    print(f"{duration(total_sec)} of audio")
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
