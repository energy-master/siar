# Vixen Intelligence c.2026
"""``siar-app serve``'s own decisions, before any socket is involved.

Three of them are worth a test. Which folder gets served when none was named — because the answer
comes from run history and a wrong one would serve somebody else's survey. That a non-loopback bind
is refused rather than warned about. And that the flags mean what the help says, since ``--open``
defaulting the wrong way would open a browser on a headless box.
"""
from __future__ import annotations

import json
import os

import pytest

from siarapp.cli import commands
from siarapp.cli.main import build_parser
from siarapp.grid import ScannerError
from siarapp.io.output import RUN_MANIFEST_NAME


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A private ~/.siar-app, so run history can be written without touching the real one."""
    monkeypatch.setenv("SIAR_APP_HOME", str(tmp_path / "home"))
    return tmp_path


def _output_folder(root, name="scan"):
    """A folder that looks enough like a finished run for `serve` to accept it."""
    out = root / name
    out.mkdir(parents=True)
    (out / RUN_MANIFEST_NAME).write_text(json.dumps({
        "format": "siar-app-run-v1", "files": 0, "by_status": {}, "structures": 0,
        "shapes": {}, "audio_sec": 0.0, "manifest": [],
    }))
    return out


def test_a_named_folder_is_the_one_served(home):
    out = _output_folder(home)
    root, chosen = commands._serve_root(str(out))
    assert root == str(out)
    assert chosen == "argument"


def test_a_named_folder_is_taken_as_given_even_before_it_has_a_manifest(home):
    """Serving a run that has only just started is the point, so an empty folder is not refused
    here — the daemon reports `no_manifest` and the page says so."""
    empty = home / "fresh"
    empty.mkdir()
    root, chosen = commands._serve_root(str(empty))
    assert root == str(empty)
    assert chosen == "argument"


def test_with_no_folder_the_most_recent_run_is_served(home):
    from siarapp.config import record_run

    old = _output_folder(home, "old-scan")
    new = _output_folder(home, "new-scan")
    record_run({"at": "2026-08-01T00:00:00Z", "out": str(old), "algorithm": "a"})
    record_run({"at": "2026-08-13T00:00:00Z", "out": str(new), "algorithm": "a"})

    root, chosen = commands._serve_root(None)
    assert root == str(new), "history is newest first, and that is the one to serve"
    assert chosen == "history"


def test_a_recorded_run_that_has_been_deleted_is_skipped(home):
    from siarapp.config import record_run

    kept = _output_folder(home, "kept")
    record_run({"at": "2026-08-01T00:00:00Z", "out": str(kept), "algorithm": "a"})
    record_run({"at": "2026-08-13T00:00:00Z", "out": str(home / "gone"), "algorithm": "a"})

    root, _chosen = commands._serve_root(None)
    assert root == str(kept)


def test_with_nothing_to_serve_the_message_names_the_command_that_helps(home):
    with pytest.raises(ScannerError, match="siar-app runs"):
        commands._serve_root(None)


# -- the flags -----------------------------------------------------------------------------


def test_the_defaults_are_the_tunnel_shaped_ones():
    args = build_parser().parse_args(["serve"])
    assert args.folder is None
    assert args.port == 8420
    assert args.bind == "127.0.0.1"
    assert args.allow_remote is False
    # Not opening is right on a headless box, which is the machine this command is for.
    assert args.open is False
    assert args.no_audio is False
    assert args.verbose is False
    assert args.token is None
    assert args.allow_origin is None


def test_the_flags_parse(tmp_path):
    args = build_parser().parse_args([
        "serve", str(tmp_path), "--port", "9000", "--bind", "0.0.0.0", "--allow-remote",
        "--token", "abc", "--open", "--no-audio", "--verbose",
        "--allow-origin", "https://goident.ai", "--allow-origin", "https://www.goident.ai",
    ])
    assert (args.folder, args.port, args.bind) == (str(tmp_path), 9000, "0.0.0.0")
    assert args.allow_remote and args.open and args.no_audio and args.verbose
    assert args.token == "abc"
    assert args.allow_origin == ["https://goident.ai", "https://www.goident.ai"]


def test_a_remote_bind_is_refused_rather_than_warned_about(home, capsys):
    """A warning on stderr is exactly the thing that scrolls past, and a bearer token over plain
    HTTP on a survey LAN is a different proposition from the loopback default."""
    out = _output_folder(home)
    args = build_parser().parse_args(["serve", str(out), "--bind", "0.0.0.0", "--port", "0"])
    assert commands.cmd_serve(args) == 1
    error = capsys.readouterr().err
    assert "--allow-remote" in error
    assert "ssh -L" in error


def test_a_folder_that_is_not_one_is_a_message_not_a_traceback(home, capsys):
    args = build_parser().parse_args(["serve", str(home / "nope"), "--port", "0"])
    assert commands.cmd_serve(args) == 1
    assert "not a folder" in capsys.readouterr().err


def test_serve_is_gated_by_the_licence_like_every_other_working_command():
    from siarapp.cli.main import _UNGATED

    assert "serve" not in _UNGATED


def test_the_command_is_dispatched():
    from siarapp.cli.main import _DISPATCH

    assert _DISPATCH["serve"] is commands.cmd_serve


def test_the_parsers_default_port_matches_the_daemons():
    """The parser spells the number out so that building it does not import `http.server`; this is
    the test that stops the two copies drifting."""
    from siarapp.cli.main import SERVE_PORT
    from siarapp.serve.http import DEFAULT_PORT

    assert SERVE_PORT == DEFAULT_PORT


def test_building_the_parser_does_not_import_the_server():
    """Every `siar-app version` builds the parser. None of them should pay for `http.server`."""
    import subprocess
    import sys

    code = (
        "import sys; from siarapp.cli.main import build_parser; build_parser();"
        "print('http.server' in sys.modules or 'socketserver' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_page_it_serves_is_packaged():
    """`serve` is pointless without the page, so the wheel has to carry it."""
    from siarapp import docs

    for name in ("viewer.html", "viewer.css", "viewer.js"):
        assert os.path.isfile(docs.local_web_path(name))
