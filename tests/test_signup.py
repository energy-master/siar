# Vixen Intelligence c.2026
"""Account creation from the command line — the one command that runs before there is a token.

Two things are worth pinning. The first is the request body: ``register.php`` is a public
endpoint that must receive no ``Authorization`` header, because a stale token in the config
would otherwise decide whether a stranger can sign up. The second is that the command refuses
locally what the server would refuse anyway — a short password, a malformed username — since
the cost of a round trip here is a real account half-created in someone's head.

The prompting is tested through ``builtins.input``/``getpass`` rather than a pty: what matters
is the order the fields are asked in and that the confirmation actually has to match, neither
of which needs a terminal to observe.
"""
from __future__ import annotations

import getpass
import json

import pytest

from siarapp.api import ApiError, Client
from siarapp.cli import commands
from siarapp.cli.main import build_parser


def _args(**over):
    """A ``signup`` namespace with everything supplied, so nothing prompts unless asked."""
    args = build_parser().parse_args(["signup"])
    args.server = "https://example.test"
    args.email = "new@example.test"
    args.username = "newcomer"
    args.display_name = "A Newcomer"
    for key, value in over.items():
        setattr(args, key, value)
    return args


class _FakeClient:
    """Stands in for :class:`Client`, recording the one call the command should make.

    What ``register`` does next is set on the class before the command runs — ``result`` for a
    server that accepted, ``raises`` for one that did not.
    """

    instances: list["_FakeClient"] = []
    result: dict = {}
    raises: Exception | None = None

    def __init__(self, base_url):
        self.base_url = base_url
        self.calls = []
        _FakeClient.instances.append(self)

    def register(self, email, username, password, *, display_name=""):
        self.calls.append((email, username, password, display_name))
        if _FakeClient.raises is not None:
            raise _FakeClient.raises
        return _FakeClient.result


@pytest.fixture
def fake_client(monkeypatch):
    """Swap the real client out and hand back the class, so a test can set its answer."""
    _FakeClient.instances.clear()
    _FakeClient.raises = None
    _FakeClient.result = {
        "ok": True,
        "verification_sent": True,
        "user": {"id": 42, "username": "newcomer", "email": "new@example.test"},
    }
    monkeypatch.setattr(commands, "Client", _FakeClient)
    monkeypatch.setattr(commands, "load_credentials", lambda: {})
    monkeypatch.setenv("SIAR_APP_PASSWORD", "correct-horse")
    return _FakeClient


# -- the happy path -------------------------------------------------------------------------


def test_signup_posts_the_fields_and_reports_the_email(fake_client, capsys):
    assert commands.cmd_signup(_args()) == 0

    client = fake_client.instances[-1]
    assert client.base_url == "https://example.test"
    assert client.calls == [("new@example.test", "newcomer", "correct-horse", "A Newcomer")]

    out = capsys.readouterr().out
    assert "Account created on https://example.test as newcomer." in out
    assert "new@example.test" in out
    assert "siar-app login" in out


def test_signup_does_not_log_in(fake_client, monkeypatch, capsys):
    """A new account is unverified, so saving credentials here would save an unusable state."""
    saved = []
    monkeypatch.setattr(commands, "save_credentials",
                        lambda *a, **k: saved.append(a) or "/dev/null")

    assert commands.cmd_signup(_args()) == 0
    assert saved == []


def test_unsent_verification_mail_is_reported_not_failed(fake_client, capsys):
    """The account exists; the user needs the resend link, not an error."""
    fake_client.result = {"ok": True, "verification_sent": False,
                          "user": {"username": "newcomer", "email": "new@example.test"}}

    assert commands.cmd_signup(_args()) == 0

    out = capsys.readouterr().out
    assert "could not be sent" in out
    assert "resend_verification.php?email=new%40example.test" in out


# -- what the command refuses before the network ---------------------------------------------


@pytest.mark.parametrize("field,value,message", [
    ("email", "not-an-address", "does not look like an email"),
    ("username", "ab", "3-64 characters"),
    ("username", "has spaces", "3-64 characters"),
    ("username", "no/slashes", "3-64 characters"),
])
def test_bad_fields_never_reach_the_server(fake_client, capsys, field, value, message):
    assert commands.cmd_signup(_args(**{field: value})) == 1
    assert fake_client.instances == []
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("field,message", [
    ("email", "no email given"),
    ("username", "no username given"),
])
def test_empty_answer_at_a_required_prompt_stops(fake_client, monkeypatch, capsys,
                                                 field, message):
    """A required field left blank is refused, rather than posted as an empty string.

    An empty ``--email ''`` prompts instead of failing — it reads as "ask me", the way a
    missing flag does. Only the answer to the prompt can be finally empty.
    """
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    assert commands.cmd_signup(_args(**{field: None})) == 1
    assert fake_client.instances == []
    assert message in capsys.readouterr().err


def test_short_password_is_refused_locally(fake_client, monkeypatch, capsys):
    monkeypatch.setenv("SIAR_APP_PASSWORD", "short")
    assert commands.cmd_signup(_args()) == 1
    assert fake_client.instances == []
    assert "at least 8 characters" in capsys.readouterr().err


def test_mismatched_confirmation_is_refused(fake_client, monkeypatch, capsys):
    monkeypatch.delenv("SIAR_APP_PASSWORD", raising=False)
    typed = iter(["correct-horse", "correct-hose"])
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(typed))

    assert commands.cmd_signup(_args()) == 1
    assert fake_client.instances == []
    assert "don't match" in capsys.readouterr().err


def test_prompts_in_field_order_when_nothing_is_given(fake_client, monkeypatch, capsys):
    prompts = []

    def _input(prompt=""):
        prompts.append(prompt)
        return ["new@example.test", "newcomer", "A Newcomer"][len(prompts) - 1]

    monkeypatch.setattr("builtins.input", _input)
    args = _args(email=None, username=None, display_name=None)

    assert commands.cmd_signup(args) == 0
    assert [p.split()[0] for p in prompts] == ["Email:", "Username", "Display"]
    assert fake_client.instances[-1].calls[0][3] == "A Newcomer"


def test_no_tty_says_which_flags_to_pass(fake_client, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    assert commands.cmd_signup(_args(email=None)) == 1
    err = capsys.readouterr().err
    assert "no interactive input" in err
    assert "--email" in err


# -- what the server refuses ------------------------------------------------------------------


def test_server_refusal_is_printed_verbatim(fake_client, capsys):
    """The server writes the sentence; the CLI must not paraphrase it into something vaguer."""
    fake_client.raises = ApiError(
        "POST /api/idapi/register.php failed (409): An account with that email already "
        "exists. Try signing in, or reset your password.",
        status=409, code="email_taken",
    )

    assert commands.cmd_signup(_args()) == 1
    assert "already exists" in capsys.readouterr().err


# -- the request on the wire ------------------------------------------------------------------


def test_register_sends_no_authorization_header(monkeypatch):
    """A saved token must not decide whether a stranger may create an account."""
    seen = {}

    def _request(self, path, *, method="GET", body=None, raw=False, on_progress=None):
        seen["path"] = path
        seen["method"] = method
        seen["body"] = body
        seen["token"] = self.token
        return {"ok": True, "verification_sent": True, "user": {}}

    monkeypatch.setattr(Client, "_request", _request)

    Client("https://example.test").register(
        "new@example.test", "newcomer", "correct-horse", display_name="A Newcomer"
    )

    assert seen["path"] == "/api/idapi/register.php"
    assert seen["method"] == "POST"
    assert seen["token"] is None
    assert seen["body"] == {
        "email": "new@example.test",
        "username": "newcomer",
        "password": "correct-horse",
        "display_name": "A Newcomer",
    }
    # The body is what urllib will encode; a non-serialisable value would fail at request time.
    assert json.loads(json.dumps(seen["body"]))["username"] == "newcomer"
