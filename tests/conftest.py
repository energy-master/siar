# Vixen Intelligence c.2026
"""What every test in this suite is given before it starts: a machine of its own.

Two directories decide what this package thinks is installed — its own workspace, and
siar-build's next door. Left alone, a test that asks "what can this machine run" answers out of
the *developer's* home directory: the suite then passes on the laptop that has three models in it
and fails on the one that has none, and an import test writes into a real workspace somebody is
using. Both are pointed somewhere temporary here, once, for all of it.

A test that wants to say something about those directories still sets them itself; a
:func:`monkeypatch.setenv` inside a test wins over this one.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path_factory, monkeypatch):
    """Point this package's workspace and siar-build's index at empty temporary directories."""
    monkeypatch.setenv("SIAR_APP_HOME", str(tmp_path_factory.mktemp("siar-app")))
    monkeypatch.setenv("SIAR_BUILD_HOME", str(tmp_path_factory.mktemp("siar-build")))
