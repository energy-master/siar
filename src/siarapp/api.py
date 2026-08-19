# Vixen Intelligence c.2026
"""Talking to an IDent Dynamics install.

All under ``/api/idapi/``, and all bearer-token rather than cookie-session — the two public
ones because a new customer has no token yet, the rest because a CLI has no browser:

* ``register.php`` (public) — create an account. The headless half of the app's signup form,
  sharing its validation and its per-IP cap. Hands back no token: the account is unverified
  until the emailed link is used.
* ``login.php`` (public) — username/email + password in, token out. Delegates to the app's own
  ``login()``, so lockout, disabled accounts and unverified email all still apply.
* ``scanner_list.php`` — the published algorithm catalogue: slug, description, shapes, params,
  and which platform builds exist. Any active user sees everything the super has published.
* ``scanner_get.php`` — one build's bundle, as a zip.
* ``soc_model_list.php`` — the societies this *account* may run: what it published from its own
  build box, everything on the installation if it is a super, and what a super has released and
  granted to it otherwise. A different question from ``scanner_list.php``'s, which is about the
  installation rather than the account, and so a different endpoint rather than a flag.
* ``soc_model_get.php`` — one society's package, as a zip rooted at the importable directory.
* ``scanner_feedback.php`` — post a rating for a build, or read back this account's own.

Stdlib ``urllib`` rather than ``requests``, for the reason the PNG encoder is stdlib too: this
package's dependency list is part of what it is. That costs a little ceremony around errors,
handled once in :meth:`Client._request`, and buys a customer install that is numpy, soundfile
and nothing else.

Errors arrive as JSON with an ``error`` key (see the endpoints' docstrings), so an HTTP failure
carries a sentence worth printing rather than "HTTP 403".
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from siarapp.config import DEFAULT_BASE_URL, default_platform_tag, load_credentials

__all__ = ["ApiError", "AuthError", "Client", "client_from_credentials"]

#: Seconds before a request gives up. Generous: a bundle download crosses the public internet
#: and the box may be rendering thumbnails for someone else at the time.
_TIMEOUT = 120

#: Read size for a download being reported on. Small enough that the bar moves on a slow link,
#: large enough that the callback is not the expensive part of the transfer.
_CHUNK = 64 * 1024


class ApiError(RuntimeError):
    """A request to the IDent Dynamics API failed.

    Attributes:
        status: The HTTP status, or 0 for a transport-level failure.
        code: The API's short error key (``"forbidden"``, ``"not_found"``, ...) when it sent one.
    """

    def __init__(self, message: str, *, status: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class AuthError(ApiError):
    """The token is missing, expired, or not accepted — the user needs to log in again."""


class Client:
    """A thin client over the endpoints the CLI needs.

    Args:
        base_url: The install, e.g. ``"https://goident.ai"``. A trailing slash is fine.
        token: A bearer token from :meth:`login`, or ``None`` for the login call itself.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.token = token or None

    # -- transport ------------------------------------------------------------------

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict | None = None,
        raw: bool = False,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> Any:
        """Perform one request and return parsed JSON, or raw bytes when ``raw``.

        Args:
            path: Path under the install root, e.g. ``"/api/idapi/scanner_list.php"``.
            method: HTTP method.
            body: JSON body for a POST.
            raw: Return the response bytes instead of parsing them. Used for bundle downloads.
            on_progress: Called as ``(bytes_so_far, total_or_None)`` while the body is read.
                Passing it switches the read from one blocking call to a chunked loop, which is
                the only difference between a download that looks hung and one that does not.

        Returns:
            The parsed JSON object, or the response bytes.

        Raises:
            AuthError: On 401/403 — the actionable case, and worth its own type so the CLI can
                say "run siar-app login" instead of printing a status code.
            ApiError: On any other failure, transport included.
        """
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/octet-stream" if raw else "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                if on_progress is None:
                    payload = resp.read()
                else:
                    payload = _read_reporting(resp, on_progress)
        except urllib.error.HTTPError as e:
            detail, code = _error_detail(e)
            message = f"{method} {path} failed ({e.code}): {detail}"
            if e.code in (401, 403):
                raise AuthError(message, status=e.code, code=code) from None
            raise ApiError(message, status=e.code, code=code) from None
        except urllib.error.URLError as e:
            raise ApiError(f"could not reach {self.base_url}: {e.reason}") from None
        except socket.timeout:
            raise ApiError(f"{self.base_url} timed out after {_TIMEOUT}s") from None

        if raw:
            return payload
        try:
            return json.loads(payload.decode("utf-8"))
        except ValueError:
            raise ApiError(
                f"{path} returned {len(payload)} bytes that are not JSON — "
                "is this URL really an IDent Dynamics install?"
            ) from None

    # -- endpoints ------------------------------------------------------------------

    def login(self, login_id: str, password: str, *, device: str = "") -> dict:
        """Exchange credentials for a bearer token, and keep it on this client.

        Args:
            login_id: Username or email.
            password: The account password.
            device: Label the token is filed under in the user's account page. Defaults to
                ``"siar-app (<hostname>)"`` so a user with several machines can tell their
                tokens apart and revoke one.

        Returns:
            The endpoint's response: ``{"ok", "token", "prefix", "user": {...}}``.

        Raises:
            AuthError: Bad credentials, disabled account, unverified email, or IP lockout.
        """
        label = device or f"siar-app ({socket.gethostname()})"
        res = self._request(
            "/api/idapi/login.php",
            method="POST",
            body={"login": login_id, "password": password, "device": label},
        )
        token = str(res.get("token") or "")
        if not token:
            raise AuthError("the server accepted the login but returned no token")
        self.token = token
        return res

    def register(
        self,
        email: str,
        username: str,
        password: str,
        *,
        display_name: str = "",
    ) -> dict:
        """Create an account on this install.

        The headless half of the web app's signup form, and the only endpoint here that is
        useful without a token. It does not return one: the new account is created unverified
        and cannot sign in until the emailed link is used, so there is nothing to mint a token
        for yet. The caller verifies, then calls :meth:`login`.

        Args:
            email: Where the verification link is sent. Must not already have an account.
            username: 3-64 characters of letters, digits and ``. _ -``.
            password: At least 8 characters. Sent once — the confirmation prompt is the CLI's
                business, and a second copy over the wire proves nothing the first did not.
            display_name: Optional; the server falls back to the username.

        Returns:
            ``{"ok", "verification_sent", "user": {...}, "detail"}``. ``verification_sent`` is
            False when the account exists but the mail did not go out, which is a thing to
            print rather than a failure — the user can request another link.

        Raises:
            ApiError: Signup closed on this install, a taken email or username, a malformed
                field, or the per-IP hourly cap. ``code`` carries which (``"email_taken"``,
                ``"username_taken"``, ``"weak_password"``, ...) and the message is the
                server's own sentence.
        """
        return self._request(
            "/api/idapi/register.php",
            method="POST",
            body={
                "email": email,
                "username": username,
                "password": password,
                "display_name": display_name,
            },
        )

    def algorithms(self) -> list[dict]:
        """The published scanning algorithms this user may download.

        Each row carries ``slug``, ``title``, ``description``, ``shapes``, ``params_schema``,
        ``expects``, ``version`` and ``platforms`` (the build tags that exist). A ``runnable``
        flag is added here rather than server-side, because whether a build fits depends on the
        machine asking.

        Returns:
            The rows, ordered as the server sent them.
        """
        res = self._request("/api/idapi/scanner_list.php")
        rows = res.get("algorithms") if isinstance(res, dict) else None
        rows = rows if isinstance(rows, list) else []
        local = default_platform_tag()
        for r in rows:
            tags = r.get("platforms") or []
            r["runnable"] = any(
                (not t) or str(t).lower() in ("any", local) for t in tags
            )
            r["local_platform"] = local
        return rows

    def bundle(
        self,
        slug: str,
        platform_tag: str | None = None,
        *,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> bytes:
        """Download one algorithm build's bundle zip.

        Args:
            slug: The algorithm slug.
            platform_tag: Which build; defaults to this machine's tag.
            on_progress: Called as ``(bytes_so_far, total_or_None)`` as the zip arrives.

        Returns:
            The zip bytes.

        Raises:
            ApiError: If no such build exists for that platform — the common case being a
                macOS user before the macOS build has been published, so the message matters.
        """
        query = urllib.parse.urlencode(
            {"slug": slug, "platform": platform_tag or default_platform_tag()}
        )
        return self._request(
            f"/api/idapi/scanner_get.php?{query}", raw=True, on_progress=on_progress
        )

    def published_models(self) -> list[dict]:
        """The societies this account may download — models other people bred with siar-build.

        Not a variant of :meth:`algorithms`. That catalogue is the installation's: obfuscated
        bundles, published once, offered to everybody who can sign in. These are packages an
        account uploaded from its own build box, and who may run one is a per-account grant — so
        two people signed in to the same install see different lists, and a model missing from
        this one is a grant nobody has made rather than a fault.

        Each row carries ``slug``, ``owner``, ``access`` (``"owner"``, ``"super"`` or
        ``"granted"``), ``published``, ``target``, ``title``, ``description``, ``version``,
        ``vote``, ``expects``, ``unseen``, ``bytes``, ``sha256`` and the package's own
        ``manifest``.

        Returns:
            The rows, newest build first, as the server sent them.
        """
        res = self._request("/api/idapi/soc_model_list.php")
        rows = res.get("models") if isinstance(res, dict) else None
        return rows if isinstance(rows, list) else []

    def published_bundle(
        self,
        slug: str = "",
        *,
        model_id: int = 0,
        owner: str = "",
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> bytes:
        """Download one published society's package zip.

        Args:
            slug: The society's name. Resolved against what this account may run, its own
                uploads first.
            model_id: The row id from :meth:`published_models`, which names one society exactly.
                Preferred where it is known: a society is identified by (owner, name), and two
                boxes can breed the same target and call it the same thing.
            owner: Narrow a slug to one account's upload.
            on_progress: Called as ``(bytes_so_far, total_or_None)`` as the zip arrives.

        Returns:
            The zip bytes, rooted at the importable package directory.

        Raises:
            ApiError: If there is no such society, or none this account may run — the server
                answers the two the same way on purpose.
        """
        fields: dict[str, str | int] = {}
        if model_id:
            fields["id"] = int(model_id)
        else:
            fields["slug"] = slug
            if owner:
                fields["owner"] = owner
        query = urllib.parse.urlencode(fields)
        return self._request(
            f"/api/idapi/soc_model_get.php?{query}", raw=True, on_progress=on_progress
        )

    def send_feedback(self, slug: str, score: int, *, version: str = "",
                      comment: str = "", platform: str = "") -> dict:
        """Rate how an algorithm performed, 0-9.

        Args:
            slug: The algorithm rated.
            score: 0-9, where 0 is "found nothing useful".
            version: Which build is being rated. Defaults server-side to the one currently
                published, but the CLI sends the installed one — that is the build whose
                output the user is actually reacting to.
            comment: Optional sentence.
            platform: The rater's build tag, for telling a bad port from a bad algorithm.

        Returns:
            The server's response, including ``build`` and ``algorithm`` rating totals.
        """
        return self._request(
            "/api/idapi/scanner_feedback.php",
            method="POST",
            body={
                "slug": slug,
                "score": int(score),
                "version": version,
                "comment": comment,
                "platform": platform,
            },
        )

    def my_feedback(self) -> list[dict]:
        """Every rating this account has given, newest first."""
        res = self._request("/api/idapi/scanner_feedback.php?mine=1")
        rows = res.get("feedback") if isinstance(res, dict) else None
        return rows if isinstance(rows, list) else []


def _read_reporting(resp: Any, on_progress: Callable[[int, int | None], None]) -> bytes:
    """Read a response body in chunks, reporting progress as it goes.

    ``Content-Length`` is what the bar needs and ``scanner_get.php`` always sends it, but a
    proxy that re-encodes the response may not, so the total is optional the whole way down
    and the caller renders bytes-so-far when it is missing. The first call is made before any
    bytes arrive, so something appears on screen while the server is still decrypting.
    """
    raw_total = resp.headers.get("Content-Length")
    try:
        total: int | None = int(raw_total) if raw_total else None
    except (TypeError, ValueError):
        total = None

    on_progress(0, total)
    chunks: list[bytes] = []
    done = 0
    while True:
        chunk = resp.read(_CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
        done += len(chunk)
        on_progress(done, total)
    return b"".join(chunks)


def _error_detail(e: urllib.error.HTTPError) -> tuple[str, str]:
    """Pull ``detail``/``error`` out of a JSON error body; fall back to the HTTP reason."""
    try:
        payload = json.loads(e.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - an error page that is not JSON is still an error
        return (e.reason or "unknown error"), ""
    if not isinstance(payload, dict):
        return (e.reason or "unknown error"), ""
    code = str(payload.get("error") or "")
    detail = str(payload.get("detail") or payload.get("message") or code or e.reason or "")
    return detail, code


def client_from_credentials(base_url: str | None = None) -> Client:
    """Build a client from the saved login.

    Args:
        base_url: Override the saved install URL.

    Returns:
        An authenticated :class:`Client`.

    Raises:
        AuthError: If no token is saved and none is in the environment.
    """
    creds = load_credentials()
    token = creds.get("token")
    if not token:
        raise AuthError(
            "not signed in — run `siar-app login`, or set $SIAR_APP_TOKEN"
        )
    return Client(base_url or creds.get("base_url") or DEFAULT_BASE_URL, token)
