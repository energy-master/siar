# Vixen Intelligence c.2026
"""Talking to an IDent Dynamics install.

Three endpoints, all under ``/api/idapi/`` and all authenticated by a bearer token rather than a
cookie session:

* ``login.php`` — username/email + password in, token out. Delegates to the app's own
  ``login()``, so lockout, disabled accounts and unverified email all still apply.
* ``scanner_list.php`` — the published algorithm catalogue: slug, description, shapes, params,
  and which platform builds exist. Any active user sees everything the super has published.
* ``scanner_get.php`` — one build's bundle, as a zip.

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
from typing import Any

from siarscan.config import DEFAULT_BASE_URL, default_platform_tag, load_credentials

__all__ = ["ApiError", "AuthError", "Client", "client_from_credentials"]

#: Seconds before a request gives up. Generous: a bundle download crosses the public internet
#: and the box may be rendering thumbnails for someone else at the time.
_TIMEOUT = 120


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
    """A thin client over the three endpoints the CLI needs.

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
    ) -> Any:
        """Perform one request and return parsed JSON, or raw bytes when ``raw``.

        Args:
            path: Path under the install root, e.g. ``"/api/idapi/scanner_list.php"``.
            method: HTTP method.
            body: JSON body for a POST.
            raw: Return the response bytes instead of parsing them. Used for bundle downloads.

        Returns:
            The parsed JSON object, or the response bytes.

        Raises:
            AuthError: On 401/403 — the actionable case, and worth its own type so the CLI can
                say "run siar-scanner login" instead of printing a status code.
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
                payload = resp.read()
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
                ``"siar-scanner (<hostname>)"`` so a user with several machines can tell their
                tokens apart and revoke one.

        Returns:
            The endpoint's response: ``{"ok", "token", "prefix", "user": {...}}``.

        Raises:
            AuthError: Bad credentials, disabled account, unverified email, or IP lockout.
        """
        label = device or f"siar-scanner ({socket.gethostname()})"
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

    def bundle(self, slug: str, platform_tag: str | None = None) -> bytes:
        """Download one algorithm build's bundle zip.

        Args:
            slug: The algorithm slug.
            platform_tag: Which build; defaults to this machine's tag.

        Returns:
            The zip bytes.

        Raises:
            ApiError: If no such build exists for that platform — the common case being a
                macOS user before the macOS build has been published, so the message matters.
        """
        query = urllib.parse.urlencode(
            {"slug": slug, "platform": platform_tag or default_platform_tag()}
        )
        return self._request(f"/api/idapi/scanner_get.php?{query}", raw=True)


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
            "not signed in — run `siar-scanner login`, or set $SIAR_SCANNER_TOKEN"
        )
    return Client(base_url or creds.get("base_url") or DEFAULT_BASE_URL, token)
