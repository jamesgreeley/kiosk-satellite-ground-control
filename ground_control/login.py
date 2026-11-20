"""Log in with Home Assistant, for the web UI of a standalone container.

As a Home Assistant app the panel opens through ingress and Home Assistant
has already signed the user in. A standalone container has no such proxy, so
the web UI sends the browser to Home Assistant's own login page (its OAuth
flow, ``/auth/authorize``) and back to ``/auth/callback`` here with a code.

The service trades the code for tokens itself: Home Assistant only answers
token requests from other origins when they are listed in its
``cors_allowed_origins``, so the browser can't. The client id is the address
the browser opened Ground Control at, which Home Assistant accepts because
the redirect is on the same host and port. PKCE ties the code to this
service.

The browser keeps the tokens and sends the access token like any other
Home Assistant token. Home Assistant gives out access tokens for 30 minutes,
so the web UI asks for a new one with the refresh token through
``/auth/refresh``. Signing out revokes the refresh token.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

_LOG = logging.getLogger(__name__)

PENDING_FOR_S = 600.0
MAX_PENDING = 100


class LoginError(Exception):
    """Home Assistant refused, or couldn't be reached."""

    def __init__(self, message: str, refused: bool) -> None:
        super().__init__(message)
        self.refused = refused


@dataclass
class _Pending:
    verifier: str
    client_id: str
    redirect_uri: str
    next: str
    until: float


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class HomeAssistantLogin:
    def __init__(self, ha_url: str) -> None:
        self.ha_url = ha_url.rstrip("/")
        self._pending: dict[str, _Pending] = {}
        self._http: aiohttp.ClientSession | None = None

    @property
    def available(self) -> bool:
        """Only standalone: an app finds Home Assistant through the Supervisor, at an
        address the browser can't open."""
        return self.ha_url not in ("", "auto")

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()

    def start(self, origin: str, next_hash: str = "") -> str:
        """Where to send the browser: Home Assistant's login page."""
        now = time.monotonic()
        for state in [s for s, p in self._pending.items() if p.until < now]:
            del self._pending[state]
        while len(self._pending) >= MAX_PENDING:
            del self._pending[next(iter(self._pending))]
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        client_id = f"{origin}/"
        redirect_uri = f"{origin}/auth/callback"
        if not next_hash.startswith("#") or len(next_hash) > 200:
            next_hash = ""
        self._pending[state] = _Pending(verifier, client_id, redirect_uri, next_hash, now + PENDING_FOR_S)
        query = urlencode({"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
                           "state": state, "code_challenge": _challenge(verifier),
                           "code_challenge_method": "S256"})
        return f"{self.ha_url}/auth/authorize?{query}"

    async def finish(self, state: str, code: str) -> tuple[dict[str, Any], str]:
        """Trade the code Home Assistant sent back for tokens. Returns them and
        the page to go back to."""
        pending = self._pending.pop(state, None)
        if pending is None or pending.until < time.monotonic():
            raise LoginError("this login expired or was already used", refused=True)
        got = await self._token({"grant_type": "authorization_code", "code": code,
                                 "client_id": pending.client_id, "code_verifier": pending.verifier})
        return {"access_token": got["access_token"], "refresh_token": got["refresh_token"],
                "expires_in": int(got.get("expires_in", 1800)), "client_id": pending.client_id}, pending.next

    async def refresh(self, refresh_token: str, client_id: str) -> dict[str, Any]:
        got = await self._token({"grant_type": "refresh_token", "refresh_token": refresh_token,
                                 "client_id": client_id})
        return {"access_token": got["access_token"], "expires_in": int(got.get("expires_in", 1800))}

    async def revoke(self, refresh_token: str) -> None:
        try:
            async with self._session().post(f"{self.ha_url}/auth/revoke", data={"token": refresh_token}):
                pass
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOG.info("couldn't revoke a sign-in at Home Assistant: %s", err)

    def _session(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._http

    async def _token(self, data: dict[str, str]) -> dict[str, Any]:
        try:
            async with self._session().post(f"{self.ha_url}/auth/token", data=data) as r:
                body = await r.json(content_type=None)
                if r.status != 200 or not isinstance(body, dict) or "access_token" not in body:
                    reason = body.get("error_description") or body.get("error") if isinstance(body, dict) else None
                    raise LoginError(f"Home Assistant refused: {reason or r.status}", refused=r.status < 500)
                return body
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            raise LoginError(f"couldn't reach Home Assistant at {self.ha_url}: {err}", refused=False) from err
