"""Kiosks sign in with the Home Assistant token they already have.

The service asks Home Assistant whether the token is good (``GET /api/``)
and remembers the answer for a while, so a session start costs no round
trip most of the time. Only a hash of the token is kept.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable

import aiohttp

_LOG = logging.getLogger(__name__)

GOOD_FOR_S = 600.0
BAD_FOR_S = 30.0


class TokenCheck:
    def __init__(self, ha_url: str, ask: Callable[[str], Awaitable[bool]] | None = None,
                 supervisor_token: str | None = None) -> None:
        self.ha_url = ha_url
        self._supervisor_token = supervisor_token
        self._ssl: bool | None = None
        self._ask = ask or self._ask_home_assistant
        self._cache: dict[str, tuple[bool, float]] = {}
        self._lock = asyncio.Lock()
        self._http: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()

    async def _resolve(self) -> None:
        """As a Home Assistant app on the host network: Core's port and SSL from the Supervisor."""
        if self.ha_url != "auto":
            return
        async with self._http.get("http://supervisor/core/info",
                                  headers={"Authorization": f"Bearer {self._supervisor_token}"}) as r:
            info = (await r.json())["data"]
        ssl = bool(info.get("ssl"))
        self.ha_url = f"{'https' if ssl else 'http'}://127.0.0.1:{info.get('port', 8123)}"
        # Core's certificate names the host, not 127.0.0.1.
        self._ssl = False if ssl else None
        _LOG.info("checking kiosk tokens against Home Assistant at %s", self.ha_url)

    async def _ask_home_assistant(self, token: str) -> bool:
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        try:
            await self._resolve()
            async with self._http.get(f"{self.ha_url}/api/", headers={"Authorization": f"Bearer {token}"},
                                      ssl=self._ssl) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
            _LOG.warning("could not reach Home Assistant at %s to check a token: %s", self.ha_url, err)
            raise

    def forget(self, token: str) -> None:
        """A token that was signed out: ask Home Assistant again next time."""
        self._cache.pop(hashlib.sha256(token.encode()).hexdigest(), None)

    async def valid(self, token: str | None) -> bool:
        if not token:
            return False
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and hit[1] > now:
            return hit[0]
        async with self._lock:
            hit = self._cache.get(key)
            if hit and hit[1] > now:
                return hit[0]
            try:
                ok = await self._ask(token)
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
                # Home Assistant down: keep a token that was good before, refuse a new one.
                return bool(hit and hit[0])
            self._cache[key] = (ok, now + (GOOD_FOR_S if ok else BAD_FOR_S))
            return ok


def bearer(header: str | None) -> str | None:
    if header and header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None
