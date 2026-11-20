"""A person's picture from Home Assistant.

A person can be linked to a Home Assistant person entity (``things.person``,
such as ``person.ana``). Its ``entity_picture`` is the picture set on that
person in Home Assistant, and the web UI shows it in place of the initial.

As a Home Assistant app the service reads it through the Supervisor's proxy
to Core (``homeassistant_api``). Standalone it asks with the token of whoever
wants the picture, so nobody sees a picture their own token couldn't fetch.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

import aiohttp

_LOG = logging.getLogger(__name__)

ENTITY = re.compile(r"^person\.[a-z0-9_]+$")
GOOD_FOR_S = 600.0
MISSING_FOR_S = 60.0
MAX_BYTES = 2 << 20


class Pictures:
    def __init__(self, ha_url: str, supervisor_token: str | None) -> None:
        self._supervisor_token = supervisor_token
        self._base = "http://supervisor/core" if supervisor_token else ha_url.rstrip("/")
        self._cache: dict[str, tuple[float, tuple[bytes, str] | None]] = {}
        self._http: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()

    async def get(self, entity_id: str, token: str | None) -> tuple[bytes, str] | None:
        """The picture and its content type, or None when there is none to show."""
        if not ENTITY.match(entity_id or ""):
            return None
        token = self._supervisor_token or token
        if not token or self._base == "auto":
            return None
        key = f"{entity_id}:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        try:
            found = await self._fetch(entity_id, token)
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
            _LOG.info("no picture for %s: %s", entity_id, err)
            found = None
        self._cache[key] = (now + (GOOD_FOR_S if found else MISSING_FOR_S), found)
        return found

    async def _fetch(self, entity_id: str, token: str) -> tuple[bytes, str] | None:
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        auth = {"Authorization": f"Bearer {token}"}
        async with self._http.get(f"{self._base}/api/states/{entity_id}", headers=auth) as r:
            if r.status != 200:
                return None
            picture = ((await r.json()).get("attributes") or {}).get("entity_picture")
        if not isinstance(picture, str) or not picture:
            return None
        # A picture Home Assistant serves itself needs the token, one elsewhere doesn't get it.
        url, headers = (picture, {}) if picture.startswith(("http://", "https://")) else (f"{self._base}{picture}", auth)
        async with self._http.get(url, headers=headers) as r:
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
            if r.status != 200 or not ctype.startswith("image/") or (r.content_length or 0) > MAX_BYTES:
                return None
            # The whole body: a single read returns only what has arrived so far.
            body = bytearray()
            async for part in r.content.iter_chunked(64 << 10):
                body += part
                if len(body) > MAX_BYTES:
                    return None
        return bytes(body), ctype
