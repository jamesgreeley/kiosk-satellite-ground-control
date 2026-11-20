"""Suggestions for a person's things, from Home Assistant.

The People page offers them as the fields are typed in: Home Assistant's
people, to-do lists and calendars (entity ids with their names) and Music
Assistant's playlists (by name, the way Kiosk Satellite plays them).

As a Home Assistant app the service asks through the Supervisor's proxy to
Core (``homeassistant_api``). Standalone it asks with the token of whoever
wants them, so nobody is offered an entity their own token can't read, the
same as person pictures. Anything Home Assistant won't answer just leaves
that field without suggestions.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

import aiohttp

_LOG = logging.getLogger(__name__)

DOMAINS = {"person": "person", "todo_list": "todo", "calendar": "calendar"}
"""Which entities each of a person's things can be."""
GOOD_FOR_S = 60.0
MAX_PLAYLISTS = 500


class Suggestions:
    def __init__(self, ha_url: str, supervisor_token: str | None) -> None:
        self._supervisor_token = supervisor_token
        self._base = "http://supervisor/core" if supervisor_token else ha_url.rstrip("/")
        self._cache: dict[str, tuple[float, dict[str, list[dict[str, str]]]]] = {}
        self._http: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()

    @staticmethod
    def empty() -> dict[str, list[dict[str, str]]]:
        return {key: [] for key in (*DOMAINS, "playlist")}

    async def get(self, token: str | None) -> dict[str, list[dict[str, str]]]:
        """For each thing, ``[{"value", "label"}]``: what goes in the field and what it is."""
        token = self._supervisor_token or token
        if not token or self._base == "auto":
            return self.empty()
        key = hashlib.sha256(token.encode()).hexdigest()[:16]
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        found = self.empty()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            found.update(await self._entities(headers))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            _LOG.info("no entity suggestions from Home Assistant: %s", err)
        try:
            found["playlist"] = await self._playlists(headers)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as err:
            _LOG.info("no playlist suggestions from Music Assistant: %s", err)
        self._cache[key] = (now + GOOD_FOR_S, found)
        return found

    def _session(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._http

    async def _json(self, method: str, path: str, headers: dict[str, str], **kw: Any) -> Any:
        async with self._session().request(method, f"{self._base}{path}", headers=headers, **kw) as r:
            if r.status != 200:
                raise ValueError(f"{path} answered {r.status}")
            return await r.json()

    async def _entities(self, headers: dict[str, str]) -> dict[str, list[dict[str, str]]]:
        states = await self._json("GET", "/api/states", headers)
        out: dict[str, list[dict[str, str]]] = {key: [] for key in DOMAINS}
        by_domain = {domain: key for key, domain in DOMAINS.items()}
        for s in states:
            entity = s.get("entity_id", "")
            key = by_domain.get(entity.split(".", 1)[0])
            if key:
                out[key].append({"value": entity, "label": (s.get("attributes") or {}).get("friendly_name") or entity})
        for rows in out.values():
            rows.sort(key=lambda r: r["label"].lower())
        return out

    async def _playlists(self, headers: dict[str, str]) -> list[dict[str, str]]:
        entries = await self._json("GET", "/api/config/config_entries/entry", headers,
                                   params={"domain": "music_assistant"})
        loaded = [e for e in entries if e.get("domain") == "music_assistant" and e.get("state") == "loaded"]
        if not loaded:
            return []
        reply = await self._json(
            "POST", "/api/services/music_assistant/get_library?return_response", headers,
            json={"config_entry_id": loaded[0]["entry_id"], "media_type": "playlist", "limit": MAX_PLAYLISTS})
        names = sorted({i["name"] for i in reply["service_response"]["items"] if i.get("name")}, key=str.lower)
        return [{"value": n, "label": ""} for n in names]
