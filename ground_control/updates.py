"""Whether a newer Ground Control is out, for standalone installs.

A Home Assistant app gets its updates from Home Assistant. A Docker container
has nobody to tell it, so it asks GitHub for the repository's version tags
a minute after it starts and every 12 hours after that, and Overview shows a
card when one is newer. Only the request goes out: GitHub sees the address it
comes from and the Ground Control version, nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

_LOG = logging.getLogger(__name__)

REPO = "jxlarrea/kiosk-satellite-ground-control"
REPO_URL = f"https://github.com/{REPO}"
CHANGELOG_URL = f"{REPO_URL}/blob/main/CHANGELOG.md"
TAGS_URL = f"https://api.github.com/repos/{REPO}/tags?per_page=100"
FIRST_AFTER_S = 60.0
EVERY_S = 12 * 3600.0

_VERSION = re.compile(r"^v?(\d+(?:\.\d+)*)$")


def parse(version: str) -> tuple[int, ...] | None:
    """``2026.11.6`` or ``v2026.11.6`` as numbers to compare, else None."""
    m = _VERSION.match(version.strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def newest(tags: list[Any]) -> str | None:
    """The highest version among GitHub's tags, without its ``v``."""
    found = [(v, t["name"].lstrip("v")) for t in tags
             if isinstance(t, dict) and isinstance(t.get("name"), str) and (v := parse(t["name"])) is not None]
    return max(found)[1] if found else None


class UpdateCheck:
    def __init__(self, current: str, fetch: Callable[[], Awaitable[list[Any]]] | None = None) -> None:
        self.current = current
        self.latest: str | None = None
        self.checked: float | None = None
        self._fetch = fetch or self._github
        self._task: asyncio.Task[None] | None = None
        self._http: aiohttp.ClientSession | None = None

    def available(self) -> dict[str, str] | None:
        """The newer version and where to read about it, or None."""
        mine, latest = parse(self.current), parse(self.latest or "")
        if mine is None or latest is None or latest <= mine:
            return None
        return {"version": self.latest or "", "repo_url": REPO_URL, "changelog_url": CHANGELOG_URL}

    async def check(self) -> None:
        try:
            latest = newest(await self._fetch())
        except Exception as err:  # noqa: BLE001 - offline or rate limited: try again next time
            _LOG.debug("update check failed: %s", err)
            return
        self.checked = time.time()
        if latest and latest != self.latest:
            self.latest = latest
            if self.available():
                _LOG.info("Ground Control %s is available, this is %s: %s", latest, self.current, CHANGELOG_URL)

    async def _github(self) -> list[Any]:
        if self._http is None:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": f"KioskSatelliteGroundControl/{self.current}"})
        async with self._http.get(TAGS_URL) as r:
            r.raise_for_status()
            return await r.json()

    async def _run(self) -> None:
        await asyncio.sleep(FIRST_AFTER_S)
        while True:
            await self.check()
            await asyncio.sleep(EVERY_S)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._http is not None:
            await self._http.close()
            self._http = None
