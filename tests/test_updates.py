from __future__ import annotations

import pytest

from ground_control import config, updates
from ground_control.auth import TokenCheck
from ground_control.server import UPDATES, build_app
from ground_control.service import GroundControl
from ground_control.updates import UpdateCheck, newest, parse

from .kiosks import CLIENTS

TAGS = [{"name": "v2026.11.6"}, {"name": "v2026.11.10"}, {"name": "v2026.11.9"}, {"name": "edge"}, {"name": "v2026.2.1"}]


def test_versions_compare_as_numbers():
    assert parse("v2026.11.10") > parse("2026.11.9")
    assert parse("2026.12.1") > parse("2026.11.30")
    assert parse("edge") is None
    assert newest(TAGS) == "2026.11.10"
    assert newest([{"name": "latest"}, "junk"]) is None


async def test_a_newer_version_is_offered():
    async def fetch():
        return TAGS

    check = UpdateCheck("2026.11.6", fetch)
    assert check.available() is None
    await check.check()
    assert check.available() == {"version": "2026.11.10", "repo_url": updates.REPO_URL,
                                 "changelog_url": updates.CHANGELOG_URL}


async def test_the_same_or_an_older_version_is_not():
    async def fetch():
        return [{"name": "v2026.11.6"}, {"name": "v2026.11.5"}]

    check = UpdateCheck("2026.11.6", fetch)
    await check.check()
    assert check.available() is None


async def test_a_failed_check_changes_nothing():
    async def fetch():
        raise OSError("offline")

    check = UpdateCheck("2026.11.6", fetch)
    await check.check()
    assert check.available() is None and check.checked is None


def test_the_check_is_off_as_a_home_assistant_app(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTIONS_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    assert config.load().update_check
    monkeypatch.setenv("UPDATE_CHECK", "false")
    assert not config.load().update_check
    monkeypatch.delenv("UPDATE_CHECK")
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    assert not config.load().update_check


@pytest.mark.parametrize("on", [True, False])
async def test_status_shows_an_update_only_when_checking(aiohttp_client, cfg, on):
    async def ask(token):
        return token == "good"

    cfg.update_check = on
    app = build_app(GroundControl(cfg), TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS)
    assert (UPDATES in app) == on
    if on:
        app[UPDATES].latest = "2099.1.1"
        app[UPDATES].start = lambda: None  # no request to GitHub from a test
    client = await aiohttp_client(app)
    status = await (await client.get("/api/status", headers={"Authorization": "Bearer good"})).json()
    assert (status["update"] or {}).get("version") == ("2099.1.1" if on else None)
