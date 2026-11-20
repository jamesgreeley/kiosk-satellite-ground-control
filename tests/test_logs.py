"""The Logs page: the recent log, detailed logs and the report to attach to an issue."""

from __future__ import annotations

import asyncio
import logging

import pytest

from ground_control.audio import float_to_pcm16, wav_bytes
from ground_control.auth import TokenCheck
from ground_control.logs import LOGS, LogBuffer, install, redact
from ground_control.server import build_app
from ground_control.service import GroundControl

from .conftest import SR, chunks
from .kiosks import CLIENTS, approve, hello
from .test_sockets import GOOD, next_of


@pytest.fixture
async def client(aiohttp_client, cfg):
    install()

    async def ask(token):
        return token == "good"

    return await aiohttp_client(build_app(GroundControl(cfg), TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS))


def test_the_buffer_keeps_the_latest_and_filters_by_level():
    buf = LogBuffer(size=3)
    log = logging.getLogger("ground_control.test_buffer")
    log.addHandler(buf)
    log.setLevel(logging.DEBUG)
    try:
        log.debug("one")
        log.info("two")
        log.warning("three")
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            log.exception("four")
    finally:
        log.removeHandler(buf)
    every = buf.since()
    assert [e["message"] for e in every] == ["two", "three", "four"]
    assert "RuntimeError: boom" in every[-1]["exc"]
    assert [e["message"] for e in buf.since(level="WARNING")] == ["three", "four"]
    assert [e["message"] for e in buf.since(after=every[1]["seq"])] == ["four"]


def test_tokens_never_reach_the_log():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhYmNkZWYxMjM0In0.c2lnbmF0dXJlc2lnbmF0dXJl"
    assert redact(f"token {jwt} refused") == "token [redacted] refused"
    assert redact("Authorization: Bearer abc.def") == "Authorization: [redacted]"


async def test_a_session_is_logged_and_detailed_logs_add_its_events(client, speakers, voice):
    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    ana = await r.json()
    await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]), headers=GOOD)
    first = (await (await client.get("/api/logs", headers=GOOD)).json())["last"]
    ticket = approve(client)

    async def session() -> None:
        ws = await client.ws_connect("/api/session")
        await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "device_id": "k1", "device_name": "Kitchen"})
        await next_of(ws, "ready")
        for c in chunks(voice[speakers[0]]["test"][:4 * SR]):
            await ws.send_bytes(float_to_pcm16(c))
        await ws.send_json({"type": "end"})
        await next_of(ws, "stats")
        await asyncio.sleep(0.3)

    await session()
    got = await (await client.get(f"/api/logs?after={first}", headers=GOOD)).json()
    lines = [e["message"] for e in got["entries"]]
    assert any("from Kitchen: live mode, answers anyone" in m for m in lines)
    assert any("ended: Ana (voice" in m for m in lines)
    assert not any(": speaker" in m for m in lines)
    assert got["detailed"] is False

    r = await client.patch("/api/logs", json={"detailed": True}, headers=GOOD)
    assert (await r.json())["detailed"] is True
    try:
        mark = (await (await client.get("/api/logs", headers=GOOD)).json())["last"]
        await session()
        detailed = await (await client.get(f"/api/logs?after={mark}&level=DEBUG", headers=GOOD)).json()
        assert any(e["level"] == "DEBUG" and ": speaker at " in e["message"] for e in detailed["entries"])
        info = await (await client.get(f"/api/logs?after={mark}&level=INFO", headers=GOOD)).json()
        assert all(e["level"] != "DEBUG" for e in info["entries"])
    finally:
        await client.patch("/api/logs", json={"detailed": False}, headers=GOOD)
    assert not LOGS.detailed


async def test_the_report_has_what_an_issue_needs(client):
    ws = await client.ws_connect("/api/kiosk")
    approve(client)
    await hello(ws, name="Kitchen", model="Echo Show 8", version="2026.11.23")
    await next_of(ws, "people")
    r = await client.get("/api/logs/report", headers=GOOD)
    assert r.status == 200
    assert r.headers["Content-Disposition"].startswith('attachment; filename="ground-control-logs-')
    text = await r.text()
    assert text.startswith("Kiosk Satellite Ground Control ")
    assert "Speaker model " in text
    assert "  Kitchen: Echo Show 8, Kiosk Satellite 2026.11.23, connected" in text
    assert "kiosk Kitchen connected from" in text
    await ws.close()


async def test_logs_need_a_sign_in_and_refusals_are_logged(client):
    assert (await client.get("/api/logs")).status == 401
    got = await (await client.get("/api/logs?level=WARNING", headers=GOOD)).json()
    assert any("sign-in refused for /api/logs" in e["message"] for e in got["entries"])
    assert (await client.patch("/api/logs", json={"detailed": "yes"}, headers=GOOD)).status == 400
