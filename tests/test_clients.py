"""Only Kiosk Satellite connects, and a new kiosk waits for approval."""

from __future__ import annotations

import json
import time

import pytest
from aiohttp import WSMsgType
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ground_control.auth import TokenCheck
from ground_control.hub import Hub
from ground_control.server import VM, build_app
from ground_control.service import GroundControl

from .kiosks import CLIENTS, approve, hello, proof, receive

GOOD = {"Authorization": "Bearer good"}


@pytest.fixture
async def client(aiohttp_client, cfg):
    async def ask(token):
        return token == "good"

    return await aiohttp_client(build_app(GroundControl(cfg), TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS))


async def closed_with(ws, code: str) -> None:
    assert (await receive(ws, "error"))["code"] == code
    msg = await ws.receive()
    assert msg.type == WSMsgType.CLOSE and ws.close_code == 4403


async def test_a_client_without_kiosk_satellites_key_is_refused(client):
    # No proof at all.
    ws = await client.ws_connect("/api/kiosk")
    await receive(ws, "challenge")
    await ws.send_json({"type": "hello", "token": "good", "device_id": "k1"})
    await closed_with(ws, "not_kiosk_satellite")

    # Another key, the right key over another challenge, and another kiosk's signature.
    other = Ed25519PrivateKey.generate()
    for make in (lambda n: proof(n, "k1", key=other), lambda n: proof("old", "k1"), lambda n: proof(n, "k2")):
        ws = await client.ws_connect("/api/kiosk")
        nonce = (await receive(ws, "challenge"))["nonce"]
        await ws.send_json({"type": "hello", "token": "good", "device_id": "k1", "proof": make(nonce)})
        await closed_with(ws, "not_kiosk_satellite")
    assert client.app[VM].hub.kiosks == {}


async def test_sessions_need_a_ticket_from_an_approved_kiosk(client):
    for start in ({}, {"ticket": "made-up"}):
        ws = await client.ws_connect("/api/session")
        await ws.send_json({"type": "start", "token": "good", "device_id": "k1", **start})
        await closed_with(ws, "not_approved")
    ticket = approve(client)
    ws = await client.ws_connect("/api/enroll")
    await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "device_id": "k2", "name": "Ana"})
    await closed_with(ws, "not_approved")
    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "device_id": "k1"})
    assert (await receive(ws, "ready"))["type"] == "ready"
    await ws.close()


async def test_a_new_kiosk_waits_for_approval(client):
    kiosk = await client.ws_connect("/api/kiosk")
    await hello(kiosk, name="Kitchen", features={"voice_match": True, "trimming": True, "isolation": False,
                                                  "policy": "people", "people": ["p1"], "junk": 1})
    await receive(kiosk, "pending")
    listed = (await (await client.get("/api/kiosks", headers=GOOD)).json())["kiosks"][0]
    assert listed["access"] == "pending" and "_ticket" not in listed
    assert listed["features"] == {"voice_match": True, "trimming": True, "isolation": False,
                                  "policy": "people", "people": ["p1"]}
    # Nobody's names go to a kiosk nobody approved.
    leo = await (await client.post("/api/people", json={"name": "Leo"}, headers=GOOD)).json()
    assert (await client.post("/api/kiosks/k1/enroll", json={"person_id": leo["id"]}, headers=GOOD)).status == 409

    r = await client.post("/api/kiosks/k1/access", json={"access": "approved"}, headers=GOOD)
    assert r.status == 200 and (await r.json())["access"] == "approved"
    ticket = (await receive(kiosk, "ticket"))["ticket"]
    assert [p["name"] for p in (await receive(kiosk, "people"))["people"]] == ["Leo"]

    # Declined: the ticket stops working and the kiosk hears it.
    await client.post("/api/kiosks/k1/access", json={"access": "declined"}, headers=GOOD)
    await receive(kiosk, "declined")
    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "device_id": "k1"})
    await closed_with(ws, "not_approved")
    assert (await client.post("/api/kiosks/k1/access", json={"access": "maybe"}, headers=GOOD)).status == 400
    assert (await client.post("/api/kiosks/nope/access", json={"access": "approved"}, headers=GOOD)).status == 400


async def test_a_kiosk_says_when_its_features_change(client):
    approve(client)
    kiosk = await client.ws_connect("/api/kiosk")
    await hello(kiosk, name="Kitchen", features={"voice_match": True, "trimming": True, "isolation": True})
    await receive(kiosk, "people")
    events = await client.ws_connect("/api/events")
    await events.send_json({"type": "listen", "token": "good"})
    await receive(events, "kiosks")
    await kiosk.send_json({"type": "settings", "enroll": False,
                           "features": {"voice_match": True, "trimming": True, "isolation": False}})
    assert (await receive(events, "kiosk"))["features"]["isolation"] is False


def test_access_and_features_are_kept(tmp_path):
    hub = Hub(tmp_path / "kiosks.json")
    assert hub.new
    hub.approve_known([{"device_id": "old", "device_name": "Kitchen", "started": 5.0}, {"device_id": None}])
    hub.kiosks["old"].features = {"trimming": True}
    hub.save()
    again = Hub(tmp_path / "kiosks.json")
    assert not again.new and again.approved("old") and again.kiosks["old"].features == {"trimming": True}
    assert again.kiosks["old"].name == "Kitchen" and again.kiosks["old"].last_seen == 5.0
    assert again.issue_ticket("missing") is None


def test_kiosks_that_sent_sessions_are_approved_on_the_first_start(cfg):
    vm = GroundControl(cfg)
    vm.sessions.save({"device_id": "kitchen", "mode": "live", "started": time.time(), "events": []})
    vm.close()
    (cfg.data_dir / "kiosks.json").unlink()
    vm = GroundControl(cfg)
    try:
        assert vm.hub.approved("kitchen")
        assert json.loads((cfg.data_dir / "kiosks.json").read_text())["kitchen"]["access"] == "approved"
    finally:
        vm.close()
