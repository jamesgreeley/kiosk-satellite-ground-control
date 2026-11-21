"""The enrollment, kiosk and events sockets, saved sessions and the import."""

from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import pytest
from aiohttp import WSMsgType

from ground_control.audio import float_to_pcm16, wav_bytes
from ground_control.auth import TokenCheck
from ground_control.enroll import DEFAULT_PASSAGE
from ground_control.server import VM, build_app
from ground_control.service import GroundControl

from .conftest import SR, chunks
from .kiosks import CLIENTS, approve, hello

GOOD = {"Authorization": "Bearer good"}


@pytest.fixture
async def client(aiohttp_client, cfg):
    async def ask(token):
        return token == "good"

    return await aiohttp_client(build_app(GroundControl(cfg), TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS))


async def next_of(ws, kind, timeout=10):
    while True:
        msg = await asyncio.wait_for(ws.receive(), timeout)
        if msg.type != WSMsgType.TEXT:
            raise AssertionError(f"socket ended before {kind}")
        data = msg.json()
        if data["type"] == kind:
            return data


async def test_guided_enrollment_saves_a_voice(client, speakers, voice):
    events = await client.ws_connect("/api/events")
    await events.send_json({"type": "listen", "token": "good"})
    await next_of(events, "kiosks")

    ws = await client.ws_connect("/api/enroll")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "name": "Ana", "device_id": "k1",
                        "target_s": 15})
    ready = await next_of(ws, "ready")
    for c in chunks(voice[speakers[0]]["enroll"][:30 * SR], 4000):
        if ws.closed:
            break
        await ws.send_bytes(float_to_pcm16(c))
    saved = await next_of(ws, "saved")
    assert saved["speech_s"] >= 15
    assert (await next_of(events, "enroll_saved"))["person_id"] == ready["person_id"]
    people = await (await client.get("/api/people", headers=GOOD)).json()
    assert people["people"][0]["name"] == "Ana" and people["people"][0]["has_voice"]


async def test_the_app_gets_the_passage_the_kiosk_shows(client):
    events = await client.ws_connect("/api/events")
    await events.send_json({"type": "listen", "token": "good"})
    await next_of(events, "kiosks")
    passage = ["Enciende las luces de la cocina.", "Buenas noches."]
    ticket = approve(client)
    for name, sent, shown in (("Ana", passage, passage), ("Leo", "not a list", DEFAULT_PASSAGE),
                              ("Mia", [1, 2], DEFAULT_PASSAGE)):
        ws = await client.ws_connect("/api/enroll")
        await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "name": name, "device_id": "k1",
                            "passage": sent})
        await next_of(ws, "ready")
        assert (await next_of(events, "enroll_start")).get("passage") == shown
        await ws.close()
        await next_of(events, "enroll_error")


async def test_enrollment_refuses_a_bad_token(client):
    ws = await client.ws_connect("/api/enroll")
    await ws.send_json({"type": "start", "token": "bad", "name": "Eve"})
    assert (await ws.receive()).json()["code"] == "unauthorized"


async def test_kiosk_gets_people_and_enroll_commands(client):
    kiosk = await client.ws_connect("/api/kiosk")
    await hello(kiosk, name="Kitchen", enroll=True)
    await next_of(kiosk, "pending")
    assert (await client.post("/api/kiosks/k1/access", json={"access": "approved"}, headers=GOOD)).status == 200
    await next_of(kiosk, "ticket")
    first = await next_of(kiosk, "people")
    assert first["people"] == [] and first["app_path"] is None
    r = await client.post("/api/people", json={"name": "Leo"}, headers=GOOD)
    leo = await r.json()
    pushed = await next_of(kiosk, "people")
    assert [p["name"] for p in pushed["people"]] == ["Leo"]
    kiosks = await (await client.get("/api/kiosks", headers=GOOD)).json()
    k = kiosks["kiosks"][0]
    assert k["connected"] and k["name"] == "Kitchen" and k["enroll"]
    r = await client.post("/api/kiosks/k1/enroll", json={"person_id": leo["id"]}, headers=GOOD)
    assert r.status == 200
    cmd = await next_of(kiosk, "enroll")
    assert cmd["person_id"] == leo["id"] and cmd["name"] == "Leo" and cmd["color"] == leo["color"]

    # A kiosk that can't start says why, and the web UI hears it.
    events = await client.ws_connect("/api/events")
    await events.send_json({"type": "listen", "token": "good"})
    await next_of(events, "kiosks")
    await kiosk.send_json({"type": "enroll_error", "person_id": leo["id"], "code": "busy", "message": "In a voice turn"})
    err = await next_of(events, "enroll_error")
    assert err["person_id"] == leo["id"] and err["code"] == "busy" and err["device_id"] == "k1"

    # Allow enrollment here turned off: no more enrollments there.
    await kiosk.send_json({"type": "settings", "enroll": False})
    assert (await next_of(events, "kiosk"))["enroll"] is False
    assert (await client.post("/api/kiosks/k1/enroll", json={"person_id": leo["id"]},
                              headers=GOOD)).status == 409
    assert (await client.post("/api/kiosks/nope/enroll", json={"person_id": leo["id"]},
                              headers=GOOD)).status == 409


async def test_a_session_is_saved_with_its_audio(client, speakers, voice):
    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    ana = await r.json()
    await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]), headers=GOOD)
    leo = await (await client.post("/api/people", json={"name": "Leo"}, headers=GOOD)).json()

    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "device_id": "k1",
                        "device_name": "Kitchen"})
    await next_of(ws, "ready")
    audio = np.concatenate([voice[speakers[0]]["test"][:4 * SR], voice[speakers[2]]["test"][:4 * SR]])
    for c in chunks(audio):
        await ws.send_bytes(float_to_pcm16(c))
    await ws.send_json({"type": "end"})
    await next_of(ws, "stats")
    await asyncio.sleep(0.3)

    listed = (await (await client.get("/api/sessions", headers=GOOD)).json())["sessions"]
    assert len(listed) == 1 and listed[0]["device_name"] == "Kitchen" and listed[0]["end"] == "other_voice"
    sid = listed[0]["id"]
    record = await (await client.get(f"/api/sessions/{sid}", headers=GOOD)).json()
    assert any(t["kind"] == "check" and t["low"] for t in record["timeline"])
    assert record["stats"]["speaker"]["name"] == "Ana"
    clean = await (await client.get(f"/api/sessions/{sid}/audio/clean", headers=GOOD)).read()
    with wave.open(io.BytesIO(clean)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    cut = next(e for e in record["events"] if e["type"] == "other_voice")["cut"]
    assert np.all(x[int(cut * SR) + 1:] == 0)

    r = await client.post(f"/api/sessions/{sid}/enroll", json={"person_id": leo["id"]}, headers=GOOD)
    assert r.status == 200 and (await r.json())["has_voice"]
    assert (await client.delete(f"/api/sessions/{sid}", headers=GOOD)).status == 204


async def test_enrollment_records_the_kiosk_and_can_be_deleted(client, speakers, voice):
    kiosk = await client.ws_connect("/api/kiosk")
    await hello(kiosk, name="Kitchen", model="Echo Show 8 (2nd gen)")
    await next_of(kiosk, "pending")
    await client.post("/api/kiosks/k1/access", json={"access": "approved"}, headers=GOOD)
    ticket = (await next_of(kiosk, "ticket"))["ticket"]
    await next_of(kiosk, "people")
    ws = await client.ws_connect("/api/enroll")
    await ws.send_json({"type": "start", "token": "good", "ticket": ticket, "name": "Ana", "device_id": "k1",
                        "target_s": 12})
    ready = await next_of(ws, "ready")
    for c in chunks(voice[speakers[0]]["enroll"][:25 * SR], 4000):
        if ws.closed:
            break
        await ws.send_bytes(float_to_pcm16(c))
    await next_of(ws, "saved")
    r = await client.post(f"/api/people/{ready['person_id']}/enroll?device_name=Phone",
                          data=wav_bytes(voice[speakers[0]]["enroll"][25 * SR:40 * SR]), headers=GOOD)
    ana = await r.json()
    assert ana["models"] == ["Echo Show 8 (2nd gen)"]
    by_place = {e["device_name"]: e for e in ana["enrollments"]}
    assert by_place["Kitchen"]["model"] == "Echo Show 8 (2nd gen)" and by_place["Kitchen"]["device_id"] == "k1"
    assert "Phone" in by_place
    r = await client.delete(f"/api/people/{ana['id']}/enrollments/{by_place['Kitchen']['id']}", headers=GOOD)
    after = await r.json()
    assert after["models"] == [] and [e["device_name"] for e in after["enrollments"]] == ["Phone"]
    vm = client.app[VM]
    assert (await vm.run(vm.identify, voice[speakers[0]]["test"]))["name"] == "Ana"
    assert (await client.delete(f"/api/people/{ana['id']}/enrollments/123", headers=GOOD)).status == 404


async def test_follow_sends_only_the_users_speech(client, speakers, voice):
    import base64

    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    ana = await r.json()
    await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]), headers=GOOD)
    user, tv = voice[speakers[0]]["test"], voice[speakers[2]]["test"]
    anchor = voice[speakers[0]]["enroll"][-int(1.2 * SR):]
    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "device_id": "k1", "mode": "follow",
                        "anchor": base64.b64encode(float_to_pcm16(anchor)).decode()})
    await next_of(ws, "ready")
    audio = np.concatenate([tv[:2 * SR], user[:3 * SR], tv[2 * SR:4 * SR], np.zeros(SR, dtype=np.float32)])
    for c in chunks(audio):
        await ws.send_bytes(float_to_pcm16(c))
    got: list[str] = []
    received = 0
    while "turn_end" not in got:
        msg = await asyncio.wait_for(ws.receive(), 10)
        if msg.type == WSMsgType.BINARY:
            assert "turn_start" in got, "audio before the turn started"
            received += len(msg.data) // 2
        elif msg.type == WSMsgType.TEXT:
            got.append(msg.json()["type"])
        else:
            raise AssertionError("socket ended before turn_end")
    assert "speaker" in got
    # The user's 3 s, give or take the edges: not the 4 s of TV around it.
    assert 2.4 * SR <= received <= 3.8 * SR
    await ws.send_json({"type": "end"})
    stats = await next_of(ws, "stats")
    assert stats["turns"] == 1
    await asyncio.sleep(0.3)
    sessions = (await (await client.get("/api/sessions", headers=GOOD)).json())["sessions"]
    assert sessions[0]["mode"] == "follow"


async def test_a_follow_up_is_saved_in_the_same_session(client, speakers, voice):
    import base64

    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    ana = await r.json()
    await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]), headers=GOOD)
    user = voice[speakers[0]]["test"]
    anchor = voice[speakers[0]]["enroll"][-int(1.2 * SR):]

    tickets = {"k1": approve(client), "k2": approve(client, "k2", "Hallway")}

    async def turn(device_id="k1", **start):
        ws = await client.ws_connect("/api/session")
        await ws.send_json({"type": "start", "token": "good", "ticket": tickets[device_id], "device_id": device_id,
                            "mode": "follow"} | start)
        sid = (await next_of(ws, "ready"))["session"]
        for c in chunks(np.concatenate([user[:2 * SR], np.zeros(SR, dtype=np.float32)])):
            await ws.send_bytes(float_to_pcm16(c))
        await ws.send_json({"type": "end"})
        while True:
            # Follow mode sends the user's speech back as binary frames.
            msg = await asyncio.wait_for(ws.receive(), 10)
            if msg.type == WSMsgType.TEXT and msg.json()["type"] == "stats":
                break
            assert msg.type in (WSMsgType.TEXT, WSMsgType.BINARY), "socket ended before stats"
        await asyncio.sleep(0.3)
        return sid

    first = await turn(anchor=base64.b64encode(float_to_pcm16(anchor)).decode())
    again = await turn(continues=first, speaker=ana["id"])
    assert again == first
    sessions = (await (await client.get("/api/sessions", headers=GOOD)).json())["sessions"]
    assert [s["id"] for s in sessions] == [first]
    record = await (await client.get(f"/api/sessions/{first}", headers=GOOD)).json()
    assert record["stats"]["turns"] == 2 and len(record["followups"]) == 1
    assert record["wake_s"] == 1.2
    assert (await client.get(f"/api/sessions/{first}/audio/wake", headers=GOOD)).status == 200
    # Another kiosk can't join it.
    other = await turn(continues=first, device_id="k2")
    assert other != first


async def test_a_session_downloads_as_one_zip(client, speakers, voice):
    import io
    import json
    import zipfile

    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "device_id": "k1",
                        "device_name": "Kitchen"})
    await next_of(ws, "ready")
    for c in chunks(voice[speakers[0]]["test"][:3 * SR]):
        await ws.send_bytes(float_to_pcm16(c))
    await ws.send_json({"type": "end"})
    await next_of(ws, "stats")
    await asyncio.sleep(0.3)
    sid = (await (await client.get("/api/sessions", headers=GOOD)).json())["sessions"][0]["id"]
    r = await client.get(f"/api/sessions/{sid}/download", headers=GOOD)
    assert r.status == 200 and r.headers["Content-Type"] == "application/zip"
    assert r.headers["Content-Disposition"] == f'attachment; filename="ground-control-session-{sid}.zip"'
    z = zipfile.ZipFile(io.BytesIO(await r.read()))
    assert "session.json" in z.namelist() and "as-heard.wav" in z.namelist()
    assert json.loads(z.read("session.json"))["id"] == sid
    text = z.read("ground-control.txt").decode()
    assert text.startswith("Kiosk Satellite Ground Control ") and f"Session {sid}" in text and "Kitchen" in text
    assert (await client.get("/api/sessions/nope/download", headers=GOOD)).status == 404


async def test_a_kiosk_answering_people_ground_control_doesnt_have_is_logged_once(client, caplog):
    """As after moving to another Ground Control: every command would be rejected without a word why."""
    from ground_control.server import _session_text

    approve(client)
    ana = client.app[VM].people.add("Ana")
    kiosk = await client.ws_connect("/api/kiosk")
    with caplog.at_level("WARNING", logger="ground_control.server"):
        await hello(kiosk, name="Kitchen", features={"policy": "people", "people": ["gone"]})
        await next_of(kiosk, "people")
        await kiosk.send_json({"type": "settings", "features": {"policy": "people", "people": ["gone"]}})
        await kiosk.send_json({"type": "settings", "features": {"policy": "people", "people": [ana.id]}})
        await kiosk.send_json({"type": "settings", "features": {"policy": "people", "people": [ana.id, "gone"]}})
        await asyncio.sleep(0.3)
    warned = [r.getMessage() for r in caplog.records if "doesn't have" in r.getMessage()]
    assert warned == [
        "kiosk Kitchen answers 1 person Ground Control doesn't have, so it answers nobody: choose them again in "
        "Kiosk Satellite, under Voice Satellite > Ground Control Voice Match > People",
        "kiosk Kitchen answers 1 person Ground Control doesn't have: choose them again in "
        "Kiosk Satellite, under Voice Satellite > Ground Control Voice Match > People",
    ]
    await kiosk.close()
    assert _session_text({"rejected": True, "speaker": {"name": "Ana"}}, []).startswith(
        "rejected Ana, not one of the people this kiosk answers")
