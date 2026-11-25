"""Vision Match: frames, the models' readouts and the vision socket."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
from PIL import Image

from ground_control.auth import TokenCheck
from ground_control.models import VISION_FEATURES, default_vision_keys
from ground_control.server import VM, build_app
from ground_control.service import GroundControl
from ground_control.vision import FACE_3D, FINGERS, FRONTAL_MAX_DEG, BadFrame, Look, Vision, decode, head_yaw, nms, turn_of

from .conftest import MODELS
from .kiosks import CLIENTS, approve, receive

GOOD = {"Authorization": "Bearer good"}


def jpeg(rgb: np.ndarray, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, quality=95)
    return buf.getvalue()


def halves(w: int = 64, h: int = 32) -> np.ndarray:
    """Red on the left half, blue on the right."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = (255, 0, 0)
    rgb[:, w // 2:] = (0, 0, 255)
    return rgb


# ── frames and readouts ───────────────────────────────────────────────────


def test_a_frame_is_turned_upright_clockwise():
    upright = decode(jpeg(halves()), 90)
    assert upright.shape[:2] == (64, 32)
    # Turned 90 degrees clockwise, the left half (red) ends up on top.
    assert upright[8, 16, 0] > 200 and upright[56, 16, 2] > 200


def test_a_frame_that_isnt_a_jpeg_is_refused():
    with pytest.raises(BadFrame):
        decode(jpeg(halves(), "PNG"))
    with pytest.raises(BadFrame):
        decode(b"not an image")
    with pytest.raises(BadFrame):
        decode(jpeg(halves()), 45)


def test_every_finger_count_has_a_gesture():
    assert set(FINGERS.values()) == {0, 1, 2, 3, 4, 5}


def test_overlapping_faces_count_once():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]


def _project(yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """The generic face turned ``yaw_deg`` and rolled ``roll_deg``, as a camera sees it."""
    a, b = np.radians(yaw_deg), np.radians(roll_deg)
    yaw = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    roll = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return (FACE_3D @ yaw.T @ roll.T)[:, :2] * 2 + [320, 240]


def test_a_head_turned_away_is_not_frontal():
    assert head_yaw(_project(0)) < 1
    assert abs(head_yaw(_project(20, roll_deg=25)) - 20) < 1
    assert abs(head_yaw(_project(-50)) - 50) < 1
    assert head_yaw(_project(-50)) > FRONTAL_MAX_DEG > head_yaw(_project(15))


def test_the_nose_check_still_applies():
    eye_a, eye_b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    assert abs(turn_of(eye_a, eye_b, np.array([5.0, 6.0]))) < 0.01
    assert turn_of(eye_a, eye_b, np.array([9.0, 6.0])) > 0.3


@pytest.fixture(scope="module")
def light() -> Vision:
    v = Vision(MODELS, {f: "light" for f in VISION_FEATURES}, threads=2)
    yield v
    v.close()


def test_the_light_models_see_nothing_in_an_empty_frame(light):
    look = light.look(jpeg(np.full((480, 640, 3), 128, np.uint8)))
    assert look.faces == [] and look.hand is None
    assert set(look.ms) == {"decode", "face", "hands"}
    assert light.max_looks["face"] > 0 and light.max_looks["hands"] > 0
    assert set(light.ms) == {"face", "hands", "recognition"} and light.label == "light"


def test_a_portrait_frame_and_a_single_look(light):
    look = light.look(jpeg(np.full((640, 480, 3), 90, np.uint8)), rotation=90, want=("face",))
    assert look.faces == [] and look.hand is None and "hands" not in look.ms


# ── the socket ────────────────────────────────────────────────────────────


class FakeVision:
    """Answers one face and a hand showing three, after ``delay`` seconds."""

    def __init__(self, delay: float = 0.0) -> None:
        self.keys = {f: "light" for f in VISION_FEATURES}
        self.label = "light"
        self.rec_key = "light"
        self.max_looks = {"face": 12.5, "hands": 10.0}
        self.on_gpu = False
        self.delay = delay
        self.face_embedding = np.eye(8, dtype=np.float32)[0]

    def look(self, data, rotation=0, want=("face", "hands"), recognize=False, hand_min=0.0):
        decode(data, rotation)
        time.sleep(self.delay)
        face = {"box": [0.4, 0.2, 0.1, 0.15], "score": 0.9, "frontal": True, "yaw": 1.5, "points": []}
        hand = {"box": [0.6, 0.3, 0.05, 0.1], "score": 0.88, "gesture": "three", "fingers": 3}
        self.hand_min = hand_min
        if max(hand["box"][2], hand["box"][3]) < hand_min:
            hand = None
        self.recognized = getattr(self, "recognized", 0) + bool(recognize)
        return Look([face] if "face" in want else None, hand if "hands" in want else None, {"decode": 1.0},
                    [(face["box"], self.face_embedding)] if recognize and "face" in want else None)


@pytest.fixture
async def client(aiohttp_client, cfg):
    async def ask(token):
        return token == "good"

    return await aiohttp_client(build_app(GroundControl(cfg), TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS))


def use(client, vision) -> None:
    client.app[VM].vision = lambda: vision


async def open_vision(client, **start):
    ws = await client.ws_connect("/api/vision")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "device_id": "k1", **start})
    return ws


async def test_a_frame_comes_back_with_its_faces_and_hand(client):
    use(client, FakeVision())
    ws = await open_vision(client)
    ready = await receive(ws, "ready")
    assert ready["model"] == "light" and ready["want"] == ["face", "hands"]
    assert ready["max_looks"] == {"face": 12.5, "hands": 10.0}
    await ws.send_json({"type": "frame", "id": 7, "rotation": 0})
    await ws.send_bytes(jpeg(halves()))
    result = await receive(ws, "result")
    assert result["id"] == 7 and result["faces"][0]["frontal"] and result["hand"]["fingers"] == 3


async def test_a_frame_can_ask_for_one_look(client):
    use(client, FakeVision())
    ws = await open_vision(client, want=["face"])
    await receive(ws, "ready")
    await ws.send_json({"type": "frame", "id": 1, "want": ["face"]})
    await ws.send_bytes(jpeg(halves()))
    result = await receive(ws, "result")
    assert "faces" in result and "hand" not in result
    await ws.send_json({"type": "frame", "id": 2, "want": ["hands"]})
    assert (await receive(ws, "error"))["code"] == "bad_want"


async def test_a_bad_frame_is_answered_and_the_next_one_works(client):
    use(client, FakeVision())
    ws = await open_vision(client)
    await receive(ws, "ready")
    await ws.send_json({"type": "frame", "id": 1})
    await ws.send_bytes(b"not a jpeg")
    error = await receive(ws, "error")
    assert error["id"] == 1 and error["code"] == "bad_frame"
    await ws.send_json({"type": "frame", "id": 2})
    await ws.send_bytes(jpeg(halves()))
    assert (await receive(ws, "result"))["id"] == 2


async def test_a_frame_waiting_behind_a_newer_one_is_dropped(client):
    use(client, FakeVision(delay=0.3))
    ws = await open_vision(client)
    await receive(ws, "ready")
    for i in (1, 2, 3):
        await ws.send_json({"type": "frame", "id": i})
        await ws.send_bytes(jpeg(halves()))
        await asyncio.sleep(0.05)
    seen = []
    while len(seen) < 3:
        msg = await asyncio.wait_for(ws.receive_json(), 5)
        seen.append((msg["type"], msg["id"]))
    assert seen == [("dropped", 2), ("result", 1), ("result", 3)]


async def test_vision_needs_an_approved_kiosk(client):
    use(client, FakeVision())
    ws = await client.ws_connect("/api/vision")
    await ws.send_json({"type": "start", "token": "good", "ticket": "nope", "device_id": "k1"})
    assert (await receive(ws, "error"))["code"] == "not_approved"


async def test_vision_says_when_its_models_cant_run(client):
    use(client, None)
    ws = await open_vision(client)
    assert (await receive(ws, "error"))["code"] == "vision_unavailable"


def test_the_default_models_follow_the_machine():
    assert default_vision_keys("x86_64", cuda=True) == {"face": "accurate", "hands": "accurate",
                                                        "recognition": "accurate"}
    # An N100 or an i9 and an Apple chip: Accurate only for Face Recognition.
    assert default_vision_keys("x86_64") == {"face": "light", "hands": "light", "recognition": "accurate"}
    assert default_vision_keys("aarch64", apple=True)["recognition"] == "accurate"
    # A Raspberry Pi: Light everywhere.
    assert default_vision_keys("aarch64") == {"face": "light", "hands": "light", "recognition": "light"}


async def test_settings_choose_a_model_for_each_feature(client):
    s = await (await client.get("/api/settings", headers=GOOD)).json()
    assert s["vision_models"] == s["vision_defaults"]
    assert all(set(s["vision_choices"][f]) == {"light", "accurate"} for f in VISION_FEATURES)
    r = await client.patch("/api/settings", json={"vision_models": {"hands": "accurate"}}, headers=GOOD)
    assert r.status == 200
    s = await (await client.get("/api/settings", headers=GOOD)).json()
    assert s["vision_models"]["hands"] == "accurate"
    assert s["vision_models"]["face"] == s["vision_defaults"]["face"]
    assert (await client.patch("/api/settings", json={"vision_models": {"hands": "huge"}}, headers=GOOD)).status == 400
    assert (await client.patch("/api/settings", json={"vision_models": {"eyes": "light"}}, headers=GOOD)).status == 400
    # The old single key still sets every feature.
    assert (await client.patch("/api/settings", json={"vision_model": "light"}, headers=GOOD)).status == 200
    s = await (await client.get("/api/settings", headers=GOOD)).json()
    assert set(s["vision_models"].values()) == {"light"}


def test_a_saved_single_model_carries_over_to_every_feature(cfg):
    import json

    from ground_control.service import GroundControl

    (cfg.data_dir / "settings.json").write_text(json.dumps({"vision_model": "accurate"}))
    try:
        assert set(GroundControl(cfg).vision_keys.values()) == {"accurate"}
    finally:
        (cfg.data_dir / "settings.json").unlink()


async def test_a_look_with_a_face_saves_a_vision_session_when_the_socket_closes(client):
    use(client, FakeVision())
    ws = await open_vision(client)
    await receive(ws, "ready")
    await ws.send_json({"type": "frame", "id": 1, "encode_ms": 9.5})
    await ws.send_bytes(jpeg(halves()))
    await receive(ws, "result")
    await ws.send_json({"type": "outcome", "what": "screensaver_dismissed"})
    await ws.send_json({"type": "frame", "id": 2, "rtt_ms": 41})
    await ws.send_bytes(jpeg(halves()))
    await receive(ws, "result")
    await ws.close()
    store = client.app[VM].sessions
    for _ in range(50):
        if store.count(kind="vision"):
            break
        await asyncio.sleep(0.05)
    r = await (await client.get("/api/sessions?kind=vision", headers=GOOD)).json()
    assert r["total"] == 1
    row = r["sessions"][0]
    assert row["kind"] == "vision" and row["summary"] == ["Screensaver dismissed"] and row["looks"] == 2
    record = await (await client.get(f"/api/sessions/{row['id']}", headers=GOOD)).json()
    assert record["looks"][0]["enc"] == 9.5 and record["looks"][0]["rtt"] == 41.0
    assert record["stats"]["end"] == "closed"
    assert [f["reason"] for f in record["frames"]] == ["first", "last"]
    assert record["frames"][0]["faces"][0][4] is True
    frame = await client.get(f"/api/sessions/{row['id']}/frames/0", headers=GOOD)
    assert frame.status == 200 and (await frame.read())[:2] == b"\xff\xd8"
    counts = store.looks(device_id="k1")["k1"][0]
    assert counts["looks"] == 2 and counts["sessions"] == 1 and counts["empty"] == 0


async def test_a_kiosk_with_face_recognition_hears_who_is_in_front_of_it(client):
    from tests.test_faces import FakeEmbedder

    fake = FakeVision()
    use(client, fake)
    vm = client.app[VM]
    ana = vm.people.add("Ana")
    # A faceprint pointing where the fake look's embedding does.
    vm.faces.enroll(ana.id, [np.full((112, 112, 3), 10, np.uint8)], {"light": FakeEmbedder()})
    ws = await open_vision(client, recognize=True)
    assert (await receive(ws, "ready"))["recognize"] is True
    await ws.send_json({"type": "frame", "id": 1, "want": ["face"]})
    await ws.send_bytes(jpeg(halves()))
    result = await receive(ws, "result")
    assert result["person"]["name"] == "Ana" and result["person"]["score"] > 0.9
    assert [p["name"] for p in result["people"]] == ["Ana"]
    assert fake.recognized == 1
    # Who is at the kiosk, for Confirm voices with faces, until the socket closes.
    assert vm.at_kiosk["k1"] == ana.id
    await ws.close()
    for _ in range(50):
        if "k1" not in vm.at_kiosk:
            break
        await asyncio.sleep(0.02)
    assert "k1" not in vm.at_kiosk


async def test_without_asking_for_it_a_kiosk_gets_no_recognition(client):
    fake = FakeVision()
    use(client, fake)
    vm = client.app[VM]
    from tests.test_faces import FakeEmbedder

    vm.faces.enroll(vm.people.add("Ana").id, [np.full((112, 112, 3), 10, np.uint8)], {"light": FakeEmbedder()})
    ws = await open_vision(client)
    assert (await receive(ws, "ready"))["recognize"] is False
    await ws.send_json({"type": "frame", "id": 1})
    await ws.send_bytes(jpeg(halves()))
    assert "person" not in await receive(ws, "result")
    assert fake.recognized == 0


async def test_a_face_is_enrolled_through_the_guided_capture(client, monkeypatch):
    from ground_control import faces
    from tests.test_faces import FakeEmbedder, _face

    monkeypatch.setattr(faces, "MIN_GAP_S", 0.0)
    plan = ([_face(2, 0.15)] * 4 + [_face(25, 0.15)] * 3 + [_face(25, 0.15, flip=True)] * 3 + [_face(2, 0.07)] * 3)
    fake = FakeVision()
    fake.recognizer = FakeEmbedder()
    fake.face = lambda rgb: [plan.pop(0)] if plan else []
    use(client, fake)
    vm = client.app[VM]
    ws = await client.ws_connect("/api/face_enroll")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client), "device_id": "k1", "name": "Mia"})
    ready = await receive(ws, "ready")
    assert [s["step"] for s in ready["steps"]] == ["look", "side_a", "side_b", "far"]
    for i in range(13):
        await ws.send_json({"type": "frame", "id": i, "rotation": 0})
        await ws.send_bytes(jpeg(halves()))
        progress = await receive(ws, "progress")
        assert progress["hint"] == "ok", progress
    saved = await receive(ws, "saved")
    assert saved["crops"] == 13 and saved["complete"]
    mia = next(p for p in vm.people.all() if p.name == "Mia")
    people = (await (await client.get("/api/people", headers=GOOD)).json())["people"]
    person = next(p for p in people if p["id"] == mia.id)
    assert person["has_face"] and person["face_enrollments"][0]["crops"] == 13
    eid = person["face_enrollments"][0]["id"]
    crop = await client.get(f"/api/people/{mia.id}/faces/{eid}/00.jpg", headers=GOOD)
    assert crop.status == 200 and (await crop.read())[:2] == b"\xff\xd8"
    r = await client.delete(f"/api/people/{mia.id}/faces/{eid}", headers=GOOD)
    assert r.status == 200 and not (await r.json())["has_face"]


async def test_hand_sensitivity_sets_the_smallest_hand_counted(client):
    vm = client.app[VM]
    assert vm.hand_sensitivity == 50 and vm.hand_min_size == pytest.approx(0.09)
    r = await client.patch("/api/settings", json={"hand_sensitivity": 0}, headers=GOOD)
    assert r.status == 400
    r = await client.patch("/api/settings", json={"hand_sensitivity": 1}, headers=GOOD)
    assert r.status == 200 and vm.hand_min_size == pytest.approx(0.188)
    settings = await (await client.get("/api/settings", headers=GOOD)).json()
    assert settings["hand_sensitivity"] == 1 and settings["hand_min_size"] == 0.188
    # The fake hand is 10% of the frame on its longer side: too small now.
    fake = FakeVision()
    use(client, fake)
    ws = await open_vision(client)
    await receive(ws, "ready")
    await ws.send_json({"type": "frame", "id": 1, "want": ["hands"]})
    await ws.send_bytes(jpeg(halves()))
    result = await receive(ws, "result")
    assert result["hand"] is None and fake.hand_min == pytest.approx(0.188)
    await client.patch("/api/settings", json={"hand_sensitivity": 100}, headers=GOOD)
    assert vm.hand_min_size == 0.0
