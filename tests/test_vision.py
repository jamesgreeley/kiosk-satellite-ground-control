"""Vision Match: frames, the models' readouts and the vision socket."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
from PIL import Image

from ground_control.auth import TokenCheck
from ground_control.models import VISION_MODELS
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
    v = Vision(MODELS, VISION_MODELS["light"], threads=2)
    yield v
    v.close()


def test_the_light_models_see_nothing_in_an_empty_frame(light):
    look = light.look(jpeg(np.full((480, 640, 3), 128, np.uint8)))
    assert look.faces == [] and look.hand is None
    assert set(look.ms) == {"decode", "face", "hands"}
    assert light.max_looks["face"] > 0 and light.max_looks["hands"] > 0


def test_a_portrait_frame_and_a_single_look(light):
    look = light.look(jpeg(np.full((640, 480, 3), 90, np.uint8)), rotation=90, want=("face",))
    assert look.faces == [] and look.hand is None and "hands" not in look.ms


# ── the socket ────────────────────────────────────────────────────────────


class FakeVision:
    """Answers one face and a hand showing three, after ``delay`` seconds."""

    def __init__(self, delay: float = 0.0) -> None:
        self.model = VISION_MODELS["light"]
        self.max_looks = {"face": 12.5, "hands": 10.0}
        self.on_gpu = False
        self.delay = delay

    def look(self, data, rotation=0, want=("face", "hands")):
        decode(data, rotation)
        time.sleep(self.delay)
        face = {"box": [0.4, 0.2, 0.1, 0.15], "score": 0.9, "frontal": True, "yaw": 1.5, "points": []}
        hand = {"box": [0.6, 0.3, 0.05, 0.1], "score": 0.88, "gesture": "three", "fingers": 3}
        return Look([face] if "face" in want else None, hand if "hands" in want else None, {"decode": 1.0})


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


async def test_settings_offer_accurate_only_with_cuda(client):
    s = await (await client.get("/api/settings", headers=GOOD)).json()
    assert s["vision_model"] == "light" and list(s["vision_models"]) == ["light"]
    r = await client.patch("/api/settings", json={"vision_model": "accurate"}, headers=GOOD)
    assert r.status == 400


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
