from __future__ import annotations

import numpy as np

from ground_control.faces import (
    GONE_S, MARGIN, RECHECK_S, SURE_ABOVE, UNKNOWN_RECHECK_S, TEMPLATE, FaceEnrollment, Faces, Recognizer, STEPS, align, similarity,
)
from ground_control.vision import FACE_3D


class FakeEmbedder:
    """An embedding from a crop's mean color: crops of one shade are one person."""

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        v = np.zeros(8, np.float32)
        v[int(crop.mean()) // 32] = 1.0
        return v


def unit(i: int, n: int = 8) -> np.ndarray:
    v = np.zeros(n, np.float32)
    v[i] = 1
    return v


def test_the_alignment_puts_the_points_on_the_template():
    src = TEMPLATE * 2.5 + [100, 40]
    m = similarity(src, TEMPLATE)
    moved = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(moved, TEMPLATE, atol=1e-6)
    crop = align(np.full((480, 640, 3), 200, np.uint8), src)
    assert crop.shape == (112, 112, 3)


def test_enrollments_build_faceprints_and_deleting_one_rebuilds(tmp_path):
    faces = Faces(tmp_path)
    emb = {"light": FakeEmbedder()}
    a = faces.enroll("p1", [np.full((112, 112, 3), 40, np.uint8)] * 3, emb, {"device_id": "k1"})
    b = faces.enroll("p1", [np.full((112, 112, 3), 200, np.uint8)] * 2, emb)
    ids, m = faces.faceprints("light")
    assert ids == ["p1"] and m.shape == (1, 8)
    assert [r["crops"] for r in faces.enrollments("p1")] == [3, 2] and faces.has_face("p1")
    faces.delete_enrollment("p1", b["id"], emb)
    ids, m = faces.faceprints("light")
    assert np.allclose(m[0], unit(1))
    assert faces.crop_path("p1", a["id"], "00.jpg") is not None
    assert faces.crop_path("p1", a["id"], "../x.jpg") is None
    # Another model's faceprints come from the kept crops.
    assert faces.needs_rebuild("accurate", ["p1"])
    faces.rebuild("accurate", FakeEmbedder(), ["p1"])
    assert faces.faceprints("accurate")[0] == ["p1"]
    faces.forget("p1")
    assert faces.faceprints("light")[0] == [] and not faces.has_face("p1")


BOX = [0.4, 0.2, 0.1, 0.15]
LEFT, RIGHT = [0.1, 0.2, 0.1, 0.15], [0.7, 0.2, 0.1, 0.15]


def test_a_name_needs_two_looks_unless_the_first_is_sure():
    ids, prints = ["ana", "leo"], np.stack([unit(0), unit(1)])
    near = (unit(0) * 0.45 + unit(2) * 0.89)
    near /= np.linalg.norm(near)
    r = Recognizer(threshold=0.4)
    assert r.add([(BOX, near)], ids, prints, 0.0).person_id is None
    assert r.add([(BOX, near)], ids, prints, 0.5).person_id == "ana"
    sure = Recognizer(threshold=0.4)
    v = unit(0) * (0.4 + SURE_ABOVE + 0.05) + unit(2) * 0.6
    assert sure.add([(BOX, v / np.linalg.norm(v))], ids, prints, 0.0).person_id == "ana"


def test_two_people_too_close_to_tell_apart_stay_unknown():
    ids, prints = ["ana", "leo"], np.stack([unit(0), unit(1)])
    v = unit(0) + unit(1) * (1 - MARGIN / 2)
    r = Recognizer(threshold=0.3)
    for t in range(3):
        m = r.add([(BOX, v / np.linalg.norm(v))], ids, prints, t * 0.3)
    assert m.person_id is None


def test_a_named_face_is_rechecked_and_forgotten_when_gone():
    ids, prints = ["ana"], unit(0)[None]
    r = Recognizer(threshold=0.4)
    r.add([(BOX, unit(0))], ids, prints, 0.0)
    assert r.current.person_id == "ana"
    assert not r.wants(RECHECK_S / 2) and r.wants(RECHECK_S + 0.1)
    r.seen([BOX], 1.0)
    assert not r.tick(1.0 + GONE_S - 0.1)
    assert r.tick(1.0 + GONE_S + 0.1) and r.current is None and r.people == []


def test_an_unknown_face_is_rechecked_once_a_second_after_two_looks():
    ids, prints = ["ana"], unit(0)[None]
    r = Recognizer(threshold=0.4)
    r.add([(BOX, unit(3))], ids, prints, 0.0)
    # The second look a name needs comes right away.
    assert r.wants(0.1)
    r.add([(BOX, unit(3))], ids, prints, 0.2)
    assert not r.people
    assert not r.wants(0.2 + UNKNOWN_RECHECK_S / 2)
    assert r.wants(0.2 + UNKNOWN_RECHECK_S + 0.01)


def _toward(i: int, score: float) -> np.ndarray:
    v = unit(i) * score + unit(7) * (1 - score ** 2) ** 0.5
    return v / np.linalg.norm(v)


def test_each_face_is_named_and_the_highest_score_is_at_the_kiosk():
    ids, prints = ["ana", "leo"], np.stack([unit(0), unit(1)])
    r = Recognizer(threshold=0.4)
    r.add([(LEFT, _toward(0, 0.95)), (RIGHT, _toward(1, 0.7))], ids, prints, 0.0)
    assert r.current.person_id == "ana"
    assert [m.person_id for m in r.people] == ["ana", "leo"]


def test_a_close_second_takes_over_only_after_two_checks_in_a_row():
    ids, prints = ["ana", "leo"], np.stack([unit(0), unit(1)])
    r = Recognizer(threshold=0.4)
    r.add([(LEFT, _toward(0, 0.8)), (RIGHT, _toward(1, 0.6))], ids, prints, 0.0)
    assert r.current.person_id == "ana"
    # Leo steps closer and scores higher, one check is not enough.
    r.add([(LEFT, _toward(0, 0.6)), (RIGHT, _toward(1, 0.95))], ids, prints, 2.1)
    assert r.current.person_id == "ana"
    r.add([(LEFT, _toward(0, 0.6)), (RIGHT, _toward(1, 0.95))], ids, prints, 4.2)
    assert r.current.person_id == "leo"


def test_when_the_person_at_the_kiosk_leaves_the_next_one_takes_over():
    ids, prints = ["ana", "leo"], np.stack([unit(0), unit(1)])
    r = Recognizer(threshold=0.4)
    r.add([(LEFT, _toward(0, 0.9)), (RIGHT, _toward(1, 0.7))], ids, prints, 0.0)
    r.seen([RIGHT], 2.0)
    assert r.tick(GONE_S + 0.5)
    assert r.current.person_id == "leo" and [m.person_id for m in r.people] == ["leo"]


def _face(yaw_deg: float, width: float, cx: float = 0.5, flip: bool = False) -> dict:
    """A detected face with its five points, turned ``yaw_deg``, about ``width`` of a 640 px frame wide."""
    a = np.radians(yaw_deg) * (-1 if flip else 1)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    scale = width * 640 / 90
    pts = (FACE_3D @ rot.T)[:, :2] * scale + [cx * 640, 240]
    return {"box": [cx - width / 2, 0.4, width, width * 1.3], "frontal": yaw_deg <= 30, "points": (pts / [640, 480]).tolist()}


class FakeVision:
    recognizer = FakeEmbedder()


def test_the_guided_capture_walks_its_steps_and_says_what_is_wrong():
    rgb = np.full((480, 640, 3), 120, np.uint8)
    g = FaceEnrollment(FakeVision())
    t = 0.0

    def feed(face, faces=None):
        nonlocal t
        t += 0.3
        return g.feed(rgb, faces if faces is not None else [face], t)

    assert feed(None, [])["hint"] == "no_face"
    assert feed(None, [_face(0, 0.15), _face(0, 0.12, cx=0.2)])["hint"] == "several_faces"
    assert feed(_face(0, 0.03))["hint"] == "closer"
    assert feed(_face(25, 0.15))["hint"] == "look_at_screen"
    for _ in range(STEPS[0][1]):
        e = feed(_face(2, 0.15))
    assert e["step"] == "side_a" and e["have"] == 0
    assert feed(_face(5, 0.15))["hint"] == "turn_more"
    assert feed(_face(60, 0.15))["hint"] == "turn_less"
    for _ in range(STEPS[1][1]):
        feed(_face(25, 0.15))
    assert feed(_face(25, 0.15))["hint"] == "other_side"
    for _ in range(STEPS[2][1]):
        e = feed(_face(25, 0.15, flip=True))
    assert e["step"] == "far"
    assert feed(_face(2, 0.14))["hint"] == "farther"
    for _ in range(STEPS[3][1]):
        e = feed(_face(2, 0.07))
    assert e["done"] and g.done and len(g.all_crops()) == sum(n for _, n in STEPS)


def test_a_different_face_is_left_out_of_the_capture():
    g = FaceEnrollment(FakeVision())
    for i in range(STEPS[0][1] - 1):
        g.feed(np.full((480, 640, 3), 60, np.uint8), [_face(0, 0.15)], i * 0.3)
    e = g.feed(np.full((480, 640, 3), 230, np.uint8), [_face(0, 0.15)], 5.0)
    assert e["hint"] == "other_face" and len(g.crops["look"]) == STEPS[0][1] - 1
