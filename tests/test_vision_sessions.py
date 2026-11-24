from __future__ import annotations

from ground_control.vision_sessions import GAP_S, MAX_FRAMES, MAX_S, VisionTracker, summary

FACE = {"box": [0.4, 0.2, 0.1, 0.15], "score": 0.9, "frontal": True, "turn": 0.02}
AWAY = {**FACE, "frontal": False, "turn": 0.9}
MS = {"decode": 1.0, "face": 3.0}


def hand(fingers):
    return {"box": [0.6, 0.3, 0.05, 0.1], "score": 0.8, "gesture": "x", "fingers": fingers}


def tracker(keep=True):
    saved = []
    t = VisionTracker("k1", "Kitchen", "light", False, keep_frames=lambda: keep,
                      save=lambda record, frames: saved.append((record, frames)))
    return t, saved


def test_looks_that_find_nothing_open_no_session():
    t, saved = tracker()
    for i in range(20):
        t.look(i, ("face",), [], None, MS, b"j", now=float(i))
    t.close("closed", 30.0)
    assert saved == [] and t.take_counts() == {"looks": 20, "empty": 20, "sessions": 0}


def test_a_session_ends_after_the_gap_where_its_last_find_was():
    t, saved = tracker()
    t.look(1, ("face",), [FACE], None, MS, b"a", now=100.0)
    t.look(2, ("face",), [], None, MS, b"b", now=101.0)
    t.tick(100.0 + GAP_S - 0.1)
    assert saved == []
    t.tick(100.0 + GAP_S + 3)
    record, frames = saved[0]
    assert record["stats"]["duration_s"] == GAP_S and record["stats"]["end"] == "idle"
    assert record["stats"]["looks"] == 2 and record["stats"]["faces"] == 1
    assert [n for n, _ in frames] == [0, 1]
    # An empty look after it opens nothing.
    t.look(3, ("face",), [], None, MS, b"c", now=130.0)
    assert t.current is None and t.take_counts() == {"looks": 3, "empty": 1, "sessions": 1}


def test_steps_frames_and_the_summary():
    t, saved = tracker()
    t.look(1, ("face", "hands"), [AWAY], None, MS, b"0", now=0.0)
    t.look(2, ("face", "hands"), [FACE], None, MS, b"1", now=0.5)
    t.look(3, ("face", "hands"), [FACE], hand(3), MS, b"2", now=1.0)
    t.look(4, ("face", "hands"), [FACE], hand(3), MS, b"3", now=1.5)
    t.outcome({"what": "gesture", "fingers": 3, "action": "Refresh the dashboard"}, now=1.6)
    t.look(5, ("face", "hands"), [FACE], hand(2), MS, b"4", now=2.0)
    t.look(6, ("face", "hands"), [], None, MS, b"5", now=2.5)
    t.close("closed", 3.0)
    record, frames = saved[0]
    assert [e["type"] for e in record["events"]] == ["face", "facing", "hand", "fingers", "gesture",
                                                      "fingers", "end"]
    assert [f["reason"] for f in record["frames"]] == ["first", "facing", "reading", "reading", "last"]
    assert [jpeg for _, jpeg in frames] == [b"0", b"1", b"2", b"4", b"5"]
    assert record["summary"] == ["3 fingers → Refresh the dashboard"]
    assert record["stats"]["readings"] == {"3": 2, "2": 1}


def test_frames_stay_under_the_cap_and_off_when_not_kept():
    t, saved = tracker()
    for i in range(30):
        t.look(i, ("hands",), None, hand(i % 6), MS, b"x", now=i * 0.5)
    t.close("closed", 20.0)
    assert len(saved[0][1]) == MAX_FRAMES
    t, saved = tracker(keep=False)
    t.look(1, ("face",), [FACE], None, MS, b"x", now=0.0)
    t.close("closed", 1.0)
    assert saved[0][1] == [] and saved[0][0]["frames"] == []


def test_a_long_stretch_goes_on_in_a_new_session():
    t, saved = tracker()
    now = 0.0
    while now <= MAX_S + 5:
        t.look(now, ("face",), [FACE], None, MS, b"x", now=now)
        now += 5.0
    first = saved[0][0]
    assert first["stats"]["end"] == "long" and t.current.continues == first["id"]


def test_the_kiosk_times_go_with_their_looks():
    t, saved = tracker()
    t.look(1, ("face",), [FACE], None, MS, b"x", encode_ms=8.0, now=0.0)
    t.look(2, ("face",), [FACE], None, MS, b"x", encode_ms=12.0, rtt_ms=40.0, now=0.3)
    t.lost("dropped")
    t.close("closed", 1.0)
    stats = saved[0][0]["stats"]
    assert saved[0][0]["looks"][0]["rtt"] == 40.0
    assert stats["ms"]["encode"]["max"] == 12.0 and stats["ms"]["round_trip"]["p50"] == 40.0
    assert stats["dropped"] == 1


def test_a_summary_without_outcomes_says_what_was_seen():
    stats = {"facing": 0, "faces": 3, "readings": {}, "hands": 2}
    assert summary([], stats) == ["Face turned away", "Hand seen"]


def test_a_postponed_screensaver_counts_once_a_session():
    t, saved = tracker()
    t.look(1, ("face",), [FACE], None, MS, b"x", now=0.0)
    for i in range(3):
        t.outcome({"what": "screensaver_postponed"}, now=1.0 + i)
    t.outcome({"what": "something_else"}, now=4.0)
    t.close("closed", 5.0)
    assert [e["type"] for e in saved[0][0]["events"]] == ["face", "facing", "screensaver_postponed", "end"]
