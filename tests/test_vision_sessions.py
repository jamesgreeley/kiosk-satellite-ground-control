from __future__ import annotations

from ground_control.vision_sessions import GAP_S, MAX_FRAMES, MAX_S, VisionTracker, summary

FACE = {"box": [0.4, 0.2, 0.1, 0.15], "score": 0.9, "frontal": True, "yaw": 1.5}
AWAY = {**FACE, "frontal": False, "yaw": 55.0}
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
    t.look(6, ("face", "hands"), [FACE], hand(2), MS, b"5", now=2.2)
    t.look(7, ("face", "hands"), [], None, MS, b"6", now=2.5)
    t.close("closed", 3.0)
    record, frames = saved[0]
    assert [e["type"] for e in record["events"]] == ["face", "facing", "hand", "fingers", "gesture",
                                                      "fingers", "end"]
    assert [f["reason"] for f in record["frames"]] == ["first", "facing", "reading", "gesture", "reading", "last"]
    assert [jpeg for _, jpeg in frames] == [b"0", b"1", b"2", b"3", b"4", b"6"]
    assert record["summary"] == ["3 fingers → Refresh the dashboard"]
    assert record["stats"]["readings"] == {"3": 2, "2": 2}


def test_frames_stay_under_the_cap_and_off_when_not_kept():
    t, saved = tracker()
    for i in range(30):
        t.look(i, ("hands",), None, hand((i // 2) % 6), MS, b"x", now=i * 0.5)
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


def test_the_frame_a_gesture_acted_on_is_kept():
    t, saved = tracker()
    t.look(1, ("hands",), None, hand(3), MS, b"a", now=0.0)
    t.look(2, ("hands",), None, hand(3), MS, b"b", now=0.5)
    t.look(3, ("hands",), None, hand(3), MS, b"c", now=1.0)
    t.outcome({"what": "gesture", "fingers": 3, "action": "Refresh the dashboard"}, now=1.1)
    t.look(4, ("hands",), None, None, MS, b"d", now=1.5)
    t.close("closed", 2.0)
    record, frames = saved[0]
    assert [(f["n"], f["reason"]) for f in record["frames"]] == [(0, "first"), (2, "gesture"), (3, "last")]
    assert [j for _, j in frames] == [b"a", b"c", b"d"]


def test_a_kiosks_wake_word_switches_are_kept():
    from ground_control.server import _features

    said = {"voice_match": True, "trimming": True, "wake_word_verifier": True, "verifier_trusts_enrolled": False,
            "wake_word_aligner": False}
    assert _features(said) == said
    # A kiosk from before these switches says nothing about them, which the Kiosks page reads as on.
    assert "wake_word_verifier" not in _features({"voice_match": True, "trimming": True})


def test_a_kiosks_vision_match_features_are_kept():
    from ground_control.server import _features

    out = _features({"vision_match": True, "face_detection": True, "hand_gestures": False,
                     "dismiss_on_face": True, "postpone_on_face": "yes", "finger_gestures": 3, "other": 1})
    assert out == {"vision_match": True, "face_detection": True, "hand_gestures": False,
                   "dismiss_on_face": True, "postpone_on_face": False, "finger_gestures": 3}
    assert "finger_gestures" not in _features({"finger_gestures": True})


def test_a_reading_on_one_look_is_a_hand_moving():
    # A closing fist blurs into "5 fingers" for one look, then reads as a fist.
    t, saved = tracker()
    t.look(1, ("hands",), None, hand(5), MS, b"blur", now=0.0)
    for i in range(4):
        t.look(2 + i, ("hands",), None, hand(0), MS, b"fist", now=0.3 * (i + 1))
    t.close("closed", 2.0)
    record, frames = saved[0]
    readings = [(e["t"], e["fingers"]) for e in record["events"] if e["type"] == "fingers"]
    assert readings == [(0.3, 0)]
    assert [f["reason"] for f in record["frames"]] == ["first", "reading", "last"]


def test_a_skipped_pin_and_lockdown_by_face_are_recorded():
    t, saved = tracker()
    t.look(1, ("face",), [FACE], None, MS, b"x", now=0.0)
    t.outcome({"what": "pin_skipped", "mode": "lockdown", "name": "Ana"}, now=1.0)
    t.outcome({"what": "lockdown_off", "name": "Ana"}, now=1.5)
    t.outcome({"what": "pin_skipped", "mode": "other", "name": "Leo"}, now=2.0)
    t.outcome({"what": "lockdown_on"}, now=3.0)
    t.close("closed", 5.0)
    record = saved[0][0]
    events = [e for e in record["events"] if e["type"] in ("pin_skipped", "lockdown_off", "lockdown_on")]
    assert events == [
        {"type": "pin_skipped", "t": 1.0, "name": "Ana", "mode": "lockdown"},
        {"type": "lockdown_off", "t": 1.5, "name": "Ana"},
        {"type": "pin_skipped", "t": 2.0, "name": "Leo", "mode": "kiosk"},
        {"type": "lockdown_on", "t": 3.0},
    ]
    assert record["summary"][:4] == ["Lockdown PIN skipped", "Lockdown off", "Kiosk PIN skipped", "Lockdown on"]


def test_a_looks_time_is_its_total_not_its_stages_added_up():
    """Face Detection and Hand Gestures overlap in a look, so the stages add up to more than the look took."""
    from ground_control.vision_sessions import look_ms

    assert look_ms({"decode": 1.0, "face": 5.0, "hands": 11.0, "total": 13.0}) == 13.0
    # A look saved before the stages could overlap has no total.
    assert look_ms({"decode": 1.0, "face": 5.0, "hands": 11.0}) == 17.0


def test_someone_recognized_on_a_sessions_first_look_is_in_its_timeline():
    """The first look names the person a moment before it opens the session,
    and the recognition was lost: a screensaver dismissed for someone the
    timeline never showed."""
    saved = []
    tracker = VisionTracker("k1", "Kitchen", "accurate", False, keep_frames=lambda: False,
                            save=lambda record, frames: saved.append(record), clock=lambda: 100.0)
    xavier = {"id": "p1", "name": "Xavier", "score": 0.45}
    face = [{"box": [0.4, 0.3, 0.2, 0.3], "score": 0.9, "frontal": True, "yaw": 2.0, "pitch": -10.0}]
    tracker.person(xavier, [xavier], now=100.0)          # told before the look below starts the session
    tracker.look(1, ("face",), face, None, {"decode": 1.0, "face": 5.0, "total": 6.0}, b"", now=100.0)
    events = tracker.current.events
    assert [e["type"] for e in events if e["type"] == "recognized"] == ["recognized"]
    assert next(e for e in events if e["type"] == "recognized")["name"] == "Xavier"
    # Once they are gone, the next session doesn't name them again.
    tracker.person(None, [], now=101.0)
    tracker.close("idle", now=130.0)
    tracker.look(2, ("face",), face, None, {"decode": 1.0, "face": 5.0, "total": 6.0}, b"", now=140.0)
    assert not [e for e in tracker.current.events if e["type"] == "recognized"]
