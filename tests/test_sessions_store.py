from __future__ import annotations

import time

import numpy as np

from ground_control.audio import read_wav
from ground_control.sessions import SessionStore


def test_save_list_get_and_audio(tmp_path):
    store = SessionStore(tmp_path)
    raw = np.full(16000, 0.25, dtype=np.float32)
    clean = raw.copy()
    clean[8000:] = 0
    sid = store.save({"device_id": "k1", "stats": {"end": "other_voice", "audio_s": 1.0,
                                                    "speaker": {"name": "Ana"}}}, raw, clean)
    store.save({"device_id": "k2", "stats": {}}, raw)
    assert [s["device_id"] for s in store.list()] == ["k2", "k1"]
    assert [s["id"] for s in store.list(device_id="k1")] == [sid]
    assert store.get(sid)["stats"]["speaker"]["name"] == "Ana"
    assert np.allclose(read_wav(store.audio_path(sid, "clean"))[8000:], 0)
    assert store.audio_path(sid, "../x") is None and store.get("../x") is None
    assert store.delete(sid) and store.get(sid) is None


def test_no_audio_when_off(tmp_path):
    store = SessionStore(tmp_path, keep_audio=False)
    sid = store.save({"stats": {}}, np.zeros(1600, dtype=np.float32))
    assert store.get(sid)["has_audio"] is False
    assert store.audio_path(sid, "raw") is None


def test_ids_sort_in_save_order(tmp_path):
    store = SessionStore(tmp_path)
    ids = [store.save({"stats": {}}) for _ in range(50)]
    assert store._ids() == sorted(ids, reverse=True) == ids[::-1]


def test_old_sessions_are_pruned(tmp_path):
    store = SessionStore(tmp_path, keep_days=7)
    old = f"{1_000_000_000_000_000:014x}abcd"  # 2001
    store.save({"id": old, "stats": {}})
    assert store.get(old) is None
    new = store.save({"stats": {}})
    assert store.get(new) is not None


def test_the_wake_word_is_kept_with_the_session(tmp_path):
    store = SessionStore(tmp_path)
    wake = np.full(12000, 0.1, dtype=np.float32)
    sid = store.save({"mode": "follow", "stats": {}}, np.zeros(16000, dtype=np.float32), None, wake)
    assert store.get(sid)["wake_s"] == 0.75
    assert len(read_wav(store.audio_path(sid, "wake"))) == 12000
    assert store.delete(sid) and store.audio_path(sid, "wake") is None


def test_a_follow_up_joins_the_session_of_its_first_turn(tmp_path):
    store = SessionStore(tmp_path)
    first = {"mode": "follow", "device_id": "k1", "started": time.time(),
             "events": [{"type": "turn_start", "t": 0.3, "turn": 1}],
             "timeline": [{"t": 0.0, "kind": "anchor", "score": 0.6}], "released": [[0.3, 0.9]],
             "stats": {"audio_s": 1.0, "turns": 1, "released_s": 0.6, "speaker": {"name": "Ana"}, "end": "silence"}}
    sid = store.save(first, np.full(16000, 0.2, dtype=np.float32), np.full(9600, 0.2, dtype=np.float32))
    assert store.continuing(sid, "k1")
    assert not store.continuing(sid, "k2") and not store.continuing("nope", "k1")

    second = {"mode": "follow", "device_id": "k1",
              "events": [{"type": "turn_start", "t": 0.2, "turn": 1}, {"type": "rejected", "t": 0.5}],
              "timeline": [{"t": 0.0, "kind": "conversation", "id": "a"}, {"t": 0.4, "kind": "check", "turn": 1}],
              "released": [[0.2, 0.5]],
              "stats": {"audio_s": 0.8, "turns": 1, "released_s": 0.3, "speaker": {"name": None}, "end": "rejected",
                        "rejected": True}}
    assert store.append(sid, second, np.full(12800, 0.4, dtype=np.float32), np.full(4800, 0.4, dtype=np.float32)) == sid

    got = store.get(sid)
    assert [s["id"] for s in store.list()] == [sid]
    assert got["followups"] == [1.5]
    assert [e["t"] for e in got["events"]] == [0.3, 1.7, 2.0]
    assert [e.get("turn") for e in got["events"]] == [1, 2, None]
    assert got["timeline"][-1] == {"t": 1.9, "kind": "check", "turn": 2}
    assert got["released"] == [[0.3, 0.9], [1.7, 2.0]]
    assert got["stats"]["turns"] == 2 and got["stats"]["audio_s"] == 2.3 and got["stats"]["released_s"] == 0.9
    assert got["stats"]["speaker"]["name"] == "Ana" and got["stats"]["end"] == "rejected"
    raw = read_wav(store.audio_path(sid, "raw"))
    assert len(raw) == 16000 + 8000 + 12800 and np.allclose(raw[16000:24000], 0)


def test_sessions_saved_as_json_files_are_imported_once(tmp_path):
    import json

    (tmp_path / "sessions").mkdir()
    old = {"id": "0005f00000000000ab12", "started": 1.0, "device_id": "k1", "device_name": "Kitchen",
           "mode": "command", "stats": {"end": "released", "speaker": {"id": "p1", "name": "Ana"}}}
    (tmp_path / "sessions" / f"{old['id']}.json").write_text(json.dumps(old))
    (tmp_path / "sessions" / "broken.json").write_text("{")
    store = SessionStore(tmp_path, keep_days=100000)
    assert store.get(old["id"])["stats"]["speaker"]["name"] == "Ana"
    assert (tmp_path / "sessions" / "imported-json" / f"{old['id']}.json").is_file()
    store.close()
    # A second start finds nothing left to import and keeps what it has.
    store = SessionStore(tmp_path, keep_days=100000)
    assert [s["id"] for s in store.list()] == [old["id"]]


def test_the_list_filters_and_pages_in_the_database(tmp_path):
    store = SessionStore(tmp_path)
    for i in range(5):
        store.save({"device_id": "k1", "device_name": "Kitchen",
                    "stats": {"end": "other_voice" if i % 2 else "released",
                              "speaker": {"id": "p1", "name": "Ana"} if i < 2 else None}})
    store.save({"kind": "vision", "device_id": "k2", "device_name": "Hall", "stats": {"end": "idle"},
                "summary": ["Looked at the kiosk"]})
    assert store.count() == 6
    assert store.count(kind="voice") == 5 and store.count(kind="vision") == 1
    assert store.count(person="p1") == 2 and store.count(person="unknown") == 3
    assert store.count(show="cut") == 2
    assert store.count(device="Hall") == 1 and store.count(q="kitch") == 5
    first, second = store.list(limit=4), store.list(limit=4, offset=4)
    assert len(first) == 4 and len(second) == 2
    assert not {s["id"] for s in first} & {s["id"] for s in second}
    vision = store.list(kind="vision")[0]
    assert vision["kind"] == "vision" and vision["summary"] == ["Looked at the kiosk"]
    assert store.kiosks() == ["Hall", "Kitchen"]
    counts = store.counts(0)
    assert counts["voice"]["total"] == 5 and counts["voice"]["other_voice"] == 2
    assert counts["vision"]["total"] == 1


def test_each_kind_has_its_own_cap(tmp_path, monkeypatch):
    from ground_control import sessions

    monkeypatch.setattr(sessions, "MAX_SESSIONS", {"voice": 3, "vision": 2})
    store = SessionStore(tmp_path)
    voice = [store.save({"stats": {}}) for _ in range(3)]
    for _ in range(5):
        store.save({"kind": "vision", "stats": {}})
    assert store.count(kind="vision") == 2
    assert [s["id"] for s in store.list(kind="voice")] == voice[::-1]


def test_vision_frames_are_kept_and_deleted(tmp_path):
    store = SessionStore(tmp_path)
    record = {"kind": "vision", "stats": {}, "frames": [{"n": 0, "reason": "first"}, {"n": 4, "reason": "last"}]}
    sid = store.save(record, frames=[(0, b"\xff\xd8a"), (4, b"\xff\xd8bb")])
    assert store.frame_path(sid, 4).read_bytes() == b"\xff\xd8bb"
    assert store.frame_path(sid, "../1") is None
    assert store.frames_stats() == {"frames": 2, "bytes": 7}
    assert store.delete_frames() == {"frames": 2, "bytes": 7}
    assert store.frame_path(sid, 0) is None and store.get(sid)["frames"] == []
    assert store.get(sid) is not None

    store.keep_frames = False
    sid = store.save({"kind": "vision", "stats": {}, "frames": [{"n": 0}]}, frames=[(0, b"x")])
    assert store.frame_path(sid, 0) is None and store.get(sid)["frames"] == []
