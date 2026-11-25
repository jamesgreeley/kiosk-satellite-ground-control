"""Vision sessions: one stretch of activity in front of one kiosk.

A session starts with the first look that finds a face or a hand, takes in
every look after it and ends 10 seconds after the last look that found
something, when the kiosk's vision socket closes or after 30 minutes, when a
new one continues it. Looks that find nothing (a pet, a light change, the TV)
open no session: they are only counted per kiosk and day, so the Sessions
list never fills up with empty stretches.

Each session keeps every look's result and times, what the kiosk did with
them (its outcome messages: the screensaver dismissed or postponed, a finger
gesture and the action it ran) and, while Keep frames is on, up to
``MAX_FRAMES`` of the frames as they came in: the first look, the first face
looking at the kiosk, each new finger reading and the last look.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from .sessions import new_id

GAP_S = 10.0
"""A session ends this long after the last look that found a face or a hand."""
MAX_S = 1800.0
"""A longer stretch goes on in a new session."""
MAX_FRAMES = 8


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 1)


def _spread(values: list[float]) -> dict[str, float | None]:
    return {"p50": _pct(values, 0.5), "p95": _pct(values, 0.95), "max": round(max(values), 1) if values else None}


OUTCOMES = ("screensaver_dismissed", "screensaver_postponed", "gesture", "pin_skipped", "lockdown_off", "lockdown_on")
"""What a kiosk reports doing for a vision session: the screensaver dismissed
or postponed, a finger gesture's action, the kiosk or lockdown PIN skipped
for a recognized person and Lockdown Mode turned off or on by faces."""


@dataclass
class _Frame:
    n: int
    reason: str
    t: float
    jpeg: bytes
    rotation: int
    boxes: dict[str, Any]


@dataclass
class VisionSession:
    device_id: str | None
    device_name: str | None
    model: str
    on_gpu: bool
    started: float
    continues: str | None = None
    models: dict[str, str] | None = None
    """Each feature's model when they differ (``model`` is then ``mixed``)."""
    id: str = field(default_factory=new_id)
    looks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    frames: list[_Frame] = field(default_factory=list)
    last: _Frame | None = None
    last_found: float = 0.0
    dropped: int = 0
    failed: int = 0
    _facing_seen: bool = False
    _face_seen: bool = False
    _hand_seen: bool = False
    _reading: int | None = None
    _pending: tuple[int, str, float, _Frame | None] | None = None
    _person: str | None = None
    people: dict[str, str] = field(default_factory=dict)
    """Face Recognition: everyone recognized in the session, id to name."""
    """A new finger reading seen on one hand look: (fingers, gesture, t, its frame)."""

    def t(self, now: float) -> float:
        return round(now - self.started, 2)


def summarize_look(faces: list[dict[str, Any]] | None, hand: dict[str, Any] | None) -> tuple[dict | None, dict | None]:
    """What a look found, kept small: the largest face and the shown hand."""
    face = None
    if faces:
        top = faces[0]
        face = {"count": len(faces), "frontal": any(f.get("frontal") for f in faces),
                "width": top["box"][2], "yaw": top.get("yaw")}
    hand_out = None
    if hand:
        hand_out = {"gesture": hand.get("gesture"), "fingers": hand.get("fingers"), "score": hand.get("score")}
    return face, hand_out


class VisionTracker:
    """The sessions of one kiosk's vision socket.

    ``save(record, frames)`` stores a finished session (``frames``: a list of
    ``(n, jpeg)``), ``keep_frames()`` says whether frames are kept at all and
    ``publish(event)`` tells the web UI's live view. The kiosk's look counts
    for the day add up in ``counts`` until ``take_counts`` hands them over.
    """

    def __init__(self, device_id: str | None, device_name: str | None, model: str, on_gpu: bool,
                 keep_frames, save, publish=None, clock=time.time, models: dict[str, str] | None = None) -> None:
        self.device_id = device_id
        self.device_name = device_name
        self.model = model
        self.models = models
        self.on_gpu = on_gpu
        self._keep_frames = keep_frames
        self._save = save
        self._publish = publish or (lambda event: None)
        self._clock = clock
        self.current: VisionSession | None = None
        self.counts = {"looks": 0, "empty": 0, "sessions": 0}

    def take_counts(self) -> dict[str, int]:
        """The looks, empty looks and sessions counted since the last call."""
        out, self.counts = self.counts, {"looks": 0, "empty": 0, "sessions": 0}
        return out

    # ── looks ─────────────────────────────────────────────────────────────

    def look(self, frame_id: Any, want: tuple[str, ...], faces: list[dict[str, Any]] | None,
             hand: dict[str, Any] | None, ms: dict[str, float], jpeg: bytes, rotation: int = 0,
             encode_ms: float | None = None, rtt_ms: float | None = None, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        found = bool(faces) or hand is not None
        self.counts["looks"] += 1
        s = self.current
        # The kiosk's round trip for its previous frame comes with this one.
        if s is not None and rtt_ms is not None and s.looks:
            s.looks[-1]["rtt"] = round(float(rtt_ms), 1)
        if s is not None and now - s.last_found > GAP_S:
            self.close("idle", now)
            s = None
        if s is not None and now - s.started > MAX_S:
            prev = s.id
            self.close("long", now)
            s = self._start(now, continues=prev) if found else None
        if s is None:
            if not found:
                # Nothing in front of the kiosk and no session going on.
                self.counts["empty"] += 1
                return
            s = self._start(now)
        face, hand_out = summarize_look(faces, hand)
        t = s.t(now)
        entry = {"t": t, "want": "".join(w[0] for w in want), "face": face, "hand": hand_out,
                 "ms": {k: v for k, v in ms.items()}}
        if encode_ms is not None:
            entry["enc"] = round(float(encode_ms), 1)
        s.looks.append(entry)
        if found:
            s.last_found = now
        boxes = {"faces": [[*f["box"], bool(f.get("frontal"))] for f in faces or []],
                 "hand": [*hand["box"], hand.get("fingers")] if hand and hand.get("box") else None}
        frame = _Frame(len(s.looks) - 1, "look", t, jpeg, rotation, boxes) if self._keep_frames() else None
        reasons = self._steps(s, t, face, hand_out, "h" in entry["want"], frame)
        self._frame(s, frame, reasons)
        self._publish({"type": "vision_look", "device_id": self.device_id, "session": s.id, "t": t,
                       "want": entry["want"], "face": face, "hand": hand_out, "ms": entry["ms"]})

    def _start(self, now: float, continues: str | None = None) -> VisionSession:
        s = VisionSession(self.device_id, self.device_name, self.model, self.on_gpu, started=now,
                          continues=continues, last_found=now, models=self.models)
        self.current = s
        self.counts["sessions"] += 1
        self._publish({"type": "vision_start", "device_id": self.device_id, "session": s.id})
        return s

    def _steps(self, s: VisionSession, t: float, face: dict | None, hand: dict | None, hand_look: bool,
               frame: _Frame | None) -> list[str]:
        """The moments worth a step, and a frame: the first face, the first
        face looking at the kiosk, the first hand and each new finger reading.

        A new reading counts once the next hand look reads the same, and its
        step and frame are the first of the two. A hand moving between two
        shapes is blurred and reads as anything for one look (a closing fist
        as 5 fingers), and the kiosk never acts on one look either: its
        gesture needs the count held."""
        reasons = []
        if face and not s._face_seen:
            s._face_seen = True
            s.events.append({"type": "face", "t": t, "frontal": face["frontal"], "width": face["width"]})
        if face and face["frontal"] and not s._facing_seen:
            s._facing_seen = True
            s.events.append({"type": "facing", "t": t, "width": face["width"]})
            reasons.append("facing")
        if hand and not s._hand_seen:
            s._hand_seen = True
            s.events.append({"type": "hand", "t": t})
        if not hand_look:
            return reasons
        fingers = hand["fingers"] if hand else None
        pending = s._pending
        if fingers is None or fingers == s._reading:
            s._pending = None
        elif pending is not None and pending[0] == fingers:
            n, gesture, first_t, first_frame = pending
            s.events.append({"type": "fingers", "t": first_t, "fingers": n, "gesture": gesture})
            s._reading, s._pending = n, None
            if first_frame is not None and len(s.frames) < MAX_FRAMES - 1 \
                    and all(f.n != first_frame.n for f in s.frames):
                first_frame.reason = "reading"
                s.frames.append(first_frame)
        else:
            s._pending = (fingers, hand["gesture"], t, frame)
        return reasons

    def _frame(self, s: VisionSession, frame: _Frame | None, reasons: list[str]) -> None:
        if frame is None:
            return
        n = frame.n
        if reasons:
            frame.reason = reasons[0]
        if n == 0:
            frame.reason = "first"
            s.frames.append(frame)
        elif reasons and len(s.frames) < MAX_FRAMES - 1:
            s.frames.append(frame)
        s.last = frame

    # ── what the kiosk did, and how it went ──────────────────────────────

    def outcome(self, message: dict[str, Any], now: float | None = None) -> None:
        """An outcome message from the kiosk, in the session it belongs to."""
        s = self.current
        if s is None:
            return
        now = self._clock() if now is None else now
        what = message.get("what")
        if what not in OUTCOMES:
            return
        # Postponing repeats while someone stays: the first one says it.
        if what == "screensaver_postponed" and any(e["type"] == what for e in s.events):
            return
        event = {"type": what, "t": s.t(now)}
        if what == "gesture":
            if isinstance(message.get("fingers"), int):
                event["fingers"] = message["fingers"]
            if isinstance(message.get("action"), str):
                event["action"] = message["action"][:120]
        if what in ("pin_skipped", "lockdown_off"):
            if isinstance(message.get("name"), str):
                event["name"] = message["name"][:60]
            if what == "pin_skipped":
                event["mode"] = "lockdown" if message.get("mode") == "lockdown" else "kiosk"
        s.events.append(event)
        # The frame the kiosk acted on: the last look before the outcome.
        if what in ("gesture", "screensaver_dismissed") and s.last is not None and self._keep_frames() \
                and len(s.frames) < MAX_FRAMES - 1:
            kept = next((f for f in s.frames if f.n == s.last.n), None)
            if kept is None:
                s.frames.append(replace(s.last, reason=what))
            elif kept.reason in ("look", "first", "facing", "reading") and what == "gesture":
                kept.reason = what
        self._publish({**event, "type": "vision_outcome", "what": what, "device_id": self.device_id, "session": s.id})

    def person(self, person: dict[str, Any] | None, people: list[dict[str, Any]] | None = None,
               now: float | None = None) -> None:
        """Face Recognition named the person at the kiosk, or nobody, and
        everyone else in view. Each person new to the session, or back at
        the kiosk after someone else, is a ``recognized`` event."""
        s = self.current
        if s is None:
            return
        now = self._clock() if now is None else now
        pid = (person or {}).get("id")
        for p in ([person] if pid else []) + [p for p in people or [] if p.get("id") and p["id"] != pid]:
            first = p["id"] not in s.people
            if first or (p["id"] == pid and pid != s._person):
                s.events.append({"type": "recognized", "t": s.t(now), "person_id": p["id"], "name": p.get("name"),
                                 "score": p.get("score"), "at_kiosk": p["id"] == pid})
                s.people[p["id"]] = p.get("name") or p["id"]
        s._person = pid

    def lost(self, kind: str) -> None:
        """A frame that got no result: ``dropped`` or ``failed``."""
        if self.current is not None:
            if kind == "dropped":
                self.current.dropped += 1
            else:
                self.current.failed += 1

    def tick(self, now: float | None = None) -> None:
        """Ends the session once nothing was found for ``GAP_S``."""
        now = self._clock() if now is None else now
        if self.current is not None and now - self.current.last_found > GAP_S:
            self.close("idle", now)

    # ── the saved session ─────────────────────────────────────────────────

    def close(self, reason: str, now: float | None = None) -> str | None:
        s, self.current = self.current, None
        if s is None:
            return None
        now = self._clock() if now is None else now
        # An idle session ends where its last look that found something was.
        end_t = s.t(min(now, s.last_found + GAP_S)) if reason == "idle" else s.t(now)
        s.events.append({"type": "end", "t": end_t, "reason": reason})
        frames = list(s.frames)
        if s.last is not None and all(f.n != s.last.n for f in frames):
            s.last.reason = "last"
            frames.append(s.last)
        record = self._record(s, end_t, sorted(frames, key=lambda f: f.n))
        self._save(record, [(f.n, f.jpeg) for f in frames])
        self._publish({"type": "vision_end", "device_id": self.device_id, "session": s.id, "reason": reason})
        return s.id

    def _record(self, s: VisionSession, end_t: float, frames: list[_Frame]) -> dict[str, Any]:
        looks = s.looks
        gc = [sum(v for v in lk["ms"].values()) for lk in looks]
        readings: dict[str, int] = {}
        for lk in looks:
            n = (lk.get("hand") or {}).get("fingers")
            if n is not None:
                readings[str(n)] = readings.get(str(n), 0) + 1
        stats = {
            "duration_s": end_t,
            "looks": len(looks),
            "face_looks": sum(1 for lk in looks if "f" in lk["want"]),
            "hand_looks": sum(1 for lk in looks if "h" in lk["want"]),
            "faces": sum(1 for lk in looks if lk.get("face")),
            "facing": sum(1 for lk in looks if (lk.get("face") or {}).get("frontal")),
            "hands": sum(1 for lk in looks if lk.get("hand")),
            "readings": readings,
            "dropped": s.dropped,
            "failed": s.failed,
            "ms": {
                "ground_control": _spread(gc),
                "decode": _spread([lk["ms"]["decode"] for lk in looks if "decode" in lk["ms"]]),
                "face": _spread([lk["ms"]["face"] for lk in looks if "face" in lk["ms"]]),
                "hands": _spread([lk["ms"]["hands"] for lk in looks if "hands" in lk["ms"]]),
                "recognize": _spread([lk["ms"]["recognize"] for lk in looks if "recognize" in lk["ms"]]),
                "encode": _spread([lk["enc"] for lk in looks if "enc" in lk]),
                "round_trip": _spread([lk["rtt"] for lk in looks if "rtt" in lk]),
            },
            "end": next(e["reason"] for e in reversed(s.events) if e["type"] == "end"),
        }
        record = {
            "id": s.id, "kind": "vision", "started": s.started,
            "device_id": s.device_id, "device_name": s.device_name,
            "model": s.model, **({"models": s.models} if s.models else {}), "on_gpu": s.on_gpu, "stats": stats,
            "events": s.events, "looks": looks,
            "frames": [{"n": f.n, "reason": f.reason, "t": f.t, "rotation": f.rotation, **f.boxes}
                       for f in frames],
            "summary": summary(s.events, stats),
        }
        if s.people:
            record["people"] = [{"id": pid, "name": name} for pid, name in s.people.items()]
        if s.continues:
            record["continues"] = s.continues
        return record


def summary(events: list[dict[str, Any]], stats: dict[str, Any]) -> list[str]:
    """A few words for the Sessions list: who was recognized, what the kiosk did, else what was seen."""
    names = list(dict.fromkeys(e["name"] for e in events if e["type"] == "recognized" and e.get("name")))
    out = [f"{name} recognized" for name in names]
    seen = len(out)
    for e in events:
        if e["type"] == "screensaver_dismissed":
            out.append("Screensaver dismissed")
        elif e["type"] == "screensaver_postponed" and "Screensaver postponed" not in out:
            out.append("Screensaver postponed")
        elif e["type"] == "pin_skipped":
            out.append(f"{'Lockdown' if e.get('mode') == 'lockdown' else 'Kiosk'} PIN skipped")
        elif e["type"] == "lockdown_off":
            out.append("Lockdown off")
        elif e["type"] == "lockdown_on":
            out.append("Lockdown on")
        elif e["type"] == "gesture":
            n = e.get("fingers")
            what = f"{n} finger{'s' if n != 1 else ''}" if n is not None else "Gesture"
            out.append(f"{what} → {e['action']}" if e.get("action") else what)
    if len(out) > seen:
        return out
    if stats["facing"] and not names:
        out.append("Looked at the kiosk")
    elif stats["faces"]:
        out.append("Face turned away")
    if stats["readings"]:
        top = max(stats["readings"].items(), key=lambda kv: kv[1])[0]
        out.append(f"Showed {top} finger{'s' if top != '1' else ''}")
    elif stats["hands"]:
        out.append("Hand seen")
    return out
