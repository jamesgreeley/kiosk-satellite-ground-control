"""Saved sessions, for the Sessions page.

Each session keeps its events, the score timeline, its stats and, unless
audio keeping is off, the raw and cleaned audio and the wake word. A
follow-up in a conversation is appended to the session of its first turn,
so a conversation is one session. Sessions older than the retention are
deleted when a new one is saved. Files, under the data directory:
``sessions/<id>.json``, ``<id>.raw.wav``, ``<id>.clean.wav`` and
``<id>.wake.wav``. Ids sort by time.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav, write_wav

MAX_SESSIONS = 2000
TURN_GAP_S = 0.5
"""Silence between the turns of a conversation in its saved audio."""
CONTINUE_WITHIN_S = 900.0
"""A follow-up joins a session from the same kiosk at most this long after it started."""
_SUMMED = ("audio_s", "turns", "released_s", "dropped_s", "checks", "cpu_ms", "wall_ms")
_seq_lock = threading.Lock()
_last = [0]


def new_id() -> str:
    """Microseconds since the epoch (14 hex digits), strictly increasing, then 4 random hex digits."""
    with _seq_lock:
        us = max(time.time_ns() // 1000, _last[0] + 1)
        _last[0] = us
    return f"{us:014x}{os.urandom(2).hex()}"


def id_time(sid: str) -> float | None:
    """When a session id was made, in seconds."""
    try:
        return int(sid[:14], 16) / 1e6
    except ValueError:
        return None


class SessionStore:
    def __init__(self, data_dir: Path, keep_days: int = 7, keep_audio: bool = True) -> None:
        self.dir = data_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_days = keep_days
        self.keep_audio = keep_audio
        self._lock = threading.Lock()

    def save(self, record: dict[str, Any], raw: np.ndarray | None = None, clean: np.ndarray | None = None,
             wake: np.ndarray | None = None) -> str:
        sid = record.setdefault("id", new_id())
        record["has_audio"] = bool(self.keep_audio and raw is not None)
        if record["has_audio"]:
            write_wav(self.dir / f"{sid}.raw.wav", raw)
            if clean is not None:
                write_wav(self.dir / f"{sid}.clean.wav", clean)
        if self.keep_audio and wake is not None and len(wake):
            write_wav(self.dir / f"{sid}.wake.wav", wake)
            record["wake_s"] = round(len(wake) / 16000, 2)
        tmp = self.dir / f"{sid}.part"
        tmp.write_text(json.dumps(record))
        tmp.replace(self.dir / f"{sid}.json")
        self.prune()
        return sid

    def continuing(self, sid: str | None, device_id: str | None) -> bool:
        """Whether a follow-up from ``device_id`` can join session ``sid``."""
        prev = self.get(sid) if isinstance(sid, str) else None
        return bool(prev and prev.get("mode") == "follow" and prev.get("device_id") == device_id
                    and time.time() - (prev.get("started") or 0) < CONTINUE_WITHIN_S)

    def append(self, sid: str, record: dict[str, Any], raw: np.ndarray | None = None,
               clean: np.ndarray | None = None) -> str:
        """A follow-up turn, after the turns already saved under ``sid``: its
        audio after theirs, its times moved to match and the stats added up."""
        if self.get(sid) is None:
            return self.save({**record, "id": sid}, raw, clean)
        with self._lock:
            prev = self.get(sid) or {}
            raw_prev = self.audio_path(sid, "raw")
            heard = read_wav(raw_prev) if raw_prev else None
            before = len(heard) / 16000 if heard is not None else float((prev.get("stats") or {}).get("audio_s") or 0)
            offset = round(before + TURN_GAP_S, 3)
            turns = int((prev.get("stats") or {}).get("turns") or 0)

            def moved(item: dict[str, Any]) -> dict[str, Any]:
                out = dict(item)
                for key in ("t", "cut"):
                    if isinstance(out.get(key), (int, float)):
                        out[key] = round(out[key] + offset, 3)
                if isinstance(out.get("turn"), int) and out["turn"]:
                    out["turn"] += turns
                return out

            prev["events"] = prev.get("events", []) + [moved(e) for e in record.get("events", [])]
            prev["timeline"] = prev.get("timeline", []) + [moved(t) for t in record.get("timeline", [])]
            prev["released"] = prev.get("released", []) + [[round(a + offset, 3), round(b + offset, 3)]
                                                            for a, b in record.get("released", [])]
            prev["followups"] = prev.get("followups", []) + [offset]
            old, new = prev.get("stats") or {}, record.get("stats") or {}
            stats = {**old, **{k: new[k] for k in ("end", "rejected", "model") if k in new}}
            for key in _SUMMED:
                stats[key] = round(float(old.get(key) or 0) + float(new.get(key) or 0), 3)
            stats["audio_s"] = round(stats["audio_s"] + TURN_GAP_S, 3)
            stats["turns"] = int(stats["turns"])
            # Who the conversation is with: the first turn that recognized
            # them, since a follow-up only carries that on.
            if not (old.get("speaker") or {}).get("name"):
                stats["speaker"] = new.get("speaker")
            prev["stats"] = stats
            gap = np.zeros(int(TURN_GAP_S * 16000), dtype=np.float32)
            if self.keep_audio and heard is not None and raw is not None:
                write_wav(self.dir / f"{sid}.raw.wav", np.concatenate([heard, gap, raw]))
                clean_prev = self.audio_path(sid, "clean")
                if clean is not None:
                    parts = [read_wav(clean_prev), gap] if clean_prev else []
                    write_wav(self.dir / f"{sid}.clean.wav", np.concatenate([*parts, clean]))
            tmp = self.dir / f"{sid}.part"
            tmp.write_text(json.dumps(prev))
            tmp.replace(self.dir / f"{sid}.json")
            return sid

    def _ids(self) -> list[str]:
        return sorted((p.stem for p in self.dir.glob("*.json")), reverse=True)

    def list(self, limit: int = 100, device_id: str | None = None) -> list[dict[str, Any]]:
        out = []
        for sid in self._ids():
            r = self.get(sid)
            if r is None or (device_id and r.get("device_id") != device_id):
                continue
            stats = r.get("stats") or {}
            out.append({
                "id": sid, "started": r.get("started"), "device_id": r.get("device_id"),
                "device_name": r.get("device_name"), "mode": r.get("mode"), "policy": r.get("policy"),
                "speaker": stats.get("speaker"), "end": stats.get("end"),
                "audio_s": stats.get("audio_s"), "dropped_s": stats.get("dropped_s"),
                "has_audio": r.get("has_audio", False),
            })
            if len(out) >= limit:
                break
        return out

    def get(self, sid: str) -> dict[str, Any] | None:
        if not sid.isalnum():
            return None
        try:
            return json.loads((self.dir / f"{sid}.json").read_text())
        except (OSError, ValueError):
            return None

    def audio_path(self, sid: str, kind: str) -> Path | None:
        if not sid.isalnum() or kind not in ("raw", "clean", "wake"):
            return None
        p = self.dir / f"{sid}.{kind}.wav"
        return p if p.is_file() else None

    def delete(self, sid: str) -> bool:
        if not sid.isalnum():
            return False
        found = False
        for suffix in (".json", ".raw.wav", ".clean.wav", ".wake.wav"):
            p = self.dir / f"{sid}{suffix}"
            if p.exists():
                p.unlink()
                found = True
        return found

    def prune(self) -> None:
        with self._lock:
            cutoff = time.time() - self.keep_days * 86400
            for i, sid in enumerate(self._ids()):
                made = id_time(sid)
                if (made is not None and made < cutoff) or i >= MAX_SESSIONS:
                    self.delete(sid)
