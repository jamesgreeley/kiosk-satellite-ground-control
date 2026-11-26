"""Saved sessions, for the Sessions page.

Two kinds: ``voice`` (Voice Match: one command, or a conversation with its
follow-ups) and ``vision`` (Vision Match: one stretch of activity in front of
a kiosk, see ground_control.vision_sessions).

Sessions live in an SQLite database, ``sessions.db`` in the data directory.
Each row keeps the fields the list shows and filters on as columns, the
list's summary of the session and the whole record, which is only read when
one session opens. The files stay next to it in ``sessions/``: a voice
session's audio (``<id>.raw.wav``, ``<id>.clean.wav``, ``<id>.wake.wav``) and
a vision session's frames (``<id>.f<n>.jpg``). Ids sort by time.

Each kind is kept for the retention and has its own cap, so a busy week of
vision sessions never pushes voice sessions out. Sessions saved as JSON files
by earlier versions are imported once, on the first start, and the files are
moved to ``sessions/imported-json``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav, write_wav

MAX_SESSIONS = {"voice": 2000, "vision": 3000}
TURN_GAP_S = 0.5
"""Silence between the turns of a conversation in its saved audio."""
CONTINUE_WITHIN_S = 900.0
"""A follow-up joins a session from the same kiosk at most this long after it started."""
_SUMMED = ("audio_s", "turns", "released_s", "dropped_s", "checks", "cpu_ms", "wall_ms")
_AUDIO = ("raw", "clean", "wake")
_seq_lock = threading.Lock()
_last = [0]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'voice',
    started REAL,
    saved REAL,
    device_id TEXT,
    device_name TEXT,
    speaker_id TEXT,
    speaker_name TEXT,
    outcome TEXT,
    listing TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_kind ON sessions (kind, id);
CREATE INDEX IF NOT EXISTS sessions_device ON sessions (device_id, id);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions (started);
CREATE TABLE IF NOT EXISTS vision_days (
    device_id TEXT NOT NULL,
    day TEXT NOT NULL,
    looks INTEGER NOT NULL DEFAULT 0,
    empty INTEGER NOT NULL DEFAULT 0,
    sessions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (device_id, day)
);
"""
LOOK_DAYS = 31
"""How many days of each kiosk's look counts are kept."""


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


def listing(record: dict[str, Any]) -> dict[str, Any]:
    """What the Sessions list shows of a session, without its details."""
    stats = record.get("stats") or {}
    out = {"id": record["id"], "kind": record.get("kind", "voice"), "started": record.get("started"),
           "device_id": record.get("device_id"), "device_name": record.get("device_name")}
    if out["kind"] == "vision":
        out.update({"summary": record.get("summary") or [], "end": stats.get("end"),
                    "duration_s": stats.get("duration_s"), "looks": stats.get("looks"),
                    "round_trip_ms": ((stats.get("ms") or {}).get("round_trip") or {}).get("p50"),
                    "ground_control_ms": ((stats.get("ms") or {}).get("ground_control") or {}).get("p50"),
                    "frames": len(record.get("frames") or []), "model": record.get("model")})
        return out
    out.update({"mode": record.get("mode"), "policy": record.get("policy"),
                "speaker": stats.get("speaker"), "end": stats.get("end"),
                "audio_s": stats.get("audio_s"), "dropped_s": stats.get("dropped_s"),
                "has_audio": record.get("has_audio", False)})
    return out


class SessionStore:
    def __init__(self, data_dir: Path, keep_days: int = 7, keep_audio: bool = True,
                 keep_frames: bool = True) -> None:
        self.dir = data_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_days = keep_days
        self.keep_audio = keep_audio
        self.keep_frames = keep_frames
        self._lock = threading.RLock()
        self._db = sqlite3.connect(data_dir / "sessions.db", check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._import_json()
        self.prune()

    # ── writing ───────────────────────────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        stats = record.get("stats") or {}
        speaker = stats.get("speaker") or {}
        record["saved_at"] = time.time()
        kind = record.get("kind", "voice")
        outcome = stats.get("end")
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO sessions (id, kind, started, saved, device_id, device_name, speaker_id,"
                " speaker_name, outcome, listing, record) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (record["id"], kind, record.get("started") or id_time(record["id"]), record["saved_at"], record.get("device_id"),
                 record.get("device_name"), speaker.get("id"), speaker.get("name"), outcome,
                 json.dumps(listing(record)), json.dumps(record)))

    def save(self, record: dict[str, Any], raw: np.ndarray | None = None, clean: np.ndarray | None = None,
             wake: np.ndarray | None = None, frames: list[tuple[int, bytes]] | None = None) -> str:
        sid = record.setdefault("id", new_id())
        if record.get("kind") == "vision":
            kept = []
            if self.keep_frames:
                for n, jpeg in frames or []:
                    (self.dir / f"{sid}.f{int(n)}.jpg").write_bytes(jpeg)
                    kept.append(int(n))
            record["frames"] = [f for f in record.get("frames") or [] if f.get("n") in kept]
        else:
            record["has_audio"] = bool(self.keep_audio and raw is not None)
            if record["has_audio"]:
                write_wav(self.dir / f"{sid}.raw.wav", raw)
                if clean is not None:
                    write_wav(self.dir / f"{sid}.clean.wav", clean)
            if self.keep_audio and wake is not None and len(wake):
                write_wav(self.dir / f"{sid}.wake.wav", wake)
                record["wake_s"] = round(len(wake) / 16000, 2)
        self._write(record)
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
            self._write(prev)
            return sid

    # ── reading ───────────────────────────────────────────────────────────

    def _ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT id FROM sessions ORDER BY id DESC")]

    def _where(self, kind: str | None, device_id: str | None, device: str | None, person: str | None,
               show: str | None, q: str | None, since: float | None) -> tuple[str, list[Any]]:
        sql, args = [], []
        if kind:
            sql.append("kind = ?")
            args.append(kind)
        if device_id:
            sql.append("device_id = ?")
            args.append(device_id)
        if device:
            # A kiosk by the name the list shows, or its id when it has none.
            sql.append("COALESCE(NULLIF(device_name, ''), device_id) = ?")
            args.append(device)
        if person == "unknown":
            sql.append("kind = 'voice' AND speaker_id IS NULL")
        elif person:
            sql.append("speaker_id = ?")
            args.append(person)
        if show == "cut":
            sql.append("outcome = 'other_voice'")
        elif show == "rej":
            sql.append("outcome = 'rejected'")
        if q:
            sql.append("(COALESCE(device_name, device_id, '') || ' ' || COALESCE(speaker_name, "
                       "CASE WHEN kind = 'voice' THEN 'unknown' ELSE 'vision' END)) LIKE ?")
            args.append(f"%{q}%")
        if since is not None:
            sql.append("started >= ?")
            args.append(since)
        return (" WHERE " + " AND ".join(sql)) if sql else "", args

    def list(self, limit: int = 100, device_id: str | None = None, *, offset: int = 0, kind: str | None = None,
             device: str | None = None, person: str | None = None, show: str | None = None,
             q: str | None = None, since: float | None = None) -> list[dict[str, Any]]:
        where, args = self._where(kind, device_id, device, person, show, q, since)
        with self._lock:
            rows = self._db.execute(f"SELECT listing FROM sessions{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                                    (*args, int(limit), int(offset))).fetchall()
        return [json.loads(r[0]) for r in rows]

    def count(self, **filters: Any) -> int:
        where, args = self._where(filters.get("kind"), filters.get("device_id"), filters.get("device"),
                                  filters.get("person"), filters.get("show"), filters.get("q"), filters.get("since"))
        with self._lock:
            return self._db.execute(f"SELECT COUNT(*) FROM sessions{where}", args).fetchone()[0]

    def kiosks(self) -> list[str]:
        """Every kiosk with a session, by the name the list shows."""
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT DISTINCT COALESCE(NULLIF(device_name, ''), device_id) AS k FROM sessions"
                " WHERE k IS NOT NULL ORDER BY k")]

    def counts(self, since: float) -> dict[str, Any]:
        """Sessions since ``since`` by kind and outcome, for the Overview."""
        with self._lock:
            rows = self._db.execute("SELECT kind, outcome, COUNT(*) FROM sessions WHERE started >= ?"
                                    " GROUP BY kind, outcome", (since,)).fetchall()
        out: dict[str, Any] = {"voice": {"total": 0}, "vision": {"total": 0}}
        for kind, outcome, n in rows:
            k = out.setdefault(kind, {"total": 0})
            k["total"] += n
            if outcome:
                k[outcome] = k.get(outcome, 0) + n
        return out

    def get(self, sid: str) -> dict[str, Any] | None:
        if not isinstance(sid, str) or not sid.isalnum():
            return None
        with self._lock:
            row = self._db.execute("SELECT record FROM sessions WHERE id = ?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def saved_at(self, sid: str) -> float | None:
        """When a session was last written: its last turn, or when a vision session ended."""
        with self._lock:
            row = self._db.execute("SELECT saved FROM sessions WHERE id = ?", (sid,)).fetchone()
        return row[0] if row else None

    def audio_path(self, sid: str, kind: str) -> Path | None:
        if not sid.isalnum() or kind not in _AUDIO:
            return None
        p = self.dir / f"{sid}.{kind}.wav"
        return p if p.is_file() else None

    def frame_path(self, sid: str, n: Any) -> Path | None:
        if not sid.isalnum() or not str(n).isdigit():
            return None
        p = self.dir / f"{sid}.f{int(n)}.jpg"
        return p if p.is_file() else None

    # ── deleting ──────────────────────────────────────────────────────────

    def _files(self, sid: str) -> list[Path]:
        return [*(self.dir / f"{sid}.{k}.wav" for k in _AUDIO), *self.dir.glob(f"{sid}.f*.jpg")]

    def delete(self, sid: str) -> bool:
        if not isinstance(sid, str) or not sid.isalnum():
            return False
        with self._lock:
            found = self._db.execute("DELETE FROM sessions WHERE id = ?", (sid,)).rowcount > 0
        for p in self._files(sid):
            if p.exists():
                p.unlink()
                found = True
        return found

    def frames_stats(self) -> dict[str, int]:
        files = list(self.dir.glob("*.f*.jpg"))
        return {"frames": len(files), "bytes": sum(p.stat().st_size for p in files)}

    def delete_frames(self) -> dict[str, int]:
        """Every vision session's frames, at once. The sessions stay."""
        gone = self.frames_stats()
        for p in self.dir.glob("*.f*.jpg"):
            p.unlink(missing_ok=True)
        with self._lock:
            rows = self._db.execute("SELECT record FROM sessions WHERE kind = 'vision'").fetchall()
            for (raw,) in rows:
                record = json.loads(raw)
                if record.get("frames"):
                    record["frames"] = []
                    self._write(record)
        return gone

    # ── each kiosk's looks, by day ────────────────────────────────────────

    def add_looks(self, device_id: str | None, counts: dict[str, int], day: str | None = None) -> None:
        """Looks, empty looks and vision sessions for the kiosk's day (local time)."""
        if not device_id or not any(counts.values()):
            return
        day = day or time.strftime("%Y-%m-%d")
        with self._lock:
            self._db.execute(
                "INSERT INTO vision_days (device_id, day, looks, empty, sessions) VALUES (?,?,?,?,?)"
                " ON CONFLICT (device_id, day) DO UPDATE SET looks = looks + excluded.looks,"
                " empty = empty + excluded.empty, sessions = sessions + excluded.sessions",
                (device_id, day, int(counts.get("looks", 0)), int(counts.get("empty", 0)),
                 int(counts.get("sessions", 0))))

    def looks(self, days: int = 7, device_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """Each kiosk's look counts for its last ``days`` days, newest first."""
        since = time.strftime("%Y-%m-%d", time.localtime(time.time() - (days - 1) * 86400))
        sql, args = "SELECT device_id, day, looks, empty, sessions FROM vision_days WHERE day >= ?", [since]
        if device_id:
            sql += " AND device_id = ?"
            args.append(device_id)
        out: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for d, day, looks, empty, sessions in self._db.execute(sql + " ORDER BY day DESC", args):
                out.setdefault(d, []).append({"day": day, "looks": looks, "empty": empty, "sessions": sessions})
        return out

    def prune(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM vision_days WHERE day < ?",
                             (time.strftime("%Y-%m-%d", time.localtime(time.time() - LOOK_DAYS * 86400)),))
            cutoff = time.time() - self.keep_days * 86400
            # By the time in the id, like the list's order.
            old = [sid for (sid,) in self._db.execute("SELECT id FROM sessions")
                   if (made := id_time(sid)) is not None and made < cutoff]
            for kind, cap in MAX_SESSIONS.items():
                old += [r[0] for r in self._db.execute(
                    "SELECT id FROM sessions WHERE kind = ? ORDER BY id DESC LIMIT -1 OFFSET ?", (kind, cap))]
            for sid in set(old):
                self.delete(sid)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── sessions from earlier versions ────────────────────────────────────

    def _import_json(self) -> None:
        """Sessions saved as ``<id>.json`` files, into the database once."""
        files = sorted(self.dir.glob("*.json"))
        if not files:
            return
        moved = self.dir / "imported-json"
        moved.mkdir(exist_ok=True)
        for f in files:
            try:
                record = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            record.setdefault("id", f.stem)
            if record["id"] != f.stem or not f.stem.isalnum():
                continue
            with self._lock:
                exists = self._db.execute("SELECT 1 FROM sessions WHERE id = ?", (f.stem,)).fetchone()
            if not exists:
                self._write(record)
            f.replace(moved / f.name)
