"""Face Recognition: who is looking at the kiosk.

A face the detector found is lined up with its five points into a 112x112
crop (``align``) and turned into a unit-length embedding (``FaceEmbedder``).
Each person's faceprint is the normalized sum of the embeddings of their
enrolled crops, per model, the way a voiceprint is built from windows.

On disk, under the data directory:

- ``faces/<id>/<enrollment>/*.jpg``: the aligned crops of each enrollment,
  kept so a model change rebuilds the faceprints without anyone enrolling
  again. 112x112, a few kilobytes each.
- ``faces/<id>/enrollments.json``: when, where and how many crops.
- ``faceprints/<model>.json``: per person, the sum of crop embeddings and
  their count.

``Recognizer`` follows the face in front of one kiosk across looks: one look
of a face 3 m away in dim light is not enough to name someone, so the
embeddings of the last few looks are averaged first.
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .engine import LEAN_CUDA, GpuThread, open_session, run_options

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)
"""Where the eyes, the nose and the mouth corners sit in a 112x112 crop."""
CROP = 112


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 similarity transform (rotation, scale, shift) taking ``src`` points onto ``dst``."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[1, 1] = -1
    r = u @ e @ vt
    var = s.var(0).sum()
    scale = (sig * np.diag(e)).sum() / var if var > 0 else 1.0
    return np.hstack([scale * r, (dm - scale * r @ sm)[:, None]])


def align(rgb: np.ndarray, points_px: Any) -> np.ndarray:
    """The face lined up in a 112x112 RGB crop, from its five points in pixels."""
    m = np.vstack([similarity(np.asarray(points_px, np.float64), TEMPLATE), [0, 0, 1]])
    inv = np.linalg.inv(m)[:2]
    img = Image.fromarray(rgb).transform((CROP, CROP), Image.AFFINE, tuple(inv.ravel()), resample=Image.BILINEAR)
    return np.asarray(img)


def points_px(face: dict[str, Any], width: int, height: int) -> np.ndarray:
    """A detected face's five points, from fractions of the frame to pixels."""
    return np.asarray(face["points"], np.float64) * [width, height]


def jpeg(crop: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class FaceEmbedder:
    """A face crop to a unit-length embedding."""

    def __init__(self, path: Path, threads: int, use_cuda: bool, gpu: GpuThread, keep_memory: bool = False) -> None:
        from .vision import session_options

        self.session = gpu(open_session, path, session_options(threads), use_cuda, LEAN_CUDA)
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self._input = self.session.get_inputs()[0].name
        self._gpu = gpu
        # On a GPU the memory of a run goes back after it, unless KEEP_GPU_MEMORY is on.
        self._after_run = run_options(self.on_gpu and not keep_memory)

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        x = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        v = self._gpu(self.session.run, None, {self._input: x}, self._after_run)[0][0].astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)


# ── people's faces ────────────────────────────────────────────────────────

class Faces:
    """The enrolled faces of the People, and their faceprints per model."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "faces"
        self.prints_dir = data_dir / "faceprints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prints_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._prints: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
        self.on_change: list[Callable[[], None]] = []

    def _changed(self) -> None:
        for callback in self.on_change:
            callback()

    # ── enrollments ───────────────────────────────────────────────────────

    def _records_file(self, person_id: str) -> Path:
        return self.dir / person_id / "enrollments.json"

    def enrollments(self, person_id: str) -> list[dict[str, Any]]:
        f = self._records_file(person_id)
        if not f.is_file():
            return []
        try:
            return json.loads(f.read_text())
        except ValueError:
            return []

    def _write_records(self, person_id: str, records: list[dict[str, Any]]) -> None:
        f = self._records_file(person_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(records, indent=1))
        tmp.replace(f)

    def crops(self, person_id: str, enrollment_id: str | None = None) -> list[Path]:
        base = self.dir / person_id
        if not base.is_dir():
            return []
        dirs = [base / enrollment_id] if enrollment_id else [d for d in base.iterdir() if d.is_dir()]
        return sorted(p for d in dirs if d.is_dir() for p in d.glob("*.jpg"))

    def crop_path(self, person_id: str, enrollment_id: str, name: str) -> Path | None:
        if not all(x.replace("-", "").replace("_", "").replace(".", "").isalnum() for x in (person_id, enrollment_id, name)):
            return None
        p = self.dir / person_id / enrollment_id / name
        return p if p.is_file() and p.suffix == ".jpg" else None

    def has_face(self, person_id: str) -> bool:
        return bool(self.enrollments(person_id))

    def enroll(self, person_id: str, crops: list[np.ndarray], embedders: dict[str, FaceEmbedder],
               source: dict[str, Any] | None = None) -> dict[str, Any]:
        """Keep the crops as a new enrollment and fold them into every loaded model's faceprint."""
        if not crops:
            raise ValueError("no face crops to enroll")
        eid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        d = self.dir / person_id / eid
        d.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crops):
            (d / f"{i:02d}.jpg").write_bytes(jpeg(crop))
        record = {"id": eid, "at": time.time(), "crops": len(crops), **({"source": source} if source else {})}
        with self._lock:
            self._write_records(person_id, [*self.enrollments(person_id), record])
            for key, embedder in embedders.items():
                self._fold(key, person_id, [embedder(c) for c in crops])
        self._changed()
        return record

    def delete_enrollment(self, person_id: str, enrollment_id: str, embedders: dict[str, FaceEmbedder]) -> None:
        records = self.enrollments(person_id)
        if not any(r["id"] == enrollment_id for r in records):
            raise KeyError(enrollment_id)
        d = self.dir / person_id / enrollment_id
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()
        with self._lock:
            self._write_records(person_id, [r for r in records if r["id"] != enrollment_id])
            for key, embedder in embedders.items():
                self.rebuild_person(person_id, key, embedder)
        self._changed()

    def forget(self, person_id: str) -> None:
        with self._lock:
            for key in list(self._prints) + [p.stem for p in self.prints_dir.glob("*.json")]:
                prints = self._model_prints(key)
                if prints.pop(person_id, None) is not None:
                    self._save_prints(key)
        base = self.dir / person_id
        if base.is_dir():
            for f in sorted(base.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            base.rmdir()
        self._changed()

    # ── faceprints ────────────────────────────────────────────────────────

    def _print_file(self, key: str) -> Path:
        return self.prints_dir / f"{key}.json"

    def _model_prints(self, key: str) -> dict[str, tuple[np.ndarray, int]]:
        with self._lock:
            if key not in self._prints:
                f = self._print_file(key)
                data = json.loads(f.read_text()) if f.is_file() else {}
                self._prints[key] = {pid: (np.asarray(v["sum"], np.float32), int(v["n"])) for pid, v in data.items()}
            return self._prints[key]

    def _save_prints(self, key: str) -> None:
        f = self._print_file(key)
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps({pid: {"sum": [round(float(x), 6) for x in s], "n": n}
                                   for pid, (s, n) in self._model_prints(key).items()}))
        tmp.replace(f)

    def _fold(self, key: str, person_id: str, embeddings: list[np.ndarray]) -> None:
        prints = self._model_prints(key)
        s, n = prints.get(person_id, (np.zeros_like(embeddings[0]), 0))
        prints[person_id] = (s + np.sum(embeddings, 0), n + len(embeddings))
        self._save_prints(key)

    def _decode(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    def rebuild_person(self, person_id: str, key: str, embedder: FaceEmbedder) -> None:
        with self._lock:
            prints = self._model_prints(key)
            prints.pop(person_id, None)
            crops = self.crops(person_id)
            if crops:
                prints[person_id] = (np.sum([embedder(self._decode(c)) for c in crops], 0), len(crops))
            self._save_prints(key)

    def rebuild(self, key: str, embedder: FaceEmbedder, people: list[str]) -> None:
        """Every person's faceprint for a model, from the kept crops."""
        with self._lock:
            self._prints[key] = {}
            for pid in people:
                crops = self.crops(pid)
                if crops:
                    self._prints[key][pid] = (np.sum([embedder(self._decode(c)) for c in crops], 0), len(crops))
            self._save_prints(key)

    def needs_rebuild(self, key: str, people: list[str]) -> bool:
        prints = self._model_prints(key)
        return any(self.crops(pid) and pid not in prints for pid in people)

    def faceprints(self, key: str) -> tuple[list[str], np.ndarray]:
        """The enrolled people's ids and their unit-length faceprints, one row each."""
        with self._lock:
            prints = self._model_prints(key)
            ids = [pid for pid, (s, n) in prints.items() if n > 0]
            if not ids:
                return [], np.zeros((0, 0), np.float32)
            m = np.stack([prints[pid][0] for pid in ids]).astype(np.float32)
            return ids, m / np.linalg.norm(m, axis=1, keepdims=True)


# ── who is in front of a kiosk ────────────────────────────────────────────

LOOKS = 5
"""Face looks averaged before a name is given."""
WINDOW_S = 4.0
"""Only looks this recent count toward the average."""
SURE_ABOVE = 0.15
"""A single look this far over the threshold names someone without waiting for a second one."""
MARGIN = 0.05
"""The best person has to beat the next one by this much."""
GONE_S = 3.0
"""With no face looking at the kiosk this long, nobody is in front of it."""
RECHECK_S = 2.0
"""Once someone is named, their face is checked again this often, not on every look."""
UNKNOWN_RECHECK_S = 1.0
"""A face nobody enrolled matches, after the two looks a name needs, is
checked again this often. A guest standing at the kiosk then costs one face
recognized a second instead of one on every look."""
MAX_FACES = 3
"""Faces looking at the kiosk recognized in one look, the largest first."""
SWITCH_AFTER = 2
"""Checks in a row another person has to score higher before they become the person at the kiosk."""
SAME_FACE = 0.6
"""A face in the next look is the same one when its center moved less than this many face widths."""


@dataclass
class Match:
    person_id: str | None
    score: float
    looks: int


@dataclass
class _Track:
    """One face followed from look to look."""

    id: int
    box: list[float]
    last_seen: float
    recent: list[tuple[float, np.ndarray]] = field(default_factory=list)
    match: Match | None = None
    last_check: float = float("-inf")


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


@dataclass
class Recognizer:
    """Who is looking at one kiosk, from its face looks.

    Each face looking at the kiosk is followed from look to look by where it
    is, up to ``MAX_FACES``, and named on its own. ``current`` is the person
    at the kiosk: the named face with the highest score, kept until they are
    gone for ``GONE_S`` or someone else scores higher ``SWITCH_AFTER`` checks
    in a row, so two people side by side don't make it flip. ``people`` is
    everyone named.

    ``wants(now)`` says whether the next look's faces are worth embeddings,
    ``add(faces, ids, prints, now)`` takes them as ``(box, embedding)`` pairs,
    ``seen(boxes, now)`` is called for every look with faces looking at the
    kiosk and ``tick(now)`` lets faces go after ``GONE_S``."""

    threshold: float
    current: Match | None = None
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 0
    _primary: int | None = None
    _challenger: tuple[int | None, int] = (None, 0)

    # ── what the socket asks ──

    def wants(self, now: float) -> bool:
        if not self._tracks:
            return True
        return any(self._due(t, now) for t in self._tracks)

    @staticmethod
    def _due(t: _Track, now: float) -> bool:
        m = t.match
        if m is None:
            return True
        if m.person_id:
            return now - t.last_check >= RECHECK_S
        return m.looks < 2 or now - t.last_check >= UNKNOWN_RECHECK_S

    @property
    def people(self) -> list[Match]:
        """Everyone named, the highest score first, each person once."""
        best: dict[str, Match] = {}
        for t in self._tracks:
            m = t.match
            if m is not None and m.person_id and (m.person_id not in best or m.score > best[m.person_id].score):
                best[m.person_id] = m
        return sorted(best.values(), key=lambda m: -m.score)

    def seen(self, boxes: list[list[float]], now: float) -> None:
        for box in boxes[:MAX_FACES]:
            track = self._track_for(box, now, create=False)
            if track is not None:
                track.box, track.last_seen = box, now

    def add(self, faces: list[tuple[list[float], np.ndarray]], ids: list[str], prints: np.ndarray,
            now: float) -> Match | None:
        taken: set[int] = set()
        for box, embedding in faces[:MAX_FACES]:
            track = self._track_for(box, now, taken=taken)
            taken.add(track.id)
            track.box, track.last_seen, track.last_check = box, now, now
            track.recent = [(t, e) for t, e in track.recent if now - t <= WINDOW_S][-(LOOKS - 1):] + [(now, embedding)]
            track.match = self._name(track, ids, prints)
        self._choose()
        return self.current

    def tick(self, now: float) -> bool:
        """True when someone left and the people named changed."""
        before = [t.id for t in self._tracks]
        self._tracks = [t for t in self._tracks if now - t.last_seen <= GONE_S]
        if [t.id for t in self._tracks] == before:
            return False
        if self._primary not in [t.id for t in self._tracks]:
            self._primary = None
        self._choose()
        return True

    # ── inside ──

    def _track_for(self, box: list[float], now: float, create: bool = True,
                   taken: set[int] | None = None) -> _Track | None:
        cx, cy = _center(box)
        best, dist = None, float("inf")
        for t in self._tracks:
            if taken and t.id in taken:
                continue
            tx, ty = _center(t.box)
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / max(box[2], t.box[2], 1e-6)
            if d < SAME_FACE and d < dist:
                best, dist = t, d
        if best is not None or not create:
            return best
        track = _Track(self._next_id, box, now)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _name(self, track: _Track, ids: list[str], prints: np.ndarray) -> Match:
        looks = len(track.recent)
        if not ids:
            return Match(None, 0.0, looks)
        mean = np.mean([e for _, e in track.recent], 0)
        mean /= max(float(np.linalg.norm(mean)), 1e-9)
        scores = prints @ mean
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        second = float(scores[order[1]]) if len(order) > 1 else -1.0
        enough = looks >= 2 or best >= self.threshold + SURE_ABOVE
        if best >= self.threshold and best - second >= MARGIN and enough:
            return Match(ids[int(order[0])], round(best, 3), looks)
        if track.match is not None and track.match.person_id and looks < 2:
            return track.match
        return Match(None, round(best, 3), looks)

    def _choose(self) -> None:
        """The person at the kiosk: sticky, so a close second doesn't take over on one look."""
        named = sorted((t for t in self._tracks if t.match and t.match.person_id), key=lambda t: -t.match.score)
        primary = next((t for t in self._tracks if t.id == self._primary), None)
        if primary is None or not (primary.match and primary.match.person_id):
            top = named[0] if named else None
            self._primary = top.id if top else None
            self._challenger = (None, 0)
        elif named and named[0].id != primary.id and named[0].match.score > primary.match.score:
            cid, n = self._challenger
            n = n + 1 if cid == named[0].id else 1
            if n >= SWITCH_AFTER:
                self._primary, self._challenger = named[0].id, (None, 0)
            else:
                self._challenger = (named[0].id, n)
        else:
            self._challenger = (None, 0)
        chosen = next((t for t in self._tracks if t.id == self._primary), None)
        if chosen is not None:
            self.current = chosen.match
        elif self._tracks:
            # Faces in view, nobody named yet: the closest one's best guess.
            self.current = max(self._tracks, key=lambda t: t.box[2]).match
        else:
            self.current = None


# ── guided enrollment ─────────────────────────────────────────────────────

STEPS = (("look", 4), ("side_a", 3), ("side_b", 3), ("far", 3))
"""The guided capture: look at the screen, turn a little to one side, then the
other, then step back. Each needs this many good crops."""
FACING_DEG = 15.0
"""Looking at the screen for the first step: the head turned at most this much."""
SIDE_DEG = (12.0, 40.0)
"""Turned to one side: at least this, and at most this, from the camera."""
MIN_WIDTH = 0.06
"""The face's width as a fraction of the frame, for the close steps: about 1.5 m on a kiosk camera."""
FAR_OF_NEAR = 0.7
"""Stepping back: the face at most this much of its width in the first step."""
SAME_PERSON = 0.35
"""A crop has to be this similar to the first step's to be the same person."""
MIN_GAP_S = 0.25
DARK = 45.0
"""A face crop darker than this, on average out of 255, is too dark to enroll."""


class FaceEnrollment:
    """One person's guided capture, fed frames as they come.

    ``feed(rgb, faces, now)`` takes a decoded frame and what the face model
    found in it, and returns a progress event: the step, how many good crops
    it has and needs, and a ``hint`` for the person (``ok``, ``no_face``,
    ``several_faces``, ``closer``, ``farther``, ``look_at_screen``,
    ``turn_more``, ``turn_less``, ``other_side``, ``too_dark``,
    ``other_face``)."""

    def __init__(self, vision: Any) -> None:
        self.vision = vision
        self.step = 0
        self.crops: dict[str, list[np.ndarray]] = {name: [] for name, _ in STEPS}
        self._embs: list[np.ndarray] = []
        self._widths: list[float] = []
        self._side: int | None = None
        self._last = float("-inf")

    @property
    def done(self) -> bool:
        return self.step >= len(STEPS)

    def all_crops(self) -> list[np.ndarray]:
        return [c for name, _ in STEPS for c in self.crops[name]]

    def _event(self, hint: str) -> dict[str, Any]:
        name, need = STEPS[min(self.step, len(STEPS) - 1)]
        return {"type": "progress", "step": name, "step_index": self.step, "steps": len(STEPS),
                "have": len(self.crops[name]), "need": need, "hint": hint, "done": self.done}

    def feed(self, rgb: np.ndarray, faces: list[dict[str, Any]], now: float) -> dict[str, Any]:
        from .vision import head_yaw, turn_of

        if self.done:
            return self._event("ok")
        if not faces:
            return self._event("no_face")
        big = sorted(faces, key=lambda f: -f["box"][2])
        if len(big) > 1 and big[1]["box"][2] > 0.6 * big[0]["box"][2]:
            return self._event("several_faces")
        face = big[0]
        width = face["box"][2]
        h, w = rgb.shape[:2]
        px = points_px(face, w, h)
        yaw = head_yaw(px)
        side = 1 if turn_of(px[0], px[1], px[2]) > 0 else -1
        name, need = STEPS[self.step]
        if name in ("look", "far") and yaw > FACING_DEG:
            return self._event("look_at_screen")
        if name == "look" and width < MIN_WIDTH:
            return self._event("closer")
        if name == "far" and self._widths and width > FAR_OF_NEAR * float(np.median(self._widths)):
            return self._event("farther")
        if name.startswith("side"):
            if width < MIN_WIDTH:
                return self._event("closer")
            if yaw < SIDE_DEG[0]:
                return self._event("turn_more")
            if yaw > SIDE_DEG[1]:
                return self._event("turn_less")
            if name == "side_a":
                self._side = self._side or side
            if (name == "side_a" and side != self._side) or (name == "side_b" and side == self._side):
                return self._event("other_side")
        if now - self._last < MIN_GAP_S:
            return self._event("ok")
        crop = align(rgb, px)
        if float(crop.mean()) < DARK:
            return self._event("too_dark")
        emb = self.vision.recognizer(crop)
        if self._embs:
            ref = np.mean(self._embs[: STEPS[0][1]], 0)
            if float(emb @ (ref / np.linalg.norm(ref))) < SAME_PERSON:
                return self._event("other_face")
        self._last = now
        self.crops[name].append(crop)
        self._embs.append(emb)
        if name == "look":
            self._widths.append(width)
        if len(self.crops[name]) >= need:
            self.step += 1
        return self._event("ok")
