"""The running service: models, people and the active speaker model.

Changing the speaker model rebuilds every voiceprint from the kept audio in
the background. Sessions keep using the old model until the rebuild is done.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from .config import Config
from .engine import Embedder, malloc_trim
from .models import (ISOLATOR_CPU_FILE, ISOLATOR_CPU_SHA256, ISOLATOR_FILE, ISOLATOR_MODEL, ISOLATOR_SHA256,
                     SPEAKER_MODELS, VAD_FILE, VAD_SHA256, SpeakerModel, verified_path)
from .hub import Hub
from .people import People
from .session import Session, SessionConfig
from .sessions import SessionStore
from .vad import VadModel

_LOG = logging.getLogger(__name__)


@dataclass
class Status:
    model: str
    rebuilding: str | None = None
    rebuild_done: int = 0
    rebuild_total: int = 0
    sessions: int = 0
    sessions_total: int = 0
    errors: list[str] = field(default_factory=list)


class GroundControl:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.app_path: str | None = None
        """As a Home Assistant app, its page there (``/hassio/ingress/<slug>``)."""
        self.vad = VadModel(verified_path(cfg.models_dir, VAD_FILE, VAD_SHA256))
        self.people = People(cfg.data_dir, self.vad)
        self.sessions = SessionStore(cfg.data_dir, keep_days=cfg.keep_days, keep_audio=cfg.keep_audio)
        self.hub = Hub(cfg.data_dir / "kiosks.json")
        if self.hub.new:
            self.hub.approve_known(self.sessions.list(limit=100000))
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vm")
        self._embedders: dict[str, Embedder] = {}
        self._settings_file = cfg.data_dir / "settings.json"
        saved = self._saved()
        chosen = saved.get("model")
        self.status = Status(model=chosen if chosen in SPEAKER_MODELS else cfg.model)
        self.learn = bool(saved.get("learn", cfg.learn))
        self.sessions.keep_days = int(saved.get("keep_days", cfg.keep_days))
        self.sessions.keep_audio = bool(saved.get("keep_audio", cfg.keep_audio))
        self.isolate = bool(saved.get("isolator", False))
        self._isolator = None
        self._isolator_lock = threading.Lock()
        self.active = self._load(self.status.model)
        if self.people.needs_rebuild(self.status.model):
            _LOG.info("building voiceprints for %s", self.status.model)
            self.people.rebuild(self.status.model, self.active)
        if self.isolate:
            self.isolator()

    def _saved(self) -> dict[str, object]:
        try:
            return json.loads(self._settings_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, **values: object) -> None:
        data = {**self._saved(), **values}
        tmp = self._settings_file.with_suffix(".part")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._settings_file)

    @property
    def model(self) -> SpeakerModel:
        """The active model, with any thresholds set on the Settings page."""
        base = SPEAKER_MODELS[self.status.model]
        tuned = (self._saved().get("thresholds") or {}).get(base.key) or {}
        return replace(base, threshold=float(tuned.get("threshold", base.threshold)),
                       other_threshold=float(tuned.get("other_threshold", base.other_threshold)))

    def settings(self) -> dict[str, object]:
        m, base = self.model, SPEAKER_MODELS[self.status.model]
        return {"model": self.status.model, "rebuilding": self.status.rebuilding,
                "models": {k: v.label for k, v in SPEAKER_MODELS.items()},
                "threshold": m.threshold, "other_threshold": m.other_threshold,
                "default_threshold": base.threshold, "default_other_threshold": base.other_threshold,
                "learn": self.learn, "keep_days": self.sessions.keep_days,
                "keep_audio": self.sessions.keep_audio, "cuda": self.cfg.cuda, "on_gpu": self.active.on_gpu,
                "threads": self.cfg.threads, "port": self.cfg.port,
                "isolator": self.isolate, "isolator_available": self._isolator_files() != [],
                "isolator_model": ISOLATOR_MODEL,
                "isolator_on_gpu": self._isolator.on_gpu if self._isolator is not None else None}

    def update_settings(self, values: dict[str, object]) -> None:
        """Thresholds (for the active model), learning and session keeping. Checked, then saved."""
        saved = self._saved()
        thresholds = dict(saved.get("thresholds") or {})
        tuned = dict(thresholds.get(self.status.model) or {})
        for key in ("threshold", "other_threshold"):
            if key in values:
                v = float(values[key])
                if not 0.0 < v < 1.0:
                    raise ValueError(f"{key} must be between 0 and 1")
                tuned[key] = round(v, 3)
        m = SPEAKER_MODELS[self.status.model]
        if tuned.get("other_threshold", m.other_threshold) >= tuned.get("threshold", m.threshold):
            raise ValueError("the other-voice threshold must be below the threshold")
        thresholds[self.status.model] = tuned
        out: dict[str, object] = {"thresholds": thresholds}
        if "learn" in values:
            self.learn = out["learn"] = bool(values["learn"])
        if "keep_days" in values:
            days = int(values["keep_days"])
            if not 1 <= days <= 90:
                raise ValueError("keep_days must be 1 to 90")
            self.sessions.keep_days = out["keep_days"] = days
        if "keep_audio" in values:
            self.sessions.keep_audio = out["keep_audio"] = bool(values["keep_audio"])
        if "isolator" in values:
            on = bool(values["isolator"])
            if on and not self._isolator_files():
                raise ValueError("the isolation model is not in this image")
            self.isolate = out["isolator"] = on
            if on:
                self.isolator()  # load it now, so the first command doesn't wait
        self._save(**out)

    def _isolator_files(self) -> list[tuple[str, str, bool]]:
        """The isolation models in this image, in the order to try them: the
        full one on a GPU, then the one made for the CPU."""
        files = [(ISOLATOR_FILE, ISOLATOR_SHA256, True)] if self.cfg.cuda else []
        files += [(ISOLATOR_CPU_FILE, ISOLATOR_CPU_SHA256, False), (ISOLATOR_FILE, ISOLATOR_SHA256, False)]
        return [f for f in files if (self.cfg.models_dir / f[0]).is_file()]

    def isolator(self):
        """Isolation's model when Isolation is on and the Accurate model is active, else None."""
        if not self.isolate or self.status.model != ISOLATOR_MODEL:
            return None
        with self._isolator_lock:
            if self._isolator is None:
                from .isolator import Isolator

                try:
                    for file, sha, gpu in self._isolator_files():
                        candidate = Isolator(verified_path(self.cfg.models_dir, file, sha),
                                             threads=self.cfg.threads, use_cuda=gpu)
                        if candidate.on_gpu or not gpu:
                            self._isolator = candidate
                            break
                    if self._isolator is None:
                        raise FileNotFoundError("no isolation model in this image")
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self.isolate = False
                    self.status.errors.append(f"Isolation model failed to load: {err}")
                    _LOG.exception("Isolation model failed to load")
                    return None
                where = "the GPU" if self._isolator.on_gpu else "the CPU"
                if not self._isolator.on_gpu:
                    _LOG.warning("Isolation runs on the CPU: each command waits about a second for it")
                _LOG.info("Isolation model %s loaded on %s", self._isolator.version, where)
            return self._isolator

    def _load(self, key: str) -> Embedder:
        if key not in self._embedders:
            m = SPEAKER_MODELS[key]
            self._embedders[key] = Embedder(verified_path(self.cfg.models_dir, m.file, m.sha256),
                                            threads=self.cfg.threads, use_cuda=self.cfg.cuda)
            where = "the GPU" if self._embedders[key].on_gpu else "the CPU"
            _LOG.info("%s model loaded on %s", m.label, where)
        return self._embedders[key]

    def new_session(self, **options: object) -> Session:
        cfg = SessionConfig(**{k: v for k, v in options.items() if v is not None})
        if not self.learn:
            cfg.learn = False
        return Session(cfg, self.model, self.active, self.vad, self.people)

    def new_follow(self, policy: str = "anyone", anchor=None, people: tuple[str, ...] = (),
                   speaker: str | None = None, isolation: bool = True) -> "Follow":
        """``isolation``: the kiosk's own switch. Isolation runs when that and
        the service's setting are both on."""
        from .follow import Follow, FollowConfig

        return Follow(FollowConfig(policy=policy, people=people), self.model, self.active, self.vad, self.people,
                      anchor=anchor, speaker=speaker, isolator=self.isolator() if isolation else None)

    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    def enroll_models(self) -> dict[str, Embedder]:
        """Every loaded model, so an enrollment lands in each one's voiceprints."""
        return dict(self._embedders)

    async def switch_model(self, key: str) -> None:
        if key not in SPEAKER_MODELS:
            raise ValueError(f"unknown model {key}")
        if key == self.status.model or self.status.rebuilding:
            return
        self.status.rebuilding = key
        try:
            embedder = await self.run(self._load, key)
            await self.run(self.people.rebuild, key, embedder)
            old = self.status.model
            self.active = embedder
            self.status.model = key
            self._save(model=key)
            self._embedders.pop(old, None)
            malloc_trim()
            _LOG.info("speaker model is now %s", key)
        except Exception as err:  # noqa: BLE001 - reported on the status page
            self.status.errors.append(f"model switch to {key} failed: {err}")
            _LOG.exception("model switch failed")
        finally:
            self.status.rebuilding = None

    def identify(self, audio: np.ndarray) -> dict[str, object]:
        """Who is speaking in a clip, outside a session."""
        from .people import speech_only

        speech = speech_only(self.vad, audio)
        ids, rows = self.people.voiceprints(self.model.key)
        if len(speech) < 16000 or not ids:
            return {"id": None, "name": None, "score": None, "by": None}
        scores = rows @ self.active.embed(speech[-64000:])
        i = int(np.argmax(scores))
        if scores[i] < self.model.threshold:
            return {"id": None, "name": None, "score": round(float(scores[i]), 3), "by": None}
        person = self.people.get(ids[i])
        return {"id": person.id, "name": person.name, "color": person.color,
                "score": round(float(scores[i]), 3), "by": "voice"}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
