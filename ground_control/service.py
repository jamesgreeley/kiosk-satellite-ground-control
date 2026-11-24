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
from .models import (ALIGNER_FILE, ALIGNER_SHA256, ISOLATOR_CPU_FILE, ISOLATOR_CPU_SHA256, ISOLATOR_FILE,
                     ISOLATOR_MODEL, ISOLATOR_SHA256, SPEAKER_MODELS, VAD_FILE, VAD_SHA256, VISION_MODELS, SpeakerModel,
                     default_vision_key, verified_path)
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
        """As a Home Assistant app, its page there (``/app/<slug>``, ``/hassio/ingress/<slug>`` before Home Assistant 2026.2)."""
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
        self.align = bool(saved.get("aligner", True))
        self.short_answers = bool(saved.get("short_answers", True))
        """Take a short reply to a question Home Assistant asked when it
        sounds like one of the people the kiosk answers (``FollowConfig.short_answers``)."""
        """Leave the end of the wake word out of the command, for a kiosk
        with "Talk right after the wake word" on."""
        self._aligner = None
        self._aligner_failed = False
        self._aligner_lock = threading.Lock()
        choices = self.vision_choices()
        chosen_vision, default_vision = saved.get("vision_model"), default_vision_key(cfg.cuda)
        self.vision_model = (chosen_vision if chosen_vision in choices else default_vision if default_vision in choices
                             else choices[0] if choices else default_vision)
        """Vision Match's models: Light or Accurate. Accurate only with CUDA."""
        self._vision = None
        self._vision_error: str | None = None
        self._vision_lock = threading.Lock()
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
                "isolator_on_gpu": self._isolator.on_gpu if self._isolator is not None else None,
                "aligner": self.align, "aligner_missing": self._aligner_missing(),
                "short_answers": self.short_answers,
                "vision_model": self.vision_model,
                "vision_models": {k: VISION_MODELS[k].label for k in self.vision_choices()},
                "vision_loaded": self._vision is not None,
                "vision_on_gpu": self._vision.on_gpu if self._vision is not None else None,
                "vision_max_looks": self._vision.max_looks if self._vision is not None else None,
                "vision_error": self._vision_error}

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
        if "aligner" in values:
            self.align = out["aligner"] = bool(values["aligner"])
        if "short_answers" in values:
            self.short_answers = out["short_answers"] = bool(values["short_answers"])
        if "vision_model" in values:
            key = values["vision_model"]
            if key not in self.vision_choices():
                raise ValueError("vision_model must be one of " + ", ".join(self.vision_choices()))
            if key != self.vision_model:
                self.vision_model = out["vision_model"] = key
                self._switch_vision()
        self._save(**out)

    def _aligner_missing(self) -> str | None:
        """What finding the end of a wake word needs that this install doesn't
        have: ``model`` or ``espeak-ng``. None when it has both."""
        from .aligner import espeak

        if not (self.cfg.models_dir / ALIGNER_FILE).is_file():
            return "model"
        return None if espeak() else "espeak-ng"

    def aligner(self):
        """The Wake Word Aligner, loaded with the first command that needs
        it. None when it is off or can't run here."""
        if not self.align or self._aligner_failed:
            return None
        with self._aligner_lock:
            if self._aligner is None and not self._aligner_failed:
                from .aligner import Aligner

                missing = self._aligner_missing()
                if missing:
                    self._aligner_failed = True
                    _LOG.warning("the Wake Word Aligner can't run: %s", "its model is not in this install"
                                 if missing == "model" else "espeak-ng is not installed")
                    return None
                try:
                    self._aligner = Aligner(verified_path(self.cfg.models_dir, ALIGNER_FILE, ALIGNER_SHA256),
                                            threads=self.cfg.threads)
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self._aligner_failed = True
                    self.status.errors.append(f"Wake Word Aligner failed to load: {err}")
                    _LOG.exception("Wake Word Aligner failed to load")
                    return None
                _LOG.info("Wake Word Aligner %s loaded", self._aligner.version)
            return self._aligner

    def vision_choices(self) -> list[str]:
        """The Vision Match model sets this install can run: their files are
        here, and Accurate needs CUDA."""
        return [k for k, m in VISION_MODELS.items()
                if (self.cfg.cuda or not m.gpu_only)
                and all((self.cfg.models_dir / f).is_file() for f in (m.face_file, m.hand_file))]

    def vision(self):
        """Vision Match's models, loaded with the first look a kiosk asks for.
        None when they can't run here (the reason is in ``settings()``)."""
        with self._vision_lock:
            if self._vision is None and self._vision_error is None:
                from .vision import Vision

                if self.vision_model not in self.vision_choices():
                    self._vision_error = "Vision Match's models are not in this install"
                    _LOG.warning("Vision Match can't run: its models are not in this install")
                    return None
                m = VISION_MODELS[self.vision_model]
                try:
                    self._vision = Vision(self.cfg.models_dir, m, threads=self.cfg.threads, use_cuda=self.cfg.cuda)
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self._vision_error = f"Vision Match's {m.label} models failed to load: {err}"
                    self.status.errors.append(self._vision_error)
                    _LOG.exception("Vision Match's %s models failed to load", m.label)
                    return None
                looks = self._vision.max_looks
                _LOG.info("Vision Match %s models loaded on %s: %s face and %s hand looks a second", m.label,
                          "the GPU" if self._vision.on_gpu else "the CPU", looks["face"], looks["hands"])
            return self._vision

    def _switch_vision(self) -> None:
        """Drop the loaded models. The next look loads the chosen ones."""
        with self._vision_lock:
            old, self._vision, self._vision_error = self._vision, None, None
        if old is not None:
            old.close()
            malloc_trim()
            self.vision()

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
                                             threads=self.cfg.threads, use_cuda=gpu,
                                             keep_memory=self.cfg.keep_gpu_memory)
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
                _LOG.info("Isolation model %s loaded on %s%s", self._isolator.version, where,
                          ", keeping its GPU memory between commands" if self._isolator.keeps_memory else "")
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
                   speaker: str | None = None, isolation: bool = True,
                   wake: dict[str, object] | None = None, asked: bool | None = None) -> "Follow":
        """``isolation``: the kiosk's own switch. Isolation runs when that and
        the service's setting are both on.

        ``wake``: what the kiosk said about the wake word (``wake_details``).
        The end of the wake word is looked for only when the kiosk has "Talk
        right after the wake word" on for this session, which is when it
        reaches the assistant. A follow-up has no wake word of its own.

        ``asked``: the turn is a reply to something Home Assistant asked or a
        follow-up, as the kiosk says. None when it doesn't say."""
        from .follow import Follow, FollowConfig

        find = None
        phrase = (wake or {}).get("wake_word")
        if wake and wake.get("seamless") and phrase and anchor is not None and speaker is None:
            aligner = self.aligner()
            if aligner is not None:
                language = wake.get("language")

                def find(anchor_audio, head):
                    return aligner.find(anchor_audio, head, str(phrase), language if isinstance(language, str) else None)

        cfg = FollowConfig(policy=policy, people=people, short_answers=asked if self.short_answers else False)
        return Follow(cfg, self.model, self.active, self.vad, self.people,
                      anchor=anchor, speaker=speaker, isolator=self.isolator() if isolation else None,
                      wake_word_end=find)

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
        if self._vision is not None:
            self._vision.close()
