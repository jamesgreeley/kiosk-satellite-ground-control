"""Guided enrollment: a person reads aloud until there is enough speech.

Fed 16 kHz audio as it is recorded, on a kiosk or in the browser. It keeps
only speech, reports progress about four times a second and drops a stretch
of speech that sounds like someone else, so a TV or another person in the
room doesn't end up in the voiceprint.

It also paces the passage the person reads, so the kiosk and the app show
the same sentence. Each sentence fills as it is spoken, over about the time
it takes to read, and the next one comes when it is full. A pause after
most of a sentence completes it too, so a quick reader never repeats one.

Events:

- ``progress``: ``speech_s`` toward ``target_s``, counted as it is heard,
  ``sentence`` (the index in the passage), ``sentence_fill`` (0 to 1) and
  ``level`` (dBFS of the last half second) for a meter.
- ``other_voice``: the last stretch didn't sound like the person and was
  left out.
- ``done``: enough speech. The caller saves ``speech()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import SAMPLE_RATE
from .engine import Embedder
from .models import SpeakerModel
from .vad import FRAME, VadModel

SR = SAMPLE_RATE
REFERENCE_S = 4.0
"""The first speech is taken as the person and is what later stretches are compared with."""
BLOCK_S = 2.0
PROGRESS_S = 0.25
PAUSE_S = 0.5
"""This much silence after speech is a pause."""
FLUSH_S = 1.0
"""At a pause, a stretch at least this long is checked and counted without waiting for a full block."""
COMPLETE_AT = 0.5
"""A pause completes the sentence once this much of it was spoken."""
FRAME_S = FRAME / SR

DEFAULT_PASSAGE = [
    "Turn on the kitchen lights.", "Set a timer for ten minutes.", "What is the weather like tomorrow?",
    "Add milk and eggs to the shopping list.", "Play some quiet music in the living room.",
    "The morning was cold, so we stayed inside and made pancakes.",
    "Later we walked to the park and watched the dogs chase each other across the grass.",
    "In the evening the wind picked up and we lit a fire.", "Turn off all the lights downstairs.",
    "What time is it in Tokyo?", "Remind me to call the dentist on Monday.",
    "How long will it take to drive to the airport?", "Pause the music.", "Good night.",
]
"""Kiosk Satellite's English passage, for a kiosk that doesn't send its own."""


def reading_seconds(sentence: str) -> float:
    """About how long a sentence takes to read aloud, in seconds of speech: 11
    letters a second, or 3.8 Chinese characters, 1.2 s at least. On the slow
    side, so a sentence doesn't move on before the last word."""
    latin = sum(1 for ch in sentence if 0x20 < ord(ch) < 0x2E80)
    wide = sum(1 for ch in sentence if ord(ch) >= 0x2E80)
    return max(1.2, latin / 11 + wide / 3.8)


class GuidedEnrollment:
    def __init__(self, embedder: Embedder, vad: VadModel, model: SpeakerModel, target_s: float = 45.0,
                 passage: list[str] | None = None) -> None:
        self.embedder = embedder
        self.passage = passage or DEFAULT_PASSAGE
        self._reading = [reading_seconds(p) for p in self.passage]
        self.sentence = 0
        self._said = 0.0
        self._silence = 0.0
        self.model = model
        self.target_s = target_s
        self._vad = vad.stream()
        self._pending = np.zeros(0, dtype=np.float32)
        self._kept: list[np.ndarray] = []
        self._block: list[np.ndarray] = []
        self._reference: np.ndarray | None = None
        self._ref_parts: list[np.ndarray] = []
        self._since_progress = 0
        self._recent = np.zeros(0, dtype=np.float32)
        self.dropped_s = 0.0
        self.done = False
        # A stretch scoring under this on the person's own early speech is someone else.
        self.same_threshold = model.threshold - 0.1

    @property
    def speech_s(self) -> float:
        """Speech kept for the voiceprint."""
        return sum(len(k) for k in self._kept) / SR

    @property
    def heard_s(self) -> float:
        """Speech kept, and the stretch waiting to be checked: what progress shows."""
        return self.speech_s + sum(len(b) for b in self._block) / SR

    @property
    def sentence_fill(self) -> float:
        return min(1.0, self._said / self._reading[self.sentence])

    def _next_sentence(self, carry: float = 0.0) -> None:
        self.sentence = (self.sentence + 1) % len(self.passage)
        self._said = carry

    def speech(self) -> np.ndarray:
        return np.concatenate(self._kept) if self._kept else np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.done:
            return events
        self._recent = np.concatenate([self._recent, audio])[-SR // 2:]
        self._pending = np.concatenate([self._pending, audio])
        usable = len(self._pending) - len(self._pending) % FRAME
        frames, self._pending = self._pending[:usable], self._pending[usable:]
        moved = False
        for i, p in enumerate(self._vad.feed(frames)):
            if p >= 0.5:
                self._block.append(frames[i * FRAME:(i + 1) * FRAME])
                self._said += FRAME_S
                self._silence = 0.0
                # Read straight through: the next sentence once this one is full.
                if self._said >= self._reading[self.sentence]:
                    self._next_sentence(self._said - self._reading[self.sentence])
                    moved = True
            else:
                was = self._silence
                self._silence += FRAME_S
                if was < PAUSE_S <= self._silence:
                    # A pause: count what was said, and finish the sentence if most of it was.
                    if sum(len(b) for b in self._block) >= FLUSH_S * SR:
                        self._take_block(events)
                    if self._said >= COMPLETE_AT * self._reading[self.sentence]:
                        self._next_sentence()
                        moved = True
            if sum(len(b) for b in self._block) >= BLOCK_S * SR:
                self._take_block(events)
        self._since_progress += len(audio)
        if moved or self._since_progress >= PROGRESS_S * SR:
            self._since_progress = 0
            rms = float(np.sqrt(np.mean(self._recent ** 2))) if len(self._recent) else 0.0
            events.append({"type": "progress", "speech_s": round(min(self.heard_s, self.target_s), 1),
                           "target_s": self.target_s, "sentence": self.sentence,
                           "sentence_fill": round(self.sentence_fill, 2),
                           "level": round(20 * np.log10(max(rms, 1e-6)), 1)})
        if self.speech_s >= self.target_s:
            self.done = True
            events.append({"type": "done", "speech_s": round(self.speech_s, 1)})
        return events

    def _take_block(self, events: list[dict[str, Any]]) -> None:
        block = np.concatenate(self._block)
        self._block = []
        if self._reference is None:
            self._ref_parts.append(block)
            self._kept.append(block)
            if sum(len(r) for r in self._ref_parts) >= REFERENCE_S * SR:
                self._reference = self.embedder.embed(np.concatenate(self._ref_parts))
            return
        score = float(self._reference @ self.embedder.embed(block))
        if score < self.same_threshold:
            self.dropped_s += len(block) / SR
            events.append({"type": "other_voice", "score": round(score, 3)})
            return
        self._kept.append(block)
