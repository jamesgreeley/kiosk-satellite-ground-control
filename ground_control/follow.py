"""Follow mode: pass on only the user's speech, turn by turn.

The kiosk sends everything its microphone hears for a whole interaction:
one Assist command, or a realtime conversation with many turns. The user is
the person who said the wake word. The kiosk sends that second or so of
audio as the anchor, so a TV talking right after the wake word is never
taken for the user. An enrolled person is recognized from it. Anyone else
is followed by the anchor's own voice.

Every stretch of speech is labeled user or other, by a check of the second
of audio centered on it. Only the user's speech goes back to the kiosk, to
forward to the backend in place of the microphone, with long pauses
shortened and other voices left out, so (tv) (user) (tv) (user) (tv)
reaches the backend as (user) (user). A turn starts at the user's first
word and ends once the user has been quiet for ``eos_silence_s``, whatever
the TV does.

Events:

- ``speaker``: who the user is (``name`` None when unknown), ``score``,
  ``by`` (``wake_word`` or ``voice``).
- ``turn_start``: the user started talking. Released audio follows.
- ``audio``: the user's speech to forward, in order.
- ``other_voice``: someone else talked during a turn and was left out.
- ``turn_end``: the user is done. ``t`` is where their last word ended.

With Isolation on, a turn's audio is held while the user talks and
goes out at ``turn_end``, cleaned of the voices under the user's words. The
isolator runs once over the turn's continuous audio, the way it was trained,
and the released stretches are cut from its output.
- ``rejected``: the device only takes enrolled voices and this isn't one.
  Nothing more is released.
- ``stats``: from ``finish``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import SAMPLE_RATE
from .engine import Embedder
from .models import ISOLATOR_MODEL, SpeakerModel
from .people import People
from .vad import FRAME, VadModel

SR = SAMPLE_RATE
MAX_AUDIO_EVENT = SR // 2
"""The most audio in one event: what the kiosk forwards as one message to Home Assistant."""


@dataclass
class FollowConfig:
    policy: str = "anyone"
    """anyone, enrolled (anyone enrolled) or people (only ``people``)."""
    people: tuple[str, ...] = ()
    speech_threshold: float = 0.5
    speech_release: float = 0.35
    check_window_s: float = 1.0
    check_speech_fraction: float = 0.4
    eos_silence_s: float = 0.7
    pad_s: float = 0.2
    """Audio kept before the user's first word, so it isn't clipped."""
    isolation_context_s: float = 2.0
    """How much audio from before the turn the Voice Isolator also gets. With
    the TV heard on its own around the user's words, before them and in the
    second after the last one that the end of the turn waits for, it tells
    better what isn't the user: 2 to 2.5 dB more of the TV goes, measured on
    Echo Show 8 sessions."""
    tail_keep_s: float = 0.25
    """When another voice starts right as the user finishes, with no pause
    between, the check straddling the two is mostly the other voice and
    scores low, so the end of the user's last word would go with it ("outside"
    cut to "outs"). This much of the other voice's labels, right after the
    user's and in the same stretch of speech, still counts as the user's."""
    gap_keep_s: float = 0.15
    """A pause inside a turn is shortened to this, so left out voices leave no long hole.
    Under 0.25 s, Home Assistant's aggressive end of speech: the kiosk sends it only
    this audio, and a breath between two phrases mustn't end the command there. The
    end of the command is Ground Control's (``eos_silence_s``)."""
    identify_after_s: float = 1.0
    reject_after_s: float = 2.0
    min_anchor_speech_s: float = 0.3
    max_turn_s: float = 30.0
    keep_s: float = 120.0
    """Raw and released audio kept for the Sessions page."""
    blip_s: float = 0.15
    """Speech shorter than this, alone, counts as noise outside a turn."""
    anchor_slack: float = 0.08
    """The wake word alone is a short sample of a voice: below the other-voice
    line by this much still counts as the user, until their own speech is in."""
    next_turn_share: float = 0.5
    """A turn after the first starts only on a voice this far from the
    other-voice line toward the recognized one (0.395 with Accurate). The TV
    scoring just over the other-voice line once the user is done must not
    start a turn: in a realtime conversation that stops the answer."""
    own_after_s: float = 2.0
    """This much of the user's speech rebuilds the reference of a voice that
    isn't enrolled, from what they actually said."""
    household_wait_s: float = 2.5
    """Without a wake word, a device that answers anyone waits this much speech
    for an enrolled voice, then follows the voice it heard."""


class Follow:
    def __init__(self, cfg: FollowConfig, model: SpeakerModel, embedder: Embedder,
                 vad: VadModel, people: People, anchor: np.ndarray | None = None,
                 speaker: str | None = None, isolator=None) -> None:
        self.cfg = cfg
        self.isolator = isolator
        """Isolation's model (the Voice Isolator), or None. It needs the Accurate model's voiceprints."""
        self.model = model
        self.embedder = embedder
        self.people = people
        self._vad_model = vad
        self._vad = vad.stream()
        self._ids, self._rows = people.voiceprints(model.key)
        # The voices this device answers. Everyone is still identified, so a
        # session shows who was turned away.
        allowed = [i for i, pid in enumerate(self._ids) if self._allowed(pid)]
        self._house_rows = self._rows[allowed] if len(self._ids) else self._rows
        self._win = int(cfg.check_window_s * SR)
        self._cadence = int(model.cadence_s * SR)
        # A speech frame is decided once the check centered on it is in.
        self._hold = self._win // 2 + self._cadence + int(0.1 * SR)

        # Audio, kept for the last minute so checks can look back.
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0
        self.n = 0
        self._speech: list[bool] = []
        self._checks: list[tuple[int, float]] = []
        self._next_check = 0
        self._run_scores: dict[int, float] = {}

        # Who the user is.
        self._ref: np.ndarray | None = None
        self._ref_kind: str | None = None
        self._anchor_emb: np.ndarray | None = None
        self._anchor_speech = np.zeros(0, dtype=np.float32)
        """The wake word's speech: it counts toward who the user is, with what they say after."""
        self._identify_at = 0.0
        self._heard: list[tuple[int, int]] = []
        """With a likely reference: every stretch of speech since the wake word, whoever it sounds like."""
        self._heard_at = 0.0
        self.speaker: dict[str, Any] | None = None
        self._identified = False
        self._rejected = False

        # Turns and what was released.
        self._decided = 0
        self.in_turn = False
        self.turns = 0
        self._turn_start = 0
        self._last_user_end = 0
        self._last_other_end = 0
        self._heard_user_end = 0
        """The end of the last frame labeled the user's by its check, not by the tail grace."""
        self._prev_label: str | None = None
        self._pending_gap: list[tuple[int, int]] = []
        self._other_reported = False
        self._user_speech: list[tuple[int, int]] = []
        """The current first turn's user speech, for identification."""
        self._turn_user_s = 0.0
        self.released: list[tuple[int, int]] = []
        """Input sample ranges released, in order."""
        self._held: list[tuple[int, int]] = []
        """With Isolation on: this turn's released ranges, sent when it ends."""
        self.isolated_ms = 0
        self.dropped_s = 0.0
        self._raw: list[np.ndarray] = []
        self._raw_n = 0
        self._sent: list[np.ndarray] = []
        self._sent_n = 0
        self.timeline: list[dict[str, Any]] = []
        self.events_kept: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._cpu = 0.0
        self._pending_events: list[dict[str, Any]] = []
        self._household_embs: dict[int, np.ndarray] = {}
        self._household_settled = False
        self.anchor_audio = anchor.astype(np.float32) if anchor is not None else None
        """The wake word as the kiosk sent it, for the session's audio."""
        if anchor is not None and len(anchor):
            self._take_anchor(anchor.astype(np.float32))
        if speaker and not self._identified:
            self._take_speaker(speaker)
        if self._ref is None and len(self._house_rows):
            # No wake word (a manual start): the user is whoever of the
            # household talks, and a voice nobody enrolled, like a TV, is not.
            self._ref_kind = "household"

    # ── the anchor ────────────────────────────────────────────────────────

    def _take_speaker(self, person_id: str) -> None:
        """A follow-up in a conversation: an earlier turn already recognized
        the user. Their voiceprint is who to follow, so a one-word answer like
        "yes" or "never mind" needs no identifying of its own, and another
        voice is still left out."""
        if person_id not in self._ids:
            return
        self._ref, self._ref_kind = self._rows[self._ids.index(person_id)], "person"
        self._identified = True
        self.timeline.append({"t": 0.0, "kind": "conversation", "id": person_id})
        self._set_speaker(person_id, None, "conversation", self._pending_events, 0)
        if not self._allowed(person_id):
            self._reject(self._pending_events, 0, None)

    def _take_anchor(self, anchor: np.ndarray) -> None:
        probs = self._vad_model.stream().feed(anchor)
        parts = [anchor[i * FRAME:(i + 1) * FRAME] for i, p in enumerate(probs) if p >= self.cfg.speech_threshold]
        if len(parts) * FRAME < self.cfg.min_anchor_speech_s * SR:
            return
        self._anchor_speech = np.concatenate(parts)
        emb = self.embedder.embed(self._anchor_speech)
        self._anchor_emb = emb
        self._ref, self._ref_kind = emb, "anchor"
        if len(self._ids):
            scores = self._rows @ emb
            best = int(np.argmax(scores))
            score = float(scores[best])
            self.timeline.append({"t": 0.0, "kind": "anchor", "score": round(score, 3), "id": self._ids[best]})
            if score >= self.model.threshold:
                self._ref, self._ref_kind = self._rows[best], "person"
                self._identified = True
                self._set_speaker(self._ids[best], score, "wake_word", self._pending_events, 0)
                if not self._allowed(self._ids[best]):
                    self._reject(self._pending_events, 0, score)
            elif self._strict:
                self._follow_likely(scores, emb)

    def _follow_likely(self, scores: np.ndarray, emb: np.ndarray) -> None:
        """A device that answers only enrolled or chosen people, and a wake
        word too short to say who: the user is one of them, so follow the
        closest of their voiceprints with the wake word. A short wake word
        alone is a poor reference, and the user's own next sentence can score
        under the other voice threshold against it. Being followed is not
        being answered: identifying the voice still decides that."""
        allowed = [i for i, pid in enumerate(self._ids) if self._allowed(pid)]
        if not allowed:
            return
        likely = max(allowed, key=lambda i: float(scores[i]))
        ref = 2 * self._rows[likely] + emb
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "likely"

    # ── input ─────────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        events, self._pending_events = self._pending_events, []
        audio = audio.astype(np.float32)
        t0 = time.process_time()
        self._buf = np.concatenate([self._buf, audio])
        self.n += len(audio)
        if self._raw_n < self.cfg.keep_s * SR:
            self._raw.append(audio.copy())
            self._raw_n += len(audio)
        cfg = self.cfg
        for p in self._vad.feed(audio):
            was = self._speech[-1] if self._speech else False
            self._speech.append(p >= cfg.speech_threshold or (was and p >= cfg.speech_release))
            self._maybe_check(len(self._speech) * FRAME)
        if self._waiting_for_household(False) and any(self._speech):
            last = len(self._speech) - 1 - self._speech[::-1].index(True)
            if self.n - (last + 1) * FRAME >= cfg.eos_silence_s * SR:
                self._settle_household(self.n, final=True)
        self._decide(events, final=False)
        self._trim()
        self._cpu += time.process_time() - t0
        return self._merge(events)

    def finish(self) -> list[dict[str, Any]]:
        events, self._pending_events = self._pending_events, []
        self._decide(events, final=True)
        if self._ref_kind == "likely" and not self._identified and self._heard:
            self._identify_heard(events, self.n, final=True)
        if self.in_turn:
            self._end_turn(events)
        end = ("rejected" if self._rejected else "other_voice" if self.dropped_s > 0
               else "silence" if self.turns else None)
        events.append({
            "type": "stats",
            "mode": "follow",
            "end": end,
            "audio_s": round(self.n / SR, 2),
            "turns": self.turns,
            "released_s": round(sum(b - a for a, b in self.released) / SR, 2),
            "dropped_s": round(self.dropped_s, 2),
            "speaker": self.speaker,
            "rejected": self._rejected,
            "checks": sum(1 for x in self.timeline if x["kind"] == "check"),
            "cpu_ms": round(self._cpu * 1000),
            **({"isolated_ms": self.isolated_ms} if self.isolator is not None else {}),
            "wall_ms": round((time.monotonic() - self._started) * 1000),
            "model": self.model.key,
        })
        return self._merge(events)

    def _audio(self, a: int, b: int) -> np.ndarray:
        return self._buf[max(a - self._base, 0):max(b - self._base, 0)]

    def _trim(self) -> None:
        if len(self._buf) > 60 * SR:
            drop = len(self._buf) - 30 * SR
            self._buf = self._buf[drop:].copy()
            self._base += drop
            keep_from = self._base
            self._checks = [c for c in self._checks if c[0] >= keep_from - self._win]

    # ── who is talking, frame by frame ────────────────────────────────────

    def _maybe_check(self, end: int) -> None:
        if end < self._next_check or end < self._win:
            return
        start = end - self._win
        f0, f1 = start // FRAME, end // FRAME
        frames = self._speech[f0:f1]
        if not frames or sum(frames) / len(frames) < self.cfg.check_speech_fraction:
            return
        self._next_check = end + self._cadence
        emb = self.embedder.embed(self._audio(start, end))
        if self._ref_kind == "household":
            scores = self._house_rows @ emb
            score = float(np.max(scores))
            self._checks.append((end - self._win // 2, score))
            self.timeline.append({"t": round(end / SR, 3), "kind": "check", "score": round(score, 3),
                                  "low": score < self.model.other_threshold, "turn": self.turns})
            self._household_embs[end - self._win // 2] = emb
            if score >= self.model.other_threshold:
                self._household_settled = True
            self._settle_household(end, final=False)
            return
        if self._ref is None:
            # No anchor: the first second of speech is the user.
            if sum(frames) / len(frames) >= 0.6:
                self._ref, self._ref_kind = emb, "first"
            return
        score = float(self._ref @ emb)
        if self._ref_kind == "anchor":
            score += self.cfg.anchor_slack
        self._checks.append((end - self._win // 2, score))
        self.timeline.append({"t": round(end / SR, 3), "kind": "check", "score": round(score, 3),
                              "low": score < self.model.other_threshold, "turn": self.turns})

    def _settle_household(self, at: int, final: bool) -> None:
        """No wake word, a device that answers anyone and nobody enrolled heard
        yet: once enough speech is in, the voice heard is the user."""
        if self._ref_kind != "household" or self._household_settled or self.cfg.policy != "anyone":
            return
        if not final and sum(self._speech) * FRAME < self.cfg.household_wait_s * SR:
            return
        if self._household_embs:
            ref = np.mean(list(self._household_embs.values()), axis=0)
        else:
            # A word too short for a check: its own audio is the reference.
            frames = [i for i, sp in enumerate(self._speech) if sp and i * FRAME >= self._base]
            if not frames:
                return
            ref = self.embedder.embed(np.concatenate([self._audio(i * FRAME, (i + 1) * FRAME) for i in frames]))
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "first"
        self._checks = [(c, float(self._ref @ self._household_embs[c]) if c in self._household_embs else s)
                        for c, s in self._checks]
        self._run_scores = {}
        self._household_embs = {}
        self.timeline.append({"t": round(at / SR, 3), "kind": "reference", "turn": self.turns})

    def _waiting_for_household(self, final: bool) -> bool:
        return (not final and self._ref_kind == "household" and not self._household_settled
                and self.cfg.policy == "anyone")

    def _start_line(self) -> float | None:
        """The score a voice needs to start a turn after the first, or None inside a turn."""
        if self.in_turn or self.turns == 0:
            return None
        m = self.model
        return m.other_threshold + self.cfg.next_turn_share * (m.threshold - m.other_threshold)

    def _label(self, f: int, final: bool) -> str | None:
        """user or other for speech frame ``f``, or None to wait for its check."""
        fs, fe = f * FRAME, (f + 1) * FRAME
        if not final and self.n - fe < self._hold:
            return None
        if final:
            self._settle_household(self.n, final=True)
        if self._ref is None and self._ref_kind != "household":
            return "user"
        center = (fs + fe) // 2
        near = [c for c in self._checks if abs(c[0] - center) <= self._win * 6 // 10]
        if near:
            _, score = min(near, key=lambda c: abs(c[0] - center))
            if score < self.model.other_threshold and self._waiting_for_household(final):
                return None
            line = self._start_line() or self.model.other_threshold
            return "user" if score >= line else "other"
        # Too little speech around it for a check: a word on its own.
        r0, r1 = self._run(f)
        if (r1 - r0) * FRAME < self.cfg.blip_s * SR:
            return "user" if self.in_turn else "other"
        if r1 >= len(self._speech) and not final:
            return None  # the run is still going
        score = self._run_scores.get(r0)
        if score is None:
            word = self.embedder.embed(self._audio(r0 * FRAME, r1 * FRAME))
            score = float(np.max(self._house_rows @ word)) if self._ref_kind == "household" else float(self._ref @ word)
            if self._ref_kind == "anchor":
                score += self.cfg.anchor_slack
            self._run_scores[r0] = score
            self.timeline.append({"t": round(r1 * FRAME / SR, 3), "kind": "word", "score": round(score, 3),
                                  "turn": self.turns})
        # Lean to keeping a short word: a missed "yes" is worse than a TV word.
        if score < self.model.other_threshold - 0.05 and self._waiting_for_household(final):
            return None
        line = self._start_line() or self.model.other_threshold - 0.05
        return "user" if score >= line else "other"

    def _run(self, f: int) -> tuple[int, int]:
        r0 = f
        while r0 > 0 and self._speech[r0 - 1]:
            r0 -= 1
        r1 = f + 1
        while r1 < len(self._speech) and self._speech[r1]:
            r1 += 1
        return r0, r1

    # ── turns and release ─────────────────────────────────────────────────

    def _decide(self, events: list[dict[str, Any]], final: bool) -> None:
        while self._decided < len(self._speech):
            f = self._decided
            if self._rejected:
                self._decided = len(self._speech)
                return
            if self._speech[f]:
                label = self._label(f, final)
                if label is None:
                    return
            else:
                label = "gap"
            self._apply(f, label, events)
            self._decided += 1

    def _apply(self, f: int, label: str, events: list[dict[str, Any]]) -> None:
        cfg = self.cfg
        fs, fe = f * FRAME, (f + 1) * FRAME
        tail = (label == "other" and self.in_turn and self._prev_label in ("user", "tail")
                and fe - self._heard_user_end <= int(cfg.tail_keep_s * SR))
        self._prev_label = "tail" if tail else label
        if tail:
            # The end of the user's word, not the other voice yet: released, nothing more.
            self._release(fs, fe, events)
            self._last_user_end = fe
            return
        if label == "user":
            self._heard_user_end = fe
        if label in ("user", "other") and self._ref_kind == "likely" and not self._identified:
            self._heard.append((fs, fe))
            self._identify_heard(events, fe, final=False)
            if self._rejected:
                return
        if label == "user":
            if not self.in_turn:
                self.in_turn = True
                self.turns += 1
                self._turn_start = fs
                self._other_reported = False
                self._turn_user_s = 0.0
                events.append({"type": "turn_start", "t": round(fs / SR, 3), "turn": self.turns})
                pad_from = max(fs - int(cfg.pad_s * SR), self._last_other_end, self._decided_floor())
                if pad_from < fs:
                    self._release(pad_from, fs, events)
            elif self._pending_gap:
                self._release_gap(self._pending_gap, int(cfg.gap_keep_s * SR), events)
            self._pending_gap = []
            self._release(fs, fe, events)
            self._last_user_end = fe
            self._other_reported = False
            self._turn_user_s += FRAME / SR
            if self.turns == 1 or not self._identified or self._ref_kind in ("anchor", "first"):
                self._user_speech.append((fs, fe))
                self._maybe_identify(events, fe)
                self._maybe_own(fe)
        elif label == "other":
            self._last_other_end = fe
            if self.in_turn:
                self.dropped_s += FRAME / SR
                if not self._other_reported:
                    self._other_reported = True
                    events.append({"type": "other_voice", "t": round(fs / SR, 3), "turn": self.turns})
        else:
            if self.in_turn:
                self._pending_gap.append((fs, fe))
        if self.in_turn:
            quiet = fe - self._last_user_end
            if quiet >= cfg.eos_silence_s * SR or fe - self._turn_start >= cfg.max_turn_s * SR:
                self._end_turn(events)

    def _release_gap(self, gap: list[tuple[int, int]], keep: int, events: list[dict[str, Any]]) -> None:
        """The quiet frames of a pause, shortened to ``keep`` samples: half from
        each end, so neither neighboring word is clipped. Frames of a voice left
        out in the middle are not in ``gap`` and never go."""
        total = sum(b - a for a, b in gap)
        if total <= keep:
            for a, b in gap:
                self._release(a, b, events)
            return
        head, tail = keep // 2, keep - keep // 2
        out: list[tuple[int, int]] = []
        for a, b in gap:
            if head <= 0:
                break
            take = min(b - a, head)
            out.append((a, a + take))
            head -= take
        back: list[tuple[int, int]] = []
        for a, b in reversed(gap):
            if tail <= 0:
                break
            take = min(b - a, tail)
            back.append((b - take, b))
            tail -= take
        for a, b in out + list(reversed(back)):
            self._release(a, b, events)

    def _decided_floor(self) -> int:
        """Audio already passed over as another voice or an earlier turn is not padding."""
        return self.released[-1][1] if self.released else 0

    def _end_turn(self, events: list[dict[str, Any]]) -> None:
        self.in_turn = False
        self._pending_gap = []
        first = self.turns == 1
        early = first and bool(self._held)
        if early:
            # Isolation keeps the voice of the voiceprint it is given: who the
            # user is gets settled first, so a short command is cleaned with
            # their profile and not the wake word's guess, and a voice turned
            # away sends nothing.
            self._after_first_turn(events)
        self._send_held(events)
        events.append({"type": "turn_end", "t": round(self._last_user_end / SR, 3), "turn": self.turns,
                       "speech_s": round(self._turn_user_s, 2)})
        if first and not early:
            self._after_first_turn(events)

    def _release(self, a: int, b: int, events: list[dict[str, Any]]) -> None:
        if b <= a or self._rejected:
            return
        chunk = self._audio(a, b).copy()
        if not len(chunk):
            return
        if self.released and self.released[-1][1] == a:
            self.released[-1] = (self.released[-1][0], b)
        else:
            self.released.append((a, b))
        if self.isolator is not None and self.in_turn:
            if self._held and self._held[-1][1] == a:
                self._held[-1] = (self._held[-1][0], b)
            else:
                self._held.append((a, b))
            return
        self._send(chunk, a, events)

    def _send_held(self, events: list[dict[str, Any]]) -> None:
        """The turn's held audio, through the Voice Isolator when the user's
        voiceprint is one it knows."""
        held, self._held = self._held, []
        if not held or self._rejected:
            return
        start, end = held[0][0], held[-1][1]
        span = self._audio(start, end).copy()
        if len(span) < end - start:  # older than the kept audio: can't happen within max_turn_s
            return
        voiceprint = self._ref if self.model.key == ISOLATOR_MODEL else None
        if voiceprint is not None:
            t0 = time.perf_counter()
            # The isolator hears everything kept around the turn, up to now.
            ctx = max(self._base, start - int(self.cfg.isolation_context_s * SR))
            around = self.isolator.clean(self._audio(ctx, self.n).copy(), voiceprint)
            span = around[start - ctx:end - ctx]
            ms = round((time.perf_counter() - t0) * 1000)
            self.isolated_ms += ms
            self.timeline.append({"t": round(end / SR, 3), "kind": "isolated", "ms": ms, "turn": self.turns})
        cleaned = np.concatenate([span[a - start:b - start] for a, b in held])
        # In pieces no bigger than the stream's own, so a whole turn doesn't
        # reach Home Assistant as one message.
        for i in range(0, len(cleaned), MAX_AUDIO_EVENT):
            self._send(cleaned[i:i + MAX_AUDIO_EVENT], start, events)

    def _send(self, chunk: np.ndarray, a: int, events: list[dict[str, Any]]) -> None:
        if self._sent_n < self.cfg.keep_s * SR:
            self._sent.append(chunk)
            self._sent_n += len(chunk)
        events.append({"type": "audio", "t": round(a / SR, 3), "audio": chunk})

    # ── who the user is ───────────────────────────────────────────────────

    def _user_audio(self, last_s: float = 4.0) -> np.ndarray:
        parts = [self._audio(a, b) for a, b in self._user_speech]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio[-int(last_s * SR):]

    def _identity_audio(self, last_s: float = 4.0) -> np.ndarray:
        """The user's speech so far with the wake word in front, for a command
        too short on its own to tell who said it."""
        return np.concatenate([self._anchor_speech, self._user_audio(last_s)])[-int(last_s * SR):]

    def _maybe_identify(self, events: list[dict[str, Any]], at: int) -> None:
        if self._identified or not len(self._ids):
            return
        speech_s = sum(b - a for a, b in self._user_speech) / SR
        # Again with every quarter second of new speech, not every frame.
        if speech_s < self.cfg.identify_after_s or speech_s < self._identify_at:
            return
        self._identify_at = speech_s + 0.25
        emb = self.embedder.embed(self._user_audio())
        scores = self._rows @ emb
        best = int(np.argmax(scores))
        score = float(scores[best])
        self.timeline.append({"t": round(at / SR, 3), "kind": "identify", "score": round(score, 3),
                              "id": self._ids[best], "turn": self.turns})
        if score >= self.model.threshold:
            self._identified = True
            self._ref, self._ref_kind = self._rows[best], "person"
            self._set_speaker(self._ids[best], score, "voice", events, at)
            if not self._allowed(self._ids[best]):
                self._reject(events, at, score)
        elif speech_s >= self.cfg.reject_after_s:
            self._identified = True
            self._set_speaker(None, score, None, events, at)
            if self._strict:
                self._reject(events, at, score)

    def _identify_heard(self, events: list[dict[str, Any]], at: int, final: bool) -> None:
        """A strict device whose wake word didn't say who: who is talking,
        from the wake word and everything heard since, whatever the voice
        checks made of it. An allowed person is followed from here on.

        What was heard can include a TV, before or after the user's words, so
        it only ever recognizes someone. Turning a voice away is left to the
        user's own speech, and to the end of the session when nobody's speech
        was ever taken for the user's."""
        heard_s = sum(b - a for a, b in self._heard) / SR
        if self._identified or (not final and heard_s < max(self._heard_at, self.cfg.identify_after_s)):
            return
        self._heard_at = heard_s + 0.25
        heard = [self._audio(a, b) for a, b in self._heard]
        # As for a command: what was said, and the wake word in front only when
        # that is too short. Mixed in sooner, a voice close to someone enrolled
        # can be taken for them.
        if heard_s < self.cfg.identify_after_s:
            heard = [self._anchor_speech, *heard]
        audio = np.concatenate(heard)[-4 * SR:]
        if len(audio) < SR:
            if final and heard_s:
                self._identified = True
                self._set_speaker(None, None, None, events, at)
                self._reject(events, at, None)
            return
        scores = self._rows @ self.embedder.embed(audio)
        best = int(np.argmax(scores))
        score = float(scores[best])
        self.timeline.append({"t": round(at / SR, 3), "kind": "identify", "score": round(score, 3),
                              "id": self._ids[best], "turn": self.turns,
                              **({"wake_word": True} if heard_s < self.cfg.identify_after_s else {})})
        if score >= self.model.threshold:
            self._identified = True
            self._ref, self._ref_kind = self._rows[best], "person"
            self._set_speaker(self._ids[best], score, "voice", events, at)
            if not self._allowed(self._ids[best]):
                self._reject(events, at, score)
        elif final:
            self._identified = True
            self._set_speaker(None, score, None, events, at)
            self._reject(events, at, score)

    def _maybe_own(self, at: int) -> None:
        """A voice that isn't enrolled: once enough of it is in, follow what the
        user actually said rather than the wake word alone."""
        if self._ref_kind not in ("anchor", "first"):
            return
        if sum(b - a for a, b in self._user_speech) < self.cfg.own_after_s * SR:
            return
        own = self.embedder.embed(self._user_audio(4.0))
        ref = 2 * own + (self._anchor_emb if self._anchor_emb is not None else 0)
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "own"
        self.timeline.append({"t": round(at / SR, 3), "kind": "reference", "turn": self.turns})

    def _after_first_turn(self, events: list[dict[str, Any]]) -> None:
        """The first command says more about the user than the wake word did."""
        speech = self._user_audio(6.0)
        with_wake_word = False
        if len(speech) < self.cfg.reject_after_s * SR:
            # A command too short to tell who said it: with the wake word in
            # front it may not be. Only then, since the two together can lift a
            # voice close to someone enrolled over the threshold.
            speech = self._identity_audio(6.0)
            with_wake_word = len(self._anchor_speech) > 0
        if len(speech) < SR:
            if not self._identified and self._strict and len(self._ids):
                self._reject(events, self._last_user_end, None)
            return
        emb = self.embedder.embed(speech)
        if self._ref_kind in ("anchor", "first"):
            ref = 2 * emb + (self._anchor_emb if self._anchor_emb is not None else 0)
            self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
            self._ref_kind = "own"
        if not self._identified:
            self._identified = True
            score = None
            if len(self._ids):
                scores = self._rows @ emb
                best = int(np.argmax(scores))
                score = float(scores[best])
                self.timeline.append({"t": round(self._last_user_end / SR, 3), "kind": "identify",
                                      "score": round(score, 3), "id": self._ids[best], "turn": self.turns,
                                      **({"wake_word": True} if with_wake_word else {})})
                if score >= self.model.threshold:
                    self._ref, self._ref_kind = self._rows[best], "person"
                    self._set_speaker(self._ids[best], score, "voice", events, self._last_user_end)
                    if not self._allowed(self._ids[best]):
                        self._reject(events, self._last_user_end, score)
                    return
            self._set_speaker(None, score, None, events, self._last_user_end)
            if self._strict and len(self._ids):
                self._reject(events, self._last_user_end, score)

    @property
    def _strict(self) -> bool:
        """Only enrolled voices get an answer."""
        return self.cfg.policy in ("enrolled", "people")

    def _allowed(self, person_id: str) -> bool:
        return self.cfg.policy != "people" or person_id in self.cfg.people

    def _reject(self, events: list[dict[str, Any]], at: int, score: float | None) -> None:
        self._rejected = True
        events.append({"type": "rejected", "t": round(at / SR, 3),
                       "score": None if score is None else round(score, 3)})

    def _set_speaker(self, person_id: str | None, score: float | None, by: str | None,
                     events: list[dict[str, Any]], at: int) -> None:
        person = self.people.get(person_id) if person_id else None
        self.speaker = {
            "id": person.id if person else None,
            "name": person.name if person else None,
            "color": person.color if person else None,
            "score": None if score is None else round(score, 3),
            "by": by,
        }
        events.append({"type": "speaker", "t": round(at / SR, 3), **self.speaker})

    # ── output ────────────────────────────────────────────────────────────

    def _merge(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Neighboring audio events become one, so the socket sends fewer
        frames, up to MAX_AUDIO_EVENT samples each."""
        out: list[dict[str, Any]] = []
        for e in events:
            if (e["type"] == "audio" and out and out[-1]["type"] == "audio"
                    and len(out[-1]["audio"]) + len(e["audio"]) <= MAX_AUDIO_EVENT):
                out[-1] = {**out[-1], "audio": np.concatenate([out[-1]["audio"], e["audio"]])}
            else:
                out.append(e)
        return out

    def raw_audio(self) -> np.ndarray:
        return np.concatenate(self._raw) if self._raw else np.zeros(0, dtype=np.float32)

    def cleaned_audio(self) -> np.ndarray:
        """What the backend got: the user's speech, other voices left out."""
        return np.concatenate(self._sent) if self._sent else np.zeros(0, dtype=np.float32)
