"""Where the wake word ends, so its end can be left out of the command.

With "Talk right after the wake word" on, Kiosk Satellite starts the
command's audio the moment the wake word engine fires. An engine fires before
the word is over ("Alex" of "Alexa"), so the rest of it reaches the assistant
in front of the command: "xa turn on the lights".

The kiosk also sends the wake word itself, as the anchor. The anchor and the
start of the stream together hold the whole word. The Wake Word Aligner, a
small phoneme model, says which phonemes are heard every 20 ms. The wake
word's own phonemes, as espeak-ng says them, are then placed on the frames
where they fit best (forced alignment). The kiosk already heard the wake
word, so the question is where it is, not whether. The command starts at the
quietest point between the word's last phoneme and the next sound.

It works for any wake word of any engine: only the word's text is needed.
Nothing is cut when the word isn't found well enough, so the audio goes on
as it would have without this.

The Wake Word Verifier uses the same model and the same placing to say
whether the wake word was said at all (``Aligner.check``). An engine now and
then fires on speech that is nothing like its wake word, most often a TV
across the room. The word's phonemes then fit the audio poorly wherever they
are placed, and the session is stopped before anything reaches the assistant.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import SAMPLE_RATE
from .engine import malloc_trim, session_options

_LOG = logging.getLogger(__name__)

SR = SAMPLE_RATE
FRAME_S = 0.02
"""The model gives one set of phoneme log-probabilities every 20 ms."""

BEFORE_S = 1.4
"""How much of the anchor, before the stream starts, the model hears."""
AFTER_S = 0.8
"""How much of the stream the model hears. The end of the wake word and the
start of the command are in it."""
MIN_BEFORE_S = 0.5
"""With less of the anchor than this before the stream, the word isn't looked for."""

# A phoneme counts with its look-alikes, so an accent or a slip of a small
# model still lands: any open vowel for a schwa, k with g.
OPEN = set("aɑɐəʌæɜɚ")
FRONT = set("eɛiɪɨᵻy")
BACK = set("oɔɒuʊ")
PAIRS = [set("kɡ"), set("sz"), set("td"), set("pb"), set("fv"), set("ʃʒ"), set("θð"), set("rɹɾʁʀ"), set("mnŋɲ")]

MAX_GAP = 9
"""Frames between two phonemes of the word, at most (0.18 s)."""
SKIP = -6.0
"""The cost of a phoneme of the word that isn't heard at all."""
BETWEEN = -2.5
"""The most a frame between two phonemes can cost."""
LAST_AFTER = 13
"""The engine fires near the word's end: its last phoneme starts this many
frames past the stream's start at most (0.26 s)."""
MIN_FIT = -4.0
"""A word placed with a lower mean log-probability per phoneme isn't trusted."""
VERIFY_AFTER_S = 0.3
"""How much audio after the engine fired the Wake Word Verifier hears. An
engine fires before the word is over, so the check can't run on the audio
before it alone: real "Alexa"s scored -1.9 to -2.9 there and -0.4 to -0.9
with this much more. More than this doesn't help."""
VERIFY_MIN_FIT = -2.2
"""The fit under which the wake word wasn't said. On 114 real wake words from
three kiosks ("Hey Jarvis", "Ok Nova", "Alexa") the median was -1.2 and one
scored under this. Of 672 stretches of other speech from the same kiosks, one
scored over it. Six real sessions an engine fired on with no wake word in
them scored -2.7 to -3.7."""
NEXT_WITHIN_S = 0.25
"""A next sound this close after the word is the command's first."""
RISE_DB = 8.0
"""This much louder than the quietest point since the word: a new sound has started."""
QUIET_DB = 10.0
"""Within this of the room's floor for 80 ms: a pause."""

ESPEAK_VOICES = {"zh": "cmn", "nb": "nb", "no": "nb", "he": "he", "iw": "he"}
"""Language tags espeak-ng names another way."""


def espeak() -> str | None:
    """The espeak-ng program, which says how a wake word is pronounced."""
    return shutil.which("espeak-ng")


@functools.lru_cache(maxsize=256)
def _ipa(text: str, voice: str) -> str:
    program = espeak()
    if program is None:
        return ""
    try:
        # With a line end: espeak-ng 1.51 drops the last letter of text that
        # ends without one ("Alex" for "Alexa").
        done = subprocess.run([program, "-q", "-v", voice, "--ipa", "--stdin"], input=text + "\n",
                              capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def voices(language: str | None) -> list[str]:
    """The espeak-ng voices to try for a kiosk in ``language``: English, which
    most wake words are, then the kiosk's own, since a wake word is said with
    the accent of the person saying it ("Hey Jarvis" in French)."""
    out = ["en-us"]
    tag = (language or "").strip().lower().replace("_", "-")
    if re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]{2,8})*", tag):
        primary = tag.split("-")[0]
        for voice in (tag, ESPEAK_VOICES.get(primary, primary)):
            if voice not in out and not (voice == "en" or voice.startswith("en-")):
                out.append(voice)
    return out


def family(token: str, vocab: dict[str, int]) -> list[int]:
    """The ids of every token that sounds like ``token``."""
    c = token[0]
    group = next((g for g in (OPEN, FRONT, BACK) if c in g), None) or next((g for g in PAIRS if c in g), {c})
    return [i for t, i in vocab.items() if not t.startswith("<") and t[0] in group]


def tokens_of(ipa: str, vocab: dict[str, int]) -> list[str]:
    """A pronunciation in the model's tokens, longest match first."""
    # espeak-ng marks a word it says in another language: "(en)ɐlˈɛksə(fr)".
    ipa = re.sub(r"\([a-z-]+\)", "", ipa)
    for c in "ˈˌ\n -":
        ipa = ipa.replace(c, "")
    keys = sorted((k for k in vocab if not k.startswith("<")), key=len, reverse=True)
    out, i = [], 0
    while i < len(ipa):
        for k in keys:
            if ipa.startswith(k, i):
                out.append(k)
                i += len(k)
                break
        else:
            i += 1
    return out


def align(lp: np.ndarray, emit: np.ndarray, blank: int, near: int, reach: int = 30,
          after: int = LAST_AFTER) -> tuple[int | None, list[int | None], float]:
    """Places a word's phonemes on frames, in order and close to each other.

    ``lp``: (frames, vocab) log-probabilities. ``emit``: (phonemes, frames),
    how well each phoneme of the word fits each frame. The word's last phoneme
    starts from ``reach`` frames before ``near`` to ``after`` frames past it.
    Returns the frame of the last phoneme found, the frame of each phoneme
    (None for one that was skipped) and the mean score per phoneme."""
    n, T = emit.shape
    # Between two phonemes: blank or the end of a phoneme held over frames, capped.
    rest = np.maximum(lp[:, blank], BETWEEN)
    cum = np.concatenate([[0.0], np.cumsum(rest)])
    NEG = -1e9
    V = np.full((n, T), NEG)
    back = np.full((n, T), -1, dtype=np.int32)
    skipped = np.zeros((n, T), dtype=bool)
    V[0] = emit[0]
    idx = np.arange(T)
    for i in range(1, n):
        best = np.full(T, NEG)
        arg = np.full(T, -1, dtype=np.int32)
        for g in range(1, min(MAX_GAP, T - 1) + 1):
            v = np.full(T, NEG)
            # From frame t - g, with the frames between at their cost.
            v[g:] = V[i - 1, :T - g] + (cum[g:T] - cum[1:T - g + 1])
            better = v > best
            best[better] = v[better]
            arg[better] = (idx - g)[better]
        emitted = best + emit[i]
        # Or this phoneme isn't there: stay on the frame of the one before.
        stay = V[i - 1] + SKIP
        use = stay > emitted
        V[i] = np.where(use, stay, emitted)
        back[i] = np.where(use, idx, arg)
        skipped[i] = use
        if i == 1:
            # The word's first phoneme can be the one missing.
            first_skip = np.full(T, SKIP) + emit[1]
            better = first_skip > V[1]
            V[1][better] = first_skip[better]
            back[1][better] = -2
            skipped[1][better] = False
    lo, hi = max(0, near - reach), min(T, near + after + 1)
    if hi <= lo:
        return None, [None] * n, NEG
    end = lo + int(np.argmax(V[n - 1, lo:hi]))
    frames: list[int | None] = [None] * n
    t = end
    for i in range(n - 1, -1, -1):
        if i > 0 and skipped[i, t]:
            continue
        frames[i] = t
        if i > 0:
            if back[i, t] == -2:
                break
            t = int(back[i, t])
    found = [f for f in frames if f is not None]
    return (max(found) if found else None), frames, float(V[n - 1, end]) / n


def command_start(lp: np.ndarray, blank: int, after: int, last_family: list[int], limit: int = 25) -> int | None:
    """The first frame after ``after`` where a new sound starts: not blank and
    not the word's last phoneme still sounding."""
    ids = lp.argmax(-1)
    fam = set(last_family)
    for t in range(after + 1, min(len(lp), after + 1 + limit)):
        if ids[t] != blank and int(ids[t]) not in fam:
            return t
        if ids[t] != blank and t - after > 3:
            return t  # the same vowel again, later: a new word
    return None


def levels(audio: np.ndarray) -> np.ndarray:
    """The level in dB of every 10 ms."""
    n = len(audio) // 160
    return 10 * np.log10((audio[: n * 160].reshape(n, 160).astype(np.float64) ** 2).mean(1) + 1e-12)


def after_word(e: np.ndarray, floor: float, last_end: float, reach: float = 0.6) -> tuple[float, str]:
    """Where the command can start when no next sound was heard close after
    the word's last phoneme. ``e``: the level every 10 ms.

    Either the person paused and the level only falls until the room is
    quiet: the cut goes there. Or the command went on, its first sound was
    missed and the level comes back up before any quiet: the cut goes in the
    dip before it rises, never at a later pause, which would be the one after
    the command's first word."""
    lo = min(len(e) - 1, int(last_end * 100))
    hi = min(len(e) - 8, lo + int(reach * 100))
    if hi <= lo:
        return last_end, "word end"
    # The last phoneme's own peak comes first.
    start = lo + int(np.argmax(e[lo:min(hi, lo + 12)]))
    # Quiet is measured from the room's floor or from the quietest it gets
    # after the word, whichever is louder. A kiosk's noise suppression leaves
    # stretches far quieter than the room ever is. Against those, nothing
    # after the word would count as quiet.
    quiet = max(floor, float(e[start:hi + 8].min())) + QUIET_DB
    low, low_at = e[start], start
    for i in range(start, hi):
        if e[i] < low:
            low, low_at = e[i], i
        if (e[i:i + 8] < quiet).all():
            return (i + 4) / 100, "pause"
        if i - low_at >= 2 and e[i] > low + RISE_DB and e[i - 1] > low + RISE_DB / 2:
            return low_at / 100, "rise"
    return last_end, "word end"


@dataclass
class Placed:
    """A wake word placed in audio."""

    cut_s: float
    """Where the command starts, in seconds from the start of the audio."""
    fit: float
    """The mean log-probability per phoneme."""
    rule: str
    """How the cut was chosen: ``next sound``, ``pause``, ``rise`` or ``word end``."""
    phonemes: list[str]
    frames: list[int | None]


def locate(lp: np.ndarray, audio: np.ndarray, stream_s: float, targets: list[list[str]], vocab: dict[str, int],
           blank: int, min_fit: float = MIN_FIT) -> Placed | None:
    """Where the command starts in ``audio``, whose phoneme log-probabilities
    are ``lp``. None when the wake word isn't found. The stream starts
    ``stream_s`` seconds into ``audio``. ``targets``: the word's
    pronunciations. The one that fits best is used."""
    near = int(round((stream_s - 0.01) / FRAME_S))
    best: tuple[int, list[int | None], float, list[str]] | None = None
    for target in targets:
        if not target:
            continue
        emit = np.stack([np.logaddexp.reduce(lp[:, family(t, vocab)], axis=1) for t in target])
        end, frames, fit = align(lp, emit, blank, near)
        if end is None:
            continue
        placed = [f for f in frames if f is not None]
        # Most of the word is found and most of it is before the stream
        # starts. Without this a poor match lands inside the command.
        if len(placed) * 3 < len(target) * 2 or sum(f <= near + 2 for f in placed) * 2 < len(target):
            continue
        if best is None or fit > best[2]:
            best = (end, frames, fit, target)
    if best is None or best[2] < min_fit:
        return None
    end, frames, fit, target = best
    last_end = (end + 1) * FRAME_S
    nxt = command_start(lp, blank, end, family(target[-1], vocab))
    e = levels(audio)
    if not len(e):
        return None
    if nxt is None or nxt * FRAME_S - last_end > NEXT_WITHIN_S:
        cut, rule = after_word(e, float(np.percentile(e, 10)), last_end)
    else:
        lo = min(len(e) - 1, int(last_end * 100))
        hi = min(len(e), max(lo + 1, int(nxt * FRAME_S * 100)))
        cut, rule = (lo + int(np.argmin(e[lo:hi]))) / 100, "next sound"
    return Placed(cut, fit, rule, target, frames)


def stream_start(anchor: np.ndarray, head: np.ndarray, most_s: float = 1.0, least: int = 80) -> int:
    """Where the stream starts in the anchor, in samples.

    The kiosk's wake clip runs a little past the moment the stream starts, up
    to 0.4 s, so its end and the stream's start are the same audio, sample for
    sample. That is how the two are lined up. With nothing in common, the
    stream starts where the anchor ends."""
    most = min(len(anchor), len(head), int(most_s * SR))
    if most < least:
        return len(anchor)
    probe = min(16, least)
    region = anchor[len(anchor) - most:]
    # The longest end of the anchor that the stream starts with.
    hits = np.flatnonzero((np.lib.stride_tricks.sliding_window_view(region, probe) == head[:probe]).all(axis=1))
    for i in hits:
        k = most - int(i)
        if k < least:
            break
        if np.array_equal(region[i:], head[:k]):
            return len(anchor) - k
    return len(anchor)


@dataclass
class WakeWordEnd:
    """The end of the wake word in a stream."""

    samples: int
    """How much of the stream's start is still the wake word. 0 when the word
    was over before the stream started."""
    fit: float
    rule: str
    phonemes: str
    ms: int
    """How long finding it took."""


@dataclass
class WakeWordCheck:
    """Whether the wake word was said, as the Wake Word Verifier found it."""

    said: bool
    fit: float
    """The mean log-probability per phoneme of the word placed where it fits best."""
    needs: float
    """The fit a wake word that was said reaches."""
    phonemes: str
    ms: int
    """How long the check took."""


class Aligner:
    """The Wake Word Aligner model. Thread safe: onnxruntime sessions can run concurrently."""

    def __init__(self, path: Path, threads: int = 1) -> None:
        so = session_options(threads)
        so.log_severity_level = 3
        # Always on the CPU: one run of two seconds of audio takes 5 to 70 ms
        # there. A GPU session would set up again for every new length.
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.version = meta.get("version", "unknown")
        tokens = json.loads(meta["tokens"])
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.blank = int(meta.get("blank", 0))
        self._input = self.session.get_inputs()[0].name
        malloc_trim()

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        """(frames, vocab) phoneme log-probabilities. Frame ``i`` is centered
        on sample ``320 i + 200``."""
        x = np.ascontiguousarray(audio, dtype=np.float32)[None, :]
        return self.session.run(None, {self._input: x})[0][0]

    def pronunciations(self, phrase: str, language: str | None = None) -> list[list[str]]:
        """The wake word's phonemes, as each voice to try says them."""
        text = " ".join(str(phrase).split())[:80]
        out: list[list[str]] = []
        for voice in voices(language):
            target = tokens_of(_ipa(text, voice), self.vocab) if text else []
            if len(target) >= 2 and target not in out:
                out.append(target)
        return out

    def find(self, anchor: np.ndarray, head: np.ndarray, phrase: str, language: str | None = None) -> WakeWordEnd | None:
        """The end of the wake word ``phrase`` in a stream that starts with
        ``head``, after the wake clip ``anchor``. None when it isn't found."""
        t0 = time.perf_counter()
        targets = self.pronunciations(phrase, language)
        if not targets:
            return None
        start = stream_start(anchor, head)
        a0 = max(0, start - int(BEFORE_S * SR))
        if start - a0 < MIN_BEFORE_S * SR:
            return None
        audio = np.concatenate([anchor[a0:start], head[: int(AFTER_S * SR)]]).astype(np.float32)
        stream_s = (start - a0) / SR
        placed = locate(self.log_probs(audio), audio, stream_s, targets, self.vocab, self.blank)
        if placed is None:
            return None
        samples = int(round((placed.cut_s - stream_s) * SR))
        return WakeWordEnd(samples=min(max(samples, 0), len(head)), fit=round(placed.fit, 2), rule=placed.rule,
                           phonemes="".join(placed.phonemes), ms=round((time.perf_counter() - t0) * 1000))

    def check(self, anchor: np.ndarray, head: np.ndarray, phrase: str, language: str | None = None,
              min_fit: float = VERIFY_MIN_FIT) -> WakeWordCheck | None:
        """Whether the wake word ``phrase`` was said, from the wake clip
        ``anchor`` and ``head``, the start of the stream that follows it.
        None when there is nothing to judge it by: no pronunciation or too
        little of the wake clip."""
        t0 = time.perf_counter()
        targets = self.pronunciations(phrase, language)
        if not targets:
            return None
        start = stream_start(anchor, head)
        a0 = max(0, start - int(BEFORE_S * SR))
        if start - a0 < MIN_BEFORE_S * SR:
            return None
        # What came after the engine fired: the rest of the wake clip, then the stream past it.
        after = np.concatenate([anchor[start:], head[len(anchor) - start:]])[: int(VERIFY_AFTER_S * SR)]
        audio = np.concatenate([anchor[a0:start], after]).astype(np.float32)
        lp = self.log_probs(audio)
        near = min(len(lp) - 1, int(round(((start - a0) / SR - 0.01) / FRAME_S)))
        reach = min(LAST_AFTER, int(len(after) / SR / FRAME_S))
        best: tuple[float, list[str]] | None = None
        for target in targets:
            emit = np.stack([np.logaddexp.reduce(lp[:, family(t, self.vocab)], axis=1) for t in target])
            end, _, fit = align(lp, emit, self.blank, near, after=reach)
            if end is not None and (best is None or fit > best[0]):
                best = (fit, target)
        if best is None:
            return None
        return WakeWordCheck(said=best[0] >= min_fit, fit=round(best[0], 2), needs=min_fit,
                             phonemes="".join(best[1]), ms=round((time.perf_counter() - t0) * 1000))
