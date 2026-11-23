"""The end of the wake word: lining the stream up with the wake clip, placing
the word's phonemes and choosing where the command starts.

The placing is tested on made-up phoneme probabilities, so it needs no
speech. The model itself is tested on a wake word said by espeak-ng, which
skips where espeak-ng isn't installed.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from ground_control import aligner
from ground_control.aligner import Aligner, locate, stream_start, tokens_of, voices
from ground_control.audio import float_to_pcm16, pcm16_to_float, read_wav
from ground_control.models import ALIGNER_FILE, ALIGNER_SHA256, verified_path

from .conftest import MODELS, SR

VOCAB = {t: i for i, t in enumerate(["<pad>", "<s>", "ɐ", "l", "ɛ", "k", "s", "ə", "t", "ɜː", "n", "ɑː", "a", "ɡ"])}
BLANK = 0
ALEXA = ["ɐ", "l", "ɛ", "k", "s", "ə"]
STREAM_S = 1.4


def probs(frames: int, heard: dict[int, str]) -> np.ndarray:
    """Log-probabilities with a sure blank everywhere but the frames in ``heard``."""
    p = np.full((frames, len(VOCAB)), 0.001)
    p[:, BLANK] = 1.0
    for frame, token in heard.items():
        p[frame, BLANK] = 0.001
        p[frame, VOCAB[token]] = 1.0
    return np.log(p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def sound(seconds: float, loud: list[tuple[float, float]], level: float = 0.2) -> np.ndarray:
    """Room noise with something loud in each of the ``loud`` stretches."""
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(seconds * SR)).astype(np.float32) * 0.001
    for a, b in loud:
        audio[int(a * SR):int(b * SR)] = rng.standard_normal(int(b * SR) - int(a * SR)) * level
    return audio


def test_the_stream_is_lined_up_with_the_end_of_the_wake_clip():
    rng = np.random.default_rng(0)
    heard = pcm16_to_float(float_to_pcm16(rng.standard_normal(4 * SR).astype(np.float32) * 0.1))
    for tail in (0.4, 0.361, 0.05):
        start = 2 * SR
        anchor = heard[: start + int(tail * SR)]
        assert stream_start(anchor, heard[start:]) == start


def test_a_stream_with_nothing_of_the_wake_clip_starts_where_it_ends():
    rng = np.random.default_rng(0)
    anchor = rng.standard_normal(2 * SR).astype(np.float32) * 0.1
    head = rng.standard_normal(SR).astype(np.float32) * 0.1
    assert stream_start(anchor, head) == len(anchor)
    assert stream_start(anchor, head[:10]) == len(anchor)


def test_the_wake_word_is_tried_in_english_and_in_the_kiosks_language():
    assert voices(None) == ["en-us"]
    assert voices("en-GB") == ["en-us"]
    assert voices("fr") == ["en-us", "fr"]
    assert voices("pt-BR") == ["en-us", "pt-br", "pt"]
    assert voices("zh_CN") == ["en-us", "zh-cn", "cmn"]
    # Only a language tag is ever handed to espeak-ng.
    assert voices("fr; rm -rf /") == ["en-us"]
    assert voices("-f/etc/passwd") == ["en-us"]


def test_a_pronunciation_is_read_in_the_models_tokens():
    assert tokens_of("ɐlˈɛksə\n", VOCAB) == ALEXA
    # A long vowel is one token and espeak-ng's language marks are not phonemes.
    assert tokens_of("(en)nˈɑːt(fr)", VOCAB) == ["n", "ɑː", "t"]


def test_a_command_right_after_the_wake_word_starts_at_the_quietest_point_between():
    # "Alexa" ends at frame 76 (1.54 s, 0.14 s into the stream), "t" follows at frame 81.
    lp = probs(110, {56: "ɐ", 60: "l", 64: "ɛ", 68: "k", 72: "s", 76: "ə", 81: "t", 84: "n"})
    audio = sound(2.2, [(1.0, 1.55), (1.60, 2.1)])
    placed = locate(lp, audio, STREAM_S, [ALEXA], VOCAB, BLANK)
    assert placed is not None and placed.rule == "next sound"
    assert 1.55 <= placed.cut_s <= 1.60
    assert placed.frames == [56, 60, 64, 68, 72, 76]


def test_a_pause_after_the_wake_word_is_cut_where_the_room_goes_quiet():
    lp = probs(110, {56: "ɐ", 60: "l", 64: "ɛ", 68: "k", 72: "s", 76: "ə"})
    audio = sound(2.2, [(1.0, 1.62)])
    placed = locate(lp, audio, STREAM_S, [ALEXA], VOCAB, BLANK)
    assert placed is not None and placed.rule == "pause"
    assert 1.62 <= placed.cut_s <= 1.70


def test_a_command_whose_first_sound_was_missed_is_cut_before_the_level_rises():
    # Nothing heard after the word, but the level dips at 1.60 s and comes back: the command.
    lp = probs(110, {56: "ɐ", 60: "l", 64: "ɛ", 68: "k", 72: "s", 76: "ə"})
    audio = sound(2.2, [(1.0, 1.58), (1.58, 1.62), (1.62, 2.2)])
    audio[int(1.58 * SR):int(1.62 * SR)] *= 0.1
    placed = locate(lp, audio, STREAM_S, [ALEXA], VOCAB, BLANK)
    assert placed is not None and placed.rule == "rise"
    assert 1.58 <= placed.cut_s <= 1.62


def test_a_look_alike_phoneme_still_counts():
    # An accent: "a" for the schwas and a "ɡ" for the "k".
    lp = probs(110, {56: "a", 60: "l", 64: "ɛ", 68: "ɡ", 72: "s", 76: "a", 81: "t"})
    placed = locate(lp, sound(2.2, [(1.0, 1.55), (1.60, 2.1)]), STREAM_S, [ALEXA], VOCAB, BLANK)
    assert placed is not None and placed.frames[-1] == 76


def test_the_pronunciation_that_fits_best_is_used():
    lp = probs(110, {56: "ɐ", 60: "l", 64: "ɛ", 68: "k", 72: "s", 76: "ə", 81: "t"})
    placed = locate(lp, sound(2.2, [(1.0, 1.55), (1.60, 2.1)]), STREAM_S, [["n", "ɑː", "t", "ɜː"], ALEXA], VOCAB, BLANK)
    assert placed is not None and placed.phonemes == ALEXA


def test_a_wake_word_that_isnt_heard_is_not_placed():
    assert locate(probs(110, {}), sound(2.2, [(1.0, 2.0)]), STREAM_S, [ALEXA], VOCAB, BLANK) is None
    # Another word altogether.
    lp = probs(110, {56: "n", 60: "ɑː", 64: "t", 81: "t"})
    assert locate(lp, sound(2.2, [(1.0, 2.0)]), STREAM_S, [ALEXA], VOCAB, BLANK) is None


def test_the_wake_word_is_never_placed_inside_the_command():
    # The same sounds, but all of them after the stream started: that is the command.
    lp = probs(110, {74: "ɐ", 78: "l", 82: "ɛ", 86: "k", 90: "s", 94: "ə"})
    assert locate(lp, sound(2.2, [(1.4, 2.0)]), STREAM_S, [ALEXA], VOCAB, BLANK) is None


# ── the model ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> Aligner:
    return Aligner(verified_path(MODELS, ALIGNER_FILE, ALIGNER_SHA256), threads=2)


def test_the_model_gives_phoneme_probabilities_every_20_ms(model):
    lp = model.log_probs(np.zeros(int(2.2 * SR), dtype=np.float32))
    assert lp.shape == (109, len(model.vocab)) and len(model.vocab) == 392
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-3)
    assert model.version != "unknown" and model.blank == model.vocab["<pad>"]


needs_espeak = pytest.mark.skipif(aligner.espeak() is None, reason="espeak-ng is not installed")


def say(text: str) -> np.ndarray:
    wav = subprocess.run(["espeak-ng", "-v", "en-us", "--stdout", text], capture_output=True, check=True).stdout
    audio = read_wav(wav, any_rate=True)
    loud = np.flatnonzero(np.abs(audio) > 0.02 * np.abs(audio).max())
    return audio[loud[0]:loud[-1] + 1] * 0.5


def heard(wake: np.ndarray, command: np.ndarray, gap_s: float, inside_s: float):
    """A wake clip and a stream as a kiosk sends them: the stream starts
    ``inside_s`` before the wake word is over and the wake clip runs 0.4 s
    past that."""
    rng = np.random.default_rng(1)
    full = np.concatenate([np.zeros(SR), wake, np.zeros(int(gap_s * SR)), command, np.zeros(SR // 2)])
    full = full.astype(np.float32) + rng.standard_normal(len(full)).astype(np.float32) * 0.002
    full = pcm16_to_float(float_to_pcm16(full))
    start = SR + len(wake) - int(inside_s * SR)
    return full[start - 2 * SR if start > 2 * SR else 0: start + int(0.4 * SR)], full[start: start + int(0.8 * SR)]


@needs_espeak
def test_a_wake_word_is_pronounced_as_the_kiosks_language_says_it(model):
    # The whole word: espeak-ng 1.51 drops the last letter of text given without a line end.
    assert model.pronunciations("Alexa") == [["ɐ", "l", "ɛ", "k", "s", "ə"]]
    assert model.pronunciations("Ok Nova")[0][-2:] == ["v", "ə"]
    said = model.pronunciations("Hey Jarvis", "fr")
    assert len(said) == 2 and said[0][0] == "h" and said[1][0] != "h"
    assert model.pronunciations("", "fr") == []


@needs_espeak
def test_the_end_of_a_wake_word_said_in_one_breath_with_the_command_is_found(model):
    wake, command = say("Alexa"), say("what time is it")
    for inside in (0.10, 0.20):
        found = model.find(*heard(wake, command, 0.0, inside), "Alexa", "en")
        assert found is not None and found.rule == "next sound"
        # The end of the word goes and none of the command does.
        assert inside - 0.05 <= found.samples / SR <= inside + 0.03


@needs_espeak
def test_the_end_of_a_wake_word_before_a_pause_is_found(model):
    wake, command = say("Alexa"), say("what time is it")
    found = model.find(*heard(wake, command, 0.5, 0.15), "Alexa", "en")
    assert found is not None and found.rule == "pause"
    assert 0.15 <= found.samples / SR <= 0.3


@needs_espeak
def test_another_wake_word_than_the_one_said_cuts_nothing_of_the_command(model):
    wake, command = say("Alexa"), say("what time is it")
    found = model.find(*heard(wake, command, 0.0, 0.15), "Hey Jarvis", "en")
    assert found is None or found.samples / SR <= 0.18


def test_without_espeak_nothing_is_looked_for(model, monkeypatch):
    monkeypatch.setattr(aligner, "espeak", lambda: None)
    aligner._ipa.cache_clear()
    try:
        assert model.pronunciations("Alexa") == []
        assert model.find(np.zeros(2 * SR, dtype=np.float32), np.zeros(SR, dtype=np.float32), "Alexa") is None
    finally:
        aligner._ipa.cache_clear()
