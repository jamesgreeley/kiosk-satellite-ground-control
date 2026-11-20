"""The Voice Isolator, on its own and in follow mode.

Follow mode holds a turn's audio and sends it at the end of the turn, cleaned.
The real model is checked on a voice over another voice talking at the same
level, the case follow mode alone can't fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from ground_control.follow import Follow, FollowConfig
from ground_control.isolator import WINDOW, Isolator
from ground_control.models import ISOLATOR_FILE, SPEAKER_MODELS

from .conftest import MODELS, SR, chunks
from .test_follow import Script, held_out, household, of, silence  # noqa: F401 - household is a fixture

ACCURATE = SPEAKER_MODELS["accurate"]


class Halves:
    """Stands in for the isolator: half the level, and what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, np.ndarray]] = []

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        self.calls.append((len(audio), voiceprint.copy()))
        return audio * 0.5


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    proj = (est @ ref) / (ref @ ref) * ref
    return float(10 * np.log10((proj @ proj) / ((est - proj) @ (est - proj)) + 1e-12))


def test_a_turn_goes_out_cleaned_when_it_ends(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:2 * SR])
         .add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    iso = Halves()
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=iso)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    kinds = [e["type"] for e in events]
    # Nothing goes out during a turn: each turn's audio comes right before its
    # end, in pieces of half a second at most.
    for i, k in enumerate(kinds):
        if k == "audio":
            assert kinds[i + 1] in ("audio", "turn_end")
    assert max(len(e["audio"]) for e in of(events, "audio")) <= SR // 2
    assert len(iso.calls) == len(of(events, "turn_end"))
    # The isolator gets the user's own voiceprint, and what goes out is its
    # output cut to what follow mode released.
    assert np.allclose(iso.calls[0][1], f._ref)
    sent = np.concatenate([e["audio"] for e in of(events, "audio")])
    raw = s.audio
    expected = np.concatenate([raw[a:b] for a, b in f.released]) * 0.5
    assert np.allclose(sent, expected)
    assert s.released_from("tv", f.released) <= 0.5
    assert "isolated_ms" in of(events, "stats")[0]


def test_without_the_isolator_audio_goes_out_as_it_comes(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.3)).add("user", user[2 * SR:6 * SR]).add("quiet", silence(1.5))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    first_audio = None
    for i, c in enumerate(chunks(s.audio)):
        if any(e["type"] == "audio" for e in f.feed(c)) and first_audio is None:
            first_audio = (i + 1) * 1280 / SR
    assert first_audio is not None and first_audio < 4.0


def test_long_audio_runs_in_overlapping_windows():
    """Through an identity model, the windows and their fades add up to the input."""
    iso = Isolator.__new__(Isolator)
    iso._lock = __import__("threading").Lock()
    iso.calls = 0
    iso._run = lambda audio, voiceprint: audio
    x = np.random.default_rng(0).standard_normal(int(2.6 * WINDOW)).astype(np.float32) * 0.1
    assert np.allclose(iso.clean(x, np.ones(192, np.float32)), x, atol=1e-6)


@pytest.fixture(scope="module")
def isolator():
    if not (MODELS / ISOLATOR_FILE).is_file():
        pytest.skip("no Voice Isolator model")
    return Isolator(MODELS / ISOLATOR_FILE, threads=4)


def test_the_model_carries_its_license(isolator):
    meta = isolator.session.get_modelmeta().custom_metadata_map
    assert meta["license"].startswith("CC BY-NC-ND 4.0")
    assert "Copyright" in meta["copyright"]
    assert isolator.version == meta["version"]


def test_the_user_is_kept_and_another_voice_under_them_removed(household, speakers, voice, isolator):
    people, ana, _ = household
    user, other = held_out(speakers[0])[2 * SR:5 * SR], held_out(speakers[2])[2 * SR:5 * SR]
    other = other * (np.sqrt(np.mean(user ** 2)) / np.sqrt(np.mean(other ** 2)))  # the same level
    mix = (user + other).astype(np.float32)
    ids, rows = people.voiceprints("accurate")
    out = isolator.clean(mix, rows[ids.index(ana.id)])
    assert len(out) == len(mix)
    assert si_sdr(out, user) - si_sdr(mix, user) >= 6.0
    # Its level stays the user's.
    assert abs(20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(user ** 2)))) < 3.0


def test_a_short_command_is_isolated_with_the_users_profile(household, speakers, voice, vad, embedder):
    """A command too short to identify alone is identified when it ends, from
    the wake word and the command together. Isolation runs after that, so it
    gets the user's profile, not the weaker guess from the wake word."""
    people, ana, _ = household
    user = held_out(speakers[0])
    anchor, command = user[9 * SR:10 * SR], user[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    iso = Halves()
    f = Follow(FollowConfig(policy="people", people=(ana.id,)), ACCURATE, embedder, vad, people, anchor=anchor,
               isolator=iso)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id
    ids, rows = people.voiceprints("accurate")
    assert len(iso.calls) == 1 and np.allclose(iso.calls[0][1], rows[ids.index(ana.id)])
    # Who it is comes before the audio, and the audio before the end of the turn.
    kinds = [e["type"] for e in events]
    assert kinds.index("speaker") < kinds.index("audio") < kinds.index("turn_end")


def test_each_kiosk_chooses_isolation(cfg, isolator):
    """The service's setting allows Isolation, a kiosk's own switch picks it.
    A kiosk that doesn't say (an older Kiosk Satellite) gets it."""
    from ground_control.service import GroundControl

    vm = GroundControl(cfg)
    try:
        vm.update_settings({"isolator": True})
        assert vm.new_follow().isolator is not None
        assert vm.new_follow(isolation=False).isolator is None
        vm.update_settings({"isolator": False})
        assert vm.new_follow(isolation=True).isolator is None
    finally:
        vm.close()
