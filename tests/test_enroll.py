from __future__ import annotations

import numpy as np

from ground_control.enroll import FRAME_S, GuidedEnrollment, reading_seconds
from ground_control.models import SPEAKER_MODELS

from .conftest import SR, chunks


def run(guide, audio):
    events = []
    for c in chunks(audio):
        events += guide.feed(c)
    return events


def test_reaches_the_target_and_reports_progress(speakers, voice, vad, embedder):
    guide = GuidedEnrollment(embedder, vad, SPEAKER_MODELS["accurate"], target_s=20)
    events = run(guide, voice[speakers[0]]["enroll"])
    progress = [e for e in events if e["type"] == "progress"]
    assert progress and progress[-1]["speech_s"] <= 20.5
    assert [e["speech_s"] for e in progress] == sorted(e["speech_s"] for e in progress)
    assert events[-1]["type"] == "done" and guide.done
    assert guide.dropped_s == 0
    assert 20 <= guide.speech_s < 23


def test_another_voice_is_left_out(speakers, voice, vad, embedder):
    guide = GuidedEnrollment(embedder, vad, SPEAKER_MODELS["accurate"], target_s=200)
    own = voice[speakers[0]]["enroll"]
    stranger = voice[speakers[2]]["test"][:8 * SR]
    events = run(guide, np.concatenate([own[:15 * SR], stranger, own[15 * SR:30 * SR]]))
    assert any(e["type"] == "other_voice" for e in events)
    assert guide.dropped_s >= 4.0
    assert not guide.done


class ScriptedVad:
    """Speech where the script says so, one probability per 32 ms frame."""

    def __init__(self, script):
        self.script = list(script)

    def stream(self):
        script = self.script

        class Stream:
            def feed(self, frames):
                n = len(frames) // 512
                out, script[:n] = script[:n], []
                return out + [0.0] * (n - len(out))

        return Stream()


def frames(seconds, p):
    return [p] * round(seconds / FRAME_S)


def progress_of(events):
    return [e for e in events if e["type"] == "progress"]


def test_a_sentence_fills_as_it_is_read_and_a_pause_finishes_it(speakers, voice, embedder):
    passage = ["Turn on the kitchen lights.", "Good night.", "Pause the music."]
    first = reading_seconds(passage[0])
    # Most of the first sentence, a pause, then the whole second one read straight into the third.
    script = frames(first * 0.4, 1.0) + frames(0.3, 0.0) + frames(first * 0.3, 1.0) + frames(0.8, 0.0) \
        + frames(reading_seconds(passage[1]) + 0.3, 1.0) + frames(0.2, 0.0)
    guide = GuidedEnrollment(embedder, ScriptedVad(script), SPEAKER_MODELS["accurate"], target_s=60, passage=passage)
    events = run(guide, voice[speakers[0]]["enroll"][:int(len(script) * FRAME_S * SR) + SR])
    seen = [(e["sentence"], e["sentence_fill"]) for e in progress_of(events)]
    # A short pause in the middle doesn't move on, the long one after most of it does.
    assert seen[0][0] == 0 and any(s == 0 and 0.3 < f < 1 for s, f in seen)
    assert any(s == 1 for s, _ in seen)
    # Reading on past the end of a sentence moves to the next one without a pause.
    assert seen[-1][0] == 2
    # Speech counts as it is heard, not two seconds at a time.
    speech = [e["speech_s"] for e in progress_of(events)]
    assert any(0 < v < 2 for v in speech)


def test_a_short_pause_early_in_a_sentence_keeps_it(speakers, voice, embedder):
    passage = ["Later we walked to the park and watched the dogs chase each other across the grass.", "Good night."]
    script = frames(reading_seconds(passage[0]) * 0.3, 1.0) + frames(1.0, 0.0)
    guide = GuidedEnrollment(embedder, ScriptedVad(script), SPEAKER_MODELS["accurate"], target_s=60, passage=passage)
    events = run(guide, voice[speakers[0]]["enroll"][:3 * SR + int(len(script) * FRAME_S * SR)])
    assert all(e["sentence"] == 0 for e in progress_of(events))
