"""Follow mode on recorded speech: only the user's segments go on.

Ana and Leo are enrolled. The third speaker plays the TV, or a guest. The
anchor stands in for the wake word: a short clip of the user that is not
part of their enrollment.
"""

from __future__ import annotations

import numpy as np
import pytest

from ground_control.audio import read_wav
from ground_control.follow import Follow, FollowConfig
from ground_control.models import SPEAKER_MODELS
from ground_control.people import People
from ground_control.vad import FRAME

from .conftest import SR, chunks, clips

ACCURATE = SPEAKER_MODELS["accurate"]


@pytest.fixture(scope="module")
def household(tmp_path_factory, speakers, voice, vad, embedder):
    people = People(tmp_path_factory.mktemp("people"), vad)
    ana, leo = people.add("Ana"), people.add("Leo")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.enroll(leo.id, voice[speakers[1]]["enroll"], {"accurate": embedder})
    return people, ana, leo


def held_out(speaker: str) -> np.ndarray:
    """Speech from neither the enrollment nor the test clip."""
    c = clips(speaker)
    return np.concatenate([read_wav(p) for p in c[8:-1]]) if len(c) > 9 else read_wav(c[-2])


def anchor_of(audio: np.ndarray) -> np.ndarray:
    return audio[: int(1.2 * SR)]


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * SR), dtype=np.float32)


class Script:
    """Audio built from labeled parts, with where each part landed."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.spans: list[tuple[str, int, int]] = []
        self.n = 0

    def add(self, who: str, audio: np.ndarray) -> Script:
        self.parts.append(audio.astype(np.float32))
        self.spans.append((who, self.n, self.n + len(audio)))
        self.n += len(audio)
        return self

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.parts)

    def released_from(self, who: str, released: list[tuple[int, int]]) -> float:
        """Seconds of ``who`` that were released."""
        total = 0
        for name, a, b in self.spans:
            if name != who:
                continue
            for ra, rb in released:
                total += max(0, min(b, rb) - max(a, ra))
        return total / SR

    def seconds(self, who: str) -> float:
        return sum(b - a for name, a, b in self.spans if name == who) / SR

    def speech_share(self, who: str, released: list[tuple[int, int]], vad) -> float:
        """The share of ``who``'s speech (VAD frames, not their pauses) that was
        released: pauses inside a turn go out shortened, the words go out whole."""
        probs = vad.stream().feed(self.audio)
        frames = [(i * FRAME, (i + 1) * FRAME) for i, p in enumerate(probs) if p >= 0.5]
        mine = [(a, b) for a, b in frames if any(n == who and s <= a and b <= e for n, s, e in self.spans)]
        if not mine:
            return 0.0
        return sum(1 for a, b in mine if any(ra <= a and b <= rb for ra, rb in released)) / len(mine)


def run(folks, embedder, vad, script: Script, anchor=None, **kw):
    f = Follow(FollowConfig(**kw), ACCURATE, embedder, vad, folks, anchor=anchor)
    events = []
    for c in chunks(script.audio):
        events += f.feed(c)
    events += f.finish()
    return events, f


def of(events, kind):
    return [e for e in events if e["type"] == kind]


def test_tv_before_between_and_after_the_user_is_left_out(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("tv", tv[:2 * SR]).add("user", user[2 * SR:5 * SR])
         .add("tv", tv[2 * SR:4 * SR]).add("user", user[6 * SR:8 * SR]).add("tv", tv[4 * SR:6 * SR])
         .add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user))
    assert of(events, "speaker")[0]["id"] == ana.id
    assert of(events, "speaker")[0]["by"] == "wake_word"
    # Each TV stretch is longer than a pause: the user's two stretches are two turns.
    assert len(of(events, "turn_start")) == 2 and len(of(events, "turn_end")) == 2
    assert s.speech_share("user", f.released, vad) >= 0.8
    assert s.released_from("tv", f.released) <= 0.5
    released = sum(len(e["audio"]) for e in of(events, "audio") if not e.get("keepalive")) / SR
    assert released == pytest.approx(sum(b - a for a, b in f.released) / SR, abs=0.01)


def test_tv_right_after_the_wake_word_is_not_the_user(household, speakers, voice, vad, embedder):
    """Without the anchor, the TV talking first would be taken for the user."""
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("tv", tv[:2500 * SR // 1000]).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user))
    assert len(of(events, "turn_start")) == 1
    assert s.released_from("tv", f.released) <= 0.3
    assert s.speech_share("user", f.released, vad) >= 0.8
    end = of(events, "turn_end")[0]
    user_end = next(b for who, a, b in s.spans if who == "user") / SR
    assert end["t"] == pytest.approx(user_end, abs=0.5)


def test_the_turn_ends_when_the_user_stops_even_with_the_tv_on(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("user", user[2 * SR:5 * SR]).add("tv", tv[: 6 * SR])
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    ended_at = None
    fed = 0
    for c in chunks(s.audio):
        fed += len(c)
        if any(e["type"] == "turn_end" for e in f.feed(c)) and ended_at is None:
            ended_at = fed / SR
    assert ended_at is not None, "the TV kept the turn open"
    # The user stopped at 3 s: the turn ends well before the TV does.
    assert ended_at < 3.0 + 2.0
    assert s.released_from("tv", f.released) <= 0.5


def test_a_guest_is_followed_by_the_wake_word(household, speakers, voice, vad, embedder):
    """Nobody enrolled said it: the anchor's own voice is the user, Ana on the TV is not.

    Leo's voice is too close to this guest's (0.42) to tell apart from a guest's
    wake word: an enrolled voiceprint is what separates voices that close."""
    people, _, _ = household
    guest, tv = held_out(speakers[2]), voice[speakers[0]]["test"]
    s = (Script().add("user", guest[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("user", guest[6 * SR:8 * SR])
         .add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(guest))
    assert of(events, "speaker")[-1]["name"] is None
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.7
    assert not of(events, "rejected")


def test_enrolled_only_rejects_a_guest_and_releases_nothing_after(household, speakers, voice, vad, embedder):
    people, _, _ = household
    guest = held_out(speakers[2])
    s = Script().add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5)).add("user", guest[6 * SR:8 * SR])
    events, f = run(people, embedder, vad, s, anchor=anchor_of(guest), policy="enrolled")
    rejected = of(events, "rejected")
    assert rejected
    after = [e for e in events if e["type"] == "audio" and e["t"] > rejected[0]["t"] + 0.1]
    assert not after


def test_without_an_anchor_the_first_voice_is_the_user(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("user", user[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s)
    assert of(events, "speaker")[0]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5


def test_a_pause_is_shortened_but_kept(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:4 * SR]).add("quiet", silence(1.0)).add("user", user[4 * SR:6 * SR])
    s.add("quiet", silence(2.0))
    # A long end of turn, so the 1 s pause stays inside one turn.
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), eos_silence_s=1.6)
    assert len(of(events, "turn_start")) == 1
    released = sum(b - a for a, b in f.released) / SR
    assert s.speech_share("user", f.released, vad) >= 0.9
    # The 1 s pause went as 0.15 s, under the 0.25 s of silence that ends a command
    # in Home Assistant's aggressive end of speech.
    assert released < s.seconds("user") + 0.25


def test_without_a_wake_word_the_household_is_the_user(tmp_path, speakers, voice, vad, embedder):
    """A manual start has no anchor: the TV talking first is not taken for the user.

    Only Ana is enrolled here: this TV voice is close to Leo's (0.42) and
    would pass as him."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("tv", tv[: int(1.5 * SR)]).add("user", user[2 * SR:5 * SR]).add("tv", tv[2 * SR:int(4.5 * SR)])
         .add("quiet", silence(1.0)))
    events, f = run(people, embedder, vad, s)
    assert of(events, "speaker")[0]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_without_a_wake_word_a_voice_nobody_enrolled_is_answered(tmp_path, speakers, voice, vad, embedder):
    """Home Assistant asked a question, so there is no wake word, and the person
    answering isn't enrolled: a device that answers anyone follows them."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.5)).add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s)
    assert len(of(events, "turn_start")) == 1
    assert s.speech_share("user", f.released, vad) >= 0.8
    assert of(events, "speaker")[-1]["name"] is None


def test_without_a_wake_word_a_short_answer_goes_through(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    word = guest[2 * SR:int(2.7 * SR)]
    s = Script().add("quiet", silence(0.5)).add("user", word).add("quiet", silence(1.5))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people)
    ended = False
    for c in chunks(s.audio):
        ended = ended or any(e["type"] == "turn_end" for e in f.feed(c))
    assert ended, "the answer waited for the end of the session"
    assert s.speech_share("user", f.released, vad) >= 0.5


def test_enrolled_only_without_a_wake_word_leaves_a_voice_nobody_enrolled_out(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="enrolled")
    assert s.released_from("user", f.released) <= 0.3


def test_only_the_chosen_people_are_followed(household, speakers, voice, vad, embedder):
    """Ana says the wake word on a device that answers only Leo: nothing goes on."""
    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), policy="people", people=(leo.id,))
    assert of(events, "speaker")[0]["name"] == "Ana"
    assert of(events, "rejected")
    assert not f.released
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_without_a_wake_word_only_the_chosen_people_are_the_user(household, speakers, voice, vad, embedder):
    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="people", people=(leo.id,))
    assert s.released_from("user", f.released) <= 0.3


def test_a_short_command_after_a_short_wake_word_is_still_recognized(household, speakers, voice, vad, embedder):
    """Neither the wake word (under the threshold on its own) nor a command
    under a second is enough alone. Together they are, so a device that
    answers only Ana doesn't turn her away."""
    people, ana, _ = household
    user = held_out(speakers[0])
    anchor, command = user[9 * SR:10 * SR], user[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor, policy="people", people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id
    assert not of(events, "rejected")
    # The score that recognized her is on the timeline, with the wake word in it.
    found = [t for t in f.timeline if t["kind"] == "identify"]
    assert found[-1]["score"] >= ACCURATE.threshold and found[-1]["wake_word"] is True


def test_a_short_command_from_a_guest_is_still_turned_away(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest = held_out(speakers[2])
    anchor, command = guest[9 * SR:10 * SR], guest[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    events, _ = run(people, embedder, vad, s, anchor=anchor, policy="people", people=(ana.id,))
    assert of(events, "rejected")


def run_followup(folks, embedder, vad, script, anchor, speaker, **kw):
    f = Follow(FollowConfig(**kw), ACCURATE, embedder, vad, folks, anchor=anchor, speaker=speaker)
    events = []
    for c in chunks(script.audio):
        events += f.feed(c)
    events += f.finish()
    return events, f


def test_a_one_word_follow_up_goes_through(household, speakers, voice, vad, embedder):
    """Asked a follow-up, Ana says "yes": half a second, too little to tell who
    said it, but the conversation already knows it is her."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(1.0)).add("user", user[9 * SR:int(9.5 * SR)]).add("quiet", silence(1.5))
    events, f = run_followup(people, embedder, vad, s, user[:SR // 2], ana.id, policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert of(events, "speaker")[0]["id"] == ana.id and of(events, "speaker")[0]["by"] == "conversation"
    assert s.released_from("user", f.released) >= 0.3


def test_a_follow_up_still_leaves_another_voice_out(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("tv", tv[:3 * SR]).add("user", user[9 * SR:11 * SR]).add("quiet", silence(1.5))
    events, f = run_followup(people, embedder, vad, s, None, ana.id, policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.7


def test_a_short_wake_word_on_a_strict_device_follows_the_voiceprint(household, speakers, voice, vad, embedder):
    """Only Ana gets an answer, and her wake word was too short to say it was
    her: her command is still hers, followed by her voiceprint."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.3)).add("user", user[9 * SR:12 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=user[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id and not of(events, "rejected")
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_a_guest_on_a_strict_device_still_gets_no_answer(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.3)).add("user", guest[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=guest[9 * SR:int(9.6 * SR)], policy="people", people=(ana.id,))
    assert not any(e["id"] == ana.id for e in of(events, "speaker"))
    assert of(events, "rejected") or s.released_from("user", f.released) <= 0.5


def test_a_tv_around_a_short_command_does_not_turn_the_user_away(household, speakers, voice, vad, embedder):
    """Only Ana gets an answer, her wake word was too short to say it was her
    and the TV talks before and right after her command. The TV is not who she
    is: her command, with the wake word in front, is."""
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("tv", tv[:int(1.5 * SR)]).add("quiet", silence(0.3))
         .add("user", user[9 * SR:int(10.5 * SR)]).add("tv", tv[2 * SR:5 * SR]).add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=user[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert not of(events, "rejected")
    assert of(events, "speaker")[-1]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5


def test_a_guest_with_the_tv_on_is_still_turned_away(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest, tv = held_out(speakers[2]), voice[speakers[1]]["test"]
    s = (Script().add("tv", tv[:int(1.5 * SR)]).add("quiet", silence(0.3))
         .add("user", guest[9 * SR:int(10.5 * SR)]).add("tv", tv[2 * SR:5 * SR]).add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=guest[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert not any(e["id"] == ana.id for e in of(events, "speaker"))
    assert of(events, "rejected") or s.released_from("user", f.released) <= 0.5



@pytest.mark.parametrize("keep", [0.25, 0.0])
def test_the_end_of_the_last_word_is_kept_when_another_voice_starts_right_after(
        household, speakers, voice, vad, embedder, keep):
    """No pause between the user and the TV: the check straddling the two is
    mostly the TV and scores low, so the labels turn to the TV before the
    user's word ends. That much of the TV's labels still goes out as the
    user's. Here the labels are forced to turn at a known point."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("quiet", silence(1.5))
    # A point inside continuous speech, around where the TV takes over.
    speech = [p >= 0.5 for p in vad.stream().feed(s.audio)]
    tv_start = next(a for who, a, b in s.spans if who == "tv") // FRAME
    k = next(i for i in range(tv_start - 10, len(speech)) if all(speech[i - 6:i + 16]))
    turn = k * FRAME
    f = Follow(FollowConfig(tail_keep_s=keep), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    f._label = lambda frame, final: "user" if (frame + 1) * FRAME <= turn else "other"
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    end = max(b for a, b in f.released)
    assert end == pytest.approx(turn + int(keep * SR), abs=FRAME)
    assert of(events, "other_voice")[0]["t"] == pytest.approx(end / SR, abs=0.02)


def test_a_later_turn_needs_a_clearer_voice_than_staying_in_one(household, speakers, voice, vad, embedder):
    """Just over the other-voice line keeps a turn going, but once the user is
    done it doesn't start another: a TV scoring there would stop a realtime
    answer. Here the check scores are set by hand."""
    people, _, _ = household
    user = held_out(speakers[0])
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    for c in chunks(user[:2 * SR]):
        f.feed(c)
    frame = f.n // FRAME - 2
    center = frame * FRAME + FRAME // 2
    other, start = ACCURATE.other_threshold, ACCURATE.other_threshold + 0.5 * (ACCURATE.threshold - ACCURATE.other_threshold)

    def label(score, in_turn, turns):
        f._checks = [(center, score)]
        f.in_turn, f.turns = in_turn, turns
        return f._label(frame, final=True)

    assert label(other + 0.01, in_turn=True, turns=1) == "user"  # inside a turn
    assert label(other + 0.01, in_turn=False, turns=0) == "user"  # the first turn
    assert label(other + 0.01, in_turn=False, turns=1) == "other"  # a later one
    assert label(start + 0.01, in_turn=False, turns=1) == "user"
