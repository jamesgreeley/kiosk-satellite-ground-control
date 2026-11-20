from __future__ import annotations

import numpy as np
import pytest

from ground_control.people import People


def test_add_rename_and_things(tmp_path, vad):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    with pytest.raises(ValueError):
        people.add("ana")
    people.update(ana.id, name="Ana M", things={"todo_list": "todo.ana", "calendar": "calendar.ana"})
    again = People(tmp_path, vad).get(ana.id)
    assert again.name == "Ana M"
    assert again.things == {"todo_list": "todo.ana", "calendar": "calendar.ana"}
    people.update(ana.id, things={"calendar": None})
    assert people.get(ana.id).things == {"todo_list": "todo.ana"}
    with pytest.raises(ValueError):
        people.update(ana.id, things={"car": "x"})


def test_colors_are_distinct(tmp_path, vad):
    people = People(tmp_path, vad)
    colors = [people.add(n).color for n in ("A", "B", "C", "D")]
    assert len(set(colors)) == 4


def test_enroll_needs_speech(tmp_path, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    with pytest.raises(ValueError):
        people.enroll(p.id, np.zeros(16000 * 5, dtype=np.float32), {"accurate": embedder})


def test_rebuild_matches_enrollment(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    people.enroll(p.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    _, before = people.voiceprints("accurate")
    people.rebuild("accurate", embedder)
    _, after = people.voiceprints("accurate")
    assert float(before[0] @ after[0]) > 0.999


def test_forget_removes_voice(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    people.enroll(p.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.forget(p.id)
    assert people.get(p.id) is None
    assert people.voiceprints("accurate")[0] == []
    assert not (tmp_path / "voices" / p.id).exists()
