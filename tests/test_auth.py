from __future__ import annotations

import aiohttp

from ground_control.auth import TokenCheck, bearer


async def test_answers_are_cached():
    asked = []

    async def ask(token):
        asked.append(token)
        return token == "good"

    check = TokenCheck("http://ha.invalid", ask=ask)
    assert await check.valid("good")
    assert await check.valid("good")
    assert not await check.valid("bad")
    assert not await check.valid(None)
    assert asked == ["good", "bad"]


async def test_home_assistant_down_keeps_known_tokens():
    up = {"on": True}

    async def ask(token):
        if not up["on"]:
            raise aiohttp.ClientError("down")
        return True

    check = TokenCheck("http://ha.invalid", ask=ask)
    assert await check.valid("known")
    check._cache = {k: (v, 0.0) for k, (v, _) in check._cache.items()}  # expire
    up["on"] = False
    assert await check.valid("known")
    assert not await check.valid("new")
    assert await check.check("new") is None


def test_bearer():
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer abc ") == "abc"
    assert bearer("Basic x") is None
    assert bearer(None) is None
