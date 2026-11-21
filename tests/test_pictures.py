"""A person's picture, read from Home Assistant with the asker's token."""

from __future__ import annotations

import asyncio

from aiohttp import web

from ground_control.pictures import Pictures

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


async def home_assistant(aiohttp_server, picture="/api/image/serve/abc/512x512"):
    calls = []

    def signed_in(request):
        calls.append(request.path)
        if request.headers.get("Authorization") != "Bearer good":
            raise web.HTTPUnauthorized()

    async def state(request):
        signed_in(request)
        if request.match_info["entity"] != "person.ana":
            raise web.HTTPNotFound()
        return web.json_response({"entity_id": "person.ana", "attributes": {"entity_picture": picture}})

    async def image(request):
        signed_in(request)
        return web.Response(body=PNG, content_type="image/png")

    app = web.Application()
    app.router.add_get("/api/states/{entity}", state)
    app.router.add_get("/api/image/serve/abc/512x512", image)
    server = await aiohttp_server(app)
    return str(server.make_url("")).rstrip("/"), calls


async def test_the_picture_comes_with_the_askers_token(aiohttp_server):
    url, calls = await home_assistant(aiohttp_server)
    pictures = Pictures(url, None)
    try:
        assert await pictures.get("person.ana", "good") == (PNG, "image/png")
        assert await pictures.get("person.ana", "bad") is None
        # A second ask is answered from the cache.
        n = len(calls)
        assert await pictures.get("person.ana", "good") == (PNG, "image/png")
        assert len(calls) == n
    finally:
        await pictures.close()


async def test_no_picture_without_a_person_entity_or_a_token(aiohttp_server):
    url, calls = await home_assistant(aiohttp_server)
    pictures = Pictures(url, None)
    try:
        assert await pictures.get("person.leo", "good") is None
        assert await pictures.get("light.kitchen", "good") is None
        assert await pictures.get("person.ana", None) is None
        assert not [c for c in calls if "light" in c]
    finally:
        await pictures.close()


async def test_a_person_without_a_picture(aiohttp_server):
    url, _ = await home_assistant(aiohttp_server, picture=None)
    pictures = Pictures(url, None)
    try:
        assert await pictures.get("person.ana", "good") is None
    finally:
        await pictures.close()


async def test_a_large_picture_arrives_whole(aiohttp_server):
    big = b"\xff\xd8" + bytes(range(256)) * 2000 + b"\xff\xd9"

    async def state(request):
        return web.json_response({"attributes": {"entity_picture": "/api/image/serve/big/512x512"}})

    async def image(request):
        # Sent in pieces with pauses, so the body comes over many reads.
        r = web.StreamResponse(headers={"Content-Type": "image/jpeg"})
        await r.prepare(request)
        for i in range(0, len(big), 16 << 10):
            await r.write(big[i:i + (16 << 10)])
            await asyncio.sleep(0.01)
        await r.write_eof()
        return r

    app = web.Application()
    app.router.add_get("/api/states/{entity}", state)
    app.router.add_get("/api/image/serve/big/512x512", image)
    server = await aiohttp_server(app)
    pictures = Pictures(str(server.make_url("")).rstrip("/"), None)
    try:
        assert await pictures.get("person.ana", "good") == (big, "image/jpeg")
    finally:
        await pictures.close()


async def test_an_app_gets_a_picture_in_the_www_folder_straight_from_core(aiohttp_server):
    """The Supervisor's proxy only passes /api, and Core serves /local without a login."""
    async def avatar(request):
        assert "Authorization" not in request.headers
        return web.Response(body=PNG, content_type="image/png")

    core = web.Application()
    core.router.add_get("/local/avatar.png", avatar)
    core_server = await aiohttp_server(core)

    async def state(request):
        assert request.headers.get("Authorization") == "Bearer supervisor"
        return web.json_response({"attributes": {"entity_picture": "/local/avatar.png"}})

    async def info(request):
        return web.json_response({"data": {"port": core_server.port, "ssl": False}})

    async def refused(request):
        raise web.HTTPForbidden()

    supervisor = web.Application()
    supervisor.router.add_get("/core/api/states/{entity}", state)
    supervisor.router.add_get("/core/info", info)
    supervisor.router.add_get("/core/local/{name}", refused)
    sup_server = await aiohttp_server(supervisor)
    pictures = Pictures("auto", "supervisor", supervisor=str(sup_server.make_url("")).rstrip("/"))
    try:
        assert await pictures.get("person.ana", None) == (PNG, "image/png")
    finally:
        await pictures.close()


# ── suggestions for a person's things ────────────────────────────────────

from ground_control.suggestions import Suggestions  # noqa: E402

STATES = [
    {"entity_id": "person.ana", "attributes": {"friendly_name": "Ana"}},
    {"entity_id": "todo.ana", "attributes": {"friendly_name": "Ana's list"}},
    {"entity_id": "todo.groceries", "attributes": {"friendly_name": "Groceries"}},
    {"entity_id": "calendar.family", "attributes": {}},
    {"entity_id": "light.kitchen", "attributes": {"friendly_name": "Kitchen"}},
]


async def home_assistant_with_music(aiohttp_server, music=True):
    calls = []

    def signed_in(request):
        calls.append(request.path)
        if request.headers.get("Authorization") != "Bearer good":
            raise web.HTTPUnauthorized()

    async def states(request):
        signed_in(request)
        return web.json_response(STATES)

    async def entries(request):
        signed_in(request)
        found = [{"entry_id": "ma1", "domain": "music_assistant", "state": "loaded"}] if music else []
        return web.json_response(found)

    async def library(request):
        signed_in(request)
        body = await request.json()
        assert body == {"config_entry_id": "ma1", "media_type": "playlist", "limit": 500}
        assert "return_response" in request.query
        items = [{"name": "Chill"}, {"name": "Anadel"}, {"name": "Chill"}]
        return web.json_response({"changed_states": [], "service_response": {"items": items}})

    app = web.Application()
    app.router.add_get("/api/states", states)
    app.router.add_get("/api/config/config_entries/entry", entries)
    app.router.add_post("/api/services/music_assistant/get_library", library)
    server = await aiohttp_server(app)
    return str(server.make_url("")).rstrip("/"), calls


async def test_suggestions_are_the_entities_and_playlists_the_asker_can_read(aiohttp_server):
    url, calls = await home_assistant_with_music(aiohttp_server)
    s = Suggestions(url, None)
    try:
        got = await s.get("good")
        assert got["person"] == [{"value": "person.ana", "label": "Ana"}]
        assert [r["value"] for r in got["todo_list"]] == ["todo.ana", "todo.groceries"]
        assert got["calendar"] == [{"value": "calendar.family", "label": "calendar.family"}]
        assert got["playlist"] == [{"value": "Anadel", "label": ""}, {"value": "Chill", "label": ""}]
        # Someone Home Assistant turns away gets nothing, and a second ask comes from the cache.
        assert all(v == [] for v in (await s.get("bad")).values())
        n = len(calls)
        await s.get("good")
        assert len(calls) == n
    finally:
        await s.close()


async def test_without_music_assistant_there_are_no_playlists(aiohttp_server):
    url, _ = await home_assistant_with_music(aiohttp_server, music=False)
    s = Suggestions(url, None)
    try:
        got = await s.get("good")
        assert got["playlist"] == [] and got["person"]
        assert all(v == [] for v in (await Suggestions(url, None).get(None)).values())
    finally:
        await s.close()
