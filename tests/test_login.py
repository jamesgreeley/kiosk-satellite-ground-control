"""Log in with Home Assistant: the standalone web UI's sign-in."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import pytest
from aiohttp import web

from ground_control.auth import TokenCheck
from ground_control.server import build_app
from ground_control.service import GroundControl


@pytest.fixture
async def home_assistant(aiohttp_server):
    """A Home Assistant that signs in one user: a code, then tokens."""
    state = {"challenge": None, "client_id": None, "revoked": [], "valid": set(), "down": False}

    async def token(request):
        if state["down"]:
            return web.Response(status=502, text="Bad Gateway")
        data = await request.post()
        if data["grant_type"] == "authorization_code":
            digest = hashlib.sha256(data["code_verifier"].encode()).digest()
            if (data["code"] != "the-code" or data["client_id"] != state["client_id"]
                    or base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != state["challenge"]):
                return web.json_response({"error": "invalid_request"}, status=400)
            state["valid"].add("access-1")
            return web.json_response({"access_token": "access-1", "refresh_token": "refresh-1",
                                      "expires_in": 1800, "token_type": "Bearer"})
        if data["grant_type"] == "refresh_token":
            if data["refresh_token"] != "refresh-1" or "refresh-1" in state["revoked"]:
                return web.json_response({"error": "invalid_grant"}, status=400)
            state["valid"].add("access-2")
            return web.json_response({"access_token": "access-2", "expires_in": 1800, "token_type": "Bearer"})
        return web.json_response({"error": "unsupported_grant_type"}, status=400)

    async def revoke(request):
        state["revoked"].append((await request.post())["token"])
        state["valid"].clear()
        return web.Response()

    async def api_root(request):
        if state["down"]:
            return web.Response(status=502, text="Bad Gateway")
        ok = request.headers.get("Authorization", "").removeprefix("Bearer ") in state["valid"]
        return web.json_response({"message": "API running."}, status=200 if ok else 401)

    app = web.Application()
    app.router.add_post("/auth/token", token)
    app.router.add_post("/auth/revoke", revoke)
    app.router.add_get("/api/", api_root)
    server = await aiohttp_server(app)
    return str(server.make_url("")).rstrip("/"), state


@pytest.fixture
async def client(aiohttp_client, cfg, home_assistant):
    url, _ = home_assistant
    cfg = replace(cfg, ha_url=url)
    return await aiohttp_client(build_app(GroundControl(cfg), TokenCheck(url)))


async def log_in(client, home_assistant, next_hash="#/enroll"):
    _, state = home_assistant
    r = await client.get(f"/auth/login?next={next_hash.replace('#', '%23')}", allow_redirects=False)
    assert r.status == 302
    where = urlparse(r.headers["Location"])
    q = {k: v[0] for k, v in parse_qs(where.query).items()}
    state["challenge"], state["client_id"] = q["code_challenge"], q["client_id"]
    return where, q


async def test_login_goes_to_home_assistant_and_comes_back_signed_in(client, home_assistant):
    url, _ = home_assistant
    assert (await (await client.get("/auth/info")).json()) == {"login": True, "ha_url": url, "token": None}
    assert (await client.get("/api/people")).status == 401

    where, q = await log_in(client, home_assistant)
    assert f"{where.scheme}://{where.netloc}" == url and where.path == "/auth/authorize"
    origin = f"http://{client.host}:{client.port}"
    assert q["client_id"] == f"{origin}/" and q["redirect_uri"] == f"{origin}/auth/callback"
    assert q["code_challenge_method"] == "S256" and q["response_type"] == "code"

    r = await client.get(f"/auth/callback?code=the-code&state={q['state']}", allow_redirects=False)
    assert r.status == 302 and r.headers["Location"] == "../#/enroll"
    # The login is a cookie only Ground Control reads, never in the page.
    assert "refresh-1" not in await r.text()
    kept = r.cookies["gc_refresh"]
    assert kept.value == "refresh-1" and kept["httponly"] and kept["path"] == "/auth/"
    assert kept["samesite"] == "Lax" and not kept["secure"] and int(kept["max-age"]) == 90 * 86400
    assert r.cookies["gc_client"].value == q["client_id"]

    # The web UI gets its access token from the cookie.
    r = await client.post("/auth/refresh")
    assert (await r.json())["access_token"] == "access-2"
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-2"})).status == 200

    # A code works once.
    again = await (await client.get(f"/auth/callback?code=the-code&state={q['state']}")).text()
    assert "Not signed in" in again and "access-1" not in again


async def test_refresh_and_sign_out(client, home_assistant):
    _, state = home_assistant
    _, q = await log_in(client, home_assistant)
    await client.get(f"/auth/callback?code=the-code&state={q['state']}")

    r = await client.post("/auth/refresh")
    assert (await r.json())["access_token"] == "access-2"
    assert r.cookies["gc_refresh"].value == "refresh-1"  # kept another 90 days
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-2"})).status == 200

    r = await client.post("/auth/logout", json={"access_token": "access-2"})
    assert state["revoked"] == ["refresh-1"]
    assert r.cookies["gc_refresh"].value == "" and r.cookies["gc_client"].value == ""
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-2"})).status == 401
    assert (await client.post("/auth/refresh")).status == 401


async def test_a_login_from_before_the_cookie_moves_to_it(client, home_assistant):
    _, q = await log_in(client, home_assistant)
    await client.get(f"/auth/callback?code=the-code&state={q['state']}")
    client.session.cookie_jar.clear()

    # A browser that kept its refresh token in localStorage sends it once.
    r = await client.post("/auth/refresh", json={"refresh_token": "refresh-1", "client_id": q["client_id"]})
    assert r.status == 200 and r.cookies["gc_refresh"].value == "refresh-1"
    assert (await client.post("/auth/refresh")).status == 200


async def test_a_refused_refresh_drops_the_cookie(client, home_assistant):
    _, state = home_assistant
    _, q = await log_in(client, home_assistant)
    await client.get(f"/auth/callback?code=the-code&state={q['state']}")
    state["revoked"].append("refresh-1")  # signed out in Home Assistant
    r = await client.post("/auth/refresh")
    assert r.status == 401 and r.cookies["gc_refresh"].value == ""


async def test_the_cookie_is_secure_behind_an_https_proxy(client, home_assistant):
    _, state = home_assistant
    proxy = {"X-Forwarded-Proto": "https", "X-Forwarded-Host": "gc.example.com"}
    r = await client.get("/auth/login?next=%23%2F", headers=proxy, allow_redirects=False)
    q = {k: v[0] for k, v in parse_qs(urlparse(r.headers["Location"]).query).items()}
    assert q["redirect_uri"] == "https://gc.example.com/auth/callback"
    state["challenge"], state["client_id"] = q["code_challenge"], q["client_id"]
    r = await client.get(f"/auth/callback?code=the-code&state={q['state']}", headers=proxy,
                         allow_redirects=False)
    assert r.status == 302 and r.cookies["gc_refresh"]["secure"]


async def test_a_refused_login_and_a_wrong_state(client, home_assistant):
    page = await (await client.get("/auth/callback?error=access_denied")).text()
    assert "Not signed in" in page
    page = await (await client.get("/auth/callback?code=the-code&state=made-up")).text()
    assert "Not signed in" in page


async def test_an_app_has_no_login(aiohttp_client, cfg):
    app = build_app(GroundControl(replace(cfg, ha_url="auto")), TokenCheck("auto"))
    client = await aiohttp_client(app)
    assert (await (await client.get("/auth/info")).json())["login"] is False
    assert (await client.get("/auth/login", allow_redirects=False)).status == 404


async def test_ha_token_opens_the_web_ui_but_not_the_kiosk_sockets(aiohttp_client, cfg, home_assistant):
    url, state = home_assistant
    state["valid"].add("long-lived")
    app = build_app(GroundControl(replace(cfg, ha_url=url, ha_token="long-lived")), TokenCheck(url))
    client = await aiohttp_client(app)
    assert (await (await client.get("/auth/info")).json())["token"] == "good"
    assert (await client.get("/api/people")).status == 200

    async with client.ws_connect("/api/events") as ws:
        await ws.send_json({"type": "listen"})
        assert (await ws.receive_json())["type"] == "kiosks"

    async with client.ws_connect("/api/kiosk") as ws:
        assert (await ws.receive_json())["type"] == "challenge"
        await ws.send_json({"type": "hello", "device_id": "kitchen"})
        assert (await ws.receive_json()) == {"type": "error", "code": "unauthorized"}


async def test_a_refused_ha_token_asks_to_log_in(aiohttp_client, cfg, home_assistant):
    url, _ = home_assistant
    app = build_app(GroundControl(replace(cfg, ha_url=url, ha_token="made-up")), TokenCheck(url))
    client = await aiohttp_client(app)
    info = await (await client.get("/auth/info")).json()
    assert info["token"] == "refused" and info["login"] is True
    assert (await client.get("/api/people")).status == 401


async def test_ha_token_unchecked_while_home_assistant_is_down(aiohttp_client, cfg):
    app = build_app(GroundControl(replace(cfg, ha_url="http://127.0.0.1:9", ha_token="long-lived")),
                    TokenCheck("http://127.0.0.1:9"))
    client = await aiohttp_client(app)
    assert (await (await client.get("/auth/info")).json())["token"] == "unreachable"
    assert (await client.get("/api/people")).status == 503


async def test_home_assistant_down_keeps_the_login(client, home_assistant):
    _, state = home_assistant
    _, q = await log_in(client, home_assistant)
    await client.get(f"/auth/callback?code=the-code&state={q['state']}")
    state["down"] = True

    # A token Ground Control hasn't checked yet: wait for Home Assistant, don't ask to log in.
    r = await client.get("/api/people", headers={"Authorization": "Bearer access-1"})
    assert r.status == 503 and "can't be reached" in await r.text()
    r = await client.post("/auth/refresh")
    assert r.status == 502 and "gc_refresh" not in r.cookies

    async with client.ws_connect("/api/events") as ws:
        await ws.send_json({"type": "listen", "token": "access-1"})
        assert (await ws.receive_json())["code"] == "home_assistant_unreachable"

    # A kiosk socket is refused as before.
    async with client.ws_connect("/api/kiosk") as ws:
        await ws.receive_json()
        await ws.send_json({"type": "hello", "device_id": "kitchen", "token": "access-1"})
        assert (await ws.receive_json())["code"] == "unauthorized"

    state["down"] = False
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-1"})).status == 200
    assert (await client.post("/auth/refresh")).status == 200


async def test_ha_token_unchecked_waits_for_home_assistant(aiohttp_client, cfg, home_assistant):
    url, state = home_assistant
    state["valid"].add("long-lived")
    state["down"] = True
    app = build_app(GroundControl(replace(cfg, ha_url=url, ha_token="long-lived")), TokenCheck(url))
    client = await aiohttp_client(app)
    assert (await client.get("/api/people")).status == 503
    state["down"] = False
    assert (await client.get("/api/people")).status == 200
