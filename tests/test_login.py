"""Log in with Home Assistant: the standalone web UI's sign-in."""

from __future__ import annotations

import base64
import hashlib
import json
import re
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
    state = {"challenge": None, "client_id": None, "revoked": [], "valid": set()}

    async def token(request):
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
    assert (await (await client.get("/auth/info")).json()) == {"login": True, "ha_url": url}
    assert (await client.get("/api/people")).status == 401

    where, q = await log_in(client, home_assistant)
    assert f"{where.scheme}://{where.netloc}" == url and where.path == "/auth/authorize"
    origin = f"http://{client.host}:{client.port}"
    assert q["client_id"] == f"{origin}/" and q["redirect_uri"] == f"{origin}/auth/callback"
    assert q["code_challenge_method"] == "S256" and q["response_type"] == "code"

    page = await (await client.get(f"/auth/callback?code=the-code&state={q['state']}")).text()
    tokens = json.loads(re.search(r"const t=(\{.*?\});", page).group(1))
    assert tokens["access_token"] == "access-1" and tokens["refresh_token"] == "refresh-1"
    assert tokens["client_id"] == q["client_id"]
    assert 'location.replace("../#/enroll")' in page
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-1"})).status == 200

    # A code works once.
    again = await (await client.get(f"/auth/callback?code=the-code&state={q['state']}")).text()
    assert "Not signed in" in again and "access-1" not in again


async def test_refresh_and_sign_out(client, home_assistant):
    _, state = home_assistant
    _, q = await log_in(client, home_assistant)
    await client.get(f"/auth/callback?code=the-code&state={q['state']}")

    r = await client.post("/auth/refresh", json={"refresh_token": "refresh-1", "client_id": q["client_id"]})
    assert (await r.json())["access_token"] == "access-2"
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-2"})).status == 200

    await client.post("/auth/logout", json={"refresh_token": "refresh-1", "access_token": "access-2"})
    assert state["revoked"] == ["refresh-1"]
    assert (await client.get("/api/people", headers={"Authorization": "Bearer access-2"})).status == 401
    r = await client.post("/auth/refresh", json={"refresh_token": "refresh-1", "client_id": q["client_id"]})
    assert r.status == 401


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
