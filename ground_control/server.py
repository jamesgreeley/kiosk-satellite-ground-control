"""HTTP and WebSocket API: the web UI's, and Kiosk Satellite's sockets.

Everything but ``/health`` and ``/auth`` needs a Home Assistant token. As a
Home Assistant app the panel needs none, since ingress has signed the user in.
The kiosk sockets are for Kiosk Satellite only (ground_control.clients).
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .logs import LOGS, line as log_line
from .audio import float_to_pcm16, pcm16_to_float, read_wav
from .auth import TokenCheck, bearer
from .clients import ClientCheck, challenge
from .enroll import GuidedEnrollment
from .models import SPEAKER_MODELS
from .login import HomeAssistantLogin, LoginError
from .pictures import Pictures
from .suggestions import Suggestions
from .service import GroundControl
from .sessions import new_id
from .updates import UpdateCheck

_LOG = logging.getLogger(__name__)

VM = web.AppKey("vm", GroundControl)
AUTH = web.AppKey("auth", TokenCheck)
OPEN = web.AppKey("open", bool)
PICTURES = web.AppKey("pictures", Pictures)
SUGGESTIONS = web.AppKey("suggestions", Suggestions)
LOGIN = web.AppKey("login", HomeAssistantLogin)
UPDATES = web.AppKey("updates", UpdateCheck)
CLIENTS = web.AppKey("clients", ClientCheck)
SOCKETS = ("/api/session", "/api/enroll", "/api/kiosk", "/api/events")
MIN_PARTIAL_ENROLL_S = 10.0
WEB = Path(__file__).resolve().parent / "web"
INGRESS_PROXY = "172.30.32.2"
"""The Supervisor's address. Home Assistant has signed the user in before it proxies."""


def _from_ingress(request: web.Request) -> bool:
    return request.remote == INGRESS_PROXY and "X-Ingress-Path" in request.headers


_REFUSED: dict[tuple[str, str], float] = {}


async def _signed_in(request: web.Request, token: str | None = None) -> bool:
    if request.app[OPEN] or _from_ingress(request):
        return True
    token = token or bearer(request.headers.get("Authorization"))
    ok = await request.app[AUTH].valid(token)
    if not ok:
        # Once a minute per address and path, so a kiosk retrying doesn't flood the log.
        key = (request.remote or "?", request.path)
        if time.monotonic() - _REFUSED.get(key, -60.0) >= 60.0:
            _REFUSED[key] = time.monotonic()
            _LOG.warning("sign-in refused for %s from %s: %s", request.path, key[0],
                         "no token" if not token else "Home Assistant did not accept the token")
    return ok


@web.middleware
async def require_token(request: web.Request, handler):
    public = request.path in ("/", "/health") or request.path.startswith(("/static/", "/auth/"))
    if public or request.path in SOCKETS or await _signed_in(request):
        return await handler(request)
    raise web.HTTPUnauthorized(text="sign in with a Home Assistant token")


async def _open_socket(request: web.Request, first_type: str,
                       nonce: str | None = None) -> tuple[web.WebSocketResponse, dict[str, Any] | None]:
    """Accept a socket and check its first message and token, after sending ``nonce`` to sign."""
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
    await ws.prepare(request)
    if nonce is not None:
        await ws.send_json({"type": "challenge", "nonce": nonce})
    try:
        first = await ws.receive(timeout=5)
        start = json.loads(first.data) if first.type == WSMsgType.TEXT else {}
    except (asyncio.TimeoutError, ValueError):
        start = {}
    if not isinstance(start, dict) or start.get("type") != first_type:
        await ws.close(code=4400, message=f"send {first_type} first".encode())
        return ws, None
    if not await _signed_in(request, start.get("token")):
        await ws.send_json({"type": "error", "code": "unauthorized"})
        await ws.close(code=4401, message=b"unauthorized")
        return ws, None
    return ws, start


async def _refuse(request: web.Request, ws: web.WebSocketResponse, code: str, why: str) -> None:
    """Close a kiosk socket that isn't Kiosk Satellite, or a kiosk that isn't approved."""
    key = (request.remote or "?", request.path, code)
    if time.monotonic() - _REFUSED.get(key, -60.0) >= 60.0:
        _REFUSED[key] = time.monotonic()
        _LOG.warning("%s refused from %s: %s", request.path, key[0], why)
    await ws.send_json({"type": "error", "code": code})
    await ws.close(code=4403, message=code.encode())


async def _kiosk_start(request: web.Request) -> tuple[web.WebSocketResponse, dict[str, Any] | None]:
    """A session or enrollment socket: its start must carry the ticket of an approved kiosk."""
    ws, start = await _open_socket(request, "start")
    if start is None:
        return ws, None
    vm = request.app[VM]
    device_id = start.get("device_id")
    if not vm.hub.ticket_ok(device_id if isinstance(device_id, str) else None, start.get("ticket")):
        await _refuse(request, ws, "not_approved",
                      f"{device_id or 'no device_id'} has no ticket from an approved kiosk socket")
        return ws, None
    return ws, start


def _features(value: Any) -> dict[str, Any] | None:
    """The Voice Match settings a kiosk says it has on, kept to what the Kiosks page shows."""
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {k: value[k] is True for k in
                           ("voice_match", "trimming", "isolation", "speaker_sensor", "show_speaker") if k in value}
    if value.get("policy") in ("anyone", "enrolled", "people"):
        out["policy"] = value["policy"]
    people = value.get("people")
    if isinstance(people, list):
        out["people"] = [p for p in people if isinstance(p, str)][:50]
    return out


def _person(vm: GroundControl, p) -> dict[str, Any]:
    enrollments = vm.people.enrollments(p.id)
    return {**asdict(p), "speech_s": round(vm.people.speech_seconds(p.id), 1),
            "clips": len(vm.people.clips(p.id)), "has_voice": vm.people.has_voice(p.id),
            "enrollments": enrollments, "models": sorted({e["model"] for e in enrollments if e.get("model")}),
            "picture": bool((p.things or {}).get("person"))}


async def person_picture(request: web.Request) -> web.Response:
    """The picture of the Home Assistant person a person is linked to."""
    vm = request.app[VM]
    person = vm.people.get(request.match_info["id"])
    entity = (person.things or {}).get("person") if person else None
    found = await request.app[PICTURES].get(entity or "", bearer(request.headers.get("Authorization")))
    if found is None:
        raise web.HTTPNotFound(text="no picture")
    body, ctype = found
    return web.Response(body=body, content_type=ctype, headers={"Cache-Control": "private, max-age=600"})


async def suggestions(request: web.Request) -> web.Response:
    """What a person's things can be, from Home Assistant, for the People page."""
    return web.json_response(await request.app[SUGGESTIONS].get(bearer(request.headers.get("Authorization"))))


def _source(vm: GroundControl, device_id: str | None, name: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Where an enrollment was recorded, from the kiosk when it is known."""
    k = vm.hub.kiosks.get(device_id or "")
    return {"device_id": device_id, "device_name": (k.name if k else None) or name,
            "model": (k.model if k else None) or model}


def _known(vm: GroundControl, request: web.Request) -> str:
    pid = request.match_info["id"]
    if vm.people.get(pid) is None:
        raise web.HTTPNotFound(text="no such person")
    return pid


async def _json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as err:
        raise web.HTTPBadRequest(text="send JSON") from err
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="send a JSON object")
    return body


# ── status ────────────────────────────────────────────────────────────────

# ── log in with Home Assistant ──────────────────────────────────────────────

def _origin(request: web.Request) -> str:
    """The address the browser opened Ground Control at, behind a reverse proxy too."""
    proto = request.headers.get("X-Forwarded-Proto", request.scheme).split(",")[0].strip()
    host = request.headers.get("X-Forwarded-Host", request.host).split(",")[0].strip()
    return f"{proto}://{host}"


def _page(title: str, body: str) -> web.Response:
    return web.Response(content_type="text/html", headers={"Cache-Control": "no-store"}, text=(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        '<body style="font-family:system-ui,sans-serif;margin:48px 24px;max-width:520px">'
        f"{body}</body></html>"))


async def auth_info(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    return web.json_response({"login": login.available and not request.app[OPEN],
                              "ha_url": login.ha_url if login.available else None})


async def auth_login(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    if not login.available:
        raise web.HTTPNotFound(text="open Ground Control from the Home Assistant sidebar")
    raise web.HTTPFound(login.start(_origin(request), request.query.get("next", "")))


async def auth_callback(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    if "error" in request.query or "code" not in request.query:
        return _page("Not signed in", "<h1>Not signed in</h1><p>Home Assistant didn't sign you in.</p>"
                     '<p><a href="../">Try again</a></p>')
    try:
        tokens, back = await login.finish(request.query.get("state", ""), request.query["code"])
    except LoginError as err:
        _LOG.warning("login with Home Assistant failed: %s", err)
        return _page("Not signed in", f"<h1>Not signed in</h1><p>{html.escape(str(err))}</p>"
                     '<p><a href="../">Try again</a></p>')
    _LOG.info("signed in with Home Assistant from %s", request.remote)
    # The web UI keeps the tokens in this browser, then goes back to where it was.
    data = json.dumps({**tokens, "expires": None}).replace("</", "<\\/")
    return _page("Signing in", "<p>Signing in…</p><script>"
                 f"const t={data};t.expires=Date.now()+t.expires_in*1000;delete t.expires_in;"
                 'try{localStorage.setItem("ground-control-auth",JSON.stringify(t))}catch(e){}'
                 f"location.replace({json.dumps('../' + back)});</script>")


async def auth_refresh(request: web.Request) -> web.Response:
    body = await _json(request)
    token, client_id = body.get("refresh_token"), body.get("client_id")
    if not isinstance(token, str) or not isinstance(client_id, str):
        raise web.HTTPBadRequest(text="refresh_token and client_id are needed")
    try:
        return web.json_response(await request.app[LOGIN].refresh(token, client_id))
    except LoginError as err:
        if err.refused:
            raise web.HTTPUnauthorized(text=str(err)) from err
        raise web.HTTPBadGateway(text=str(err)) from err


async def auth_logout(request: web.Request) -> web.Response:
    body = await _json(request)
    if isinstance(body.get("access_token"), str):
        request.app[AUTH].forget(body["access_token"])
    if isinstance(body.get("refresh_token"), str):
        await request.app[LOGIN].revoke(body["refresh_token"])
    _LOG.info("signed out from %s", request.remote)
    return web.json_response({"signed_out": True})


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "version": __version__})


async def status(request: web.Request) -> web.Response:
    vm = request.app[VM]
    return web.json_response({**asdict(vm.status), "version": __version__, "people": len(vm.people.all()),
                              "model_label": vm.model.label, "threshold": vm.model.threshold,
                              "other_threshold": vm.model.other_threshold, "cuda": vm.cfg.cuda,
                              "on_gpu": vm.active.on_gpu, "threads": vm.cfg.threads,
                              "kiosks": sum(k.connected for k in vm.hub.kiosks.values()),
                              "isolation": vm.isolator() is not None,
                              "isolation_on_gpu": bool(vm.isolator() and vm.isolator().on_gpu),
                              "update": request.app[UPDATES].available() if UPDATES in request.app else None})

MISSING_PEOPLE_FIX = "choose them again in Kiosk Satellite, under Voice Satellite > Ground Control Voice Match > People"


def _check_people(vm: GroundControl, device_id: str | None, people: Any) -> list[str]:
    """The people a kiosk answers that Ground Control doesn't have, as after
    moving to another Ground Control. Logged once for each set, since
    otherwise every command is rejected without a word why."""
    ids = [p for p in people if isinstance(p, str)] if isinstance(people, (list, tuple)) else []
    missing = [p for p in ids if vm.people.get(p) is None]
    k = vm.hub.kiosks.get(device_id or "")
    if k is not None and tuple(missing) != k._missing_warned:
        k._missing_warned = tuple(missing)
        if missing:
            n = f"{len(missing)} {'person' if len(missing) == 1 else 'people'}"
            known = len(ids) - len(missing)
            _LOG.warning("kiosk %s answers %s Ground Control doesn't have%s: %s", k.name, n,
                         "" if known else ", so it answers nobody", MISSING_PEOPLE_FIX)
    return missing


def _check_features(vm: GroundControl, device_id: str | None, features: dict[str, Any] | None) -> None:
    if features and features.get("policy") == "people":
        _check_people(vm, device_id, features.get("people"))


# ── sessions ──────────────────────────────────────────────────────────────

async def session_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    mode = start.get("mode", "live")
    policy = start.get("policy", "anyone")
    people = start.get("people") or []
    if (mode not in ("live", "gated", "follow") or policy not in ("anyone", "enrolled", "people")
            or not isinstance(people, list) or not all(isinstance(p, str) for p in people)):
        await ws.close(code=4400, message=b"bad mode or policy")
        return ws
    people = tuple(people) if policy == "people" else ()
    if mode == "follow":
        anchor = None
        if isinstance(start.get("anchor"), str):
            try:
                anchor = pcm16_to_float(base64.b64decode(start["anchor"]))[: 3 * 16000]
            except ValueError:
                anchor = None
        speaker = start.get("speaker") if isinstance(start.get("speaker"), str) else None
        isolation = start.get("isolation") is not False
        session = await vm.run(vm.new_follow, policy, anchor, people, speaker, isolation)
    else:
        session = vm.new_session(mode=mode, policy=policy, people=people, learn=start.get("learn"))
    device_id = start.get("device_id")
    # A follow-up in a conversation joins the session of its first turn.
    joins = start.get("continues") if mode == "follow" else None
    joins = joins if await vm.run(vm.sessions.continuing, joins, device_id) else None
    sid = joins or new_id()
    kiosk = vm.hub.seen(device_id, start.get("device_name"))
    if kiosk:
        kiosk.sessions += 1
        kiosk.in_session = True
    vm.status.sessions += 1
    vm.status.sessions_total += 1
    started = time.time()
    kept: list[dict[str, Any]] = []
    who = kiosk.name if kiosk else device_id or "an unknown kiosk"
    _LOG.info("session %s from %s: %s mode, answers %s%s%s", sid, who, mode, policy,
              f" ({len(people)} chosen)" if policy == "people" else "", ", a follow-up" if joins else "")
    if policy == "people":
        _check_people(vm, device_id, people)
    vm.hub.publish({"type": "session_start", "session": sid, "device_id": device_id, "mode": mode})

    async def send(events: list[dict[str, Any]]) -> None:
        for e in events:
            if e["type"] == "audio":
                if not ws.closed:
                    await ws.send_bytes(float_to_pcm16(e["audio"]))
                continue
            if e["type"] == "keepalive":
                # For the kiosk only: not part of the session's record or the live view.
                if not ws.closed:
                    await ws.send_json(e)
                continue
            kept.append(e)
            if e["type"] != "stats":
                _LOG.debug("session %s: %s", sid, _event_text(e))
            vm.hub.publish({**e, "session": sid, "device_id": device_id})
            if not ws.closed:
                await ws.send_json(e)

    published = 0

    def live_scores() -> None:
        nonlocal published
        for item in session.timeline[published:]:
            vm.hub.publish({**item, "type": "score", "kind": item["kind"], "session": sid, "device_id": device_id})
        published = len(session.timeline)

    try:
        await ws.send_json({"type": "ready", "session": sid, "model": vm.model.key,
                            "threshold": vm.model.threshold})
        if mode == "follow":
            _LOG.debug("session %s: wake word anchor %s", sid,
                       f"{len(anchor) / 16000:.1f} s" if anchor is not None else "none")
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await send(await vm.run(session.feed, pcm16_to_float(msg.data)))
                live_scores()
            elif msg.type == WSMsgType.TEXT:
                kind = json.loads(msg.data).get("type")
                if kind == "turn" and mode != "follow":
                    await send(session.new_turn())
                elif kind == "end":
                    break
            else:
                break
    except Exception:
        _LOG.exception("session %s from %s failed", sid, who)
        raise
    finally:
        vm.status.sessions -= 1
        if kiosk:
            kiosk.in_session = False
        await send(await vm.run(session.finish))
        stats = next((e for e in kept if e["type"] == "stats"), {})
        _LOG.info("session %s from %s ended: %s", sid, who, _session_text(stats, kept))
        record = {"id": sid, "started": started, "device_id": device_id,
                  "device_name": kiosk.name if kiosk else None, "mode": mode, "policy": policy, "people": list(people),
                  "model": vm.model.key, "threshold": vm.model.threshold,
                  "other_threshold": vm.model.other_threshold,
                  "events": [e for e in kept if e["type"] != "stats"], "timeline": session.timeline,
                  "stats": stats}
        if mode == "follow":
            # What went to the assistant, in seconds of what the kiosk sent.
            record["released"] = [[round(a / 16000, 3), round(b / 16000, 3)] for a, b in session.released]
        if joins:
            await vm.run(vm.sessions.append, sid, record, session.raw_audio(), session.cleaned_audio())
        else:
            await vm.run(vm.sessions.save, record, session.raw_audio(), session.cleaned_audio(),
                         getattr(session, "anchor_audio", None))
        vm.hub.publish({"type": "session_end", "session": sid, "device_id": device_id, "stats": stats})
        if not ws.closed:
            await ws.close()
    return ws


async def sessions_list(request: web.Request) -> web.Response:
    vm = request.app[VM]
    limit = min(int(request.query.get("limit", 100)), 500)
    return web.json_response({"sessions": await vm.run(vm.sessions.list, limit, request.query.get("device"))})


async def session_get(request: web.Request) -> web.Response:
    record = request.app[VM].sessions.get(request.match_info["sid"])
    if record is None:
        raise web.HTTPNotFound(text="no such session")
    return web.json_response(record)


async def session_audio(request: web.Request) -> web.StreamResponse:
    path = request.app[VM].sessions.audio_path(request.match_info["sid"], request.match_info["kind"])
    if path is None:
        raise web.HTTPNotFound(text="no audio for that session")
    return web.FileResponse(path, headers={"Content-Type": "audio/wav"})


SESSION_FILES = (("raw", "as-heard.wav"), ("clean", "sent-to-assistant.wav"), ("wake", "wake-word.wav"))


def _session_zip(vm: GroundControl, record: dict[str, Any]) -> bytes:
    """Everything about one session for an issue report: its audio, its record
    and the service's details with the log lines from while it ran."""
    import io
    import zipfile

    sid = record["id"]
    start = float(record.get("started") or 0.0) - 2.0
    # The record is written when its last turn ends, follow-ups included.
    saved = vm.sessions.dir / f"{sid}.json"
    end = (saved.stat().st_mtime if saved.is_file() else start) + 5.0
    lines = [log_line(e) for e in LOGS.since(0, "DEBUG") if start <= e["t"] <= end]
    text = [*_report_header(vm), "", f"Session {sid}, log lines from while it ran:" if lines
            else f"Session {sid}: the log from while it ran is gone (the service restarted since).", *lines]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps(record, indent=1))
        z.writestr("ground-control.txt", "\n".join(text) + "\n")
        for kind, name in SESSION_FILES:
            path = vm.sessions.audio_path(sid, kind)
            if path is not None:
                z.write(path, name)
    return buf.getvalue()


async def session_download(request: web.Request) -> web.Response:
    """One zip with the session's audio, its record and the log from while it ran."""
    vm = request.app[VM]
    record = vm.sessions.get(request.match_info["sid"])
    if record is None:
        raise web.HTTPNotFound(text="no such session")
    body = await vm.run(_session_zip, vm, record)
    return web.Response(body=body, content_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="ground-control-session-{record["id"]}.zip"'})


async def session_delete(request: web.Request) -> web.Response:
    if not request.app[VM].sessions.delete(request.match_info["sid"]):
        raise web.HTTPNotFound(text="no such session")
    return web.Response(status=204)


async def session_enroll(request: web.Request) -> web.Response:
    """Add a saved session's voice to a person: the cleaned audio, speech only."""
    vm = request.app[VM]
    sid = request.match_info["sid"]
    body = await _json(request)
    pid = body.get("person_id")
    if not pid or vm.people.get(pid) is None:
        raise web.HTTPBadRequest(text="person_id must name a person")
    path = vm.sessions.audio_path(sid, "clean") or vm.sessions.audio_path(sid, "raw")
    if path is None:
        raise web.HTTPNotFound(text="no audio for that session")
    record = vm.sessions.get(sid) or {}
    source = _source(vm, record.get("device_id"), record.get("device_name"))
    try:
        seconds = await vm.run(vm.people.enroll, pid, read_wav(path), vm.enroll_models(), source)
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("added session %s to %s: %.1f s of speech", sid, vm.people.get(pid).name, seconds)
    return web.json_response({"added_s": round(seconds, 1), **_person(vm, vm.people.get(pid))})


# ── enrollment ────────────────────────────────────────────────────────────

def _passage(value: Any) -> list[str] | None:
    """The sentences the kiosk shows while the person reads, so the app shows
    the same text. Anything that isn't a short list of sentences is ignored."""
    if not isinstance(value, list) or not 0 < len(value) <= 40:
        return None
    if not all(isinstance(v, str) and 0 < len(v.strip()) <= 400 for v in value):
        return None
    return [v.strip() for v in value]


async def enroll_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    pid = start.get("person_id")
    try:
        target = float(start.get("target_s", 45.0))
        if not 10.0 <= target <= 300.0:
            raise ValueError("target_s must be 10 to 300 seconds")
        if not pid:
            pid = vm.people.add(str(start.get("name", ""))).id
        elif vm.people.get(pid) is None:
            raise ValueError("no such person")
    except (ValueError, TypeError) as err:
        await ws.send_json({"type": "error", "code": "bad_request", "message": str(err)})
        await ws.close(code=4400)
        return ws
    person = vm.people.get(pid)
    device_id = start.get("device_id")
    passage = _passage(start.get("passage"))
    guide = GuidedEnrollment(vm.active, vm.vad, vm.model, target_s=target, passage=passage)
    await ws.send_json({"type": "ready", "person_id": pid, "name": person.name, "target_s": target})

    def live(event: dict[str, Any]) -> None:
        vm.hub.publish({**event, "type": f"enroll_{event['type']}", "person_id": pid, "name": person.name,
                        "device_id": device_id})

    live({"type": "start", "target_s": target, "passage": guide.passage, "sentence": 0})
    kiosk = vm.hub.kiosks.get(device_id) if device_id else None
    where = kiosk.name if kiosk else start.get("device_name") or device_id or "the browser"
    _LOG.info("enrolling %s on %s, %.0f s of speech to go", person.name, where, target)
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            for e in await vm.run(guide.feed, pcm16_to_float(msg.data)):
                live(e)
                await ws.send_json(e)
            if guide.done:
                break
        elif msg.type == WSMsgType.TEXT and json.loads(msg.data).get("type") == "end":
            break
        elif msg.type != WSMsgType.TEXT:
            break
    if guide.done or guide.speech_s >= MIN_PARTIAL_ENROLL_S:
        try:
            source = _source(vm, device_id, start.get("device_name"), start.get("model"))
            added = await vm.run(vm.people.enroll, pid, guide.speech(), vm.enroll_models(), source)
            saved: dict[str, Any] = {"type": "saved", "added_s": round(added, 1),
                                     "speech_s": round(vm.people.speech_seconds(pid), 1),
                                     "dropped_s": round(guide.dropped_s, 1)}
        except ValueError as err:
            saved = {"type": "error", "code": "not_saved", "message": str(err)}
    else:
        saved = {"type": "error", "code": "too_short",
                 "message": f"only {guide.speech_s:.0f} s of speech, nothing was saved"}
    live(saved)
    if saved["type"] == "saved":
        _LOG.info("enrollment for %s saved: %.1f s added, %.1f s left out as another voice", person.name,
                  saved["added_s"], saved["dropped_s"])
    else:
        _LOG.warning("enrollment for %s not saved: %s", person.name, saved["message"])
    if not ws.closed:
        await ws.send_json(saved)
        await ws.close()
    return ws


# ── kiosks and live events ────────────────────────────────────────────────

async def kiosk_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    nonce = challenge()
    ws, hello = await _open_socket(request, "hello", nonce)
    if hello is None:
        return ws
    device_id = str(hello.get("device_id") or "")
    if not device_id:
        await ws.close(code=4400, message=b"device_id required")
        return ws
    if not request.app[CLIENTS].verify(nonce, device_id, hello.get("proof")):
        await _refuse(request, ws, "not_kiosk_satellite", f"{device_id} did not prove it is Kiosk Satellite")
        return ws
    k = vm.hub.register(device_id, str(hello.get("name") or device_id), str(hello.get("version") or ""), ws,
                        model=str(hello.get("model") or ""), enroll=hello.get("enroll") is True,
                        features=_features(hello.get("features")))
    _LOG.info("kiosk %s connected from %s: %s, Kiosk Satellite %s, enrollment %s%s", k.name, request.remote,
              k.model or "unknown model", k.version or "unknown version", "allowed" if k.enroll else "off",
              "" if k.access == "approved" else f", {'waiting for approval' if k.access == 'pending' else 'declined'}")
    _check_features(vm, device_id, k.features)
    try:
        await _send_access(request, k.device_id)
        async for msg in ws:
            if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                break
            vm.hub.seen(device_id)
            if msg.type == WSMsgType.TEXT:
                _from_kiosk(vm, device_id, msg.data)
    finally:
        vm.hub.unregister(device_id, ws)
        _LOG.info("kiosk %s disconnected%s", k.name, f" ({ws.close_code})" if ws.close_code else "")
    return ws


async def _send_access(request: web.Request, device_id: str) -> None:
    """Tell a connected kiosk where it stands: a ticket and everyone, or wait."""
    vm = request.app[VM]
    k = vm.hub.kiosks.get(device_id)
    if k is None:
        return
    if k.access == "approved":
        ticket = vm.hub.issue_ticket(device_id)
        await vm.hub.send(device_id, {"type": "ticket", "ticket": ticket})
        await vm.hub.send(device_id, await _people_message(request))
    else:
        await vm.hub.send(device_id, {"type": k.access})


async def _people_message(request: web.Request) -> dict[str, Any]:
    vm = request.app[VM]
    return {"type": "people", "people": await vm.run(vm.people.summary), "model": vm.model.key,
            "version": __version__, "app_path": vm.app_path}


def _from_kiosk(vm: GroundControl, device_id: str, data: str) -> None:
    try:
        msg = json.loads(data)
    except ValueError:
        return
    if not isinstance(msg, dict):
        return
    if msg.get("type") == "settings":
        before = vm.hub.kiosks.get(device_id)
        was = before.enroll if before else None
        features = _features(msg.get("features"))
        vm.hub.update(device_id, msg.get("enroll") is True, features)
        _check_features(vm, device_id, features)
        if was != (msg.get("enroll") is True):
            _LOG.info("kiosk %s: enrollment %s", _kiosk_name(vm, device_id), "allowed" if msg.get("enroll") is True else "off")
    elif msg.get("type") == "enroll_error":
        _LOG.warning("kiosk %s could not start an enrollment: %s %s", _kiosk_name(vm, device_id),
                     msg.get("code") or "refused", msg.get("message") or "")
        # The kiosk couldn't start: the web UI waiting on it hears why.
        vm.hub.publish({"type": "enroll_error", "person_id": msg.get("person_id"), "device_id": device_id,
                        "code": str(msg.get("code") or "refused"), "message": str(msg.get("message") or "")})


async def events_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _open_socket(request, "listen")
    if start is None:
        return ws
    vm.hub.listen(ws)
    try:
        await ws.send_json({"type": "kiosks", "kiosks": [k.public() for k in vm.hub.kiosks.values()]})
        async for msg in ws:
            if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                break
    finally:
        vm.hub.unlisten(ws)
    return ws


async def kiosks_list(request: web.Request) -> web.Response:
    return web.json_response({"kiosks": [k.public() for k in request.app[VM].hub.kiosks.values()]})


async def kiosk_access(request: web.Request) -> web.Response:
    """Approve or decline a kiosk, from the Kiosks page."""
    vm = request.app[VM]
    access = (await _json(request)).get("access")
    device_id = request.match_info["device_id"]
    k = vm.hub.set_access(device_id, access) if access in ("approved", "declined", "pending") else None
    if k is None:
        raise web.HTTPBadRequest(text="access must be approved, declined or pending, for a known kiosk")
    _LOG.info("kiosk %s %s", k.name, {"approved": "approved", "declined": "declined", "pending": "waiting for approval"}[access])
    if k.connected:
        await _send_access(request, device_id)
    return web.json_response(k.public())


async def kiosk_enroll(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    person = vm.people.get(str(body.get("person_id", "")))
    if person is None:
        raise web.HTTPBadRequest(text="person_id must name a person")
    kiosk = vm.hub.kiosks.get(request.match_info["device_id"])
    if kiosk is None or kiosk.access != "approved":
        raise web.HTTPConflict(text="that kiosk isn't approved")
    if kiosk.connected and not kiosk.enroll:
        raise web.HTTPConflict(text="Allow enrollment here is off on that kiosk")
    sent = await vm.hub.send(request.match_info["device_id"], {
        "type": "enroll", "person_id": person.id, "name": person.name, "color": person.color,
        "target_s": float(body.get("target_s", 45.0))})
    if not sent:
        raise web.HTTPConflict(text="that kiosk isn't connected")
    _LOG.info("asked %s to enroll %s", _kiosk_name(vm, request.match_info["device_id"]), person.name)
    return web.json_response({"sent": True})


# ── people ────────────────────────────────────────────────────────────────

async def people_list(request: web.Request) -> web.Response:
    vm = request.app[VM]
    return web.json_response({"people": [_person(vm, p) for p in vm.people.all()]})


async def people_add(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    try:
        p = vm.people.add(str(body.get("name", "")), body.get("color"))
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("added %s", p.name)
    return web.json_response(_person(vm, p), status=201)


async def people_update(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    body = await _json(request)
    try:
        before = vm.people.get(pid).name
        p = vm.people.update(pid, name=body.get("name"), color=body.get("color"), things=body.get("things"))
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    changed = [k for k in ("name", "color", "things") if body.get(k) is not None]
    _LOG.info("changed %s%s: %s", before, f" (now {p.name})" if p.name != before else "", ", ".join(changed) or "nothing")
    return web.json_response(_person(vm, p))


async def people_forget(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    name = vm.people.get(pid).name
    vm.people.forget(pid)
    _LOG.info("forgot %s and their voice", name)
    return web.Response(status=204)


async def people_enroll(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    try:
        audio = read_wav(await request.read(), any_rate=True)
        source = {"device_id": None, "device_name": request.query.get("device_name") or "Uploaded file",
                  "model": request.query.get("model")}
        seconds = await vm.run(vm.people.enroll, pid, audio, vm.enroll_models(), source)
    except (ValueError, EOFError) as err:
        _LOG.warning("a file for %s was not enrolled: %s", vm.people.get(pid).name, err)
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("enrolled %s from a file: %.1f s of speech", vm.people.get(pid).name, seconds)
    return web.json_response({"added_s": round(seconds, 1), **_person(vm, vm.people.get(pid))})


async def enrollment_delete(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    try:
        await vm.run(vm.people.delete_enrollment, pid, request.match_info["eid"], vm.enroll_models())
    except KeyError as err:
        raise web.HTTPNotFound(text="no such enrollment") from err
    _LOG.info("deleted an enrollment of %s, their voice was rebuilt", vm.people.get(pid).name)
    return web.json_response(_person(vm, vm.people.get(pid)))


# ── settings ──────────────────────────────────────────────────────────────

async def settings_get(request: web.Request) -> web.Response:
    return web.json_response(request.app[VM].settings())


async def settings_update(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    rest = {k: v for k, v in body.items() if k != "model"}
    if rest:
        try:
            await vm.run(vm.update_settings, rest)
        except (ValueError, TypeError) as err:
            raise web.HTTPBadRequest(text=str(err)) from err
        _LOG.info("settings changed: %s", ", ".join(f"{k} {v}" for k, v in rest.items()))
    model = body.get("model")
    if model is not None:
        if model not in SPEAKER_MODELS:
            raise web.HTTPBadRequest(text="unknown model")
        _LOG.info("switching the speaker model to %s", model)
        asyncio.get_running_loop().create_task(vm.switch_model(model))
        await asyncio.sleep(0)  # let the switch mark itself as rebuilding
    return await settings_get(request)


# ── logs ──────────────────────────────────────────────────────────────────

def _kiosk_name(vm: GroundControl, device_id: str | None) -> str:
    k = vm.hub.kiosks.get(device_id or "")
    return k.name if k else device_id or "an unknown kiosk"


def _event_text(e: dict[str, Any]) -> str:
    """A session event in a few words, for detailed logs."""
    rest = {k: v for k, v in e.items() if k not in ("type", "t") and v is not None}
    at = f" at {e['t']:.2f} s" if isinstance(e.get("t"), (int, float)) else ""
    return f"{e['type']}{at}" + (f" {json.dumps(rest, separators=(',', ':'))}" if rest else "")


def _session_text(stats: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """How a session went, in one line: who, why it ended and how much audio."""
    speaker = stats.get("speaker") or {}
    if stats.get("rejected"):
        who = (f"rejected {speaker['name']}, not one of the people this kiosk answers" if speaker.get("name")
               else "rejected an unknown voice")
    elif speaker.get("name"):
        score = f" {speaker['score']}" if speaker.get("score") is not None else ""
        who = f"{speaker['name']} ({speaker.get('by') or 'voice'}{score})"
    elif any(e["type"] == "speaker" for e in events):
        who = "an unknown voice"
    else:
        who = "nobody talked"
    parts = [who, f"ended by {stats.get('end') or 'the kiosk'}", f"{stats.get('audio_s', 0)} s of audio"]
    if stats.get("turns"):
        parts.append(f"{stats['turns']} turns, {stats.get('released_s', 0)} s to the assistant")
    if stats.get("dropped_s"):
        parts.append(f"{stats['dropped_s']} s of other voices left out")
    if stats.get("cpu_ms") is not None:
        parts.append(f"{stats['cpu_ms']} ms of CPU")
    return ", ".join(parts)


async def logs_get(request: web.Request) -> web.Response:
    try:
        after = int(request.query.get("after", 0))
    except ValueError as err:
        raise web.HTTPBadRequest(text="after must be a number") from err
    entries = LOGS.since(after, request.query.get("level", "DEBUG"))[-2000:]
    return web.json_response({"entries": entries, "last": LOGS.last, "detailed": LOGS.detailed,
                              "started": LOGS.started})


async def logs_update(request: web.Request) -> web.Response:
    body = await _json(request)
    if not isinstance(body.get("detailed"), bool):
        raise web.HTTPBadRequest(text="detailed must be true or false")
    LOGS.detailed = body["detailed"]
    return web.json_response({"detailed": LOGS.detailed})


def _report_header(vm: GroundControl) -> list[str]:
    import platform

    s = vm.settings()
    kiosks = sorted(vm.hub.kiosks.values(), key=lambda k: (not k.connected, k.name.lower()))
    up = time.time() - LOGS.started
    lines = [
        f"Kiosk Satellite Ground Control {__version__}",
        f"Report made {time.strftime('%Y-%m-%d %H:%M:%S %z')}, running for {int(up // 3600)} h {int(up % 3600 // 60)} min",
        f"Runs {'as a Home Assistant app' if vm.cfg.supervisor_token else 'standalone'}"
        f" on {platform.system()} {platform.machine()}, Python {platform.python_version()}",
        f"Speaker model {vm.model.label} on {'the GPU' if vm.active.on_gpu else 'the CPU'}"
        f" ({vm.cfg.threads} threads, CUDA {'on' if vm.cfg.cuda else 'off'}),"
        f" thresholds {vm.model.threshold} and {vm.model.other_threshold}",
        f"Settings: {json.dumps({k: v for k, v in s.items() if k not in ('models', 'default_threshold', 'default_other_threshold')}, separators=(', ', ': '))}",
        f"People: {len(vm.people.all())}, {sum(vm.people.has_voice(p.id) for p in vm.people.all())} with a voice",
        f"Detailed logs {'on' if LOGS.detailed else 'off'}",
        "Kiosks:" if kiosks else "Kiosks: none",
    ]
    for k in kiosks:
        state = "connected" if k.connected else f"last seen {time.strftime('%Y-%m-%d %H:%M', time.localtime(k.last_seen))}"
        lines.append(f"  {k.name}: {k.model or 'unknown model'}, Kiosk Satellite {k.version or '?'}, {state},"
                     f" {k.sessions} session{'' if k.sessions == 1 else 's'}, enrollment {'allowed' if k.enroll else 'off'}")
    return lines


async def logs_report(request: web.Request) -> web.Response:
    vm = request.app[VM]
    text = "\n".join([*_report_header(vm), "", *map(log_line, LOGS.since(0, request.query.get("level", "DEBUG")))])
    name = f"ground-control-logs-{time.strftime('%Y%m%d-%H%M')}.txt"
    return web.Response(text=text + "\n", content_type="text/plain", charset="utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


ADVERTISER = web.AppKey("advertiser", object)


async def _find_app_path(vm: GroundControl, supervisor_token: str) -> None:
    """As a Home Assistant app: where its page is, for the kiosks' Open."""
    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as http, \
                http.get("http://supervisor/addons/self/info",
                         headers={"Authorization": f"Bearer {supervisor_token}"}) as r:
            slug = (await r.json())["data"]["slug"]
        vm.app_path = f"/hassio/ingress/{slug}"
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
        _LOG.warning("could not read this app's slug from the Supervisor: %s", err)


def build_app(vm: GroundControl, token_check: TokenCheck | None = None, mdns: bool = False,
              clients: ClientCheck | None = None) -> web.Application:
    app = web.Application(middlewares=[require_token], client_max_size=512 << 20)
    app[VM] = vm
    app[AUTH] = token_check or TokenCheck(vm.cfg.ha_url, supervisor_token=vm.cfg.supervisor_token)
    app[OPEN] = vm.cfg.auth == "none"
    app[CLIENTS] = clients or ClientCheck()
    app[PICTURES] = Pictures(vm.cfg.ha_url, vm.cfg.supervisor_token)
    app[SUGGESTIONS] = Suggestions(vm.cfg.ha_url, vm.cfg.supervisor_token)
    app[LOGIN] = HomeAssistantLogin(vm.cfg.ha_url)
    if vm.cfg.update_check:
        app[UPDATES] = UpdateCheck(__version__)
    r = app.router

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    r.add_get("/", index)
    r.add_static("/static/", WEB, append_version=False)
    r.add_get("/health", health)
    r.add_get("/auth/info", auth_info)
    r.add_get("/auth/login", auth_login)
    r.add_get("/auth/callback", auth_callback)
    r.add_post("/auth/refresh", auth_refresh)
    r.add_post("/auth/logout", auth_logout)
    r.add_get("/api/status", status)
    r.add_get("/api/session", session_ws)
    r.add_get("/api/enroll", enroll_ws)
    r.add_get("/api/kiosk", kiosk_ws)
    r.add_get("/api/events", events_ws)
    r.add_get("/api/people", people_list)
    r.add_get("/api/people/{id}/picture", person_picture)
    r.add_get("/api/suggestions", suggestions)
    r.add_post("/api/people", people_add)
    r.add_patch("/api/people/{id}", people_update)
    r.add_delete("/api/people/{id}", people_forget)
    r.add_post("/api/people/{id}/enroll", people_enroll)
    r.add_delete("/api/people/{id}/enrollments/{eid}", enrollment_delete)
    r.add_get("/api/sessions", sessions_list)
    r.add_get("/api/sessions/{sid}", session_get)
    r.add_get("/api/sessions/{sid}/audio/{kind}", session_audio)
    r.add_get("/api/sessions/{sid}/download", session_download)
    r.add_post("/api/sessions/{sid}/enroll", session_enroll)
    r.add_delete("/api/sessions/{sid}", session_delete)
    r.add_get("/api/kiosks", kiosks_list)
    r.add_post("/api/kiosks/{device_id}/enroll", kiosk_enroll)
    r.add_post("/api/kiosks/{device_id}/access", kiosk_access)
    r.add_get("/api/settings", settings_get)
    r.add_patch("/api/settings", settings_update)
    r.add_get("/api/logs", logs_get)
    r.add_patch("/api/logs", logs_update)
    r.add_get("/api/logs/report", logs_report)

    async def on_startup(app: web.Application) -> None:
        loop = asyncio.get_running_loop()

        def people_changed() -> None:
            async def push() -> None:
                await vm.hub.broadcast({"type": "people", "people": await vm.run(vm.people.summary),
                                        "app_path": vm.app_path})
            loop.call_soon_threadsafe(lambda: loop.create_task(push()))

        vm.people.on_change.append(people_changed)
        if UPDATES in app:
            app[UPDATES].start()
        if vm.cfg.supervisor_token:
            await _find_app_path(vm, vm.cfg.supervisor_token)
        if mdns:
            from .mdns import Advertiser

            advertiser = Advertiser(vm.cfg.port, vm.model.key, vm.cfg.mdns_address)
            try:
                await advertiser.start()
                app[ADVERTISER] = advertiser
            except Exception as err:  # noqa: BLE001
                # Kiosks can still be given the address: never stop the service over it.
                _LOG.warning("mDNS advertising failed, kiosks need the address entered: %s", err)

    async def on_cleanup(app: web.Application) -> None:
        if ADVERTISER in app:
            await app[ADVERTISER].stop()
        await app[AUTH].close()
        await app[PICTURES].close()
        await app[SUGGESTIONS].close()
        await app[LOGIN].close()
        if UPDATES in app:
            await app[UPDATES].close()
        vm.close()

    async def revalidate(request: web.Request, response: web.StreamResponse) -> None:
        # The UI's files change with every update: make browsers check, not guess.
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"

    app.on_response_prepare.append(revalidate)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
