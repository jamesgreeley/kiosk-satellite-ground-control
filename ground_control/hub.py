"""Connected kiosks, and live events for the web UI.

A kiosk keeps a control socket open (``/api/kiosk``). Through it the
service sends everyone's names, colors and things, so the kiosk can show
the speaker and resolve "my" on its own, and commands such as starting an
enrollment from the web UI.

A new kiosk waits for approval on the Kiosks page before it gets anything.
Each kiosk's access and the last things it said about itself (name, model,
version, the Voice Match features it has on) are kept in ``kiosks.json``.

The web UI listens on ``/api/events`` for what happens right now: sessions
as they run (the Live view) and enrollment progress.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

_LOG = logging.getLogger(__name__)

ACCESS = ("pending", "approved", "declined")
KEPT = ("name", "model", "version", "enroll", "features", "access", "last_seen")
"""What ``kiosks.json`` keeps of each kiosk."""


@dataclass
class Kiosk:
    device_id: str
    name: str
    model: str = ""
    """The kind of device, as Kiosk Satellite reports it (Echo Show 8 (2nd gen), Galaxy Tab S8)."""
    version: str = ""
    enroll: bool = False
    """Allow enrollment here is on: the web UI can record a voice through this kiosk."""
    access: str = "pending"
    """``approved`` on the Kiosks page, ``declined``, or ``pending`` until then."""
    features: dict[str, Any] = field(default_factory=dict)
    """The Voice Match settings the kiosk has on, as it last said."""
    connected: bool = False
    connected_at: float = 0.0
    last_seen: float = 0.0
    sessions: int = 0
    in_session: bool = False
    _ws: web.WebSocketResponse | None = field(default=None, repr=False)
    _ticket: str | None = field(default=None, repr=False)
    _missing_warned: tuple[str, ...] = field(default=(), repr=False)

    def public(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Hub:
    def __init__(self, path: Path | None = None) -> None:
        self.kiosks: dict[str, Kiosk] = {}
        self._listeners: set[web.WebSocketResponse] = set()
        self._path = path
        self.new = path is None or not path.is_file()
        """No ``kiosks.json`` yet: the service approves the kiosks it already knows."""
        if not self.new:
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError) as err:
                _LOG.warning("could not read %s: %s", path, err)
                saved = {}
            for device_id, v in (saved if isinstance(saved, dict) else {}).items():
                if not isinstance(v, dict):
                    continue
                k = Kiosk(device_id=device_id, name=str(v.get("name") or device_id))
                k.model, k.version = str(v.get("model") or ""), str(v.get("version") or "")
                k.enroll = v.get("enroll") is True
                k.features = v.get("features") if isinstance(v.get("features"), dict) else {}
                k.access = v.get("access") if v.get("access") in ACCESS else "pending"
                k.last_seen = float(v.get("last_seen") or 0.0)
                self.kiosks[device_id] = k

    def save(self) -> None:
        if self._path is None:
            return
        data = {d: {f: getattr(k, f) for f in KEPT} for d, k in self.kiosks.items()}
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=1))
            tmp.replace(self._path)
        except OSError as err:
            _LOG.warning("could not save %s: %s", self._path, err)

    # ── kiosks ────────────────────────────────────────────────────────────

    def register(self, device_id: str, name: str, version: str, ws: web.WebSocketResponse,
                 model: str = "", enroll: bool = False, features: dict[str, Any] | None = None) -> Kiosk:
        now = time.time()
        k = self.kiosks.get(device_id) or Kiosk(device_id=device_id, name=name)
        k.name, k.version, k.connected, k.connected_at, k.last_seen, k._ws = name, version, True, now, now, ws
        k.model = model or k.model
        k.enroll = enroll
        if features is not None:
            k.features = features
        self.kiosks[device_id] = k
        self.save()
        self.publish({"type": "kiosk", **k.public()})
        return k

    def unregister(self, device_id: str, ws: web.WebSocketResponse) -> None:
        k = self.kiosks.get(device_id)
        if k and k._ws is ws:
            k.connected, k._ws, k.last_seen = False, None, time.time()
            self.save()
            self.publish({"type": "kiosk", **k.public()})

    def update(self, device_id: str, enroll: bool, features: dict[str, Any] | None = None) -> None:
        k = self.kiosks.get(device_id)
        if k is None or (k.enroll == enroll and (features is None or k.features == features)):
            return
        k.enroll = enroll
        if features is not None:
            k.features = features
        self.save()
        self.publish({"type": "kiosk", **k.public()})

    # ── access ────────────────────────────────────────────────────────────

    def approved(self, device_id: str | None) -> bool:
        k = self.kiosks.get(device_id or "")
        return k is not None and k.access == "approved"

    def set_access(self, device_id: str, access: str) -> Kiosk | None:
        k = self.kiosks.get(device_id)
        if k is None or access not in ACCESS:
            return None
        k.access = access
        if access != "approved":
            k._ticket = None
        self.save()
        self.publish({"type": "kiosk", **k.public()})
        return k

    def approve_known(self, sessions: list[dict[str, Any]]) -> None:
        """The first start with approvals: kiosks that already sent sessions keep working."""
        for r in sorted(sessions, key=lambda r: r.get("started") or 0.0):
            d = r.get("device_id")
            if not d:
                continue
            k = self.kiosks.get(d) or Kiosk(device_id=d, name=d)
            k.name = r.get("device_name") or k.name
            k.last_seen = max(k.last_seen, float(r.get("started") or 0.0))
            k.access = "approved"
            self.kiosks[d] = k
        self.new = False
        self.save()

    def issue_ticket(self, device_id: str) -> str | None:
        """A ticket for the session and enrollment sockets of an approved kiosk."""
        k = self.kiosks.get(device_id)
        if k is None or k.access != "approved":
            return None
        k._ticket = secrets.token_urlsafe(24)
        return k._ticket

    def ticket_ok(self, device_id: str | None, ticket: Any) -> bool:
        k = self.kiosks.get(device_id or "")
        return (k is not None and k.access == "approved" and k._ticket is not None
                and isinstance(ticket, str) and hmac.compare_digest(k._ticket, ticket))

    def seen(self, device_id: str | None, name: str | None = None) -> Kiosk | None:
        """A session from a kiosk that never opened the control socket still lists it."""
        if not device_id:
            return None
        k = self.kiosks.get(device_id)
        if k is None:
            k = self.kiosks[device_id] = Kiosk(device_id=device_id, name=name or device_id)
        k.last_seen = time.time()
        return k

    async def send(self, device_id: str, message: dict[str, Any]) -> bool:
        k = self.kiosks.get(device_id)
        if not k or not k._ws or k._ws.closed:
            return False
        await k._ws.send_json(message)
        return True

    async def broadcast(self, message: dict[str, Any]) -> None:
        """To every approved kiosk that is connected."""
        for k in list(self.kiosks.values()):
            if k.access == "approved" and k._ws is not None and not k._ws.closed:
                try:
                    await k._ws.send_json(message)
                except ConnectionError:
                    pass

    # ── live events for the web UI ────────────────────────────────────────

    def listen(self, ws: web.WebSocketResponse) -> None:
        self._listeners.add(ws)

    def unlisten(self, ws: web.WebSocketResponse) -> None:
        self._listeners.discard(ws)

    def publish(self, event: dict[str, Any]) -> None:
        for ws in list(self._listeners):
            if ws.closed:
                self._listeners.discard(ws)
                continue
            asyncio.get_running_loop().create_task(self._safe_send(ws, event))

    async def _safe_send(self, ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except (ConnectionError, RuntimeError):
            self._listeners.discard(ws)
