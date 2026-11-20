"""Test kiosks: a test client key in place of Kiosk Satellite's, and approval."""

from __future__ import annotations

import asyncio
import base64

from aiohttp import WSMsgType
from cryptography.hazmat.primitives import serialization as s
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ground_control.clients import ClientCheck, signed_text
from ground_control.hub import Kiosk
from ground_control.server import VM

KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
CLIENTS = ClientCheck({"test": base64.b64encode(
    KEY.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw)).decode()})
"""Pass to ``build_app``: the service then accepts the test key only."""


def proof(nonce: str, device_id: str, key: Ed25519PrivateKey = KEY, key_id: str = "test") -> dict[str, str]:
    return {"key": key_id, "sig": base64.b64encode(key.sign(signed_text(nonce, device_id))).decode()}


async def receive(ws, kind: str, timeout: float = 10) -> dict:
    while True:
        msg = await asyncio.wait_for(ws.receive(), timeout)
        if msg.type != WSMsgType.TEXT:
            raise AssertionError(f"socket ended before {kind}")
        data = msg.json()
        if data["type"] == kind:
            return data


async def hello(ws, device_id: str = "k1", **fields) -> None:
    """Say hello on a kiosk socket as Kiosk Satellite does: signing the challenge."""
    nonce = (await receive(ws, "challenge"))["nonce"]
    await ws.send_json({"type": "hello", "token": "good", "device_id": device_id,
                        "proof": proof(nonce, device_id), **fields})


def approve(client, device_id: str = "k1", name: str = "Kitchen") -> str:
    """Approve a kiosk without its socket and return the ticket for its sessions."""
    hub = client.app[VM].hub
    hub.kiosks.setdefault(device_id, Kiosk(device_id=device_id, name=name))
    hub.set_access(device_id, "approved")
    return hub.issue_ticket(device_id)
