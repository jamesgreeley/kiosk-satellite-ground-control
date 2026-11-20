"""Only Kiosk Satellite connects to Ground Control."""

from __future__ import annotations

import base64
import logging
import secrets
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_LOG = logging.getLogger(__name__)

PROTOCOL = "kiosk-satellite/ground-control/1"

KEYS = {
    "ks1": "ag3FVaLcLHSkmScKGMb8BBrmXBqBdRJw6WST/6pyfBA=",
}
"""Kiosk Satellite's client keys. A new key goes here a release before Kiosk Satellite uses it."""


def challenge() -> str:
    return secrets.token_urlsafe(24)


def signed_text(nonce: str, device_id: str) -> bytes:
    """What Kiosk Satellite signs: the protocol, the challenge and the kiosk."""
    return f"{PROTOCOL}\n{nonce}\n{device_id}".encode()


class ClientCheck:
    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self._keys = {kid: Ed25519PublicKey.from_public_bytes(base64.b64decode(pub))
                      for kid, pub in (KEYS if keys is None else keys).items()}

    def verify(self, nonce: str, device_id: str, proof: Any) -> bool:
        """Whether ``proof`` (``{"key", "sig"}``) is Kiosk Satellite's signature of the challenge."""
        if not isinstance(proof, dict):
            return False
        key = self._keys.get(str(proof.get("key") or ""))
        if key is None:
            return False
        try:
            sig = base64.b64decode(str(proof.get("sig") or ""), validate=True)
            key.verify(sig, signed_text(nonce, device_id))
        except (InvalidSignature, ValueError):
            return False
        return True
