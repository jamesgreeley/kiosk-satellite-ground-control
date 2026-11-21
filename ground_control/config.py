"""Settings from the Home Assistant app's options, or from the environment.

As a Home Assistant app the options are in ``/data/options.json`` and
the data is in the app's own folder, ``/config``, which Samba and the File
editor show under ``addon_configs``. As a standalone container they come
from environment variables:

=================  ===========================================  ==================
Variable           Meaning                                      Default
=================  ===========================================  ==================
``DATA_DIR``       people, voices and voiceprints               ``/data``, or ``/config`` as an app
``MODELS_DIR``     the ONNX models                              ``/app/models``
``MODEL``          ``accurate``, ``light`` or ``auto``          ``auto``
``THREADS``        threads per inference                        half the CPUs, 1 to 4
``CUDA``           run the speaker model on an NVIDIA GPU       ``false``
``HA_URL``         Home Assistant: kiosk tokens and the web     ``http://homeassistant:8123``, or
                   UI's login, so a browser must reach it too   ``auto`` (from the Supervisor) as an app
``PORT``           the service port                             ``10900``
``LEARN``          learn from confident commands                ``true``
``AUTH``           ``homeassistant`` or ``none`` (testing)      ``homeassistant``
``KEEP_DAYS``      days saved sessions are kept                 ``7``
``KEEP_AUDIO``     keep each session's audio                    ``true``
``MDNS``           advertise the service to kiosks              ``true``
``MDNS_ADDRESS``   the address to advertise, for kiosks that    the main network's
                   can't reach the main network's address
=================  ===========================================  ==================
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from .models import SPEAKER_MODELS, default_model_key


@dataclass
class Config:
    data_dir: Path
    models_dir: Path
    model: str
    threads: int
    cuda: bool
    ha_url: str
    port: int
    learn: bool
    auth: str
    keep_days: int = 7
    keep_audio: bool = True
    mdns: bool = True
    mdns_address: str | None = None
    supervisor_token: str | None = None


def _bool(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load() -> Config:
    options: dict[str, object] = {}
    options_file = Path(os.environ.get("OPTIONS_FILE", "/data/options.json"))
    if options_file.is_file():
        options = json.loads(options_file.read_text())

    def get(key: str, default: object) -> object:
        return options.get(key.lower(), os.environ.get(key, default))

    cpus = os.cpu_count() or 2
    cuda = _bool(get("CUDA", "false"))
    model = str(get("MODEL", "auto")).lower()
    if model == "auto":
        model = default_model_key(platform.machine(), cpus, cuda)
    if model not in SPEAKER_MODELS:
        raise ValueError(f"MODEL must be one of {', '.join(SPEAKER_MODELS)} or auto")
    threads = int(get("THREADS", 0) or max(1, min(4, cpus // 2)))
    mdns_address = str(get("MDNS_ADDRESS", "") or "").strip() or None
    if mdns_address:
        try:
            ipaddress.IPv4Address(mdns_address)
        except ValueError:
            raise ValueError(f"MDNS_ADDRESS must be an IPv4 address like 192.168.1.20, not {mdns_address!r}") from None
    auth = str(get("AUTH", "homeassistant")).lower()
    if auth not in ("homeassistant", "none"):
        raise ValueError("AUTH must be homeassistant or none")
    return Config(
        data_dir=Path(str(get("DATA_DIR", "/config" if os.environ.get("SUPERVISOR_TOKEN") else "/data"))),
        models_dir=Path(str(get("MODELS_DIR", "/app/models"))),
        model=model,
        threads=threads,
        cuda=cuda,
        ha_url=str(get("HA_URL", "auto" if os.environ.get("SUPERVISOR_TOKEN") else "http://homeassistant:8123")).rstrip("/"),
        port=int(get("PORT", 10900)),
        learn=_bool(get("LEARN", "true")),
        auth=auth,
        keep_days=int(get("KEEP_DAYS", 7)),
        keep_audio=_bool(get("KEEP_AUDIO", "true")),
        mdns=_bool(get("MDNS", "true")),
        mdns_address=mdns_address,
        supervisor_token=os.environ.get("SUPERVISOR_TOKEN"),
    )
