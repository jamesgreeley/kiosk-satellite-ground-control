"""Advertise the service on the network, so kiosks find it on their own.

Service type ``_ks-groundctl._tcp``. The TXT record carries the version,
the speaker model and the API path. As a Home Assistant app this needs the
host network, which the app's config.yaml asks for.
"""

from __future__ import annotations

import logging
import socket

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from . import __version__

_LOG = logging.getLogger(__name__)

SERVICE_TYPE = "_ks-groundctl._tcp.local."
BRIDGES = ("docker", "br-", "veth", "hassio", "virbr", "lxcbr", "cni", "flannel")
"""Container bridges a kiosk can't reach (hassio is the Supervisor's on HAOS)."""


def _addresses() -> list[bytes]:
    import ifaddr

    out = []
    for adapter in ifaddr.get_adapters():
        if str(adapter.name).startswith(BRIDGES):
            continue
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and not ip.ip.startswith("127.") and not ip.ip.startswith("169.254."):
                out.append(socket.inet_aton(ip.ip))
    return out


class Advertiser:
    def __init__(self, port: int, model: str) -> None:
        self.port = port
        self.model = model
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        host = socket.gethostname().split(".")[0]
        addresses = _addresses()
        if not addresses:
            _LOG.warning("no network address to advertise on, kiosks need the address entered")
            return
        self._info = AsyncServiceInfo(
            SERVICE_TYPE,
            f"Kiosk Satellite Ground Control ({host}).{SERVICE_TYPE}",
            addresses=addresses,
            port=self.port,
            properties={"version": __version__, "model": self.model, "path": "/api"},
            server=f"{host}.local.",
        )
        self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._zc.async_register_service(self._info, allow_name_change=True)
        _LOG.info("advertised on mDNS as %s, port %d", self._info.name, self.port)

    async def stop(self) -> None:
        if self._zc is not None:
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
