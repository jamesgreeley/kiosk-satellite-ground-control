"""Advertise the service on the network, so kiosks find it on their own.

Service type ``_ks-groundctl._tcp``. The TXT record carries the version,
the speaker model and the API path. As a Home Assistant app this needs the
host network, which the app's config.yaml asks for.
"""

from __future__ import annotations

import logging
import os
import socket

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from . import __version__

_LOG = logging.getLogger(__name__)

SERVICE_TYPE = "_ks-groundctl._tcp.local."
BRIDGES = ("docker", "br-", "veth", "hassio", "virbr", "lxcbr", "cni", "flannel")
"""Container bridges a kiosk can't reach (hassio is the Supervisor's on HAOS)."""


def _interface_ips() -> list[str]:
    """The machine's IPv4 addresses a kiosk could be on the network of."""
    import ifaddr

    ips = []
    for adapter in ifaddr.get_adapters():
        if str(adapter.name).startswith(BRIDGES):
            continue
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and not ip.ip.startswith(("127.", "169.254.")) and ip.ip not in ips:
                ips.append(ip.ip)
    return ips


def _primary() -> str | None:
    """The address of the interface with the default route. No packet is sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return None


def pick(ips: list[str], primary: str | None, address: str | None = None) -> list[str]:
    """The addresses to advertise.

    ``address`` (the MDNS_ADDRESS setting) when it is set. Otherwise only the
    default route's address when it's one of this machine's: a machine on
    several networks would send kiosks addresses they can't reach. It can't
    announce each network's own address instead, because routers that relay
    mDNS between networks (common with an IoT VLAN) pass every announcement
    to every kiosk."""
    if address:
        return [address]
    return [primary] if primary in ips else ips


def instance_name(host: str, app: bool) -> str:
    """The name kiosks see. mDNS allows 63 bytes, and an app's hostname
    (``local-kiosk-satellite-ground-control``) is too long for it."""
    name = f"Kiosk Satellite Ground Control ({'Home Assistant' if app else host})"
    if len(name.encode()) > 63:
        name = name.encode()[:62].decode(errors="ignore").rstrip(" (") + ")"
    return name


class Advertiser:
    def __init__(self, port: int, model: str, address: str | None = None) -> None:
        self.port = port
        self.model = model
        self.address = address
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        host = socket.gethostname().split(".")[0]
        addresses = pick(_interface_ips(), _primary(), self.address)
        if not addresses:
            _LOG.warning("no network address to advertise on, kiosks need the address entered")
            return
        self._info = AsyncServiceInfo(
            SERVICE_TYPE,
            f"{instance_name(host, bool(os.environ.get('SUPERVISOR_TOKEN')))}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(a) for a in addresses],
            port=self.port,
            properties={"version": __version__, "model": self.model, "path": "/api"},
            server=f"{host}.local.",
        )
        self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._zc.async_register_service(self._info, allow_name_change=True)
        _LOG.info("advertised on mDNS as %s with %s%s, port %d", self._info.name.removesuffix(f".{SERVICE_TYPE}"),
                  ", ".join(addresses), " (MDNS_ADDRESS)" if self.address else "", self.port)

    async def stop(self) -> None:
        if self._zc is not None:
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
