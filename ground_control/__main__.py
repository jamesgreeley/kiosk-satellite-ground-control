"""Run the service: ``python -m ground_control``."""

from __future__ import annotations

import logging

from aiohttp import web

from . import __version__, config, logs
from .server import build_app
from .service import GroundControl


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logs.install()
    cfg = config.load()
    log = logging.getLogger("ground_control")
    # The model chosen on the Settings page, if any, is logged once it loads.
    log.info("Kiosk Satellite Ground Control %s: %d threads, CUDA %s, default model %s", __version__,
             cfg.threads, "on" if cfg.cuda else "off", cfg.model)
    if cfg.auth == "none":
        log.warning("AUTH=none: anyone who can reach port %d can use the service", cfg.port)
    vm = GroundControl(cfg)
    web.run_app(build_app(vm, mdns=cfg.mdns), port=cfg.port, print=None)


if __name__ == "__main__":
    main()
