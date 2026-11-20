"""The service's recent log, for the Logs page and issue reports.

Every record of the service (and of the libraries it runs on) is kept in a
ring of the last few thousand, in memory only, so the web UI can show what
happened without a shell on the machine. Detailed logs turn on the
service's own debug records, such as every event of a session, until they
are turned off or the service restarts.

Each request to the web server is left out: the Logs page asks every few
seconds and would bury everything else.
"""

from __future__ import annotations

import collections
import logging
import re
import threading
import time
import traceback
from typing import Any

SIZE = 5000
SERVICE = "ground_control"
LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
SKIP = {"aiohttp.access"}

# A token that slips into a message: a JWT (Home Assistant's long-lived
# tokens) or a Bearer header.
_SECRETS = re.compile(r"eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}|(?i:bearer)\s+\S+")


def redact(text: str) -> str:
    return _SECRETS.sub("[redacted]", text)


class LogBuffer(logging.Handler):
    def __init__(self, size: int = SIZE) -> None:
        super().__init__(level=logging.DEBUG)
        self._entries: collections.deque[dict[str, Any]] = collections.deque(maxlen=size)
        self._seq = 0
        self._guard = threading.Lock()
        self.started = time.time()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in SKIP:
            return
        try:
            message = redact(record.getMessage())
            exc = None
            if record.exc_info and record.exc_info[0] is not None:
                exc = redact("".join(traceback.format_exception(*record.exc_info)).rstrip())
            with self._guard:
                self._seq += 1
                self._entries.append({"seq": self._seq, "t": round(record.created, 3), "level": record.levelname,
                                      "logger": record.name, "message": message, "exc": exc})
        except Exception:  # noqa: BLE001 - a log line must never take the service down
            self.handleError(record)

    @property
    def last(self) -> int:
        return self._seq

    def since(self, after: int = 0, level: str = "DEBUG") -> list[dict[str, Any]]:
        """The records after ``after``, at ``level`` or above, oldest first."""
        floor = LEVELS.get(level.upper(), 10)
        with self._guard:
            return [e for e in self._entries if e["seq"] > after and LEVELS.get(e["level"], 20) >= floor]

    # ── detailed logs ─────────────────────────────────────────────────────

    @property
    def detailed(self) -> bool:
        return logging.getLogger(SERVICE).level == logging.DEBUG

    @detailed.setter
    def detailed(self, on: bool) -> None:
        if on != self.detailed:
            logging.getLogger(SERVICE).setLevel(logging.DEBUG if on else logging.INFO)
            logging.getLogger(SERVICE).info("detailed logs %s", "on" if on else "off")


def line(e: dict[str, Any]) -> str:
    """One record as text: local time, level, where and the message."""
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["t"])) + f".{int(e['t'] * 1000) % 1000:03d}"
    text = f"{t} {e['level']:<7} {e['logger']}: {e['message']}"
    return f"{text}\n{e['exc']}" if e.get("exc") else text


LOGS = LogBuffer()


def install() -> LogBuffer:
    """Keep the log from here on: the service's own records at info (or debug
    with detailed logs on) and everything else at info."""
    root = logging.getLogger()
    if LOGS not in root.handlers:
        root.addHandler(LOGS)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    if logging.getLogger(SERVICE).level == logging.NOTSET:
        logging.getLogger(SERVICE).setLevel(logging.INFO)
    return LOGS
