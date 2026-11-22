"""How Ground Control is installed, for the kiosks' Kiosk Satellite Analytics.

Each kiosk hears it when it connects and puts it in its daily report with
the rest of Ground Control's details: the HA app, standalone Docker or
Python from the source, the CPU architecture, the CUDA version of the CUDA
image and whether the speaker model runs on the GPU.
"""

from __future__ import annotations

import platform
from pathlib import Path

_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "aarch64", "arm64": "aarch64"}


_CONTAINER = (Path("/.dockerenv"), Path("/run/.containerenv"))
"""What Docker and Podman put in a container."""


def kind(supervisor: bool, marks: tuple[Path, ...] = _CONTAINER) -> str:
    """``ha_app``, ``docker`` (Docker or Podman) or ``python``."""
    if supervisor:
        return "ha_app"
    return "docker" if any(m.exists() for m in marks) else "python"


def arch(machine: str | None = None) -> str:
    """The architecture as the images are named: ``amd64`` or ``aarch64``."""
    m = (machine if machine is not None else platform.machine()).lower()
    return _ARCH.get(m, m)


def cuda_version() -> str:
    """The CUDA version ONNX Runtime was built for, empty without the GPU build."""
    try:
        from onnxruntime.capi import build_and_package_info as info
    except ImportError:
        return ""
    return str(getattr(info, "cuda_version", "") or "")


def describe(supervisor: bool, on_gpu: bool) -> dict[str, object]:
    return {"install": kind(supervisor), "arch": arch(), "cuda": cuda_version(), "on_gpu": on_gpu}
