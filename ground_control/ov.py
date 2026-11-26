"""OpenVINO on an Intel GPU, for the OpenVINO image.

In that image the vision models run through OpenVINO on an Intel GPU, where
a look takes a fraction of its time on the CPU: 64 ms to 25 ms on an Iris Xe
and 284 ms to 81 ms on an Intel N100. The speaker models, the Wake Word
Aligner and Isolation stay in ONNX Runtime on the CPU, where they are as
fast or faster.

Isolation was tried on the GPU and left out. On an Iris Xe a 5 s command
took 0.8 s against 0.6 s for the model made for the CPU. The GPU needs the
model compiled for each command length, and each one held a third of a
gigabyte of memory for every second of audio: 6.4 GB for three lengths. A
look that came in while a command was cleaned waited until it was done.

The GPU plugin gave wrong results and took half a minute for a first run
with the models' own changing input shapes, so each model is compiled for the
exact shape it is run with, once for each shape. A compiled model is cached
in the data folder: compiling takes 1 to 5 seconds the first time and a
fraction of a second after that.

Compiling also leaves about 0.4 GB behind in the process for each model and
shape, which nothing gives back. Loading from the cache doesn't. So a model
that isn't in the cache yet is compiled in a process of its own first, which
fills the cache and ends. It is then loaded here from the cache.

Everything runs in half precision, which is 1.3 to 2 times faster than full
precision there. The models' outputs match the CPU's to a cosine of 0.9998 or
better. On the 2,357 frames of the kiosk test set, 6 read differently from
the CPU, each one a face or a hand whose score sits on its threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .engine import malloc_trim

_LOG = logging.getLogger(__name__)

PRECISION = "f16"

_DONE = "compiled for the cache"
"""What the process that compiles a model ahead prints when it is done."""

_core: Any = None
_core_lock = threading.Lock()


class Unavailable(RuntimeError):
    """OpenVINO is on, but there is no Intel GPU it can use. The OpenVINO
    image runs its vision models on the GPU or not at all, like the CUDA image."""


def core() -> Any:
    """OpenVINO's runtime, loaded with its first use."""
    global _core
    with _core_lock:
        if _core is None:
            # From Python 3.14 on, OpenVINO starts Python's helper process for
            # subprocesses, which would import the program's main module a
            # second time. Nothing here needs that.
            try:
                multiprocessing.set_forkserver_preload([])
            except Exception:  # noqa: BLE001 - not there or already started
                pass
            import openvino

            _core = openvino.Core()
        return _core


def gpu_name(device: str) -> str:
    """The name of the GPU ``device`` is, such as "Intel(R) Arc(TM) 140T GPU (iGPU)".
    Raises Unavailable with the reason when OpenVINO can't use it."""
    try:
        devices = list(core().available_devices)
    except ImportError:
        raise Unavailable("OpenVINO can't start: this image has no OpenVINO. Use the OpenVINO image "
                          "(ghcr.io/jxlarrea/kiosk-satellite-ground-control:openvino).") from None
    if device in devices:
        name = str(core().get_property(device, "FULL_DEVICE_NAME"))
        # OpenVINO lists any GPU with an OpenCL driver. On an NVIDIA GPU its
        # plugin took 20 s to compile a model and gave no numbers back.
        if device.startswith("GPU") and "intel" not in name.lower():
            raise Unavailable(f"OpenVINO can't start: {device} is {name}, which is not an Intel GPU. The OpenVINO "
                              "image needs an Intel GPU, integrated or Arc. For an NVIDIA GPU, use the CUDA image "
                              "(ghcr.io/jxlarrea/kiosk-satellite-ground-control:cuda).")
        return name
    gpus = [d for d in devices if d.startswith("GPU")]
    if gpus:
        reason = f"there is no device {device}. OpenVINO sees {', '.join(gpus)}"
    elif not Path("/dev/dri").exists():
        reason = "the container sees no Intel GPU: start it with --device /dev/dri or the devices section of compose.openvino.yaml"
    else:
        reason = ("OpenVINO found no Intel GPU among the devices in /dev/dri. The GPU may be too old for it or the "
                  "host may be missing its driver")
    raise Unavailable(f"OpenVINO can't start: {reason}. The OpenVINO image needs an Intel GPU, integrated or Arc. "
                      "Without one, use the CPU image (ghcr.io/jxlarrea/kiosk-satellite-ground-control:latest).")


class Model:
    """One ONNX model on the GPU, compiled for each input shape it is run with.

    ``shapes`` is how many shapes stay compiled at once, the least recently
    used one going first. One run at a time: callers put every run of a
    model through one thread (``engine.GpuThread``).
    """

    def __init__(self, path: Path, device: str, cache_dir: Path, shapes: int = 1) -> None:
        self.path, self.device = path, device
        self._config = {"INFERENCE_PRECISION_HINT": PRECISION, "PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(cache_dir)}
        self._cache_dir = cache_dir
        self._model = core().read_model(str(path))
        self.inputs = [i.get_any_name() for i in self._model.inputs]
        self.outputs = len(self._model.outputs)
        self._shapes = shapes
        self._compiled: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()

    def metadata(self, key: str) -> str | None:
        """A value from the ONNX file's metadata, such as the hand model's gesture names."""
        try:
            return str(self._model.get_rt_info(["framework", key]).astype(str))
        except Exception:  # noqa: BLE001 - OpenVINO raises when the key isn't there
            return None

    def _marker(self, shapes: dict[str, tuple[int, ...]]) -> Path:
        """A file that says this model is in the cache for these shapes."""
        import openvino

        stat = self.path.stat()
        what = [self.path.name, stat.st_size, stat.st_mtime_ns, self.device, openvino.__version__, self._config,
                {name: list(shapes[name]) for name in self.inputs}]
        return self._cache_dir / (hashlib.sha256(json.dumps(what).encode()).hexdigest()[:32] + ".compiled")

    def _fill_cache(self, shapes: dict[str, tuple[int, ...]]) -> None:
        """Compile the model for these shapes in a process of its own, unless the cache has it."""
        marker = self._marker(shapes)
        if marker.exists():
            return
        job = {"path": str(self.path), "device": self.device, "config": self._config,
               "shapes": {name: list(shape) for name, shape in shapes.items()}}
        t0 = time.perf_counter()
        try:
            done = subprocess.run([sys.executable, "-m", "ground_control.ov", json.dumps(job)], capture_output=True,
                                  text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as err:
            _LOG.debug("%s wasn't compiled ahead: %s", self.path.name, err)
            return
        if done.returncode != 0 or _DONE not in done.stdout:
            # Compiling it here says why, to whoever asked for the run.
            _LOG.debug("%s wasn't compiled ahead: %s", self.path.name, done.stderr.strip()[-300:])
            return
        marker.touch()
        _LOG.debug("%s compiled in %.1f s", self.path.name, time.perf_counter() - t0)

    def _request(self, shapes: dict[str, tuple[int, ...]]) -> Any:
        key = tuple(shapes[name] for name in self.inputs)
        with self._lock:
            request = self._compiled.get(key)
            if request is not None:
                self._compiled.move_to_end(key)
                return request
            self._fill_cache(shapes)
            if len(self._compiled) >= self._shapes:
                self._compiled.popitem(last=False)
            t0 = time.perf_counter()
            model = self._model.clone()
            model.reshape({name: list(shape) for name, shape in shapes.items()})
            request = self._compiled[key] = core().compile_model(model, self.device, self._config).create_infer_request()
            del model
            malloc_trim()  # compiling leaves gigabytes of freed memory behind
            _LOG.debug("%s loaded for %s in %.1f s", self.path.name,
                       " and ".join("x".join(map(str, s)) for s in key), time.perf_counter() - t0)
            return request

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """The model's outputs, in the ONNX file's order."""
        request = self._request({name: tuple(x.shape) for name, x in feeds.items()})
        request.infer(feeds)
        # Copies: the request's own buffers are written again by the next run.
        return [np.array(request.get_output_tensor(i).data) for i in range(self.outputs)]


class Gpu:
    """The Intel GPU the OpenVINO image runs on, with the folder its compiled models are cached in."""

    def __init__(self, device: str, cache_dir: Path) -> None:
        self.device = device
        self.name = gpu_name(device)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def model(self, path: Path, shapes: int = 1) -> Model:
        return Model(path, self.device, self.cache_dir, shapes)


def _compile(job: dict[str, Any]) -> None:
    """What ``Model._fill_cache`` starts: compile a model for its shapes, which puts it in the cache."""
    model = core().read_model(job["path"])
    model.reshape(job["shapes"])
    core().compile_model(model, job["device"], job["config"])
    print(_DONE)


if __name__ == "__main__":
    _compile(json.loads(sys.argv[1]))
