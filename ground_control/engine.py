"""Speaker embeddings from one ONNX graph: 16 kHz waveform in, embedding out.

The feature frontend is inside the graph, so only onnxruntime is needed.
Sessions run without the CPU memory arena and memory is handed back to the
system after loading, as Voice Incubator found it keeps speaker ID small.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

_LOG = logging.getLogger(__name__)

QUANTUM = 8000
"""Window lengths are whole multiples of this many samples (0.5 s)."""

OWN_SESSION_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
"""Window lengths that get a GPU session of their own: every length up to the
4 s windows, which covers nearly every call during a command. A GPU session
sets up again each time its input length changes, about 60 ms with CUDA 13
and 180 ms with CUDA 12, so one shared session switching between lengths
would add that to the checks and to the end of each command. Longer windows,
such as enrollment audio, share one session."""

LEAN_CUDA = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"}
"""Keeps each speaker model session near 100 MB of GPU memory, at the same speed.
With its own sessions, a speaker model takes about 1.3 GB of GPU memory."""


def malloc_trim() -> None:
    """Give freed heap back to the system (glibc only, a no-op elsewhere)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def providers(use_cuda: bool, cuda_options: dict | None = None) -> list:
    available = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):
            # CUDA and cuDNN from the nvidia-* wheels, when the system has none.
            ort.preload_dlls()
        # HEURISTIC skips cuDNN's exhaustive search on each new input length.
        options = {"cudnn_conv_algo_search": "HEURISTIC", **(cuda_options or {})}
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    if use_cuda:
        _LOG.warning("CUDA asked for but not available, running on CPU")
    return ["CPUExecutionProvider"]


class CudaUnavailable(RuntimeError):
    """CUDA is on, but a model can't run on the GPU. A CUDA image runs on the
    GPU or not at all: it never falls back to the CPU on its own."""


def _gpus_supported() -> str:
    if platform.machine() in ("aarch64", "arm64"):
        return "an NVIDIA GPU from the Turing generation or newer and an NVIDIA driver for CUDA 13 (580 or newer)"
    return "an NVIDIA GPU from the GTX 10 series (Pascal) or newer and an NVIDIA driver from the 570 series or newer"


def _reason(text: str) -> str:
    if "architectural feature absent" in text or "CUBLAS failure 8" in text:
        return "this GPU's architecture isn't supported by the CUDA in this image"
    if "no CUDA-capable device" in text or "CUDA failure 100" in text:
        return "the container sees no NVIDIA GPU"
    if "driver version is insufficient" in text or "CUDA failure 35" in text:
        return "the NVIDIA driver is too old for the CUDA in this image"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else "unknown error")[:300]


def open_session(path: Path, so: ort.SessionOptions, use_cuda: bool,
                 cuda_options: dict | None = None) -> ort.InferenceSession:
    """An onnxruntime session on the GPU when ``use_cuda``, else on the CPU.
    With ``use_cuda``, a model that can't run on the GPU raises
    CudaUnavailable with the reason and what the image needs."""
    if not use_cuda:
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        reason = "this image has no CUDA support"
    else:
        try:
            session = ort.InferenceSession(str(path), so, providers=providers(True, cuda_options))
        except Exception as err:  # noqa: BLE001 - every CUDA failure is reported the same way
            reason = _reason(str(err))
        else:
            if session.get_providers()[0] == "CUDAExecutionProvider":
                return session
            reason = ("the container sees no NVIDIA GPU: start it with --gpus all, or the GPU section of "
                      "compose.cuda.yaml" if not Path("/dev/nvidiactl").exists()
                      else "the CUDA provider couldn't start, so the model would run on the CPU")
    raise CudaUnavailable(f"CUDA can't start: {reason}. The CUDA image needs {_gpus_supported()}, "
                          "and the NVIDIA Container Toolkit on the host. Without a usable GPU, "
                          "use the CPU image (ghcr.io/jxlarrea/kiosk-satellite-ground-control:latest).")


def session_options(threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


class Embedder:
    """One speaker model. Thread safe: onnxruntime sessions can run concurrently."""

    def __init__(self, path: Path, threads: int = 1, use_cuda: bool = False) -> None:
        so = session_options(threads)
        so.log_severity_level = 3  # the frontend's CPU fallback nodes are expected
        self.session = open_session(path, so, use_cuda, LEAN_CUDA)
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        # On a GPU the most used lengths get sessions of their own, so none of
        # them ever changes length. The CPU has no such cost.
        self._own = {int(sec * 16000): open_session(path, so, True, LEAN_CUDA) for sec in OWN_SESSION_S} \
            if self.on_gpu else {}
        inp = self.session.get_inputs()[0]
        self._input = inp.name
        self._rank = len(inp.shape)
        self._lock = threading.Lock()
        self.calls = 0
        self.dim = int(self.embed(np.zeros(16000, dtype=np.float32)).shape[0])
        for n in self._own:
            self.embed(np.zeros(n, dtype=np.float32))
        self.calls = 0
        malloc_trim()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Unit-length embedding of float32 audio in [-1, 1].

        The window is trimmed to whole half seconds, newest audio kept. A GPU
        session pays 60 to 180 ms each time its input length changes, so the
        lengths stay few and the common ones have sessions of their own. The
        CPU path gets the same windows.
        """
        step = QUANTUM
        if len(audio) >= step:
            audio = audio[len(audio) - (len(audio) // step) * step:]
        x = np.ascontiguousarray(audio, dtype=np.float32)
        x = x[None, None, :] if self._rank == 3 else x[None, :]
        session = self._own.get(x.shape[-1], self.session)
        e = session.run(None, {self._input: x})[0][0].astype(np.float32)
        with self._lock:
            self.calls += 1
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else e
