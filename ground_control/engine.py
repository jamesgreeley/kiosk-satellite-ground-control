"""Speaker embeddings from one ONNX graph: 16 kHz waveform in, embedding out.

The feature frontend is inside the graph, so only onnxruntime is needed.
Sessions run without the CPU memory arena and memory is handed back to the
system after loading, as Voice Incubator found it keeps speaker ID small.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import logging
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
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

KEEPS_MEMORY_S = 1.0
"""The one window length whose session keeps its GPU memory between runs: the
voice checks, which run up to five times a second while someone talks. Every
other length hands its memory back after each run, for about 1 ms more. With
that, a speaker model takes about 0.6 GB of GPU memory, a quarter of what its
sessions grow to when each keeps the memory of its busiest moment."""

LEAN_CUDA = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"}
"""Keeps a GPU session from taking more memory than its runs ask for, at the same speed."""


def malloc_trim() -> None:
    """Give freed heap back to the system (glibc only, a no-op elsewhere)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


_preloaded = False


def _preload_cuda() -> None:
    """Load CUDA and cuDNN from the nvidia-* wheels when the system has none,
    once. The CUDA images have them from the base image, and onnxruntime prints
    a "Failed to load" line for each library it doesn't find, such as NVRTC and
    cuFFT, which Ground Control doesn't use. That output goes to the debug log."""
    global _preloaded
    if _preloaded or not hasattr(ort, "preload_dlls"):
        return
    _preloaded = True
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ort.preload_dlls()
    for line in out.getvalue().splitlines():
        if line.strip():
            _LOG.debug("CUDA preload: %s", line.strip())


def providers(use_cuda: bool, cuda_options: dict | None = None) -> list:
    available = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available:
        _preload_cuda()
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


def run_options(on_gpu: bool) -> ort.RunOptions:
    """Run options that hand a run's GPU memory back once it is done, for a
    session whose input length varies. Without them the session keeps what its
    longest input needed, up to gigabytes, for the rest of its life. It costs a
    few milliseconds a run."""
    options = ort.RunOptions()
    if on_gpu:
        options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return options


class GpuThread:
    """The one thread every run of a GPU model goes through, sessions being
    opened included.

    ONNX Runtime keeps cuDNN and cuBLAS handles on the GPU for each thread
    that uses a session, and a session run by two threads at once takes the
    memory of both runs and keeps it. With the service's worker threads all
    using the speaker model's sessions, that grew to twice the memory or more.
    One run at a time costs nothing: the GPU does them one after the other
    anyway.

    Without a GPU (``on`` False) calls run on the caller's thread.
    """

    def __init__(self, on: bool, name: str) -> None:
        self._pool = ThreadPoolExecutor(1, thread_name_prefix=name) if on else None

    def __call__(self, fn, *args):
        return self._pool.submit(fn, *args).result() if self._pool is not None else fn(*args)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def session_options(threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    # ONNX Runtime's worker threads spin for work after a run, for longer
    # than the 0.2 s between two voice checks. While someone talked, that
    # took about three cores with four threads, ten times what the checks
    # themselves take. With this they stop when a run ends and spin again
    # during the next one, so a run is as fast as before.
    so.add_session_config_entry("session.force_spinning_stop", "1")
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


class Embedder:
    """One speaker model. Thread safe: onnxruntime sessions can run concurrently."""

    def __init__(self, path: Path, threads: int = 1, use_cuda: bool = False) -> None:
        so = session_options(threads)
        so.log_severity_level = 3  # the frontend's CPU fallback nodes are expected
        self._gpu = GpuThread(use_cuda, "gpu-speaker")
        try:
            self.session = self._gpu(open_session, path, so, use_cuda, LEAN_CUDA)
            self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
            # On a GPU the most used lengths get sessions of their own, so none
            # of them ever changes length. The CPU has no such cost. Longer
            # audio, such as a whole turn, goes through the shared session.
            self._own = {int(sec * 16000): self._gpu(open_session, path, so, True, LEAN_CUDA)
                         for sec in OWN_SESSION_S} if self.on_gpu else {}
        except BaseException:
            self._gpu.close()
            raise
        self._keeps_memory = int(KEEPS_MEMORY_S * 16000)
        self._hand_back = run_options(self.on_gpu)
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
        n = x.shape[-1]
        session = self._own.get(n, self.session)
        if n == self._keeps_memory and n in self._own:
            e = self._gpu(session.run, None, {self._input: x})[0][0]
        else:
            e = self._gpu(session.run, None, {self._input: x}, self._hand_back)[0][0]
        e = e.astype(np.float32)
        with self._lock:
            self.calls += 1
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else e
