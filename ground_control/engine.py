"""Speaker embeddings from one ONNX graph: 16 kHz waveform in, embedding out.

The feature frontend is inside the graph, so only onnxruntime is needed.
Sessions run without the CPU memory arena and memory is handed back to the
system after loading, as Voice Incubator found it keeps speaker ID small.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

_LOG = logging.getLogger(__name__)

QUANTUM = 8000
"""Window lengths are whole multiples of this many samples (0.5 s)."""


def malloc_trim() -> None:
    """Give freed heap back to the system (glibc only, a no-op elsewhere)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def providers(use_cuda: bool) -> list:
    available = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):
            # CUDA and cuDNN from the nvidia-* wheels, when the system has none.
            ort.preload_dlls()
        # HEURISTIC skips cuDNN's exhaustive search on each new input length.
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    if use_cuda:
        _LOG.warning("CUDA asked for but not available, running on CPU")
    return ["CPUExecutionProvider"]


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
        self.session = ort.InferenceSession(str(path), so, providers=providers(use_cuda))
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        inp = self.session.get_inputs()[0]
        self._input = inp.name
        self._rank = len(inp.shape)
        self._lock = threading.Lock()
        self.calls = 0
        self.dim = int(self.embed(np.zeros(16000, dtype=np.float32)).shape[0])
        malloc_trim()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Unit-length embedding of float32 audio in [-1, 1].

        The window is trimmed to whole half seconds, newest audio kept. A GPU
        pays about 60 ms for every input length it hasn't just seen, so the
        lengths stay few. The CPU path gets the same windows.
        """
        step = QUANTUM
        if len(audio) >= step:
            audio = audio[len(audio) - (len(audio) // step) * step:]
        x = np.ascontiguousarray(audio, dtype=np.float32)
        x = x[None, None, :] if self._rank == 3 else x[None, :]
        e = self.session.run(None, {self._input: x})[0][0].astype(np.float32)
        with self._lock:
            self.calls += 1
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else e
