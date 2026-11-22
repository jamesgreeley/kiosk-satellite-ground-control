"""Isolation, Voice Match's third feature: the user's words with a TV or
other people under them removed, by the Voice Isolator model.

Trimming (follow mode) leaves other voices out between the user's words.
Isolation also removes them where they overlap. It is a band-split separator (BSRNN)
fine-tuned on kiosk audio and conditioned on the user's voiceprint from the
Accurate speaker model, so the voice it keeps is the one Voice Match
identified.

The separator looks at a whole stretch of audio at once. Follow mode holds a
turn's audio while the user talks and releases it cleaned when the turn ends.
It is meant for an NVIDIA GPU: a 3 s command takes about 50 ms on a GB10 and
about a second on four cores of an i9-13900H.

Long turns run in overlapping windows, so memory stays bounded. Lengths are
padded to whole half seconds, so a GPU sees few distinct input shapes.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .engine import LEAN_CUDA, GpuThread, malloc_trim, open_session, run_options, session_options

_LOG = logging.getLogger(__name__)

SR = SAMPLE_RATE
QUANTUM = SR // 2
WINDOW = 10 * SR
OVERLAP = SR
MIN_SAMPLES = SR // 10
"""Shorter than this, the audio goes through as it is."""


class Isolator:
    def __init__(self, path: Path, threads: int = 1, use_cuda: bool = False, keep_memory: bool = False) -> None:
        """``keep_memory``: on a GPU, keep the memory a command used for the
        next one (the ``KEEP_GPU_MEMORY`` setting). Taking it again for every
        command costs about 50 ms on a GB10 and 10 ms on an RTX 4000. It holds
        about 2 GB more."""
        so = session_options(threads)
        so.log_severity_level = 3
        # A thread of its own, not the speaker model's: voice checks go on
        # while a turn is cleaned. Two turns that end together are cleaned one
        # after the other, so the GPU never holds the memory of both.
        self._gpu = GpuThread(use_cuda, "gpu-isolator")
        try:
            self.session = self._gpu(open_session, path, so, use_cuda, LEAN_CUDA)
        except BaseException:
            self._gpu.close()
            raise
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self.keeps_memory = self.on_gpu and keep_memory
        # Without keep_memory, each command's GPU memory goes back after it.
        self._run_options = run_options(self.on_gpu and not self.keeps_memory)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.version = meta.get("version", "unknown")
        """From the model's metadata, with its copyright and license."""
        self._lock = threading.Lock()
        self.calls = 0
        if self.keeps_memory:
            # The longest input first. A command longer than any before it
            # can't use the memory kept from the shorter ones, so without this
            # the memory kept grows with every new longest command, to twice
            # as much. After it, every command fits in what is already there.
            dim = self.session.get_inputs()[1].shape[-1]
            self._run(np.zeros(WINDOW, dtype=np.float32), np.zeros(dim, dtype=np.float32))
        malloc_trim()

    def _run(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        n = len(audio)
        padded = -(-n // QUANTUM) * QUANTUM
        x = np.zeros((1, padded), dtype=np.float32)
        x[0, :n] = audio
        y = self._gpu(self.session.run, None, {"audio": x, "voiceprint": voiceprint[None].astype(np.float32)},
                      self._run_options)[0][0]
        return y[:n]

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        """The voice of ``voiceprint`` (a unit-length Accurate voiceprint) in ``audio``."""
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n < MIN_SAMPLES:
            return audio.copy()
        with self._lock:
            self.calls += 1
        if n <= WINDOW:
            return np.clip(self._run(audio, voiceprint), -1.0, 1.0)
        # Full windows, the last one ending at the end, each overlapping the
        # next by at least OVERLAP and faded in and out across it.
        starts = list(range(0, n - WINDOW + 1, WINDOW - OVERLAP))
        if starts[-1] + WINDOW < n:
            starts.append(n - WINDOW)
        ramp = np.linspace(0.0, 1.0, OVERLAP + 2, dtype=np.float32)[1:-1]
        out = np.zeros(n, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        for start in starts:
            end = start + WINDOW
            w = np.ones(WINDOW, dtype=np.float32)
            if start > 0:
                w[:OVERLAP] = ramp
            if end < n:
                w[-OVERLAP:] = np.minimum(w[-OVERLAP:], ramp[::-1])
            out[start:end] += self._run(audio[start:end], voiceprint) * w
            weight[start:end] += w
        return np.clip(out / np.maximum(weight, 1e-6), -1.0, 1.0)
