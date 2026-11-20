"""Silero VAD v5, streamed: one speech probability per 32 ms of audio.

The model takes 512 new samples with the last 64 of the previous chunk in
front, and carries a recurrent state between calls. Each stream (a session)
keeps its own state, so one model serves every session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .engine import session_options

FRAME = 512
"""Samples per VAD frame at 16 kHz (32 ms)."""
CONTEXT = 64


class VadModel:
    def __init__(self, path: Path) -> None:
        self.session = ort.InferenceSession(str(path), session_options(1), providers=["CPUExecutionProvider"])

    def stream(self) -> VadStream:
        return VadStream(self)


class VadStream:
    """Feed any amount of audio, get a probability for each whole frame."""

    def __init__(self, model: VadModel) -> None:
        self._model = model
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def feed(self, audio: np.ndarray) -> list[float]:
        self._pending = np.concatenate([self._pending, audio.astype(np.float32)])
        probs: list[float] = []
        while len(self._pending) >= FRAME:
            chunk = self._pending[:FRAME]
            self._pending = self._pending[FRAME:]
            x = np.concatenate([self._context, chunk])[None, :]
            out, self._state = self._model.session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            self._context = chunk[-CONTEXT:]
            probs.append(float(out[0][0]))
        return probs


def speech_mask(model: VadModel, audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-sample speech flags for a whole clip, frame by frame."""
    probs = model.stream().feed(audio)
    mask = np.zeros(len(audio), dtype=bool)
    for i, p in enumerate(probs):
        if p >= threshold:
            mask[i * FRAME:(i + 1) * FRAME] = True
    return mask
