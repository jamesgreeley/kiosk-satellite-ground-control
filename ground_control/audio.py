"""PCM and WAV helpers. Everything inside the service is 16 kHz mono float32."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def read_wav(source: Path | bytes, any_rate: bool = False) -> np.ndarray:
    """A PCM WAV (8, 16, 24 or 32-bit integer) as 16 kHz mono float32.

    Without ``any_rate`` only 16 kHz is taken, as kiosks send it. With it,
    other rates are resampled, for files people upload or import.
    """
    f = io.BytesIO(source) if isinstance(source, bytes) else source.open("rb")
    try:
        with f, wave.open(f, "rb") as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(w.getnframes())
    except wave.Error as err:
        raise ValueError(f"not a PCM WAV file: {err}") from err
    if rate != SAMPLE_RATE and not any_rate:
        raise ValueError("WAV must be 16 kHz")
    if width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = pcm16_to_float(raw)
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        audio = (np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)).astype(np.float32)
    elif width == 4:
        audio = (np.frombuffer(raw, dtype="<i4") / float(1 << 31)).astype(np.float32)
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels).mean(axis=1)
    return resample(audio, rate) if rate != SAMPLE_RATE else audio


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    """To 16 kHz by FFT, in 30 s blocks so long files stay small in memory."""
    if rate == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    block = rate * 30
    out = []
    for i in range(0, len(audio), block):
        x = audio[i:i + block].astype(np.float64)
        n = max(1, round(len(x) * SAMPLE_RATE / rate))
        spec = np.fft.rfft(x)
        keep = n // 2 + 1
        spec = spec[:keep] if len(spec) >= keep else np.pad(spec, (0, keep - len(spec)))
        out.append(np.fft.irfft(spec, n) * (n / len(x)))
    return np.concatenate(out).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(float_to_pcm16(audio))
    tmp.replace(path)


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(float_to_pcm16(audio))
    return buf.getvalue()
