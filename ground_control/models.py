"""The speaker models and the VAD model, with their tuned settings.

Thresholds come from the 2026-11-16 bake-off (far-field audio through the
measured Echo channels) and still need calibrating on real Echo audio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerModel:
    key: str
    label: str
    file: str
    sha256: str
    threshold: float
    """A voice at or above this score counts as the enrolled person."""
    other_threshold: float
    """A 1 s window below this score on the speaker's own voiceprint is someone else."""
    cadence_s: float
    """How often the other-voice check runs while the person talks."""


SPEAKER_MODELS: dict[str, SpeakerModel] = {
    "accurate": SpeakerModel(
        key="accurate",
        label="Accurate",
        file="redimnet2_b3_multi.onnx",
        sha256="55525c06fc909f8949b6e34783ab8d69a27c840ff43ff7dc525f9a2c05c7cb72",
        threshold=0.50,
        other_threshold=0.29,
        cadence_s=0.2,
    ),
    "light": SpeakerModel(
        key="light",
        label="Light",
        file="redimnet2_b1.onnx",
        sha256="6f437da49968616513e2432c0e6d86d8c50c524d2b51cfab8b42b2b6e6b41c44",
        threshold=0.40,
        other_threshold=0.26,
        cadence_s=0.5,
    ),
}

VAD_FILE = "silero_vad.onnx"
VAD_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"

ISOLATOR_FILE = "voice_isolator.onnx"
ISOLATOR_SHA256 = "06a044fddf048962329c1ac13a625527a7985acfebcfa5ed553dba03c2668cfb"
ISOLATOR_CPU_FILE = "voice_isolator_cpu.onnx"
ISOLATOR_CPU_SHA256 = "b23b4d93e05bbca548d830da5890cedeaaa5a184ceaa203e9855dce30863b512"
"""The same model with its LSTMs in 8 bits: twice as fast on a CPU, 0.14 dB
behind. On a GPU the full one runs, its quantized LSTMs wouldn't."""
ISOLATOR_MODEL = "accurate"
"""The Voice Isolator is conditioned on this speaker model's voiceprints."""


def default_model_key(machine: str) -> str:
    """Accurate on x86, Light on ARM, where Accurate costs too much per session."""
    return "light" if machine.lower() in ("aarch64", "arm64", "armv7l", "armv8l") else "accurate"


def verified_path(models_dir: Path, file: str, sha256: str) -> Path:
    """The model file, checked against its pinned hash."""
    path = models_dir / file
    if not path.is_file():
        raise FileNotFoundError(f"model {file} not found in {models_dir}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError(f"model {file} does not match its pinned sha256")
    return path
