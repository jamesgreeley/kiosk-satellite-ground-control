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
ISOLATOR_SHA256 = "47e9e90cf5ee8f28d0da7e59fe31c795e87a4f43f0e3f1e5d97aae27cf5c865f"
ISOLATOR_CPU_FILE = "voice_isolator_cpu.onnx"
ISOLATOR_CPU_SHA256 = "0016d71fd7d94cb3aa414b45182839a3fb5c0ee1aec4bcbd98f92911d25e370a"
"""The same model with its LSTMs in 8 bits: twice as fast on a CPU, 0.14 dB
behind. On a GPU the full one runs, its quantized LSTMs wouldn't."""
ISOLATOR_MODEL = "accurate"
"""The Voice Isolator is conditioned on this speaker model's voiceprints."""

ALIGNER_FILE = "wake_word_aligner.onnx"
ALIGNER_SHA256 = "894258d611a9c4a5175434838eecee364194fab51b2f0836b4e20b95bb89e7ca"
"""The Wake Word Aligner: the phonemes heard in audio, to find where the wake word ends."""


@dataclass(frozen=True)
class VisionModel:
    """Vision Match's face and hand models, chosen together as Light or Accurate."""
    key: str
    label: str
    face_file: str
    face_sha256: str
    face_threshold: float
    """A face at or above this score counts. SCRFD 10G's default 0.5 saw faces
    in an empty room on the test set, 0.6 doesn't."""
    hand_file: str
    hand_sha256: str
    hand_threshold: float
    """A hand at or above this score counts."""
    rec_file: str
    rec_sha256: str
    rec_threshold: float
    """Face Recognition: a face looking at the kiosk is someone enrolled at or
    above this similarity to their faceprint. On kiosk-sized faces it let 0.1%
    of strangers through, and families look more alike than strangers."""
    gpu_only: bool
    """Too slow for 4 looks a second on any CPU measured: offered only with CUDA."""


VISION_MODELS: dict[str, VisionModel] = {
    "light": VisionModel(
        key="light",
        label="Light",
        face_file="scrfd_2.5g.onnx",
        face_sha256="041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
        face_threshold=0.5,
        hand_file="yolov10n_gestures.onnx",
        hand_sha256="748b2f34f8bb45ea078c52ebc2c1344b1f1a6709e3b6104759c6f9a383b3d9f4",
        hand_threshold=0.4,
        rec_file="arcface_w600k_mbf.onnx",
        rec_sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        rec_threshold=0.35,
        gpu_only=False,
    ),
    "accurate": VisionModel(
        key="accurate",
        label="Accurate",
        face_file="scrfd_10g.onnx",
        face_sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        face_threshold=0.6,
        hand_file="yolov10x_gestures.onnx",
        hand_sha256="cab82c898c5ddf7a20c5120d5b190de669df71e47431f98266118be7aa1f85f0",
        hand_threshold=0.4,
        rec_file="arcface_w600k_r50.onnx",
        rec_sha256="3a728e738b4c0c802d3f2315ca877e76387f4fee9f240d4ae49e8da371dab25c",
        rec_threshold=0.3,
        gpu_only=True,
    ),
}


def default_vision_key(cuda: bool) -> str:
    """Accurate on an NVIDIA GPU, Light everywhere else."""
    return "accurate" if cuda else "light"


def apple_silicon(cpuinfo: Path = Path("/proc/cpuinfo")) -> bool:
    """An Apple M-series chip, also from inside Docker's VM on a Mac: Linux
    shows Apple's CPU implementer code, 0x61."""
    try:
        text = cpuinfo.read_text(errors="replace")
    except OSError:
        return False
    return any(line.split(":", 1)[0].strip() == "CPU implementer" and line.split(":", 1)[1].strip() == "0x61"
               for line in text.splitlines() if ":" in line)


def default_model_key(machine: str, cpus: int, cuda: bool = False, apple: bool = False) -> str:
    """Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with
    more than 4 cores. Light on other ARM CPUs and on a small x86 CPU such as
    an Intel N100, where Accurate is busy half the time with one kiosk talking
    and runs late with two talking at once. An M5 runs Accurate faster than an i9-13900H."""
    if cuda or apple:
        return "accurate"
    if machine.lower() in ("aarch64", "arm64", "armv7l", "armv8l"):
        return "light"
    return "accurate" if cpus > 4 else "light"


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
