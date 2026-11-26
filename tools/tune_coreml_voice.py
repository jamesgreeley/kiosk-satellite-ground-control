"""Prepare fixed speaker windows that beat CPU on the current Mac."""
import argparse
import os
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxsim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ground_control.engine import (apple_neural_engine_available, open_session, speaker_profile_dir, speaker_compute_units,
                                   session_options)
from ground_control.models import SPEAKER_MODELS, verified_path


def tune(path, threads, audio):
    options = session_options(threads)
    options.log_severity_level = 3
    cpu = open_session(path, options, False)
    folder = speaker_profile_dir(path, threads)
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(8000, 64001, 8000):
        target = folder / f"{n}.onnx"
        units = speaker_compute_units(n)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            candidate = Path(tmp) / target.name
            graph, ok = onnxsim.simplify(str(path), overwrite_input_shapes={"waveform": [1, n]})
            if not ok:
                raise ValueError("speaker graph simplification failed")
            onnx.save(graph, candidate)
            session = open_session(candidate, options, False, use_coreml=True,
                                   coreml_options={"MLComputeUnits": units})
            rng = np.random.default_rng(42)
            times = [[], []]
            similarity = []
            for i in range(22):
                x = rng.normal(0, 0.03, (1, n)).astype(np.float32)
                if i == 0:
                    x.fill(0)
                elif audio is not None and i % 2:
                    x = np.resize(np.roll(audio, i * 4000), n)[None, :]
                outputs = {}
                for j in ((0, 1) if i % 2 else (1, 0)):
                    start = time.perf_counter()
                    outputs[j] = (cpu, session)[j].run(None, {"waveform": x})[0].ravel()
                    if i >= 2:
                        times[j].append((time.perf_counter() - start) * 1000)
                a, b = outputs[0], outputs[1]
                similarity.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            base, fast = map(statistics.median, times)
            use = ("CoreMLExecutionProvider" in session.get_providers() and
                   np.all(np.asarray(similarity) >= 0.999) and fast <= base * 0.9 and
                   np.percentile(times[1], 95) <= np.percentile(times[0], 95))
            if use:
                os.replace(candidate, target)
            else:
                target.unlink(missing_ok=True)
            print(f"{path.stem} {n / 16000:g}s CPU {base:.2f} ms Core ML {fast:.2f} ms "
                  f"min cosine {np.min(similarity):.7f} selected {'Core ML' if use else 'CPU'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wav", type=Path, help="also compare a mono 16 kHz signed 16-bit PCM sample")
    args = parser.parse_args()
    if not apple_neural_engine_available():
        parser.error("requires native Apple Silicon and Core ML")
    audio = None
    if args.wav:
        with wave.open(str(args.wav)) as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2) or not wav.getnframes():
                parser.error("expected nonempty mono 16 kHz 16-bit PCM")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    for model in SPEAKER_MODELS.values():
        tune(verified_path(ROOT / "models", model.file, model.sha256), args.threads, audio)
