"""Compare CPU and Core ML model inference on Apple Silicon.

Run from the repository root with the project's ONNX Runtime and NumPy
installed, for example::

    python tools/benchmark_apple_silicon.py --suite all --iterations 40

Core ML initialization can compile models and take several seconds. The
script reports initialization separately from warmed inference and profiles
which ONNX Runtime execution provider handled each node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
COREML = (
    "CoreMLExecutionProvider",
    {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "CPUAndNeuralEngine",
        "RequireStaticInputShapes": "0",
    },
)
TUNING_OPTIONS = (
    ("CoreML-MLProgram-dynamic", COREML),
    (
        "CoreML-NeuralNetwork-dynamic",
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "NeuralNetwork",
                "MLComputeUnits": "CPUAndNeuralEngine",
                "RequireStaticInputShapes": "0",
            },
        ),
    ),
)


def cases(suite: str) -> list[tuple[str, dict[str, np.ndarray], bool, bool]]:
    rng = np.random.default_rng(20261126)
    entries: list[tuple[str, dict[str, np.ndarray], bool, bool]] = []

    def add(name: str, feed: dict[str, np.ndarray], apple_default: bool, embedding: bool = False) -> None:
        entries.append((name, feed, apple_default, embedding))

    if suite in ("face", "all"):
        for h, w in ((480, 640), (384, 640), (640, 480)):
            x = rng.uniform(-1, 1, (1, 3, h, w)).astype(np.float32)
            add(f"scrfd_2.5g {w}x{h}", {"input.1": x}, True)
        x = rng.uniform(-1, 1, (1, 3, 480, 640)).astype(np.float32)
        add("scrfd_10g 640x480", {"input.1": x}, False)
        for model in ("arcface_w600k_mbf", "arcface_w600k_r50"):
            x = rng.uniform(-1, 1, (1, 3, 112, 112)).astype(np.float32)
            add(model, {"input.1": x}, True, True)

    if suite in ("voice", "all"):
        for model in ("redimnet2_b1", "redimnet2_b3_multi"):
            for seconds in (0.5, 1, 2, 4):
                audio = rng.normal(0, 0.02, (1, int(16000 * seconds))).astype(np.float32)
                add(f"{model} {seconds:g}s", {"waveform": audio}, False, True)
        audio = rng.normal(0, 0.02, (1, 80000)).astype(np.float32)
        voiceprint = rng.normal(0, 0.05, (1, 192)).astype(np.float32)
        add("voice_isolator_cpu 5s", {"audio": audio, "voiceprint": voiceprint}, False)

    return entries


def make_options(threads: int, profile: str) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = profile
    options.add_session_config_entry("session.force_spinning_stop", "1")
    return options


def model_cache(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for block in iter(lambda: model.read(1 << 20), b""):
            digest.update(block)
    return str(Path.home() / "Library" / "Caches" / "kiosk-satellite-ground-control" / "coreml"
               / ort.__version__ / digest.hexdigest())


def profile_counts(path: str) -> dict[str, int]:
    events = json.loads(Path(path).read_text())
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time"):
            provider = event.get("args", {}).get("provider", "unknown")
            counts[provider] = counts.get(provider, 0) + 1
    Path(path).unlink(missing_ok=True)
    return counts


def compare(path: Path, feed: dict[str, np.ndarray], name: str, iterations: int, threads: int,
            providers: list, baseline: list[np.ndarray] | None, embedding: bool) -> tuple[dict, list[np.ndarray] | None]:
    with tempfile.TemporaryDirectory(prefix="ground-control-coreml-") as tmp:
        started = time.perf_counter()
        session = ort.InferenceSession(
            str(path), make_options(threads, str(Path(tmp) / "profile")), providers=providers
        )
        startup_ms = (time.perf_counter() - started) * 1000
        outputs = session.run(None, feed)
        warmups = 3
        for _ in range(warmups):
            session.run(None, feed)
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            session.run(None, feed)
            samples.append((time.perf_counter() - started) * 1000)
        profile = session.end_profiling()
        counts = profile_counts(profile)

    result: dict[str, object] = {
        "case": name,
        "provider": providers[0] if isinstance(providers[0], str) else providers[0][0],
        "threads": threads,
        "startup_ms": round(startup_ms, 1),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "nodes": counts,
    }
    if baseline is not None:
        if embedding:
            a, b = outputs[0].reshape(-1).astype(np.float64), baseline[0].reshape(-1).astype(np.float64)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            result["embedding_finite"] = finite
            result["embedding_cosine"] = (round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 7)
                                           if finite else None)
        else:
            result["max_abs_output_delta"] = round(float(np.max(np.abs(outputs[0] - baseline[0]))), 8)
    return result, outputs if baseline is None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("face", "voice", "all"), default="face")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tune", action="store_true", help="also compare Core ML's NeuralNetwork conversion")
    parser.add_argument("--tune-compute-units", action="store_true",
                        help="compare Core ML CPU, GPU, Neural Engine and ALL compute units")
    parser.add_argument("--profile-compute-plan", action="store_true",
                        help="log whether Core ML assigns operators to the Neural Engine")
    args = parser.parse_args()
    if args.iterations < 5 or args.threads < 1:
        parser.error("--iterations must be at least 5 and --threads at least 1")

    available = ort.get_available_providers()
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        parser.error("this benchmark needs native Apple Silicon macOS")
    if "CoreMLExecutionProvider" not in available:
        parser.error("this ONNX Runtime package does not include CoreMLExecutionProvider")

    print(f"device={platform.machine()} macOS={platform.mac_ver()[0]} onnxruntime={ort.__version__} "
          f"threads={args.threads} iterations={args.iterations}")
    for label, feed, apple_default, embedding in cases(args.suite):
        model = label.split()[0]
        path = MODELS / f"{model}.onnx"
        cpu, baseline = compare(path, feed, label, args.iterations, args.threads,
                                ["CPUExecutionProvider"], None, embedding)
        cpu["provider"] = "CPUExecutionProvider"
        print(json.dumps(cpu, sort_keys=True), flush=True)
        if not args.tune and not apple_default:
            continue
        production_settings = dict(COREML[1])
        if model == "arcface_w600k_mbf":
            production_settings.update(ModelFormat="NeuralNetwork")
        production_options = {**production_settings, "ModelCacheDirectory": model_cache(path)}
        if args.profile_compute_plan:
            production_options["ProfileComputePlan"] = "1"
        # Core ML uses fewer ORT workers for these small graphs; the large
        # Accurate embedding graph measured best with two.
        coreml_threads = 1 if model in ("scrfd_2.5g", "arcface_w600k_mbf") else 2
        candidates = [("CoreML-production-tuned", (COREML[0], production_options), coreml_threads)]
        if args.tune:
            candidates.extend((name, (candidate[0], {
                **candidate[1], "ModelCacheDirectory": model_cache(path),
                **({"ProfileComputePlan": "1"} if args.profile_compute_plan else {}),
            }), coreml_threads) for name, candidate in TUNING_OPTIONS
                if any(candidate[1].get(key) != production_settings.get(key)
                       for key in ("ModelFormat", "MLComputeUnits")))
        if args.tune_compute_units:
            for compute_units in ("CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"):
                if compute_units == production_settings["MLComputeUnits"]:
                    continue
                settings = {**production_settings, "MLComputeUnits": compute_units,
                            "ModelCacheDirectory": model_cache(path)}
                if args.profile_compute_plan:
                    settings["ProfileComputePlan"] = "1"
                candidates.append((f"CoreML-{settings['ModelFormat']}-{compute_units}",
                                   (COREML[0], settings), coreml_threads))
        for candidate_name, candidate, candidate_threads in candidates:
            try:
                result, _ = compare(path, feed, f"{label} / {candidate_name}", args.iterations,
                                    candidate_threads, [candidate, "CPUExecutionProvider"], baseline, embedding)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as err:  # record unsupported model/shape combinations for tuning
                print(json.dumps({"case": label, "provider": candidate_name, "error": str(err)[:500]},
                                 sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
