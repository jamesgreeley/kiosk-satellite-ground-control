"""Measure complete JPEG looks with the same face models and two gesture backends."""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ground_control.engine import apple_neural_engine_available
from ground_control.models import VISION_MODELS, verified_path
from ground_control.vision import HandModel, Vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, nargs="+", help="representative kiosk JPEG files")
    parser.add_argument("--hands", choices=("accurate", "light"), default="accurate")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not apple_neural_engine_available() or min(args.rounds, args.threads) < 1:
        parser.error("requires native Apple Silicon Core ML and positive rounds and threads")
    frames = [path.read_bytes() for path in args.frames]
    models = ROOT / "models"
    settings = {"face": "light", "recognition": "accurate", "hands": args.hands}
    vision = Vision(models, settings, threads=args.threads)
    try:
        model = VISION_MODELS["hands"][args.hands]
        accelerated = vision.hand
        cpu = HandModel(verified_path(models, model.file, model.sha256), model.threshold,
                        args.threads, False, vision._gpu_hands)
        samples = [[], []]
        matches = 0
        for hand in (cpu, accelerated):
            vision.hand = hand
            vision.look(frames[0], recognize=True)
        for i in range(args.rounds * len(frames)):
            gestures = []
            for j in ((0, 1) if i % 2 else (1, 0)):
                vision.hand = (cpu, accelerated)[j]
                look = vision.look(frames[i % len(frames)], recognize=True)
                gestures.append(look.hand and look.hand["gesture"])
                samples[j].append(look.ms["total"])
            matches += gestures[0] == gestures[1]
        if not accelerated.on_coreml:
            raise RuntimeError("gesture inference fell back to CPU")
        for label, timings in zip(("CPU gestures", "Core ML gestures"), samples):
            print(f"{label}: complete request median {statistics.median(timings):.1f} ms, "
                  f"p95 {np.percentile(timings, 95):.1f} ms")
        print(f"Gesture agreement: {matches}/{len(samples[0])}")
    finally:
        vision.close()


if __name__ == "__main__":
    main()
