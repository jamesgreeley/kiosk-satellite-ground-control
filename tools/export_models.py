"""Export ReDimNet2 B3 (multilingual) and B1 to ONNX, with the frontend inside.

Needs PyTorch, which the service never does. Run it once per model release:

    pip install torch onnxruntime
    python tools/export_models.py

Weights: PalabraAI/redimnet2 (MIT), large-margin ("lm") checkpoints, through
torch.hub. Each graph takes a [batch, samples] 16 kHz waveform and returns
[batch, embedding]. Parity with PyTorch is checked at three lengths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
EXPORTS = {
    "redimnet2_b3_multi.onnx": ("b3", "vb2+vox2+cnc2_v0"),
    "redimnet2_b1.onnx": ("b1", "vox2"),
}


def export(file: str, size: str, dataset: str) -> None:
    import onnxruntime as ort
    import torch

    model = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name=size, train_type="lm",
                           dataset=dataset, pretrained=True, trust_repo=True).eval()
    out = OUT / file
    x = torch.randn(1, 32000) * 0.1
    torch.onnx.export(model, (x,), str(out), input_names=["waveform"], output_names=["embedding"],
                      dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for n in (16000, 37123, 64000):
        w = torch.randn(1, n) * 0.1
        with torch.no_grad():
            a = model(w).numpy()[0]
        b = session.run(None, {"waveform": w.numpy()})[0][0]
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        assert cos > 0.9999, f"{file}: parity {cos} at {n} samples"
    print(file, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, (size, data) in EXPORTS.items():
        export(name, size, data)
