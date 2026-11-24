"""Fetch and convert Vision Match's models into models/.

Needs ultralytics, onnx and onnxconverter-common, which the service never
does. Run it once per model release:

    pip install ultralytics onnx onnxconverter-common
    python tools/export_vision_models.py WORK_DIR

- SCRFD 2.5G and 10G: copied unchanged from InsightFace's v0.7 model packs
  (``det_2.5g.onnx`` from buffalo_m, ``det_10g.onnx`` from buffalo_l).
- ArcFace MobileFaceNet and R50: ``w600k_mbf.onnx`` from buffalo_s unchanged,
  ``w600k_r50.onnx`` from buffalo_l with 16-bit weights (under 100 MB, the
  same embeddings).
- YOLOv10n and YOLOv10x gestures: HaGRIDv2's checkpoints, exported to ONNX at
  480x640 with opset 17. YOLOv10x then gets 16-bit weights, so it stays under
  GitHub's 100 MB file limit. It only runs on a GPU, where 16 bits is also faster.

Loading a ``.pt`` checkpoint runs code from it. Run the export step in a
container with no network and only WORK_DIR mounted.

Prints each file's sha256 for ground_control/models.py.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"
HAGRID = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/{}.pt"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def scrfd(work: Path) -> None:
    for pack, member, out in (("buffalo_m", "det_2.5g.onnx", "scrfd_2.5g.onnx"),
                              ("buffalo_l", "det_10g.onnx", "scrfd_10g.onnx")):
        with zipfile.ZipFile(fetch(INSIGHTFACE.format(pack), work / f"{pack}.zip")) as z:
            name = next(n for n in z.namelist() if n.endswith(member))
            with z.open(name) as src, (OUT / out).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def recognition(work: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    for pack, member, out, half in (("buffalo_s", "w600k_mbf.onnx", "arcface_w600k_mbf.onnx", False),
                                    ("buffalo_l", "w600k_r50.onnx", "arcface_w600k_r50.onnx", True)):
        with zipfile.ZipFile(fetch(INSIGHTFACE.format(pack), work / f"{pack}.zip")) as z:
            name = next(n for n in z.namelist() if n.endswith(member))
            with z.open(name) as src, (work / member).open("wb") as dst:
                shutil.copyfileobj(src, dst)
        if half:
            onnx.save(float16.convert_float_to_float16(onnx.load(str(work / member)), keep_io_types=True), str(OUT / out))
        else:
            shutil.copyfile(work / member, OUT / out)


def gestures(work: Path) -> None:
    import onnx
    from onnxconverter_common import float16
    from ultralytics import YOLO

    for size in ("n", "x"):
        pt = fetch(HAGRID.format(f"YOLOv10{size}_gestures"), work / f"YOLOv10{size}_gestures.pt")
        exported = Path(YOLO(str(pt)).export(format="onnx", imgsz=[480, 640], opset=17, simplify=False,
                                             dynamic=False))
        out = OUT / f"yolov10{size}_gestures.onnx"
        if size == "x":
            onnx.save(float16.convert_float_to_float16(onnx.load(str(exported)), keep_io_types=True), str(out))
        else:
            shutil.copyfile(exported, out)


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "vision-models")
    work.mkdir(parents=True, exist_ok=True)
    scrfd(work)
    recognition(work)
    gestures(work)
    for f in ("scrfd_2.5g.onnx", "scrfd_10g.onnx", "arcface_w600k_mbf.onnx", "arcface_w600k_r50.onnx",
              "yolov10n_gestures.onnx", "yolov10x_gestures.onnx"):
        print(f, hashlib.sha256((OUT / f).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
