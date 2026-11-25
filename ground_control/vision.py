"""Vision Match: faces and hands in a kiosk's camera frames.

A kiosk sends a JPEG as its camera delivered it, with the clockwise rotation
that turns it upright. The frame is decoded, turned upright and given to the
models the look asks for:

- **Faces** (SCRFD): every face, with its box, its five points and whether it
  looks at the kiosk: how far the head is turned from the camera, from a
  generic 3D face fitted to the five points (``head_yaw``). The nose alone
  between the eyes can't tell: on a head turned far, the face model puts the
  hidden eye near the nose, and a face 50 degrees away read as one looking.
  The nose check (``turn_of``) still has to pass too: on a face a dozen pixels
  wide the points are too rough for the fit, and it catches those.
- **Hands** (YOLOv10 trained on HaGRIDv2 gestures): the shown hand and its
  gesture, and the fingers that gesture holds up. The shown hand is the best
  one with a gesture, so a hand resting at the side doesn't hide a raised one.

Coordinates go back as fractions of the upright frame.
"""

from __future__ import annotations

import ast
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .engine import LEAN_CUDA, GpuThread, open_session, session_options as _session_options
from .models import VisionModel, verified_path

_LOG = logging.getLogger(__name__)

FRONTAL_MAX_DEG = 30.0
"""How far a head can be turned from the camera, in degrees, and still look at the kiosk.
On kiosk frames, faces looking at the screen measured 0 to 4 degrees and faces
looking away 45 to 77."""

TURN_MAX = 0.6
"""The nose's offset along the eye line, in eye distances, up to which a face can look at the kiosk."""

FACE_3D = np.array([[-31, -33, -30], [31, -33, -30], [0, 0, 0], [-25, 30, -25], [25, 30, -25]], np.float64)
"""A generic face in millimeters, x right, y down, z toward the camera: the eyes,
the tip of the nose and the corners of the mouth, in SCRFD's order."""
NMS_IOU = 0.4
MAX_FACES = 10
FACE_SIDE = 640
"""Faces are found on the frame with its longer side at most this long."""
HAND_SIZE = (640, 480)
"""The hand model's fixed input, width by height."""
MAX_PIXELS = 1920 * 1080

FINGERS = {"fist": 0, "one": 1, "peace": 2, "peace_inverted": 2, "two_up": 2, "two_up_inverted": 2,
           "three": 3, "three2": 3, "three3": 3, "four": 4, "palm": 5, "stop": 5, "stop_inverted": 5}
"""The gestures that hold up a number of fingers. The others have no count."""
NO_GESTURE = "no_gesture"


class BadFrame(ValueError):
    """The frame isn't a JPEG this service can read."""


def decode(jpeg: bytes, rotation: int = 0) -> np.ndarray:
    """An RGB array of the frame, turned upright: ``rotation`` is clockwise, 0, 90, 180 or 270."""
    if rotation not in (0, 90, 180, 270):
        raise BadFrame("rotation must be 0, 90, 180 or 270")
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            if img.format != "JPEG":
                raise BadFrame("not a JPEG")
            if img.width * img.height > MAX_PIXELS:
                raise BadFrame("larger than 1920x1080")
            rgb = np.asarray(img.convert("RGB"))
    except BadFrame:
        raise
    except Exception as err:  # noqa: BLE001 - anything Pillow can't read
        raise BadFrame(f"can't read the frame: {err}") from err
    if rotation:
        rgb = np.rot90(rgb, k=-(rotation // 90))
    return np.ascontiguousarray(rgb)


def turn_of(eye_a: np.ndarray, eye_b: np.ndarray, nose: np.ndarray) -> float:
    """How far the nose sits from the eyes' middle along the eye line, in eye distances. 0 is straight on."""
    e = eye_b - eye_a
    d2 = float(e @ e)
    if d2 <= 0:
        return 9.0
    return float((nose - (eye_a + eye_b) / 2) @ e / d2)


def head_yaw(points: np.ndarray) -> float:
    """How far the head is turned left or right from the camera, in degrees, from
    its five points in pixels: the generic face fitted to them with a scaled
    orthographic camera. 0 is straight at the camera."""
    model = FACE_3D - FACE_3D.mean(0)
    image = np.asarray(points, np.float64)
    image = image - image.mean(0)
    # Level the eyes first, so a tilted head doesn't read as a turned one.
    ex, ey = image[1] - image[0]
    a = -np.arctan2(ey, ex)
    image = image @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    m, *_ = np.linalg.lstsq(model, image, rcond=None)
    r1 = m[:, 0]
    if not np.isfinite(r1).all() or float(r1 @ r1) <= 0:
        return 90.0
    r1 = r1 / np.linalg.norm(r1)
    return float(np.degrees(np.arcsin(min(1.0, abs(r1[2])))))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    """Indexes of the boxes (x1, y1, x2, y2) to keep, best first."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return keep


def session_options(threads: int):
    so = _session_options(threads)
    # SCRFD declares outputs for 640x640 and warns on every other size it runs at.
    so.log_severity_level = 3
    return so


def _r(v: float) -> float:
    return round(float(v), 4)


class FaceModel:
    """SCRFD with five face points."""

    def __init__(self, path: Path, threshold: float, threads: int, use_cuda: bool, gpu: GpuThread) -> None:
        self.threshold = threshold
        self._gpu = gpu
        self.session = gpu(open_session, path, session_options(threads), use_cuda, LEAN_CUDA)
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self._input = self.session.get_inputs()[0].name
        if len(self.session.get_outputs()) != 9:
            raise ValueError(f"{path.name} is not a SCRFD model with face points")

    def __call__(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        h, w = rgb.shape[:2]
        scale = min(1.0, FACE_SIDE / max(h, w))
        img = rgb
        if scale < 1.0:
            img = np.asarray(Image.fromarray(rgb).resize((round(w * scale), round(h * scale)), Image.BILINEAR))
        ih, iw = img.shape[:2]
        ph, pw = -(-ih // 32) * 32, -(-iw // 32) * 32
        x = np.zeros((ph, pw, 3), np.float32)
        x[:ih, :iw] = img
        x = ((x - 127.5) / 128.0).transpose(2, 0, 1)[None]
        out = self._gpu(self.session.run, None, {self._input: x})
        boxes, scores, points = [], [], []
        for k, s in enumerate((8, 16, 32)):
            score = out[k].reshape(-1)
            idx = np.nonzero(score >= self.threshold)[0]
            if idx.size == 0:
                continue
            cols = pw // s
            cell = idx // 2  # two anchors per place
            cx = (cell % cols) * s
            cy = (cell // cols) * s
            d = out[3 + k].reshape(-1, 4)[idx] * s
            kp = out[6 + k].reshape(-1, 10)[idx] * s
            boxes.append(np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], axis=1))
            points.append(np.stack([cx[:, None] + kp[:, 0::2], cy[:, None] + kp[:, 1::2]], axis=2))
            scores.append(score[idx])
        if not scores:
            return []
        boxes_a = np.concatenate(boxes) / scale
        points_a = np.concatenate(points) / scale
        scores_a = np.concatenate(scores)
        faces = []
        for i in nms(boxes_a, scores_a)[:MAX_FACES]:
            x1, y1, x2, y2 = boxes_a[i]
            p = points_a[i]
            yaw = head_yaw(p)
            faces.append({
                "box": [_r(x1 / w), _r(y1 / h), _r((x2 - x1) / w), _r((y2 - y1) / h)],
                "score": _r(scores_a[i]),
                "frontal": yaw <= FRONTAL_MAX_DEG and abs(turn_of(p[0], p[1], p[2])) <= TURN_MAX,
                "yaw": round(yaw, 1),
                "points": [[_r(px / w), _r(py / h)] for px, py in p],
            })
        faces.sort(key=lambda f: -f["box"][2])
        return faces


class HandModel:
    """YOLOv10 trained on HaGRIDv2's gestures. Its class names come from the model file."""

    def __init__(self, path: Path, threshold: float, threads: int, use_cuda: bool, gpu: GpuThread) -> None:
        self.threshold = threshold
        self._gpu = gpu
        self.session = gpu(open_session, path, session_options(threads), use_cuda, LEAN_CUDA)
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self._input = self.session.get_inputs()[0].name
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get("names", "{}"))
        self.names = [names[i] for i in range(len(names))]
        if NO_GESTURE not in self.names:
            raise ValueError(f"{path.name} is not a HaGRIDv2 gesture model")
        self._none = self.names.index(NO_GESTURE)

    def __call__(self, rgb: np.ndarray) -> dict[str, Any] | None:
        h, w = rgb.shape[:2]
        tw, th = HAND_SIZE
        r = min(tw / w, th / h)
        nw, nh = round(w * r), round(h * r)
        px, py = (tw - nw) // 2, (th - nh) // 2
        canvas = np.full((th, tw, 3), 114, np.uint8)
        img = rgb if (nw, nh) == (w, h) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        canvas[py:py + nh, px:px + nw] = img
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self._gpu(self.session.run, None, {self._input: x})[0][0]  # (4 + classes, places)
        scores = out[4:]
        shown = scores.copy()
        shown[self._none] = 0
        best = shown.max(axis=0)
        i = int(best.argmax())
        if best[i] >= self.threshold:
            gesture, score = self.names[int(shown[:, i].argmax())], float(best[i])
        else:
            i = int(scores[self._none].argmax())
            gesture, score = NO_GESTURE, float(scores[self._none, i])
            if score < self.threshold:
                return None
        cx, cy, bw, bh = out[:4, i]
        x1, y1 = (cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r
        return {"box": [_r(x1 / w), _r(y1 / h), _r(bw / r / w), _r(bh / r / h)], "score": _r(score),
                "gesture": gesture, "fingers": FINGERS.get(gesture)}


@dataclass
class Look:
    faces: list[dict[str, Any]] | None
    hand: dict[str, Any] | None
    ms: dict[str, float]
    embeddings: list[tuple[list[float], np.ndarray]] | None = None
    """Face Recognition: each face looking at the kiosk, the largest first and
    at most ``faces.MAX_FACES``, as its box and embedding, when asked for."""


class Vision:
    """One Light or Accurate set of models, loaded once and shared by every kiosk."""

    def __init__(self, models_dir: Path, model: VisionModel, threads: int = 1, use_cuda: bool = False) -> None:
        self.model = model
        self._gpu = GpuThread(use_cuda, "gpu-vision")
        try:
            self.face = FaceModel(verified_path(models_dir, model.face_file, model.face_sha256),
                                  model.face_threshold, threads, use_cuda, self._gpu)
            self.hand = HandModel(verified_path(models_dir, model.hand_file, model.hand_sha256),
                                  model.hand_threshold, threads, use_cuda, self._gpu)
            from .faces import FaceEmbedder

            self.recognizer = FaceEmbedder(verified_path(models_dir, model.rec_file, model.rec_sha256),
                                           threads, use_cuda, self._gpu)
        except Exception:
            self._gpu.close()
            raise
        self.on_gpu = self.face.on_gpu and self.hand.on_gpu and self.recognizer.on_gpu
        self.max_looks = self._measure()

    def look(self, jpeg: bytes, rotation: int = 0, want: tuple[str, ...] = ("face", "hands"),
             recognize: bool = False) -> Look:
        """What a frame shows. With ``recognize``, also the embedding of each
        face looking at the kiosk, for Face Recognition."""
        t0 = time.perf_counter()
        rgb = decode(jpeg, rotation)
        t1 = time.perf_counter()
        ms = {"decode": round((t1 - t0) * 1000, 1)}
        faces = hand = embeddings = None
        if "face" in want:
            faces = self.face(rgb)
            t2 = time.perf_counter()
            ms["face"] = round((t2 - t1) * 1000, 1)
            t1 = t2
            facing = [f for f in faces if f["frontal"]]
            if recognize and facing:
                from .faces import MAX_FACES

                big = sorted(facing, key=lambda f: -f["box"][2])[:MAX_FACES]
                embeddings = [(f["box"], self.embed(rgb, f)) for f in big]
                t2 = time.perf_counter()
                ms["recognize"] = round((t2 - t1) * 1000, 1)
                t1 = t2
        if "hands" in want:
            hand = self.hand(rgb)
            ms["hands"] = round((time.perf_counter() - t1) * 1000, 1)
        return Look(faces, hand, ms, embeddings)

    def embed(self, rgb: np.ndarray, face: dict[str, Any]) -> np.ndarray:
        from .faces import align, points_px

        return self.recognizer(align(rgb, points_px(face, rgb.shape[1], rgb.shape[0])))

    def crop(self, rgb: np.ndarray, face: dict[str, Any]) -> np.ndarray:
        """A face lined up for recognition, as kept for an enrollment."""
        from .faces import align, points_px

        return align(rgb, points_px(face, rgb.shape[1], rgb.shape[0]))

    def _measure(self) -> dict[str, float]:
        """Looks a second this machine keeps up with for one kiosk, from a few warm runs."""
        rgb = np.full((HAND_SIZE[1], HAND_SIZE[0], 3), 127, np.uint8)
        out = {}
        for key, fn in (("face", self.face), ("hands", self.hand)):
            fn(rgb)
            fn(rgb)
            t0 = time.perf_counter()
            for _ in range(5):
                fn(rgb)
            ms = (time.perf_counter() - t0) * 1000 / 5
            out[key] = round(1000 / max(ms, 1e-3), 1)
        return out

    def close(self) -> None:
        self._gpu.close()
