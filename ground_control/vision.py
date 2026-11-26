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
  wide the points are too rough for the fit, and it catches those. A face the
  edge of the frame cuts off never looks at the kiosk (``whole_face``), and
  neither does one seen from far below or above (the pitch, see PITCH_MIN_DEG).
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
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .engine import (LEAN_CUDA, GpuThread, apple_neural_engine_available, open_session, run_options,
                    session_options as _session_options)
from .models import VISION_MODELS, verified_path

_LOG = logging.getLogger(__name__)

FRONTAL_MAX_DEG = 30.0
"""How far a head can be turned from the camera, in degrees, and still look at the kiosk.
On kiosk frames, faces looking at the screen measured 0 to 4 degrees and faces
looking away 45 to 77."""

PITCH_MIN_DEG, PITCH_MAX_DEG = -35.0, 20.0
"""How far up or down a face can be seen from and still look at the kiosk
(``head_angles``' pitch). Under the first, the camera sees the face from far
below: the chin and the nostrils of someone looking over the kiosk. Over the
second it sees the top of a head that is bowed. On kiosk frames of faces
looking at the screen from 1 to 3 m the pitch measured -33 to 14, around
-13: the generic face's nose isn't everyone's, and a kiosk's camera sits
below the face. On a tablet lying on a bed it ran from -55 to 48, with the
faces under -35 and over 20 not looking at it."""

TURN_MAX = 0.6
"""The nose's offset along the eye line, in eye distances, up to which a face can look at the kiosk."""

FACE_3D = np.array([[-31, -33, -30], [31, -33, -30], [0, 0, 0], [-25, 30, -25], [25, 30, -25]], np.float64)
"""A generic face in millimeters, x right, y down, z toward the camera: the eyes,
the tip of the nose and the corners of the mouth, in SCRFD's order."""
NMS_IOU = 0.4
MAX_FACES = 10
FACE_SIDE = 640
"""Faces are found on the frame with its longer side at most this long."""
FACE_SHAPES = 4
"""On a GPU, how many frame shapes the face model keeps a session for. A GPU
session sets up again each time its input shape changes, which takes 80 ms
with CUDA 12 for a model that runs in 4 ms. With one session, kiosks whose
frames differ in shape, such as one in landscape and one in portrait, would
pay that on every look. Landscape and portrait frames in 4:3 and 16:9 make four."""
HAND_SIZE = (640, 480)
"""The hand model's fixed input, width by height."""
KEEPS = "about 0.25 GB"
"""What the three Accurate models keep on a GPU between looks with
``KEEP_GPU_MEMORY`` on. Without it each model hands its memory back after
every run. Taking it again for the next run costs about 4 ms a look on an
RTX 4000 and 13 ms on a GB10, where taking GPU memory is slow."""
AT_ONCE = 4
"""Looks that can have their hand model running beside their face model at the
same moment. The service runs at most this many looks at a time."""
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


WHOLE_FACE = 0.9
"""How much of a face's box has to be inside the frame for the face to be whole."""


def whole_face(points: np.ndarray, box: tuple[float, float, float, float], width: int, height: int) -> bool:
    """Whether a face is in the frame as a whole: all five of its points, in
    pixels, and at least WHOLE_FACE of its box (x1, y1, x2, y2 in pixels).

    The face model gives five points for every face it finds, also for one
    the edge of the frame cuts off. The hidden ones are guesses that land
    outside the picture. A head angle worked out from guessed points means
    nothing (it swung from 55 to 27 degrees on a face that never moved), and
    a face with half of it missing was still recognized with the score of the
    whole face. With the points just inside and the chin and a cheek outside,
    a face in the corner of a tablet's frame still counted. So a face that is
    cut off doesn't look at the kiosk and isn't recognized. On the kiosk test
    set, 2 of 1,566 faces looking at the kiosk are cut off by this measure."""
    p = np.asarray(points, np.float64)
    if not ((p[:, 0] >= 0).all() and (p[:, 0] <= width).all() and (p[:, 1] >= 0).all() and (p[:, 1] <= height).all()):
        return False
    x1, y1, x2, y2 = box
    area = max(x2 - x1, 1e-9) * max(y2 - y1, 1e-9)
    inside = max(0.0, min(x2, width) - max(x1, 0.0)) * max(0.0, min(y2, height) - max(y1, 0.0))
    return bool(inside / area >= WHOLE_FACE)


def head_angles(points: np.ndarray) -> tuple[float, float]:
    """How far the head is turned from the camera, in degrees, from its five
    points in pixels: the generic face fitted to them with a scaled
    orthographic camera.

    The yaw is the turn left or right, 0 straight at the camera, without a
    sign. The pitch is the angle up or down: negative when the camera sees the
    face from below, positive from above or with the head bowed."""
    model = FACE_3D - FACE_3D.mean(0)
    image = np.asarray(points, np.float64)
    image = image - image.mean(0)
    # Level the eyes first, so a tilted head doesn't read as a turned one.
    ex, ey = image[1] - image[0]
    a = -np.arctan2(ey, ex)
    image = image @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    m, *_ = np.linalg.lstsq(model, image, rcond=None)
    r1, r2 = m[:, 0], m[:, 1]
    if not np.isfinite(m).all() or float(r1 @ r1) <= 0 or float(r2 @ r2) <= 0:
        return 90.0, 90.0
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    return (float(np.degrees(np.arcsin(min(1.0, abs(r1[2]))))),
            float(np.degrees(np.arcsin(float(np.clip(r2[2], -1.0, 1.0))))))


def head_yaw(points: np.ndarray) -> float:
    """How far the head is turned left or right from the camera, in degrees. 0 is straight at the camera."""
    return head_angles(points)[0]


def facing(points: np.ndarray, box: tuple[float, float, float, float], width: int,
           height: int) -> tuple[bool, float, float]:
    """Whether a face with these five points and this box (x1, y1, x2, y2), in
    pixels, looks at the kiosk,
    with its yaw and pitch: the whole face in the frame, the head turned at
    most FRONTAL_MAX_DEG left or right, seen from between PITCH_MIN_DEG and
    PITCH_MAX_DEG up or down, and the nose where a facing head's is."""
    yaw, pitch = head_angles(points)
    looks = (whole_face(points, box, width, height) and yaw <= FRONTAL_MAX_DEG and PITCH_MIN_DEG <= pitch <= PITCH_MAX_DEG
             and abs(turn_of(points[0], points[1], points[2])) <= TURN_MAX)
    return bool(looks), yaw, pitch  # a plain bool: it goes out as JSON


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


# Compile only common kiosk dimensions. Other sizes keep the fully dynamic
# session so a stream of unusual frame sizes cannot grow the compilation cache.
_COREML_FRAME_SIZES = {(480, 640), (384, 640), (640, 480)}


def fixed_scrfd_session(path: Path, shape: tuple[int, int]):
    """Specialize SCRFD's input and all nine anchor outputs for Core ML."""
    import onnx
    import onnxruntime as ort

    height, width = shape
    graph = onnx.load(path)
    dims = graph.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_value, dims[3].dim_value = height, width
    del graph.graph.value_info[:]  # Let ORT infer intermediates for this frame.
    for i, output in enumerate(graph.graph.output):
        stride = (8, 16, 32)[i % 3]
        output.type.tensor_type.shape.dim[0].dim_value = 2 * (height // stride) * (width // stride)
    opts = {"ModelFormat": "NeuralNetwork", "MLComputeUnits": "CPUAndNeuralEngine"}
    return ort.InferenceSession(graph.SerializeToString(), session_options(1),
                                providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"])


class FaceModel:
    """SCRFD with five face points."""

    def __init__(self, path: Path, threshold: float, threads: int, use_cuda: bool, gpu: GpuThread,
                 keep_memory: bool = False, openvino: Any = None, use_coreml: bool = False) -> None:
        """``openvino``: the Intel GPU to run on in the OpenVINO image (``ov.Gpu``)."""
        self.threshold = threshold
        self._gpu = gpu
        self._ov = None
        if openvino is not None:
            # Compiled for each frame shape, as many as a CUDA GPU has sessions for.
            self._ov = openvino.model(path, shapes=FACE_SHAPES)
            self.session, self.on_gpu, self._input = None, True, self._ov.inputs[0]
            self.on_coreml = False
            if self._ov.outputs != 9:
                raise ValueError(f"{path.name} is not a SCRFD model with face points")
            return
        # Core ML handles the convolution-heavy detector outside ORT's CPU
        # pool. More ORT worker threads add contention at its CPU partitions;
        # one thread was consistently as fast or faster in the M1 Pro sweep.
        self._fixed = {} if use_coreml and not use_cuda else None
        self._fixed_lock = threading.Lock()
        dynamic_coreml = use_coreml and path.stem == "scrfd_2.5g"
        self._path, self._options = path, session_options(1 if dynamic_coreml else threads)
        self.session = gpu(open_session, path, self._options, use_cuda, LEAN_CUDA, dynamic_coreml)
        self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self.on_coreml = "CoreMLExecutionProvider" in self.session.get_providers()
        self._input = self.session.get_inputs()[0].name
        if len(self.session.get_outputs()) != 9:
            raise ValueError(f"{path.name} is not a SCRFD model with face points")
        # On a GPU, a session for each frame shape (see FACE_SHAPES). The
        # first shape seen, the one the models are timed with when they load,
        # runs in the session above. With ``keep_memory`` it keeps its GPU
        # memory between looks (see KEEPS). The others always hand theirs back.
        self._first: tuple[int, int] | None = None
        self._others: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._hand_back = run_options(self.on_gpu)
        self._after_run = run_options(self.on_gpu and not keep_memory)

    def _run(self, x: np.ndarray) -> list[np.ndarray]:
        """The model on a prepared frame. Runs on the GPU thread, so sessions are opened one at a time."""
        if self._ov is not None:
            return self._ov.run({self._input: x})
        if not self.on_gpu:
            shape = x.shape[2:]
            if self._fixed is not None and shape in _COREML_FRAME_SIZES:
                with self._fixed_lock:
                    if shape not in self._fixed:
                        try:
                            self._fixed[shape] = fixed_scrfd_session(self._path, shape)
                        except Exception as err:
                            _LOG.warning("Core ML specialization failed for %s: %s", shape, err)
                            self._fixed[shape] = None
                    session = self._fixed[shape]
                if session is not None:
                    try:
                        result = session.run(None, {self._input: x})
                    except Exception as err:
                        _LOG.warning("Core ML prediction failed for %s: %s", shape, err)
                        with self._fixed_lock:
                            self._fixed[shape] = None
                    else:
                        self.on_coreml = "CoreMLExecutionProvider" in session.get_providers()
                        return result
            return self.session.run(None, {self._input: x})
        shape = (x.shape[2], x.shape[3])
        if self._first is None:
            self._first = shape
        if shape == self._first:
            return self.session.run(None, {self._input: x}, self._after_run)
        session = self._others.get(shape)
        if session is None:
            if len(self._others) >= FACE_SHAPES - 1:
                self._others.popitem(last=False)
            session = self._others[shape] = open_session(self._path, self._options, True, LEAN_CUDA)
            _LOG.debug("face model session opened for %sx%s frames", shape[1], shape[0])
        else:
            self._others.move_to_end(shape)
        return session.run(None, {self._input: x}, self._hand_back)

    def __call__(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        h, w = rgb.shape[:2]
        scale = min(1.0, FACE_SIDE / max(h, w))
        img = rgb
        if scale < 1.0:
            img = np.asarray(Image.fromarray(rgb).resize((round(w * scale), round(h * scale)), Image.BILINEAR))
        ih, iw = img.shape[:2]
        ph, pw = -(-ih // 32) * 32, -(-iw // 32) * 32
        # Straight into the layout the model takes, scaled in place. Padding
        # is black, as before it is scaled.
        x = np.empty((1, 3, ph, pw), np.float32) if (ph, pw) == (ih, iw) else np.zeros((1, 3, ph, pw), np.float32)
        x[0, :, :ih, :iw] = img.transpose(2, 0, 1)
        x -= 127.5
        x /= 128.0
        out = self._gpu(self._run, x)
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
            looks, yaw, pitch = facing(p, (x1, y1, x2, y2), w, h)
            faces.append({
                "box": [_r(x1 / w), _r(y1 / h), _r((x2 - x1) / w), _r((y2 - y1) / h)],
                "score": _r(scores_a[i]),
                "frontal": looks,
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "points": [[_r(px / w), _r(py / h)] for px, py in p],
            })
        faces.sort(key=lambda f: -f["box"][2])
        return faces


class HandModel:
    """YOLOv10 trained on HaGRIDv2's gestures. Its class names come from the model file."""

    def __init__(self, path: Path, threshold: float, threads: int, use_cuda: bool, gpu: GpuThread,
                 keep_memory: bool = False, openvino: Any = None) -> None:
        self.threshold = threshold
        self._gpu = gpu
        self._ov = openvino.model(path) if openvino is not None else None
        if self._ov is not None:
            self.session, self.on_gpu, self._input = None, True, self._ov.inputs[0]
            self.on_coreml = False
            names = ast.literal_eval(self._ov.metadata("names") or "{}")
        else:
            self.session = gpu(open_session, path, session_options(threads), use_cuda, LEAN_CUDA)
            self.on_gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
            self.on_coreml = False
            self._input = self.session.get_inputs()[0].name
            self._after_run = run_options(self.on_gpu and not keep_memory)
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
        img = rgb if (nw, nh) == (w, h) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        # Straight into the layout the model takes, scaled in place. A frame
        # of another shape sits in the middle of a gray one.
        if (nw, nh) == (tw, th):
            x = np.empty((1, 3, th, tw), np.float32)
        else:
            x = np.full((1, 3, th, tw), 114, np.float32)
        x[0, :, py:py + nh, px:px + nw] = img.transpose(2, 0, 1)
        x /= 255.0
        if self._ov is not None:
            out = self._gpu(self._ov.run, {self._input: x})[0][0]
        else:
            out = self._gpu(self.session.run, None, {self._input: x}, self._after_run)[0][0]  # (4 + classes, places)
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
    """Vision Match's models, one Light or Accurate choice for each feature
    (``keys``: face, hands and recognition), loaded once and shared by every
    kiosk."""

    def __init__(self, models_dir: Path, keys: dict[str, str], threads: int = 1, use_cuda: bool = False,
                 keep_memory: bool = False, openvino: Any = None) -> None:
        """``keep_memory``: on an NVIDIA GPU, the models keep their memory
        between looks (the ``KEEP_GPU_MEMORY`` setting, see KEEPS).
        ``openvino``: the Intel GPU the models run on in the OpenVINO image
        (``ov.Gpu``)."""
        self.keys = dict(keys)
        face, hand, rec = (VISION_MODELS[f][keys[f]] for f in ("face", "hands", "recognition"))
        on_gpu = use_cuda or openvino is not None
        self._gpu = GpuThread(on_gpu, "gpu-vision")
        # The hand model runs beside the face model in a look that asks for
        # both (see ``look``), so on a GPU it has a thread of its own.
        self._gpu_hands = GpuThread(on_gpu, "gpu-vision-hands")
        self._beside = ThreadPoolExecutor(AT_ONCE, thread_name_prefix="vision-hands")
        try:
            self.face = FaceModel(verified_path(models_dir, face.file, face.sha256),
                                  face.threshold, threads, use_cuda, self._gpu, keep_memory, openvino,
                                  use_coreml=apple_neural_engine_available())
            self.hand = HandModel(verified_path(models_dir, hand.file, hand.sha256),
                                  hand.threshold, threads, use_cuda, self._gpu_hands, keep_memory, openvino)
            from .faces import FaceEmbedder

            self.recognizer = FaceEmbedder(verified_path(models_dir, rec.file, rec.sha256),
                                           threads, use_cuda, self._gpu, keep_memory, openvino,
                                           use_coreml=apple_neural_engine_available(),
                                           coreml_options={"ModelFormat": "NeuralNetwork"} if rec.key == "light" else None,
                                           coreml_threads=1 if rec.key == "light" else 2)
        except Exception:
            self.close()
            raise
        self.on_gpu = self.face.on_gpu and self.hand.on_gpu and self.recognizer.on_gpu
        self.keeps_memory = self.on_gpu and keep_memory and openvino is None
        self.ms: dict[str, float] = {}
        """Each feature's model time on this machine, in ms: a face look, a
        hand look and one face recognized. Measured when the models load."""
        self.max_looks = self._measure()

    @property
    def on_coreml(self) -> bool:
        return self.face.on_coreml or self.recognizer.on_coreml

    @property
    def rec_key(self) -> str:
        """The Face Recognition model, which faceprints are kept for."""
        return self.keys["recognition"]

    @property
    def label(self) -> str:
        """``light`` or ``accurate`` when every feature uses it, else ``mixed``."""
        kinds = set(self.keys.values())
        return kinds.pop() if len(kinds) == 1 else "mixed"

    def look(self, jpeg: bytes, rotation: int = 0, want: tuple[str, ...] = ("face", "hands"),
             recognize: bool = False, hand_min: float = 0.0) -> Look:
        """What a frame shows. With ``recognize``, also the embedding of each
        face looking at the kiosk, for Face Recognition. A hand whose box is
        smaller than ``hand_min`` of the frame on its longer side counts as no
        hand (Hand sensitivity)."""
        t0 = time.perf_counter()
        rgb = decode(jpeg, rotation)
        t1 = time.perf_counter()
        ms = {"decode": round((t1 - t0) * 1000, 1)}
        faces = hand = embeddings = None
        # A look for both runs the hand model beside the face model: neither
        # needs the other's result, and one after the other took the sum of
        # the two.
        beside = self._beside.submit(self._hand, rgb, hand_min) if "hands" in want and "face" in want else None
        try:
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
                    ms["recognize"] = round((time.perf_counter() - t1) * 1000, 1)
        except BaseException:
            if beside is not None:
                beside.cancel()
            raise
        if beside is not None:
            hand, ms["hands"] = beside.result()
        elif "hands" in want:
            hand, ms["hands"] = self._hand(rgb, hand_min)
        # The stages overlap in a look for both, so their times no longer add up to the look's.
        ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return Look(faces, hand, ms, embeddings)

    def _hand(self, rgb: np.ndarray, hand_min: float) -> tuple[dict[str, Any] | None, float]:
        """The shown hand and the hand model's own time in ms."""
        t = time.perf_counter()
        hand = self.hand(rgb)
        if hand is not None and max(hand["box"][2], hand["box"][3]) < hand_min:
            hand = None
        return hand, round((time.perf_counter() - t) * 1000, 1)

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
        crop = np.full((112, 112, 3), 127, np.uint8)
        out = {}
        for key, fn, arg in (("face", self.face, rgb), ("hands", self.hand, rgb),
                             ("recognition", self.recognizer, crop)):
            fn(arg)
            fn(arg)
            t0 = time.perf_counter()
            for _ in range(5):
                fn(arg)
            ms = (time.perf_counter() - t0) * 1000 / 5
            self.ms[key] = round(ms, 1)
            if key != "recognition":
                out[key] = round(1000 / max(ms, 1e-3), 1)
        return out

    def close(self) -> None:
        self._beside.shutdown(wait=False)
        self._gpu.close()
        self._gpu_hands.close()
