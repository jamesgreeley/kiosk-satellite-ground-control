"""The OpenVINO image's backend: the vision models on an Intel GPU.

These run OpenVINO on its CPU device, which every machine has. The code path
is the one the GPU takes: a model compiled for each input shape.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openvino")

from ground_control import config, ov  # noqa: E402
from ground_control.models import VISION_FEATURES  # noqa: E402
from ground_control.service import GroundControl  # noqa: E402
from ground_control.vision import Vision  # noqa: E402

from .conftest import MODELS  # noqa: E402
from .test_vision import jpeg  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.fixture(scope="module")
def device(tmp_path_factory) -> ov.Gpu:
    return ov.Gpu("CPU", tmp_path_factory.mktemp("openvino"))


@pytest.fixture(scope="module")
def both(device):
    """The Light vision models through OpenVINO and through ONNX Runtime."""
    keys = {f: "light" for f in VISION_FEATURES}
    there, here = Vision(MODELS, keys, threads=2, openvino=device), Vision(MODELS, keys, threads=2)
    yield there, here
    there.close()
    here.close()


def test_a_device_that_isnt_there_says_why():
    with pytest.raises(ov.Unavailable, match="OpenVINO can't start: .*The OpenVINO image needs an Intel GPU"):
        ov.gpu_name("GPU.9")


def test_the_vision_models_give_what_they_give_on_the_cpu(both):
    there, here = both
    assert there.on_gpu and not there.keeps_memory and not here.on_gpu
    assert there.hand.names == here.hand.names
    rng = np.random.default_rng(0)
    # The face model in a landscape frame and in a portrait one: a model compiled for each.
    for shape in ((1, 3, 480, 640), (1, 3, 640, 480)):
        x = ((rng.integers(0, 256, shape) - 127.5) / 128.0).astype(np.float32)
        got, want = there.face._run(x), here.face._run(x)
        assert len(got) == len(want) == 9
        assert all(a.shape == b.shape and cosine(a, b) > 0.999 for a, b in zip(got, want))
    x = (rng.integers(0, 256, (1, 3, 480, 640)) / 255.0).astype(np.float32)
    assert cosine(there.hand._ov.run({there.hand._input: x})[0], here.hand.session.run(None, {here.hand._input: x})[0]) > 0.999
    crop = rng.integers(0, 256, (112, 112, 3)).astype(np.uint8)
    assert cosine(there.recognizer(crop), here.recognizer(crop)) > 0.999


def test_a_look_through_openvino(both):
    there, _ = both
    look = there.look(jpeg(np.full((480, 640, 3), 128, np.uint8)))
    assert look.faces == [] and look.hand is None
    assert set(look.ms) == {"decode", "face", "hands", "total"}
    assert set(there.ms) == {"face", "hands", "recognition"}


def test_only_so_many_shapes_stay_compiled(device):
    from ground_control.models import VISION_MODELS

    model = device.model(MODELS / VISION_MODELS["face"]["light"].file, shapes=2)
    name = model.inputs[0]
    for h, w in ((480, 640), (640, 480), (384, 640), (480, 640)):
        model.run({name: np.zeros((1, 3, h, w), np.float32)})
    assert list(model._compiled) == [((1, 3, 384, 640),), ((1, 3, 480, 640),)]


def test_openvino_settings_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTIONS_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv("OPENVINO", "true")
    cfg = config.load()
    assert cfg.openvino and cfg.openvino_device == "GPU" and not cfg.cuda
    monkeypatch.setenv("OPENVINO_DEVICE", "GPU.1")
    assert config.load().openvino_device == "GPU.1"
    monkeypatch.setenv("CUDA", "true")
    with pytest.raises(ValueError, match="can't both be on"):
        config.load()


def test_the_service_with_openvino(cfg):
    cfg.openvino, cfg.openvino_device = True, "CPU"
    vm = GroundControl(cfg)
    # Voices are checked on the CPU there and Isolation runs the model made for it.
    assert vm.settings()["openvino"] and not vm.active.on_gpu
    assert [gpu for _, _, gpu in vm._isolator_files()] == [False, False]


def test_without_a_gpu_the_openvino_image_stops_with_the_reason(cfg):
    cfg.openvino, cfg.openvino_device = True, "GPU.9"
    with pytest.raises(ov.Unavailable, match="OpenVINO can't start"):
        GroundControl(cfg)
