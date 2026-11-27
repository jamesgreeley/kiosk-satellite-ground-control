"""Isolation on a GPU through Vulkan.

The tests that run the shaders use the Vulkan device this machine has, Mesa's
software one (llvmpipe) included, and skip without one. They need no speech:
what is checked is that the GPU computes the model, and noise shows that as
well. The rest need no GPU.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import numpy as np
import pytest

from ground_control import vulkan
from ground_control.isolator import WINDOW, Isolator
from ground_control.models import ISOLATOR_FILE

from .conftest import MODELS, SR

MODEL = MODELS / ISOLATOR_FILE


def sound(seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise that comes and goes like talking, and a voiceprint."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    level = 0.05 + 0.15 * (np.sin(np.arange(n) * (2 * np.pi * 3 / SR)) > 0)
    voiceprint = rng.standard_normal(192)
    return (level * rng.standard_normal(n)).astype(np.float32), (voiceprint / np.linalg.norm(voiceprint)).astype(np.float32)


def apart_db(there: np.ndarray, here: np.ndarray) -> float:
    """How far the difference of the two is under the second. The model lets
    little of the noise through, so the noise itself is no measure."""
    return float(10 * np.log10(np.mean((there - here) ** 2) / np.mean(here ** 2) + 1e-20))


# ── Without a GPU ───────────────────────────────────────────────────────────

def test_messages_cross_a_pipe_whole_and_time_out():
    r, w = os.pipe()
    with os.fdopen(w, "wb") as out:
        data = np.arange(300000, dtype=np.float32).tobytes()
        writer = threading.Thread(target=vulkan._write, args=(out, {"samples": 300000}, data[:40], data[40:]))
        writer.start()
        header, got = vulkan._read(r, time.monotonic() + 5)
        writer.join()
        assert header == {"samples": 300000, "data": len(data)} and got == data
        with pytest.raises(TimeoutError):
            vulkan._read(r, time.monotonic() + 0.05)
    with pytest.raises(EOFError):
        vulkan._read(r, time.monotonic() + 1)
    os.close(r)


class FakeEngine:
    """Counts the jobs a run would be sent as, for a GPU of a given speed."""

    def __init__(self, rate: float) -> None:
        self.rate, self.sent = rate, []
        self.device = self
        self.queue = self

    def create_command_encoder(self):
        self.work = []
        return self

    def begin_compute_pass(self):
        return self

    def set_pipeline(self, pipeline): pass
    def set_bind_group(self, *args): pass
    def end(self): pass
    def finish(self): return self
    def on_submitted_work_done_sync(self): pass

    def dispatch_workgroups(self, *size):
        self.work.append(size)

    def submit(self, buffers):
        self.sent.append(self.work)


class Program:
    pipeline = None


def test_a_slow_gpu_gets_more_and_smaller_jobs(monkeypatch):
    """A job that runs for 2 s loses the GPU with Linux's AMD driver, so the
    work is cut by how much this GPU does in JOB_S."""
    # Measured rates don't move in this test: the fake GPU takes no time.
    clock = iter(np.arange(0, 1e6, 1.0))
    monkeypatch.setattr(vulkan.time, "perf_counter", lambda: next(clock))
    step = 1e7                                  # multiply-adds of one LSTM step
    counts = {}
    for rate in (1e12, 1e9, 1e7):
        engine = FakeEngine(rate)
        monkeypatch.setattr(vulkan._Jobs, "send", lambda self, real=vulkan._Jobs.send: (real(self), setattr(self._e, "rate", rate))[0])
        jobs = vulkan._Jobs(engine)
        for s in range(100):
            jobs.add(Program, None, s * vulkan.ROW, (32, 4, 2), step)
        jobs.slices(Program, None, 0, (2, 4), 100, step)
        jobs.send()
        counts[rate] = len(engine.sent)
        # Nothing is lost and no job is longer than JOB_S, unless one dispatch alone is.
        assert sum(len(job) for job in engine.sent) >= 101
        slices = [size[2] for job in engine.sent for size in job if len(size) == 3 and size[:2] == (2, 4)]
        assert sum(slices) == 100
        assert all(len(job) * step <= max(rate * vulkan.JOB_S, step) + step for job in engine.sent[:-1])
    assert counts[1e12] == 1 and counts[1e9] == 10 and counts[1e7] == 200


def test_the_model_is_cut_at_the_same_places():
    """The head gives the features, the tail takes the masks, and neither has an LSTM."""
    pytest.importorskip("onnx")
    if not MODEL.is_file():
        pytest.skip("no Voice Isolator model")
    import onnx

    parts = vulkan.split(MODEL)
    head, tail = onnx.load_from_string(parts.head), onnx.load_from_string(parts.tail)
    assert [i.name for i in head.graph.input] == ["audio", "voiceprint"]
    assert parts.features in [o.name for o in head.graph.output]
    assert [i.name for i in tail.graph.input][:vulkan.BANDS] == parts.mask_names
    assert not [n for n in (*head.graph.node, *tail.graph.node) if n.op_type == "LSTM"]
    assert len(parts.blocks) == 6 and len(parts.scales) == 5 and sum(parts.mask_rows) == len(parts.masks["w3"])
    meta = {p.key: p.value for p in head.metadata_props}
    assert meta == {p.key: p.value for p in onnx.load(str(MODEL), load_external_data=False).metadata_props}


def test_the_rewritten_layers_compute_what_they_did():
    """ONNX Runtime's WebGPU provider gets grouped convolutions with several
    outputs for a group wrong and has no PRelu, so both are rewritten. On the
    CPU the model has to give what it gave before."""
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = rng.standard_normal((12, 1, 2, 1)).astype(np.float32)            # 4 groups, 3 outputs each
    depthwise = rng.standard_normal((12, 1, 3, 3)).astype(np.float32)   # left as it is
    slope = rng.standard_normal((12, 1, 1)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["a"], group=4, kernel_shape=[2, 1], strides=[2, 1]),
         helper.make_node("PRelu", ["a", "slope"], ["b"]),
         helper.make_node("Conv", ["b", "dw"], ["y"], group=12, kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(slope, "slope"), numpy_helper.from_array(depthwise, "dw")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    x = {"x": rng.standard_normal((1, 4, 8, 6)).astype(np.float32)}
    cpu = ["CPUExecutionProvider"]
    before = ort.InferenceSession(model.SerializeToString(), providers=cpu).run(None, x)[0]
    assert vulkan.rewrite(model) == {"grouped convolutions": 1, "PRelu": 1}
    kinds = [n.op_type for n in model.graph.node]
    assert "PRelu" not in kinds and kinds.count("Conv") == 2
    assert [a.i for n in model.graph.node if n.op_type == "Conv" for a in n.attribute if a.name == "group"] == [1, 12]
    after = ort.InferenceSession(model.SerializeToString(), providers=cpu).run(None, x)[0]
    assert np.allclose(before, after, atol=1e-5)


# ── On a Vulkan device ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("wgpu")
    pytest.importorskip("onnx")
    if not MODEL.is_file():
        pytest.skip("no Voice Isolator model")
    try:
        return vulkan.choose(os.environ.get("VULKAN_DEVICE", ""), software=True)
    except vulkan.Unavailable as err:
        pytest.skip(str(err))


@pytest.fixture(scope="module")
def model(adapter):
    return vulkan.Model(adapter, MODEL, 2, 2 * SR, vulkan.SMALL_SHAPES)


@pytest.fixture(scope="module")
def on_cpu():
    return Isolator(MODEL, threads=2)


def test_the_gpu_gives_the_cpus_audio(model, on_cpu):
    # A shorter command after a longer one runs in the longer one's buffers,
    # and 0.7 s is not a whole number of the shaders' groups of frames.
    for seconds, seed in ((1.0, 1), (0.7, 2), (0.5, 3)):
        audio, voiceprint = sound(seconds, seed)
        there = model.run(audio[None], voiceprint[None])[0]
        # The same samples through the whole model: Isolator.clean would pad them to half seconds first.
        here = on_cpu.session.run(None, {"audio": audio[None], "voiceprint": voiceprint[None]})[0][0]
        assert len(there) == len(audio)
        assert apart_db(there, here) < -40.0, seconds


def test_a_slow_gpu_is_sent_short_jobs_and_gives_the_same_audio(model):
    audio, voiceprint = sound(0.5, 4)
    whole = model.run(audio[None], voiceprint[None])
    one = model.engine.jobs
    model.engine.rate = 1e7                    # taken for very slow: the first jobs are one dispatch each
    cut = model.run(audio[None], voiceprint[None])
    assert model.engine.jobs > one
    assert apart_db(cut, whole) < -80.0


def test_a_gpu_that_computes_something_else_is_refused(model, monkeypatch):
    right = vulkan.Engine.run
    monkeypatch.setattr(vulkan.Engine, "run", lambda self, features, voiceprint: 0.5 * right(self, features, voiceprint))
    with pytest.raises(vulkan.Unavailable, match="doesn't compute"):
        model.check(MODEL, software=True)


def test_the_helper_process_runs_ends_and_starts_again(adapter, on_cpu, monkeypatch):
    """Everything on the GPU runs in a helper process. When it is lost, the
    run it was working on fails, the next ones do until it is back, and
    Ground Control goes on."""
    device = vulkan.label(adapter)
    assert vulkan.probe(device, software=True)["name"] == device
    worker = vulkan.Worker(MODEL, 2, 2 * SR, device=device, software=True)
    try:
        assert worker.info["name"] == device and worker.info["db"] < -40
        iso = Isolator(MODEL, threads=2, vulkan=worker)
        assert iso.on_gpu and iso.vulkan and iso.version == on_cpu.version != "unknown"
        audio, voiceprint = sound(0.6, 5)
        assert apart_db(iso.clean(audio, voiceprint), on_cpu.clean(audio, voiceprint)) < -40.0
        os.kill(worker._proc.pid, signal.SIGKILL)
        with pytest.raises(vulkan.Failed, match="ended"):
            iso.clean(audio, voiceprint)
        deadline = time.monotonic() + 120
        while True:
            try:
                again = iso.clean(audio, voiceprint)
                break
            except vulkan.Failed:
                assert time.monotonic() < deadline
                time.sleep(0.2)
        assert apart_db(again, on_cpu.clean(audio, voiceprint)) < -40.0
    finally:
        worker.close()


def test_no_gpu_by_that_name_says_what_was_found(adapter):
    with pytest.raises(vulkan.Unavailable, match="found no GPU with 'no such gpu'"):
        vulkan.choose("no such gpu", software=True)
    with pytest.raises(vulkan.Unavailable, match="found no GPU with"):
        vulkan.probe("no such gpu", software=True)


# ── The other models, through ONNX Runtime's WebGPU build ───────────────────

@pytest.fixture(scope="module")
def gpu(adapter):
    """The GPU as the service sees it. Needs ONNX Runtime's WebGPU build:
    installed, or in the folder VULKAN_ORT names as in the Vulkan image."""
    g = vulkan.Gpu(vulkan.label(adapter), 2, software=True)
    light = g.model(MODELS / "redimnet2_b1.onnx")
    if not light.on_gpu:
        pytest.skip(f"no ONNX Runtime with a WebGPU provider: {light.why_cpu}")
    light.close()
    return g


def test_the_speaker_model_gives_the_cpus_voiceprints_on_the_gpu(gpu):
    from ground_control.engine import Embedder

    path = MODELS / "redimnet2_b1.onnx"
    here = Embedder(path, threads=2)
    there = Embedder(path, threads=2, remote=gpu.model(path))
    try:
        assert there.on_gpu and there.dim == here.dim
        for seconds, seed in ((1.0, 1), (2.5, 2), (0.05, 3)):
            audio, _ = sound(seconds, seed)
            assert float(here.embed(audio) @ there.embed(audio)) > 0.9999, seconds
        assert gpu.on_cpu() == []
    finally:
        there.close()


def test_a_model_goes_on_on_the_cpu_when_its_helper_is_lost(gpu):
    path = MODELS / "arcface_w600k_mbf.onnx"
    model = gpu.model(path)
    try:
        assert model.on_gpu and model.inputs == ["input.1"] and model.outputs == 1 and model.ranks == [4]
        x = {"input.1": np.random.default_rng(0).standard_normal((1, 3, 112, 112)).astype(np.float32)}
        first = model.run(x)[0]
        os.kill(model._helper._proc.pid, signal.SIGKILL)
        second = model.run(x)[0]                      # this one is done on the CPU
        cosine = float(first.ravel() @ second.ravel() / (np.linalg.norm(first) * np.linalg.norm(second)))
        assert first.shape == second.shape == (1, 512) and cosine > 0.999
        deadline = time.monotonic() + 120
        while not model._helper.up:                   # and the helper comes back
            assert time.monotonic() < deadline
            time.sleep(0.2)
        assert np.allclose(model.run(x)[0], first, atol=1e-3)
    finally:
        model.close()
    assert gpu.on_cpu() == []                         # a closed model isn't listed
