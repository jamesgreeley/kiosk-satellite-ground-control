"""How the models are run: quiet threads between runs, and one thread for a GPU."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from ground_control import engine
from ground_control.engine import GpuThread, session_options


def test_worker_threads_stop_spinning_when_a_run_ends():
    """Left spinning, they take about three cores while someone talks."""
    so = session_options(4)
    assert so.get_session_config_entry("session.force_spinning_stop") == "1"
    assert so.intra_op_num_threads == 4


def test_every_gpu_run_goes_through_one_thread():
    gpu = GpuThread(True, "gpu-test")
    try:
        with ThreadPoolExecutor(4) as workers:
            names = set(workers.map(lambda _: gpu(lambda: threading.current_thread().name), range(16)))
        assert len(names) == 1 and next(iter(names)).startswith("gpu-test")
    finally:
        gpu.close()


def test_without_a_gpu_a_run_stays_on_its_own_thread():
    cpu = GpuThread(False, "gpu-test")
    assert cpu(threading.get_ident) == threading.get_ident()
    assert cpu(lambda a, b: a + b, 2, 3) == 5
    cpu.close()


def test_a_gpu_run_that_fails_raises_where_it_was_called():
    gpu = GpuThread(True, "gpu-test")
    try:
        try:
            gpu(lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError("the error was swallowed")
        assert gpu(lambda: 7) == 7
    finally:
        gpu.close()


def test_coreml_provider_is_opt_in_to_apple_vision_models(monkeypatch):
    monkeypatch.setattr(engine, "apple_neural_engine_available", lambda: True)
    selected = engine.providers(False, use_coreml=True)
    assert selected[0] == ("CoreMLExecutionProvider", engine.COREML_OPTIONS)
    assert selected[1] == "CPUExecutionProvider"
    assert engine.providers(False) == ["CPUExecutionProvider"]


def test_coreml_session_creation_failure_uses_cpu(monkeypatch, tmp_path):
    calls = []
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    class Session:
        pass

    def create(path, options, providers):
        calls.append(providers)
        if providers[0][0] == "CoreMLExecutionProvider":
            raise RuntimeError("unsupported Core ML graph")
        return Session()

    monkeypatch.setattr(engine, "apple_neural_engine_available", lambda: True)
    monkeypatch.setattr(engine.ort, "InferenceSession", create)
    result = engine.open_session(model, session_options(1), False, use_coreml=True)
    assert isinstance(result, Session)
    assert calls[0][0][0] == "CoreMLExecutionProvider"
    assert calls[0][0][1] == engine.COREML_OPTIONS | {"ModelCacheDirectory": engine._coreml_cache(model)}
    assert calls[0][1] == "CPUExecutionProvider" and calls[1] == ["CPUExecutionProvider"]


def test_coreml_without_cache_option_retries_without_disk_cache(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    calls = []

    class Session:
        pass

    def create(path, options, providers):
        calls.append(providers)
        if "ModelCacheDirectory" in providers[0][1]:
            raise RuntimeError("unsupported option: ModelCacheDirectory")
        return Session()

    monkeypatch.setattr(engine, "apple_neural_engine_available", lambda: True)
    monkeypatch.setattr(engine.ort, "InferenceSession", create)
    result = engine.open_session(model, session_options(1), False, use_coreml=True)
    assert isinstance(result, Session)
    assert "ModelCacheDirectory" in calls[0][0][1]
    assert calls[1][0] == ("CoreMLExecutionProvider", engine.COREML_OPTIONS)


def test_tuned_speaker_window_retries_on_cpu_if_embedding_is_invalid():
    import numpy as np

    class Session:
        def __init__(self, vector):
            self.vector = vector

        def run(self, *args):
            return [np.array([self.vector], dtype=np.float32)]

    model = engine.Embedder.__new__(engine.Embedder)
    model._remote = None
    model._lock = threading.Lock()
    model._rank, model._input, model.calls = 2, "waveform", 0
    model.session = Session([3, 4])
    model._coreml_windows = {16000: Session([np.nan, 1])}
    model._own, model._keeps_memory = {}, 16000
    model._gpu, model._hand_back = GpuThread(False, "speaker"), None
    result = model.embed(np.zeros(16000, dtype=np.float32))
    np.testing.assert_allclose(result, [0.6, 0.8])
    assert not model.on_coreml and model.calls == 1
