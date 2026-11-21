"""A CUDA image runs on the GPU or stops with the reason: it never falls back to the CPU."""

from __future__ import annotations

import onnxruntime as ort
import pytest

from ground_control import engine
from ground_control.engine import CudaUnavailable, Embedder
from ground_control.models import SPEAKER_MODELS, verified_path

from .conftest import MODELS

PASCAL_ON_CUDA_13 = ("[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: CUBLAS failure 8: the "
                     "function requires an architectural feature absent from the device ; GPU=0 ; "
                     "expr=cublasCreate(&cublas_handle_);")


def _light():
    m = SPEAKER_MODELS["light"]
    return verified_path(MODELS, m.file, m.sha256)


def _with_cuda(monkeypatch, fail: str | None = None, lands_on_cpu: bool = False):
    real = ort.InferenceSession

    def session(path, so=None, providers=None, **kw):
        names = [p[0] if isinstance(p, tuple) else p for p in providers or []]
        if "CUDAExecutionProvider" in names and fail:
            raise ort.capi.onnxruntime_pybind11_state.Fail(fail)
        return real(path, so, providers=["CPUExecutionProvider"], **kw)

    monkeypatch.setattr(engine.ort, "InferenceSession", session)
    monkeypatch.setattr(engine.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_a_gpu_the_image_cant_use_stops_it_with_the_reason(monkeypatch):
    _with_cuda(monkeypatch, fail=PASCAL_ON_CUDA_13)
    with pytest.raises(CudaUnavailable) as err:
        Embedder(_light(), use_cuda=True)
    text = str(err.value)
    assert "architecture isn't supported" in text and "CPU image" in text
    assert "cublasCreate" not in text


def test_cuda_landing_on_the_cpu_stops_it_too(monkeypatch):
    _with_cuda(monkeypatch, lands_on_cpu=True)
    with pytest.raises(CudaUnavailable, match="no NVIDIA GPU|would run on the CPU"):
        Embedder(_light(), use_cuda=True)


def test_an_image_without_cuda_stops_when_cuda_is_on(monkeypatch):
    monkeypatch.setattr(engine.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(CudaUnavailable, match="no CUDA support"):
        Embedder(_light(), use_cuda=True)


def test_cuda_off_runs_on_the_cpu():
    assert not Embedder(_light(), use_cuda=False).on_gpu
