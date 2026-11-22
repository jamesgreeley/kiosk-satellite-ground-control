"""How the models are run: quiet threads between runs, and one thread for a GPU."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

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
