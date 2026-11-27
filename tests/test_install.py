from __future__ import annotations

from ground_control import install


def test_the_install_kind(tmp_path):
    mark = tmp_path / ".dockerenv"
    assert install.kind(True, (mark,)) == "ha_app"
    assert install.kind(False, (mark,)) == "python"
    mark.touch()
    assert install.kind(False, (mark,)) == "docker"


def test_the_architecture_as_the_images_name_it():
    assert install.arch("x86_64") == "amd64"
    assert install.arch("AMD64") == "amd64"
    assert install.arch("aarch64") == "aarch64"
    assert install.arch("arm64") == "aarch64"
    assert install.arch("armv7l") == "armv7l"


def test_the_cpu_build_has_no_cuda_version():
    assert install.cuda_version() == ""
    assert set(install.describe(False, False)) == {"install", "arch", "cuda", "on_gpu", "gpu"}
    # What the GPU is used through, for a kiosk's Runs on line.
    assert install.describe(False, False)["gpu"] == ""
    assert install.describe(False, True, "vulkan")["gpu"] == "vulkan"


def test_the_app_page_follows_the_home_assistant_version():
    from ground_control.server import app_path

    # Home Assistant 2026.2 moved an app's page from /hassio/ingress to /app (Kiosk Satellite #1563).
    assert app_path("abc_gc", "2026.11.2") == "/app/abc_gc"
    assert app_path("abc_gc", "2026.2.0") == "/app/abc_gc"
    assert app_path("abc_gc", "2026.1.3") == "/hassio/ingress/abc_gc"
    assert app_path("abc_gc", "2025.12.4") == "/hassio/ingress/abc_gc"
    assert app_path("abc_gc", None) == "/app/abc_gc"
    assert app_path("abc_gc", "dev") == "/app/abc_gc"
