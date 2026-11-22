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
    assert set(install.describe(False, False)) == {"install", "arch", "cuda", "on_gpu"}
