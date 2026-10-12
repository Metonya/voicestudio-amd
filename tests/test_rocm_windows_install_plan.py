"""`OMNIVOICE_TORCH_VARIANT=rocm` on native Windows installs AMD's own stack.

Validated by hand on an RX 7900 XT before it was automated: TheRock's stable
index serves Python 3.11 wheels, an *unpinned* `torch` is a silent no-op over
the lockfile's CPU build, each GPU needs its `[device-gfxNNNN]` kernel extra,
and CTranslate2's ROCm wheel (GitHub release zip) moves faster-whisper onto the
GPU. These tests pin that plan.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import zipfile

import pytest

from core import rocm_windows as plan
from core.gpu_inventory import HostGPU

_SETUP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "setup.py")


def _load_setup():
    spec = importlib.util.spec_from_file_location("vs_setup_win", _SETUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("name", "gfx"),
    [
        ("AMD Radeon RX 7900 XT", "gfx1100"),
        ("AMD Radeon RX 7900 XTX", "gfx1100"),
        ("AMD Radeon RX 7900 GRE", "gfx1100"),
        ("AMD Radeon PRO W7900", "gfx1100"),
        ("AMD Radeon RX 7800 XT", "gfx1101"),
        ("AMD Radeon RX 7700 XT", "gfx1101"),
        ("AMD Radeon RX 7600", "gfx1102"),
        ("AMD Radeon RX 9070 XT", "gfx1201"),
        ("AMD Radeon RX 9060 XT", "gfx1200"),
        ("AMD Radeon RX 6800 XT", "gfx1030"),
        ("AMD Radeon RX 6950 XT", "gfx1030"),
        ("AMD Radeon RX 6700 XT", "gfx1031"),
        ("AMD Radeon RX 6600", "gfx1032"),
        ("AMD Radeon(TM) Graphics", None),
        ("NVIDIA GeForce RTX 4090", None),
    ],
)
def test_radeon_names_map_to_their_kernel_target(name, gfx):
    assert plan.gfx_for_gpu_name(name) == gfx


def test_discrete_card_wins_over_the_integrated_gpu():
    gpus = (
        HostGPU(vendor="amd", name="AMD Radeon(TM) Graphics", vram_gb=0.5),
        HostGPU(vendor="amd", name="AMD Radeon RX 7900 XT", vram_gb=20.0, discrete=True),
    )
    assert plan.windows_rocm_gfx({}, gpus) == "gfx1100"
    assert plan.windows_rocm_gfx({}, gpus[:1]) is None


def test_override_wins_and_is_validated():
    assert plan.windows_rocm_gfx({"OMNIVOICE_ROCM_GFX": "GFX1101"}, ()) == "gfx1101"
    assert plan.windows_rocm_gfx({"OMNIVOICE_ROCM_GFX": "rx7900; rm -rf"}, ()) is None


def test_torch_requirements_are_exact_pins_with_the_device_extra():
    reqs = plan.windows_rocm_torch_requirements("gfx1100")
    assert reqs == [
        f"torch[device-gfx1100]=={plan.WINDOWS_ROCM_TORCH}",
        f"torchvision[device-gfx1100]=={plan.WINDOWS_ROCM_TORCHVISION}",
        f"torchaudio=={plan.WINDOWS_ROCM_TORCHAUDIO}",
    ]
    assert all("==" in r and "+rocm" in r for r in reqs)


def test_install_cmd_targets_this_venv_and_searches_amd_first():
    setup = _load_setup()
    cmd = setup.windows_rocm_torch_cmd("https://amd/", ["torch==1"], python="C:/v/python.exe")
    assert cmd == [
        "uv", "pip", "install", "--python", "C:/v/python.exe", "torch==1",
        "--extra-index-url", "https://amd/",
    ]
    # first-index (default) strategy: an AMD package can never come from PyPI.
    assert "--index-strategy" not in cmd


def test_ct2_wheel_member_matches_the_interpreter():
    assert plan.ct2_rocm_wheel_member((3, 11)) == "temp-windows/ctranslate2-4.8.2-cp311-cp311-win_amd64.whl"


def test_ct2_rocm_build_is_recognised_by_its_hip_import(tmp_path):
    pkg = tmp_path / "ctranslate2"
    pkg.mkdir()
    (pkg / "ctranslate2.dll").write_bytes(b"MZ...cublas64_12.dll...")
    assert plan.ctranslate2_is_rocm_build(str(tmp_path)) is False
    (pkg / "ctranslate2.dll").write_bytes(b"MZ...amdhip64_7.dll...hipblas.dll")
    assert plan.ctranslate2_is_rocm_build(str(tmp_path)) is True
    assert plan.ctranslate2_is_rocm_build(str(tmp_path / "missing")) is False


class _Plan:
    """The real plan with the download pointed at a local zip."""

    def __init__(self, archive_bytes, member):
        self.CT2_ROCM_ZIP_URL = "https://example.invalid/rocm-python-wheels-Windows.zip"
        self.CT2_ROCM_ZIP_SHA256 = hashlib.sha256(archive_bytes).hexdigest()
        self.CT2_ROCM_VERSION = "4.8.2"
        self._member = member

    def ct2_rocm_wheel_member(self):
        return self._member


def _zip_bytes(tmp_path, member):
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, b"wheel")
    return path.read_bytes()


def test_ct2_download_is_hash_verified_and_cached(tmp_path, monkeypatch):
    setup = _load_setup()
    member = "temp-windows/ctranslate2-4.8.2-cp311-cp311-win_amd64.whl"
    data = _zip_bytes(tmp_path, member)
    calls = []

    def fake_retrieve(url, dest):
        calls.append(url)
        with open(dest, "wb") as handle:
            handle.write(data)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    cache = tmp_path / "cache"
    wheel = setup._fetch_ct2_rocm_wheel(_Plan(data, member), str(cache))
    assert open(wheel, "rb").read() == b"wheel"
    setup._fetch_ct2_rocm_wheel(_Plan(data, member), str(cache))
    assert len(calls) == 1  # verified cache is reused


def test_ct2_download_with_a_wrong_hash_is_rejected(tmp_path, monkeypatch):
    setup = _load_setup()
    member = "temp-windows/ctranslate2-4.8.2-cp311-cp311-win_amd64.whl"
    data = _zip_bytes(tmp_path, member)
    tampered = _Plan(data, member)
    tampered.CT2_ROCM_ZIP_SHA256 = "0" * 64

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: open(dest, "wb").write(data))
    with pytest.raises(RuntimeError, match="sha256"):
        setup._fetch_ct2_rocm_wheel(tampered, str(tmp_path / "cache"))
    assert not list((tmp_path / "cache").glob("*.zip"))


def test_windows_step_without_a_recognised_gpu_keeps_the_cpu_build(monkeypatch, capsys):
    setup = _load_setup()
    monkeypatch.setattr(plan, "windows_rocm_gfx", lambda environ: None)
    monkeypatch.setattr(setup.subprocess, "check_call", lambda *a, **k: pytest.fail("installed anyway"))
    setup._ensure_rocm_windows({"OMNIVOICE_TORCH_VARIANT": "rocm"})
    assert "OMNIVOICE_ROCM_GFX" in capsys.readouterr().out


def test_windows_step_is_opt_in(monkeypatch):
    setup = _load_setup()
    monkeypatch.setattr(setup.subprocess, "check_call", lambda *a, **k: pytest.fail("not opted in"))
    setup._ensure_rocm_windows({})
    setup._ensure_rocm_windows({"OMNIVOICE_TORCH_VARIANT": "cpu"})


def test_windows_step_installs_torch_then_ct2(monkeypatch, tmp_path):
    setup = _load_setup()
    monkeypatch.setattr(plan, "windows_rocm_gfx", lambda environ: "gfx1100")
    monkeypatch.setattr(setup, "_installed_torch_version", lambda: "2.8.0+cpu")
    monkeypatch.setattr(setup, "_site_packages", lambda: str(tmp_path))
    monkeypatch.setattr(setup, "_fetch_ct2_rocm_wheel", lambda p, cache: "ct2.whl")
    calls = []
    monkeypatch.setattr(setup.subprocess, "check_call", lambda cmd: calls.append(cmd))
    setup._ensure_rocm_windows({"OMNIVOICE_TORCH_VARIANT": "rocm"})
    assert f"torch[device-gfx1100]=={plan.WINDOWS_ROCM_TORCH}" in calls[0]
    assert plan.THEROCK_STABLE_INDEX_URL in calls[0]
    assert calls[1][-1] == "ct2.whl" and "--no-deps" in calls[1]
    assert sys.executable in calls[1]
