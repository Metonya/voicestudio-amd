"""ROCm runtime DLLs must be on the search path before CTranslate2 loads.

Measured on an RX 7900 XT under native Windows (TheRock torch 2.14, CTranslate2
4.8.2 ROCm wheel): a process that imported ctranslate2 before torch failed with
"Could not find module ... ctranslate2.dll (or one of its dependencies)" — on
the GPU *and* the CPU path — while the same process with torch imported first
transcribed on the GPU. ``import torch`` is what registered the rocm-sdk wheel
folders; ``ensure_ctranslate2_loadable`` now does it itself.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from core import execstack, rocm_windows


@pytest.fixture
def site(tmp_path):
    for package in ("_rocm_sdk_core", "_rocm_sdk_libraries", "_rocm_sdk_libraries_gfx110X", "_rocm_sdk_devel"):
        (tmp_path / package / "bin").mkdir(parents=True)
    (tmp_path / "ctranslate2").mkdir()
    return tmp_path


@pytest.fixture
def windows(monkeypatch):
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda path: added.append(path) or object(), raising=False)
    monkeypatch.setattr(rocm_windows, "_registered", {})
    return added


def test_finds_core_and_library_folders_but_not_the_devel_toolchain(site):
    found = rocm_windows.rocm_sdk_bin_dirs([str(site), "", str(site / "missing")])
    assert found == [
        str(site / "_rocm_sdk_core" / "bin"),
        str(site / "_rocm_sdk_libraries" / "bin"),
        str(site / "_rocm_sdk_libraries_gfx110X" / "bin"),
    ]


def test_registers_each_folder_once(site, windows):
    first = rocm_windows.register_rocm_dll_dirs([str(site)])
    assert first == windows and len(first) == 3
    assert rocm_windows.register_rocm_dll_dirs([str(site)]) == []
    assert len(windows) == 3


def test_noop_off_windows(site, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert rocm_windows.register_rocm_dll_dirs([str(site)]) == []


def test_ctranslate2_loadable_registers_rocm_folders_on_windows(site, windows, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(site)])
    assert execstack.ensure_ctranslate2_loadable()[0] is True
    assert str(site / "_rocm_sdk_core" / "bin") in windows


def test_gpu_count_reads_the_installed_build(monkeypatch):
    monkeypatch.setattr(execstack, "ensure_ctranslate2_loadable", lambda: (True, "ok"))
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 2))
    assert execstack.ctranslate2_gpu_count() == 2


def test_gpu_count_is_zero_when_ctranslate2_cannot_load(monkeypatch):
    monkeypatch.setattr(execstack, "ensure_ctranslate2_loadable", lambda: (False, "blocked"))
    assert execstack.ctranslate2_gpu_count() == 0

    def _broken():
        raise OSError("ctranslate2.dll (or one of its dependencies)")

    monkeypatch.setattr(execstack, "ensure_ctranslate2_loadable", lambda: (True, "ok"))
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=_broken))
    assert execstack.ctranslate2_gpu_count() == 0
