"""ROCm runtime DLLs for native Windows (AMD's TheRock ``rocm-sdk`` wheels).

On Windows the ROCm torch wheels carry the HIP runtime and math libraries as
pip packages (``_rocm_sdk_core/bin``, ``_rocm_sdk_libraries*/bin``) and
``import torch`` registers those folders with ``os.add_dll_directory``. Any
other HIP-linked extension is stranded until then: the ROCm build of
CTranslate2 links ``amdhip64_7.dll`` and ``hipblas.dll`` by name, so a process
that imports it before torch (the faster-whisper sidecar, Argos translation)
fails with "Could not find module ... ctranslate2.dll (or one of its
dependencies)". Worse, when the DLLs are on ``PATH`` from another install, the
loader can pair the driver's own ``amdhip64_7.dll`` from System32 with the
wheel's libraries and kernels fail with ``hipErrorInvalidKernelFile``.

Registering the wheel folders first makes every HIP extension resolve the same
runtime torch uses, whatever the import order. A no-op off Windows and when no
ROCm wheel is installed.
"""

from __future__ import annotations

import glob
import os
import sys

# Host libraries only. `_rocm_sdk_devel` (compilers, headers) is not a runtime
# dependency of anything and stays out of the DLL search path.
_RUNTIME_PACKAGES = ("_rocm_sdk_core", "_rocm_sdk_libraries*")

# Directories already registered, and the handles keeping them registered: an
# `os.add_dll_directory` entry is removed when its handle is garbage-collected.
_registered: dict[str, object] = {}


def rocm_sdk_bin_dirs(search_path: list[str] | None = None) -> list[str]:
    """The ``bin`` folders of the ROCm runtime wheels importable from ``search_path``."""
    found: list[str] = []
    for entry in search_path if search_path is not None else sys.path:
        if not entry or not os.path.isdir(entry):
            continue
        for package in _RUNTIME_PACKAGES:
            for path in sorted(glob.glob(os.path.join(entry, package, "bin"))):
                if os.path.isdir(path) and path not in found:
                    found.append(path)
    return found


def register_rocm_dll_dirs(search_path: list[str] | None = None) -> list[str]:
    """Add the ROCm runtime ``bin`` folders to this process's DLL search path.

    Returns the folders newly registered by this call. Idempotent.
    """
    if sys.platform != "win32":
        return []
    added: list[str] = []
    for path in rocm_sdk_bin_dirs(search_path):
        if path in _registered:
            continue
        _registered[path] = os.add_dll_directory(path)
        added.append(path)
    return added
