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
import re
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


# ── Install plan for native Windows (opt-in: OMNIVOICE_TORCH_VARIANT=rocm) ──
#
# AMD publishes Windows ROCm torch only on its own TheRock index, never on
# download.pytorch.org. Exact pins: an unpinned `torch` is already "satisfied"
# by the lockfile's CPU build and the installer silently does nothing. Each GPU
# architecture's kernels ship as a separate `[device-gfxNNNN]` extra. Validated
# on an RX 7900 XT (gfx1100), Windows 11 25H2, Adrenalin 26.8.1, Python 3.11.
THEROCK_STABLE_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
WINDOWS_ROCM_TORCH = "2.14.0+rocm10.1.0"
WINDOWS_ROCM_TORCHVISION = "0.29.0a0+rocm10.1.0"
WINDOWS_ROCM_TORCHAUDIO = "2.11.0.3+rocm10.1.0"

# CTranslate2's ROCm build (faster-whisper / WhisperX on the GPU) is published
# only as a zip of wheels on its GitHub release. The zip is pinned by hash.
CT2_ROCM_VERSION = "4.8.2"
CT2_ROCM_ZIP_URL = (
    "https://github.com/OpenNMT/CTranslate2/releases/download/"
    f"v{CT2_ROCM_VERSION}/rocm-python-wheels-Windows.zip"
)
CT2_ROCM_ZIP_SHA256 = "43da4baa5feaee49f77e176277a9647f99c493173c81a0bc60f491cac97532c2"

# Radeon product names → the LLVM target TheRock names its device extras by.
# Integrated "AMD Radeon(TM) Graphics" deliberately matches nothing: it is not
# a target worth a multi-GB install, and HIP must not prefer it.
_GFX_BY_NAME: tuple[tuple[str, str], ...] = (
    (r"\bRX\s*79\d0\b|\bPRO\s*W7[89]00\b", "gfx1100"),
    (r"\bRX\s*7[78]00\b|\bPRO\s*W7700\b", "gfx1101"),
    (r"\bRX\s*76\d0\b|\bPRO\s*W7[56]00\b", "gfx1102"),
    (r"\bRX\s*9070\b|\bAI\s*PRO\s*R9[67]00\b", "gfx1201"),
    (r"\bRX\s*9060\b", "gfx1200"),
    (r"\bRX\s*6[89][05]0\b", "gfx1030"),
    (r"\bRX\s*67[05]0\b", "gfx1031"),
    (r"\bRX\s*66[05]0\b", "gfx1032"),
    (r"\bRX\s*6[45]00\b", "gfx1034"),
)


def gfx_for_gpu_name(name: str) -> str | None:
    """The gfx target of a Radeon product name, or ``None`` when unknown."""
    for pattern, gfx in _GFX_BY_NAME:
        if re.search(pattern, name, re.IGNORECASE):
            return gfx
    return None


def windows_rocm_gfx(environ=None, gpus=None) -> str | None:
    """The gfx target to install kernels for: ``OMNIVOICE_ROCM_GFX`` when set,
    else the AMD card with the most VRAM whose name maps to a target."""
    environ = os.environ if environ is None else environ
    override = environ.get("OMNIVOICE_ROCM_GFX", "").strip().lower()
    if override:
        return override if re.fullmatch(r"gfx[0-9a-f]{3,4}", override) else None
    if gpus is None:
        from core.gpu_inventory import detect_host_gpus

        gpus = detect_host_gpus()
    known = [
        (gpu.vram_gb, gfx)
        for gpu in gpus
        if gpu.vendor == "amd" and (gfx := gfx_for_gpu_name(gpu.name))
    ]
    return max(known)[1] if known else None


def windows_rocm_torch_requirements(gfx: str) -> list[str]:
    extra = f"[device-{gfx}]"
    return [
        f"torch{extra}=={WINDOWS_ROCM_TORCH}",
        f"torchvision{extra}=={WINDOWS_ROCM_TORCHVISION}",
        f"torchaudio=={WINDOWS_ROCM_TORCHAUDIO}",
    ]


def ct2_rocm_wheel_member(version_info=None) -> str:
    """The zip member holding the wheel for this interpreter."""
    major, minor = (version_info or sys.version_info)[:2]
    tag = f"cp{major}{minor}"
    return f"temp-windows/ctranslate2-{CT2_ROCM_VERSION}-{tag}-{tag}-win_amd64.whl"


def ctranslate2_is_rocm_build(site_packages: str) -> bool:
    """Whether the installed CTranslate2 links the HIP runtime."""
    dll = os.path.join(site_packages, "ctranslate2", "ctranslate2.dll")
    try:
        with open(dll, "rb") as handle:
            return b"amdhip64" in handle.read()
    except OSError:
        return False
