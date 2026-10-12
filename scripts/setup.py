#!/usr/bin/env python3
"""Post-install setup for platform-specific runtime dependencies.

1. **Windows: VC++ Redistributable** — PyTorch's native DLLs (c10.dll,
   torch_cpu.dll, etc.) link against vcruntime140.dll and msvcp140.dll from
   the Microsoft Visual C++ 2015-2022 Redistributable. Fresh Windows installs
   (especially debloated/LTSC-style) don't ship it. We detect and auto-install
   it silently before any `import torch` can fail.

2. **CUDA: cuDNN 8 compat** — Ensures cuDNN 8 libraries are available for
   CTranslate2 (faster-whisper / WhisperX) alongside PyTorch 2.8+'s cuDNN 9.

3. **AMD ROCm torch (opt-in)** — with `OMNIVOICE_TORCH_VARIANT=rocm` set,
   replace the lockfile's CUDA torch build (CPU-only on AMD cards) with the
   ROCm wheel — the same swap the packaged app's bootstrap performs, so a
   source install (`bun run desktop`) on an AMD GPU is not stuck on CPU. Runs
   AFTER `uv sync`, which always restores the locked CUDA build (#1665).
   On native Windows it installs AMD's TheRock wheels for the detected Radeon
   plus CTranslate2's ROCm build (see backend/core/rocm_windows.py).

Run automatically as part of `bun run setup:api` — no user action required.

Cross-platform:
  - Linux:   cuDNN 8 compat (.so.8 libs)
  - Windows: VC++ Redistributable + cuDNN 8 compat (.dll libs)
  - macOS:   skipped (no CUDA)
"""
import os
import sys
import subprocess
import glob

# On Windows, when this script's stdout is a *pipe* (redirected, captured by a
# parent process such as `bun run setup:api`, or CI) rather than an interactive
# console, Python defaults to the locale codepage (cp1252), which can't encode
# the ✓/⚙ status glyphs printed below — the script then dies with
# UnicodeEncodeError *before finishing setup*, taking `bun desktop` down with it.
# It only "works" in an interactive terminal by luck of the console's encoding.
# Force UTF-8 on our own streams so output is identical whether run interactively
# or piped. No-op where the streams already speak UTF-8 (macOS/Linux, modern
# Windows Terminal) or can't be reconfigured.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ── AMD ROCm torch (opt-in) ────────────────────────────────────────────────

# Keep in sync with Electron's runtime-project.ts and
# [tool.uv.constraint-dependencies].
ROCM_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.4"
ROCM_TORCH_PINS = ("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0")


def _rocm_opt_in(environ=os.environ):
    """Return the ROCm wheel index when the user opted in, else None."""
    if environ.get("OMNIVOICE_TORCH_VARIANT", "").strip().lower() != "rocm":
        return None
    return environ.get("OMNIVOICE_TORCH_INDEX") or ROCM_TORCH_INDEX


def _installed_torch_is_rocm():
    try:
        import torch  # noqa: WPS433 — deliberately lazy; torch is heavy
    except Exception:
        return False
    return bool(getattr(torch.version, "hip", None))


def rocm_torch_reinstall_cmd(index_url, python=None):
    """`uv pip install` argv targeting THIS venv (not whatever uv guesses)."""
    return [
        "uv", "pip", "install", "--reinstall",
        "--python", python or sys.executable,
        *ROCM_TORCH_PINS,
        "--index-url", index_url,
    ]


def _ensure_rocm_torch():
    index_url = _rocm_opt_in()
    if index_url is None:
        return
    if _installed_torch_is_rocm():
        print("✓ ROCm torch already installed")
        return
    print(f"⚙ OMNIVOICE_TORCH_VARIANT=rocm — swapping torch to the ROCm wheel ({index_url})")
    try:
        subprocess.check_call(rocm_torch_reinstall_cmd(index_url))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"⚠ ROCm torch install failed ({exc}); keeping the default torch build")
        return
    print("✓ ROCm torch installed")


# ── AMD ROCm on native Windows (opt-in) ───────────────────────────────────
# Same opt-in as Linux, different source: AMD's TheRock index and exact pins,
# the GPU's `[device-gfxNNNN]` kernel extra, and CTranslate2's ROCm wheel so
# faster-whisper/WhisperX run on the GPU too. The plan (pins, index, gfx
# detection, hashes) lives in backend/core/rocm_windows.py, shared with the app.

def _rocm_windows_plan():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
    from core import rocm_windows
    return rocm_windows


def windows_rocm_torch_cmd(index_url, requirements, python=None):
    """`uv pip install` argv for AMD's Windows torch stack into THIS venv.

    AMD's index is searched first and joins PyPI rather than replacing it.
    uv's default first-index strategy then takes every package AMD publishes
    (torch, rocm, rocm-sdk-*) only from AMD — a same-named PyPI project can
    never win — while anything else still resolves from PyPI.
    """
    return [
        "uv", "pip", "install",
        "--python", python or sys.executable,
        *requirements,
        "--extra-index-url", index_url,
    ]


def _installed_torch_version():
    try:
        from importlib.metadata import version
        return version("torch")
    except Exception:
        return None


def _site_packages():
    import sysconfig
    return sysconfig.get_paths()["purelib"]


def _fetch_ct2_rocm_wheel(plan, cache_dir):
    """Download (once, hash-verified) CTranslate2's ROCm zip; return the wheel path."""
    import hashlib
    import urllib.request
    import zipfile

    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, os.path.basename(plan.CT2_ROCM_ZIP_URL))

    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    if not os.path.isfile(archive) or _sha256(archive) != plan.CT2_ROCM_ZIP_SHA256:
        print(f"⚙ Downloading CTranslate2 {plan.CT2_ROCM_VERSION} ROCm wheels (~140 MB)...")
        partial = archive + ".part"
        urllib.request.urlretrieve(plan.CT2_ROCM_ZIP_URL, partial)
        if _sha256(partial) != plan.CT2_ROCM_ZIP_SHA256:
            os.remove(partial)
            raise RuntimeError("CTranslate2 ROCm download failed its sha256 check")
        os.replace(partial, archive)
    member = plan.ct2_rocm_wheel_member()
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise RuntimeError(f"no CTranslate2 ROCm wheel for this Python ({member})")
        bundle.extract(member, cache_dir)
    return os.path.join(cache_dir, *member.split("/"))


def _ensure_rocm_windows(environ=os.environ):
    if environ.get("OMNIVOICE_TORCH_VARIANT", "").strip().lower() != "rocm":
        return
    plan = _rocm_windows_plan()
    gfx = plan.windows_rocm_gfx(environ)
    if gfx is None:
        print(
            "⚠ OMNIVOICE_TORCH_VARIANT=rocm, but no supported Radeon GPU was recognised. "
            "Set OMNIVOICE_ROCM_GFX to your card's target (e.g. gfx1100 for an RX 7900 XT) "
            "and run `bun run setup:api` again; keeping the CPU build."
        )
        return
    index_url = environ.get("OMNIVOICE_TORCH_INDEX") or plan.THEROCK_STABLE_INDEX_URL
    if _installed_torch_version() == plan.WINDOWS_ROCM_TORCH:
        print(f"✓ ROCm torch {plan.WINDOWS_ROCM_TORCH} already installed")
    else:
        print(f"⚙ OMNIVOICE_TORCH_VARIANT=rocm — installing AMD's ROCm torch for {gfx} ({index_url})")
        try:
            subprocess.check_call(
                windows_rocm_torch_cmd(index_url, plan.windows_rocm_torch_requirements(gfx))
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"⚠ ROCm torch install failed ({exc}); keeping the CPU build")
            return
        print("✓ ROCm torch installed")
    if plan.ctranslate2_is_rocm_build(_site_packages()):
        print("✓ CTranslate2 ROCm build already installed")
        return
    try:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "rocm-wheels")
        wheel = _fetch_ct2_rocm_wheel(plan, cache_dir)
        subprocess.check_call([
            "uv", "pip", "install", "--python", sys.executable,
            "--no-deps", "--reinstall", wheel,
        ])
    except Exception as exc:  # noqa: BLE001 — ASR still works, on the CPU
        print(f"⚠ CTranslate2 ROCm install failed ({exc}); faster-whisper stays on the CPU")
        return
    print(f"✓ CTranslate2 {plan.CT2_ROCM_VERSION} ROCm build installed")


# ── Windows: VC++ Redistributable ─────────────────────────────────────────

def _ensure_vcredist_windows():
    """Check for and install the VC++ 2015-2022 Redistributable on Windows.

    PyTorch's native libraries (c10.dll, torch_cpu.dll, etc.) are built with
    MSVC and dynamically link against vcruntime140.dll + msvcp140.dll.  These
    ship with Visual Studio / Build Tools but are NOT part of Windows itself.
    On a fresh or debloated install the very first `import torch` crashes with:

        OSError: [WinError 126] The specified module could not be found.
        Error loading ...\\torch\\lib\\c10.dll or one of its dependencies.

    This function silently downloads and installs the official x64 redist
    package from Microsoft if the runtime DLLs are missing.
    """
    if sys.platform != "win32":
        return

    # Check if vcruntime140.dll is already loadable
    import ctypes
    try:
        ctypes.WinDLL("vcruntime140.dll")
        print("✓ VC++ Redistributable: already installed")
        return
    except OSError:
        pass

    print("⚙ VC++ Redistributable not found — installing (required for PyTorch)...")

    import tempfile
    import urllib.request

    vc_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    installer = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")

    try:
        # Download
        print("  Downloading VC++ Redistributable...")
        urllib.request.urlretrieve(vc_url, installer)

        # Silent install (/install /quiet /norestart)
        print("  Installing silently...")
        result = subprocess.run(
            [installer, "/install", "/quiet", "/norestart"],
            timeout=120,
            capture_output=True,
        )

        # Verify it worked
        try:
            ctypes.WinDLL("vcruntime140.dll")
            print("✓ VC++ Redistributable: installed successfully")
        except OSError:
            # Exit code 3010 = success but reboot required
            if result.returncode == 3010:
                print("✓ VC++ Redistributable: installed (reboot recommended)")
            else:
                print(f"⚠ VC++ Redistributable: install may have failed (exit code {result.returncode})")
                print("  Manual install: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except Exception as e:
        print(f"⚠ VC++ Redistributable: auto-install failed: {e}")
        print("  Manual install: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    finally:
        # Clean up installer
        try:
            os.remove(installer)
        except OSError:
            pass


# ── cuDNN 8 compat ────────────────────────────────────────────────────────

def _find_compat_dir():
    """Return the cudnn8_compat target directory, auto-detecting venv layout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    venv_dir = os.path.join(project_root, ".venv")

    if not os.path.isdir(venv_dir):
        return None

    if sys.platform == "win32":
        # Windows: .venv/Lib/site-packages/
        sp = os.path.join(venv_dir, "Lib", "site-packages", "cudnn8_compat")
    else:
        # Linux: .venv/lib/pythonX.Y/site-packages/
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        sp = os.path.join(venv_dir, "lib", pyver, "site-packages", "cudnn8_compat")

    return sp


def _cudnn8_lib_dir(compat_dir):
    """Return the cuDNN lib subdirectory within the compat install."""
    if sys.platform == "win32":
        return os.path.join(compat_dir, "nvidia", "cudnn", "bin")
    return os.path.join(compat_dir, "nvidia", "cudnn", "lib")


def _count_cudnn8_libs(lib_dir):
    """Count cuDNN 8 shared libraries in the given directory."""
    if sys.platform == "win32":
        return len(glob.glob(os.path.join(lib_dir, "cudnn*64_8.dll")))
    return len(glob.glob(os.path.join(lib_dir, "libcudnn*.so.8")))


def main():
    # ── Step 1: Windows VC++ Redistributable ──────────────────────────────
    _ensure_vcredist_windows()

    # macOS — no CUDA, nothing to do
    if sys.platform == "darwin":
        return

    # ── Step 2: opt-in AMD ROCm torch ─────────────────────────────────────
    if sys.platform.startswith("linux"):
        _ensure_rocm_torch()
    elif sys.platform == "win32":
        _ensure_rocm_windows()

    compat_dir = _find_compat_dir()
    if compat_dir is None:
        return

    lib_dir = _cudnn8_lib_dir(compat_dir)

    # Already installed?
    if os.path.isdir(lib_dir):
        n = _count_cudnn8_libs(lib_dir)
        if n >= 5:
            print(f"✓ cuDNN 8 compat: {n} libraries ready")
            return

    # Check if CUDA is available before installing GPU-only libs
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available() and not getattr(torch.version, 'hip', None))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or result.stdout.strip() != "True":
            print("✓ No CUDA — cuDNN 8 compat not needed")
            return
    except Exception:
        print("⚠ CUDA probe failed — skipping optional cuDNN 8 compatibility libraries")
        return

    print("⚙ Installing cuDNN 8 compatibility libraries for CTranslate2...")
    try:
        # `uv venv` doesn't seed pip into the venv, so `sys.executable -m pip`
        # fails with "No module named pip". `uv pip install --python` talks to
        # the interpreter directly without needing pip installed inside it.
        subprocess.run(
            [
                "uv", "pip", "install",
                "--no-deps", "--target", compat_dir,
                "--python", sys.executable,
                "nvidia-cudnn-cu12==8.9.7.29",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        n = _count_cudnn8_libs(lib_dir)
        print(f"✓ cuDNN 8 installed: {n} libraries")
    except subprocess.CalledProcessError as e:
        print(f"⚠ cuDNN 8 install failed (transcription may not work on CUDA):")
        print(f"  {(e.stderr or '')[:300]}")
    except Exception as e:
        print(f"⚠ cuDNN 8 install skipped: {e}")


if __name__ == "__main__":
    main()
