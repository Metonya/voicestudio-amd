"""AMD GPU on native Windows: one-shot probe for the ROCm research.

See docs/amd-windows-rocm.md. Run it with the Python that hosts
VoiceStudio (or any venv with a ROCm torch), on a short speech file:

    .venv\\Scripts\\python scripts\\amd_windows_probe.py path\\to\\speech.wav

It prints a compact report to paste back: visible GPUs, which HIP/OpenMP DLLs
the process loaded, kernel-launch overhead and HIP graph replay, fp16 matmul
throughput, SDPA backends, PyTorch Whisper timing (cold, warm GPU, CPU), and a
CTranslate2 / faster-whisper matrix. Each CTranslate2 variant runs in its own
subprocess, because DLL resolution and HIP device visibility are per process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

# Same default as backend/main.py: MIOpen's exhaustive first-shape search
# otherwise dominates the first convolution of every new input shape.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

DLL_HINTS = ("amdhip64", "hipblas", "rocblas", "miopen", "iomp5", "libomp", "ctranslate2")


def loaded_dlls(hints=DLL_HINTS) -> list[str]:
    try:
        import psutil
    except ImportError:
        return ["(psutil not installed)"]
    paths = set()
    for region in psutil.Process().memory_maps():
        base = os.path.basename(region.path).lower()
        if base.endswith(".dll") and any(h in base for h in hints):
            paths.add(region.path)
    return sorted(paths)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def run_guarded(title, fn, *args):
    section(title)
    try:
        return fn(*args)
    except Exception as exc:  # keep probing the remaining sections
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:400]}")
        return None


def environment(torch):
    print(f"os: {platform.platform()} | python {platform.python_version()}")
    print(f"torch {torch.__version__} | hip {torch.version.hip} | gpu available {torch.cuda.is_available()}")
    for name in ("transformers", "ctranslate2", "faster_whisper"):
        try:
            module = __import__(name)
            print(f"{name} {getattr(module, '__version__', '?')}")
        except Exception as exc:
            print(f"{name}: not importable ({type(exc).__name__})")
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "MIOPEN_FIND_MODE", "HSA_OVERRIDE_GFX_VERSION"):
        print(f"{var}={os.environ.get(var)}")


def gpus(torch) -> list[dict]:
    found = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        found.append({
            "index": index,
            "name": props.name,
            "arch": getattr(props, "gcnArchName", "?"),
            "vram_gb": round(props.total_memory / 2**30, 1),
            "compute_units": props.multi_processor_count,
        })
    for gpu in found:
        print(gpu)
    if len(found) > 1:
        print("NOTE: more than one HIP device is visible (integrated GPU?)")
    return found


def pick_dgpu(found: list[dict]) -> int:
    """The discrete card: the largest VRAM wins over an integrated GPU."""
    return max(found, key=lambda gpu: gpu["vram_gb"])["index"] if found else 0


def sync_time(torch, fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def launch_overhead(torch, device):
    x = torch.zeros(16, device=device)
    for _ in range(200):
        x.add_(1)
    ops = 2000
    seconds = sync_time(torch, lambda: [x.add_(1) for _ in range(ops)])
    print(f"eager launch: {seconds / ops * 1e6:.1f} us/op (small autoregressive models are bound by this)")
    try:
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.add_(1)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                x.add_(1)
        graph.replay()
        replays = 20
        seconds = sync_time(torch, lambda: [graph.replay() for _ in range(replays)])
        print(f"HIP graph replay: {seconds / (replays * 100) * 1e6:.2f} us/op (capture OK)")
    except Exception as exc:
        print(f"HIP graph capture FAILED: {type(exc).__name__}: {str(exc)[:200]}")


def matmul_throughput(torch, device):
    n = 8192
    a = torch.randn(n, n, device=device, dtype=torch.float16)
    b = torch.randn(n, n, device=device, dtype=torch.float16)
    for _ in range(3):
        a @ b
    reps = 10
    seconds = sync_time(torch, lambda: [a @ b for _ in range(reps)])
    print(f"fp16 matmul {n}x{n}: {2 * n**3 * reps / seconds / 1e12:.1f} TFLOPS")


def sdpa_backends(torch, device):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q, k, v = (torch.randn(1, 16, 2048, 64, device=device, dtype=torch.float16) for _ in range(3))
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v)
                seconds = sync_time(torch, lambda: [F.scaled_dot_product_attention(q, k, v) for _ in range(20)])
            print(f"{backend.name}: OK {seconds / 20 * 1e3:.2f} ms")
        except Exception as exc:
            print(f"{backend.name}: unavailable ({type(exc).__name__}: {str(exc)[:120]})")


def load_audio_16k(path: str):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(samples.mean(axis=1))
    if rate != 16000:
        mono = torchaudio.functional.resample(mono, rate, 16000)
    audio = mono.numpy().astype(np.float32)
    print(f"audio: {path} | {len(audio) / 16000:.2f} s")
    return audio


def whisper_timing(torch, model_id, audio, device, dtype, runs):
    from transformers import pipeline

    start = time.perf_counter()
    pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype, device=device)
    load = time.perf_counter() - start
    timings, text = [], ""
    for _ in range(runs):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        text = pipe({"raw": audio, "sampling_rate": 16000})["text"]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    duration = len(audio) / 16000
    warm = statistics.median(timings[1:]) if len(timings) > 1 else timings[0]
    print(
        f"{model_id} on {device} ({str(dtype).replace('torch.', '')}): load {load:.1f}s | "
        f"first {timings[0]:.2f}s | warm {warm:.2f}s | warm RTF {warm / duration:.3f}"
    )
    print(f"  text: {text.strip()[:100]}")
    del pipe


CT2_CHILD = r"""
import json, os, time
cfg = json.loads(os.environ["PROBE_CT2"])
res = {"variant": cfg["name"]}
try:
    if cfg["torch_first"]:
        import torch
        res["torch_gpu"] = torch.cuda.is_available()
    import numpy as np
    import ctranslate2
    res["ct2"] = ctranslate2.__version__
    res["gpu_count"] = ctranslate2.get_cuda_device_count()
    if cfg["device"] == "cuda":
        res["compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
    from faster_whisper import WhisperModel
    audio = np.load(cfg["audio"])
    start = time.perf_counter()
    model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
    res["load_s"] = round(time.perf_counter() - start, 2)
    for key in ("first_s", "warm_s"):
        start = time.perf_counter()
        segments, _ = model.transcribe(audio, beam_size=1)
        text = " ".join(s.text.strip() for s in segments)
        res[key] = round(time.perf_counter() - start, 2)
    res["text"] = text[:80]
    res["ok"] = True
except BaseException as exc:
    res["ok"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"[:400]
try:
    import psutil
    hints = ("amdhip64", "hipblas", "iomp5", "libomp", "ctranslate2")
    res["dlls"] = sorted({m.path for m in psutil.Process().memory_maps()
                          if any(h in os.path.basename(m.path).lower() for h in hints)})
except Exception:
    pass
print("PROBE_JSON " + json.dumps(res), flush=True)
"""


def ct2_matrix(audio, model, dgpu):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.npy")
        np.save(audio_path, audio)
        variants = [
            ("gpu", False, None, "cuda", "float16"),
            ("gpu, only dGPU visible", False, str(dgpu), "cuda", "float16"),
            ("gpu, torch imported first", True, None, "cuda", "float16"),
            ("gpu, torch first, only dGPU", True, str(dgpu), "cuda", "float16"),
            ("cpu int8", False, None, "cpu", "int8"),
        ]
        for name, torch_first, visible, device, compute_type in variants:
            env = dict(os.environ)
            env["PROBE_CT2"] = json.dumps({
                "name": name, "torch_first": torch_first, "audio": audio_path,
                "model": model, "device": device, "compute_type": compute_type,
            })
            if visible is not None:
                env["HIP_VISIBLE_DEVICES"] = visible
            proc = subprocess.run(
                [sys.executable, "-c", CT2_CHILD], env=env, capture_output=True, text=True, timeout=900, check=False,
            )
            line = next((l for l in proc.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
            if line is None:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
                print(f"[{name}] process died (exit {proc.returncode}): " + " | ".join(tail)[:600])
                continue
            res = json.loads(line[len("PROBE_JSON "):])
            dlls = res.pop("dlls", [])
            print(f"[{name}] " + json.dumps(res, ensure_ascii=False))
            for dll in dlls:
                print(f"    loaded: {dll}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", help="short speech file (wav/flac); ASR sections need it")
    parser.add_argument("--hf-model", default="openai/whisper-large-v3-turbo", help="PyTorch Whisper model (app default)")
    parser.add_argument("--ct2-model", default="tiny", help="faster-whisper model for the CTranslate2 matrix")
    parser.add_argument("--skip-cpu", action="store_true", help="skip the CPU Whisper baseline")
    parser.add_argument("--skip-ct2", action="store_true", help="skip the CTranslate2 matrix")
    args = parser.parse_args()

    import torch

    run_guarded("environment", environment, torch)
    found = run_guarded("HIP devices", gpus, torch) or []
    dgpu = pick_dgpu(found)
    device = f"cuda:{dgpu}"
    if torch.cuda.is_available():
        run_guarded(f"launch overhead ({device})", launch_overhead, torch, device)
        run_guarded(f"matmul ({device})", matmul_throughput, torch, device)
        run_guarded(f"SDPA backends ({device})", sdpa_backends, torch, device)
        section("DLLs loaded by this process")
        for dll in loaded_dlls():
            print(dll)
    if not args.audio:
        print("\n(no audio file given: skipping ASR sections)")
        return 0
    audio = run_guarded("audio", load_audio_16k, args.audio)
    if audio is None:
        return 1
    if torch.cuda.is_available():
        run_guarded("PyTorch Whisper GPU fp16", whisper_timing, torch, args.hf_model, audio, device, torch.float16, 4)
    if not args.skip_cpu:
        run_guarded("PyTorch Whisper CPU fp32", whisper_timing, torch, args.hf_model, audio, "cpu", torch.float32, 2)
    if not args.skip_ct2:
        run_guarded("CTranslate2 / faster-whisper matrix", ct2_matrix, audio, args.ct2_model, dgpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
