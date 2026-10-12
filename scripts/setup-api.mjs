import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { repairSourceNativeLibraries } from "../electron/scripts/native-compat.mjs";
import { selectTorchVariant, torchSyncArgs } from "../electron/scripts/torch-variant.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export async function setupApi({
  env = process.env,
  platform = process.platform,
  arch = process.arch,
  run = spawnSync,
  selectVariant = selectTorchVariant,
  log = console.log,
  repairNativeLibraries = repairSourceNativeLibraries,
} = {}) {
  const variant = await selectVariant({ env, platform, arch });
  // The matched torch stack has no native Windows ARM64 torchaudio wheel.
  const pythonArgs =
    platform === "win32" && arch === "arm64" ? ["--python", "cpython-3.11-windows-x86_64"] : [];
  // On Windows the ROCm opt-in syncs the CPU stack first; scripts/setup.py then
  // swaps in AMD's ROCm wheels, so "unavailable" would be the wrong message.
  const rocm = (env.OMNIVOICE_TORCH_VARIANT || "").trim().toLowerCase() === "rocm";
  const note =
    variant !== "cpu"
      ? ""
      : rocm
        ? " (AMD ROCm build is installed next by scripts/setup.py)"
        : " (GPU acceleration unavailable or disabled; inference will be slower)";
  log(`[setup] PyTorch: ${variant}${note}`);
  for (const args of [
    ["sync", "--locked", ...torchSyncArgs(variant), ...pythonArgs],
    ["run", "--no-sync", "python", "scripts/setup.py"],
  ]) {
    const result = run("uv", args, { cwd: root, env, stdio: "inherit" });
    if (result.error) throw result.error;
    if (result.status !== 0) return result.status ?? 1;
    if (args[0] === "sync") await repairNativeLibraries(root, platform, { env });
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  setupApi()
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error) => {
      console.error(`[setup] ${error.message}`);
      process.exitCode = 1;
    });
}
