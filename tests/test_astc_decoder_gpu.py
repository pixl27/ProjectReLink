# SPDX-License-Identifier: GPL-3.0-or-later
"""Game-free Vulkan ASTC decoder verification against Hookshot's CPU ASTC reference.

Default mode (used by run_all.py, no GPU or Vulkan loader required):
  Compiles tests/astc_decoder_gpu.cpp against gpu/vk/host_shaders/astc_decoder_comp_spv.h
  and gpu/third_party/suyu/video_core/textures/astc.cpp, validates the embedded SPIR-V
  header, and runs the CPU ASTC reference across all 14 ASTC block sizes.

GPU mode (--gpu or --gpu-device <substr>):
  Dispatches the real compiled ASTC_DECODER_COMP_SPV compute shader on Vulkan across all
  14 ASTC block sizes (4x4..12x12), odd image dimensions, linear and Switch 2 native
  tiling, 1-4 partitions, dual-plane (all 4 secondary channels), void-extent LDR, and
  error blocks, comparing every decoded RGBA8 pixel against the CPU reference
  (0 tolerance on replicated-byte void-extent and error blocks; <=1 LSB on
  interpolated LDR to account for astc.cpp's 255*C/65536 vs Vulkan UNORM8 C/65535 rounding).
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or "g++")
    parser.add_argument("--headers", type=Path, required=True, help="path to Vulkan-Headers include directory")
    parser.add_argument("--gpu", action="store_true", help="dispatch astc_decoder_comp_spv.h on a Vulkan device")
    parser.add_argument("--gpu-device", help="case-insensitive device-name substring for --gpu")
    args = parser.parse_args()
    args.gpu = args.gpu or bool(args.gpu_device)

    fixture_cpp = ROOT / "tests/astc_decoder_gpu.cpp"
    astc_cpu_cpp = ROOT / "gpu/third_party/suyu/video_core/textures/astc.cpp"
    inc_flags = [
        "-I", str(args.headers),
        "-I", str(ROOT / "gpu/vk"),
        "-I", str(ROOT / "gpu/compat"),
        "-I", str(ROOT / "gpu/third_party/suyu"),
        "-I", str(ROOT / "gpu/vk/compat"),
    ]

    with tempfile.TemporaryDirectory(prefix="hookshot-astc-gpu-") as directory:
        tmp = Path(directory)
        astc_obj = tmp / "astc_cpu.o"
        exe = tmp / ("astc_decoder_gpu.exe" if os.name == "nt" else "astc_decoder_gpu")

        # Compile third_party CPU ASTC reference with -DNDEBUG so error-block cases return FillError zeros.
        subprocess.run(
            [args.cxx, "-std=c++20", "-O1", "-DNDEBUG", *inc_flags, "-c", str(astc_cpu_cpp), "-o", str(astc_obj)],
            check=True,
        )

        link_flags = ["-pthread"]
        if os.name == "nt" and "mingw" in args.cxx.lower():
            link_flags += ["-static"]
        libraries = ["-ldl"] if os.name != "nt" else []

        subprocess.run(
            [args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
             *inc_flags, str(fixture_cpp), str(astc_obj), "-o", str(exe), *link_flags, *libraries],
            check=True,
        )

        env = os.environ.copy()
        if os.name == "nt":
            cxx_dir = str(Path(args.cxx).resolve().parent)
            if Path(cxx_dir).is_dir():
                env["PATH"] = cxx_dir + os.pathsep + env.get("PATH", "")

        cmd = [str(exe)]
        if args.gpu:
            cmd.append("--gpu")
            if args.gpu_device:
                cmd += ["--device", args.gpu_device]
        else:
            cmd.append("--cpu-only")

        subprocess.run(cmd, check=True, timeout=60 if args.gpu else 20, env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
