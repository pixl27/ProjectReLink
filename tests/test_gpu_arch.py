"""GPU architecture classification in one place (gpu/vk/gpu_arch.h, wired into gpu/vk/device_policy.h).

Architecture-specific workarounds and tuning need one classifier for AMD (GCN, Vega, RDNA 1-4), Intel (Gen12, Alchemist,
Meteor/Arrow Lake, Lunar Lake, Battlemage, Panther Lake) and NVIDIA (Pascal-Blackwell) instead of per-site PCI ID lists
or ad-hoc string checks. This compiles gpu_arch.h and checks PCI IDs, name fallbacks, cross-vendor isolation, the RADV
GCN hint, is_rdna1..4 parity, and that device_policy.h exposes DeviceFacts::arch() and logs it on the facts line.

python -I -B tests/test_gpu_arch.py [--cxx g++]
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

FIXTURE = r"""
#include <cstdio>
#include <cstring>
#include "gpu_arch.h"

using namespace mkwvk;

struct Case {
    const char* label;
    uint32_t vendor, device;
    const char* name;
    bool radv_gcn;
    GpuArch want;
};

int main() {
    const Case cases[] = {
        // AMD RDNA 4
        {"RX 9070 XT", 0x1002, 0x7550, "AMD Radeon RX 9070 XT", false, GpuArch::AmdRdna4},
        {"Radeon AI PRO R9700", 0x1002, 0x7551, nullptr, false, GpuArch::AmdRdna4},
        {"RX 9060 XT", 0x1002, 0x7590, "AMD Device 7590", false, GpuArch::AmdRdna4},
        {"RX 9080 by name", 0x1002, 0x0000, "AMD Radeon RX 9080", false, GpuArch::AmdRdna4},
        {"GFX12 by name", 0x1002, 0x0000, "AMD Radeon GFX1201", false, GpuArch::AmdRdna4},
        // AMD RDNA 3 / 3.5
        {"RX 7900 XTX", 0x1002, 0x744C, "AMD Radeon RX 7900 XTX", false, GpuArch::AmdRdna3},
        {"RX 7700S", 0x1002, 0x7480, "AMD Radeon RX 7700S", false, GpuArch::AmdRdna3},
        {"Radeon 860M", 0x1002, 0x1114, "AMD Radeon(TM) 860M Graphics", false, GpuArch::AmdRdna3},
        {"Radeon 8060S", 0x1002, 0x1586, "AMD Radeon 8060S Graphics", false, GpuArch::AmdRdna3},
        {"Radeon 780M by name", 0x1002, 0x0000, "AMD Radeon 780M", false, GpuArch::AmdRdna3},
        // AMD RDNA 2 (Navi 21-24 + APUs)
        {"RX 6800 XT (Navi 21)", 0x1002, 0x73BF, "AMD Radeon RX 6800 XT", false, GpuArch::AmdRdna2},
        {"RX 6950 XT (Navi 21)", 0x1002, 0x73A5, "AMD Radeon RX 6950 XT", false, GpuArch::AmdRdna2},
        {"RX 6700 XT (Navi 22)", 0x1002, 0x73DF, "AMD Radeon RX 6700 XT", false, GpuArch::AmdRdna2},
        {"RX 6600 (Navi 23)", 0x1002, 0x73FF, "AMD Radeon RX 6600", false, GpuArch::AmdRdna2},
        {"RX 6650 XT (Navi 23)", 0x1002, 0x73EF, "AMD Radeon RX 6650 XT", false, GpuArch::AmdRdna2},
        {"RX 6500 XT (Navi 24)", 0x1002, 0x743F, "AMD Radeon RX 6500 XT", false, GpuArch::AmdRdna2},
        {"Steam Deck (Van Gogh)", 0x1002, 0x163F, "AMD Radeon Graphics (RADV VANGOGH)", false, GpuArch::AmdRdna2},
        {"Steam Deck OLED (Sephiroth)", 0x1002, 0x1435, "AMD Custom GPU 0405", false, GpuArch::AmdRdna2},
        {"Raphael (Ryzen 7900X)", 0x1002, 0x164E, "AMD Radeon(TM) Graphics", false, GpuArch::AmdRdna2},
        {"Granite Ridge", 0x1002, 0x13C0, "AMD Radeon(TM) Graphics", false, GpuArch::AmdRdna2},
        {"Rembrandt (680M)", 0x1002, 0x1681, "AMD Radeon(TM) Graphics", false, GpuArch::AmdRdna2},
        {"RDNA 2 by name", 0x1002, 0x0000, "AMD Radeon RX 6750 GRE", false, GpuArch::AmdRdna2},
        // AMD RDNA 1
        {"RX 5700 (Navi 10)", 0x1002, 0x731F, "AMD Radeon RX 5700", false, GpuArch::AmdRdna1},
        {"RX 5500 XT (Navi 14)", 0x1002, 0x7340, "AMD Radeon RX 5500 XT", false, GpuArch::AmdRdna1},
        {"RDNA 1 by name", 0x1002, 0x0000, "AMD Radeon RX 5600 XT", false, GpuArch::AmdRdna1},
        // AMD Vega & GCN
        {"Cezanne APU (Vega)", 0x1002, 0x1638, "AMD Radeon(TM) Graphics", false, GpuArch::AmdVega},
        {"Barcelo APU (Vega)", 0x1002, 0x15E7, "AMD Radeon(TM) Graphics", false, GpuArch::AmdVega},
        {"Vega 64", 0x1002, 0x687F, "AMD Radeon RX Vega 64", false, GpuArch::AmdVega},
        {"Vega 12 (by ID, null name)", 0x1002, 0x69AF, nullptr, false, GpuArch::AmdVega},
        {"RX 580 (Polaris GCN, Windows)", 0x1002, 0x67DF, "AMD Radeon RX 580", false, GpuArch::AmdGcn},
        {"RX 580 2048SP (Polaris GCN, Windows)", 0x1002, 0x6FDF, "AMD Radeon RX 580 2048SP", false, GpuArch::AmdGcn},
        {"RX 640 (Polaris 12 GCN, must not match RDNA2 RX 6xxx)", 0x1002, 0x6987, "AMD Radeon RX 640", false, GpuArch::AmdGcn},
        {"RX 640 (Polaris 12 0x699F, must not match RDNA2)", 0x1002, 0x699F, "AMD Radeon RX 640", false, GpuArch::AmdGcn},
        {"RX 640 by name alone (3-digit must not match RDNA2)", 0x1002, 0x0000, "AMD Radeon RX 640", false, GpuArch::Unknown},
        {"Vega M GH (Polaris 22 GFX8, not Vega GFX9)", 0x1002, 0x694C, "Radeon RX Vega M GH Graphics", false, GpuArch::AmdGcn},
        {"HD 7970 (Tahiti GCN 1, Windows)", 0x1002, 0x6798, "AMD Radeon HD 7970", false, GpuArch::AmdGcn},
        {"HD 7870 (Pitcairn GCN 1, Windows)", 0x1002, 0x6818, "AMD Radeon HD 7870", false, GpuArch::AmdGcn},
        {"HD 7770 (Cape Verde GCN 1, Windows)", 0x1002, 0x683D, "AMD Radeon HD 7770", false, GpuArch::AmdGcn},
        {"R7 260X (Bonaire GCN 2, Windows)", 0x1002, 0x6650, "AMD Radeon R7 260X", false, GpuArch::AmdGcn},
        {"FirePro W9100 (Hawaii GCN 2, Windows)", 0x1002, 0x67A0, "AMD FirePro W9100", false, GpuArch::AmdGcn},
        {"R9 290X (Hawaii GCN 2, Windows)", 0x1002, 0x67B0, "AMD Radeon R9 290X", false, GpuArch::AmdGcn},
        {"R9 380 (Tonga GCN 3, Windows)", 0x1002, 0x6939, "AMD Radeon R9 380", false, GpuArch::AmdGcn},
        {"Uncatalogued RADV GCN APU (Kaveri)", 0x1002, 0x130F, "AMD KAVERI (RADV)", true, GpuArch::AmdGcn},
        {"Uncatalogued GCN APU without RADV hint", 0x1002, 0x130F, "AMD Radeon(TM) R7 Graphics", false, GpuArch::Unknown},
        {"TeraScale HD 6870 (not GCN)", 0x1002, 0x6738, "AMD Radeon HD 6870", false, GpuArch::Unknown},
        // Intel
        {"Arc B580 (Battlemage)", 0x8086, 0xE20B, "Intel(R) Arc(TM) B580 Graphics", false, GpuArch::IntelBattlemage},
        {"Arc B570 (Battlemage)", 0x8086, 0xE20C, "Intel(R) Arc(TM) B570 Graphics", false, GpuArch::IntelBattlemage},
        {"Panther Lake", 0x8086, 0xB080, "Intel(R) Graphics", false, GpuArch::IntelPanther},
        {"Lunar Lake (Arc 140V)", 0x8086, 0x64A0, "Intel(R) Arc(TM) 140V GPU", false, GpuArch::IntelLunar},
        {"Meteor Lake", 0x8086, 0x7D55, "Intel(R) Arc(TM) Graphics", false, GpuArch::IntelMeteorArrow},
        {"Arc A750 (Alchemist)", 0x8086, 0x56A1, "Intel(R) Arc(TM) A750 Graphics", false, GpuArch::IntelAlchemist},
        {"Iris Xe (Gen12)", 0x8086, 0x9A49, "Mesa Intel(R) Xe Graphics (TGL GT2)", false, GpuArch::IntelGen12},
        {"Iris Xe MAX (DG1 Gen12)", 0x8086, 0x4905, "Intel(R) Iris(R) Xe MAX Graphics", false, GpuArch::IntelGen12},
        {"Intel DG1 (0x4900)", 0x8086, 0x4900, nullptr, false, GpuArch::IntelGen12},
        // NVIDIA
        {"RTX 5080 (Blackwell)", 0x10DE, 0x2C02, "NVIDIA GeForce RTX 5080", false, GpuArch::NvidiaBlackwell},
        {"RTX 5070 (Blackwell 0x2F04)", 0x10DE, 0x2F04, nullptr, false, GpuArch::NvidiaBlackwell},
        {"RTX 4090 (Ada)", 0x10DE, 0x2684, "NVIDIA GeForce RTX 4090", false, GpuArch::NvidiaAda},
        {"RTX 4070 Laptop (Ada)", 0x10DE, 0x2860, "NVIDIA GeForce RTX 4070 Laptop GPU", false, GpuArch::NvidiaAda},
        {"RTX 3090 (Ampere)", 0x10DE, 0x2204, "NVIDIA GeForce RTX 3090", false, GpuArch::NvidiaAmpere},
        {"RTX 3060 Laptop (Ampere)", 0x10DE, 0x2520, "NVIDIA GeForce RTX 3060 Laptop GPU", false, GpuArch::NvidiaAmpere},
        {"H100 (Hopper 0x23xx, not Ampere)", 0x10DE, 0x2330, "NVIDIA H100 80GB HBM3", false, GpuArch::Unknown},
        {"GTX 1650 (Turing)", 0x10DE, 0x1F82, "NVIDIA GeForce GTX 1650", false, GpuArch::NvidiaTuring},
        {"GTX 1060 (Pascal)", 0x10DE, 0x1C03, "NVIDIA GeForce GTX 1060", false, GpuArch::NvidiaPascal},
        {"GT 1030 (Pascal GP108 0x1D01)", 0x10DE, 0x1D01, "NVIDIA GeForce GT 1030", false, GpuArch::NvidiaPascal},
        {"MX150 (Pascal GP108 0x1D10)", 0x10DE, 0x1D10, nullptr, false, GpuArch::NvidiaPascal},
        {"TITAN V (Volta 0x1D81, not Pascal)", 0x10DE, 0x1D81, "NVIDIA TITAN V", false, GpuArch::Unknown},
        {"Tesla V100 (Volta 0x1DB4, not Pascal)", 0x10DE, 0x1DB4, "Tesla V100-PCIE-16GB", false, GpuArch::Unknown},
        // Cross-vendor isolation & unknown
        {"NVIDIA with AMD RDNA2 ID", 0x10DE, 0x73DF, "NVIDIA Custom", false, GpuArch::Unknown},
        {"NVIDIA with AMD RX 9000 name", 0x10DE, 0x0000, "NVIDIA RX 9000", false, GpuArch::Unknown},
        {"llvmpipe", 0x10005, 0x0000, "llvmpipe", false, GpuArch::Unknown},
        {"Null name unknown", 0x1002, 0x0000, nullptr, false, GpuArch::Unknown},
    };
    int fails = 0;
    for (const Case& c : cases) {
        const GpuArch got = classify_gpu_arch(c.vendor, c.device, c.name, c.radv_gcn);
        if (got != c.want) {
            std::printf("FAIL %s: got %s, want %s\n", c.label, gpu_arch_name(got), gpu_arch_name(c.want));
            fails++;
        }
        // Check is_rdna1..4 agree with classify_gpu_arch
        if (is_rdna4(c.vendor, c.device, c.name) != (got == GpuArch::AmdRdna4) ||
            is_rdna3(c.vendor, c.device, c.name) != (got == GpuArch::AmdRdna3) ||
            is_rdna2(c.vendor, c.device, c.name) != (got == GpuArch::AmdRdna2) ||
            is_rdna1(c.vendor, c.device, c.name) != (got == GpuArch::AmdRdna1)) {
            std::printf("FAIL %s: is_rdna1..4 disagrees with classify_gpu_arch (%s)\n", c.label, gpu_arch_name(got));
            fails++;
        }
    }
    if (std::strcmp(gpu_arch_name(GpuArch::AmdRdna2), "amd_rdna2") != 0 ||
        std::strcmp(gpu_arch_name(GpuArch::Unknown), "unknown") != 0) {
        std::puts("FAIL: gpu_arch_name mismatch");
        fails++;
    }
    return fails ? 1 : 0;
}
"""


def default_cxx():
    return os.environ.get("CXX") or ("clang++" if shutil.which("clang++") and not shutil.which("g++") else "g++")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default=default_cxx())
    cxx = p.parse_args().cxx

    policy = (VK / "device_policy.h").read_text(encoding="utf-8")
    for need, why in (
        ('#include "gpu_arch.h"', "device_policy.h does not include gpu_arch.h"),
        ("GpuArch arch() const {", "DeviceFacts does not expose arch()"),
        ("const char* arch_name() const {", "DeviceFacts does not expose arch_name()"),
        ("(arch %s)", "vk: policy: facts does not log the classified GPU architecture"),
    ):
        if need not in policy:
            raise SystemExit("FAIL: " + why)

    with tempfile.TemporaryDirectory(prefix="gpu-arch-") as tmp:
        src, exe = Path(tmp) / "arch.cpp", Path(tmp) / "arch.exe"
        src.write_text(FIXTURE, encoding="utf-8")
        subprocess.run([cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(VK), str(src), "-o", str(exe)],
                       check=True)
        r = subprocess.run([str(exe)], text=True, capture_output=True, timeout=20)
        if r.returncode != 0:
            raise SystemExit("FAIL:\n" + r.stdout + r.stderr)
    print("PASS: gpu_arch.h classifies AMD (GCN/Vega/RDNA 1-4), Intel and NVIDIA GPUs and device_policy.h logs arch")


if __name__ == "__main__":
    main()
