"""The mesh fallback keeps its warp in workgroup memory on AMD's Windows driver where compute runs at 64 lanes (GCN: RX 400/500, Vega).

On a GPU without mesh shaders the game's task and mesh programs run as compute shaders (the mesh fallback). NVN wrote them for
32-lane warps. Where the GPU can hold a compute stage to 32 lanes (RDNA, NVIDIA, Intel) it is held there and the stage keeps the
short warp path. GCN cannot: its subgroups are 64 lanes and nothing else (minSubgroupSize = maxSubgroupSize = 64, no stage in
requiredSubgroupSizeStages), so every fallback stage with a ballot, a vote or a shuffle took the 64-lane emulation of the guest's
warp (warp_size_potentially_larger_than_guest). On AMD's Windows driver 24.9.1 an RX 580 2048SP and a Radeon Pro 580 compiled
each of those stages in 53 to 310 s, their draws left out meanwhile, and the foliage, hair and Link drew as specks (Longshot
#1204, #717, #1836). The translator can give such a stage a warp of its 32-invocation workgroup in workgroup memory
(mma_layout.h mesh_fallback_shared_warp): its SPIR-V then has no subgroup instruction, and it gives the same answers at any
lane count. Until now only Mesa's Intel driver had it by default. RADV on the same cards keeps the subgroup's warp: an RX 580
there (Mesa 26.2.6) compiled those stages in seconds and drew them right, and the shared warp gave the same answers and fps, so
it would only cost a GCN RADV player one translation of the fallback.

This cuts what coopmat_init (gpu/vk/vk_coopmat.cpp) sets mesh_fallback_shared_warp to, with the functions it calls there, and
runs it against the real device_policy() (gpu/vk/device_policy.h) on the facts of made-up PCs: the reporters' two Polaris cards
and a Vega 64 on AMD's Windows driver (the two Polaris facts from their logs), the same cards on RADV and an RX 580 on AMD's Linux
driver (all off), the RDNA 1 and RDNA 2 cards that hold compute to 32 lanes, an RTX card, Iris Xe on Mesa (as before), llvmpipe (8 lanes); then RX_VK_MESH_SHARED_WARP over the default both ways, and
the policy block's line naming the rule.

    python -I -B tests/test_gcn_mesh_shared_warp.py --headers <Vulkan-Headers/include> [--cxx g++]
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

# name, vendor, device, driver, version, windows, 16-bit stage inputs, lanes, most lanes, stages pinned to 32,
# RX_VK_MESH_SHARED_WARP (None: unset), the shared warp wanted
CASES = [
    ("RX 580 2048SP, AMD's Windows driver 24.9.1 (#1204)", 0x1002, 0x6FDF, "AMD_PROPRIETARY", 0x800117, 1, 1, 64, 64, 0x0, None, True),
    ("Radeon Pro 580, AMD's Windows driver 24.9.1 (#1836)", 0x1002, 0x67DF, "AMD_PROPRIETARY", 0x800117, 1, 1, 64, 64, 0x0, None, True),
    ("RX 580, RADV on GCN (compiles and draws right on the subgroup)", 0x1002, 0x67DF, "MESA_RADV", 0, 0, 0, 64, 64, 0x0, None, False),
    ("RX 580, RADV on GCN, RX_VK_MESH_SHARED_WARP=1", 0x1002, 0x67DF, "MESA_RADV", 0, 0, 0, 64, 64, 0x0, "1", True),
    ("RX Vega 64, RADV on GFX9", 0x1002, 0x687F, "MESA_RADV", 0, 0, 1, 64, 64, 0x0, None, False),
    ("RX Vega 64, AMD's Windows driver", 0x1002, 0x687F, "AMD_PROPRIETARY", 0x800117, 1, 1, 64, 64, 0x0, None, True),
    ("RX 580, AMD's proprietary driver on Linux (not Windows)", 0x1002, 0x67DF, "AMD_PROPRIETARY", 0x800117, 0, 1, 64, 64, 0x0, None,
     False),
    ("RX 5700, AMD's Windows driver (compute held to 32 lanes)", 0x1002, 0x731F, "AMD_PROPRIETARY", 0x800161, 1, 1, 64, 64, 0x20, None,
     False),
    ("RX 6700 XT, AMD's Windows driver", 0x1002, 0x73DF, "AMD_PROPRIETARY", 0x800161, 1, 1, 64, 64, 0xFF, None, False),
    ("RX 6800 XT, RADV", 0x1002, 0x73BF, "MESA_RADV", 0, 0, 1, 64, 64, 0xFF, None, False),
    ("RTX 3090, NVIDIA's Linux driver", 0x10DE, 0x2204, "NVIDIA_PROPRIETARY", 0, 0, 1, 32, 32, 0x0, None, False),
    ("Iris Xe, Mesa's Intel driver (as before)", 0x8086, 0x9A49, "INTEL_OPEN_SOURCE_MESA", 0, 0, 1, 32, 32, 0xFF, None, True),
    ("llvmpipe, 8 lanes", 0x10005, 0, "MESA_LLVMPIPE", 0, 0, 1, 8, 8, 0x0, None, False),
    ("RX 580 2048SP, AMD's Windows driver, RX_VK_MESH_SHARED_WARP=0", 0x1002, 0x6FDF, "AMD_PROPRIETARY", 0x800117, 1, 1, 64, 64, 0x0,
     "0", False),
    ("RX 580 2048SP, AMD's Windows driver, RX_VK_MESH_SHARED_WARP empty (the default)", 0x1002, 0x6FDF, "AMD_PROPRIETARY", 0x800117, 1,
     1, 64, 64, 0x0, "", True),
    ("RX 6700 XT, AMD's Windows driver, RX_VK_MESH_SHARED_WARP=1", 0x1002, 0x73DF, "AMD_PROPRIETARY", 0x800161, 1, 1, 64, 64, 0xFF,
     "1", True),
    ("Iris Xe, Mesa's Intel driver, RX_VK_MESH_SHARED_WARP=off", 0x8086, 0x9A49, "INTEL_OPEN_SOURCE_MESA", 0, 0, 1, 32, 32, 0xFF, "off",
     False),
]

FIXTURE = r"""
#include "device_policy.h"
#include <cstdio>
#include <cstring>
#include <string>
using namespace mkwvk;

static const char* test_env = nullptr;   // (RX_VK_MESH_SHARED_WARP as the case has it; nullptr: unset)
static const char* test_getenv(const char* name) {
    return std::strcmp(name, "RX_VK_MESH_SHARED_WARP") == 0 ? test_env : nullptr;
}
struct Dev { VkDriverId driver_id{}; DeviceFacts facts{}; DevicePolicy policy{}; } dev;

@@FUNCTIONS@@

static bool wanted() { return @@RHS@@; }

int main() {
    struct C { const char* name; uint32_t vendor, device; VkDriverId driver; uint32_t version; int windows, io16; uint32_t lanes, most,
               sg32; const char* env; bool want; };
    static const C cases[] = {
@@CASES@@
    };
    int failed = 0;
    for (const C& c : cases) {
        DeviceFacts f;
        f.vendor_id = c.vendor, f.device_id = c.device, f.driver_id = c.driver, f.driver_version = c.version;
        f.device_type = c.vendor == 0x10005 ? VK_PHYSICAL_DEVICE_TYPE_CPU
                        : c.driver == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                                                                         : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        f.driver_info = c.driver == VK_DRIVER_ID_AMD_PROPRIETARY ? "24.9.1 (AMD proprietary shader compiler)" : "";
        f.device_name = c.name;
        f.windows = c.windows != 0;
        f.vram_heap = 8192ull << 20, f.allowed_at_start = 7400ull << 20, f.ram = 16384ull << 20;
        f.cpus_all = f.cpus_usable = 12, f.cores_usable = 6;
        f.buffer_device_address = true;
        f.storage_io16 = c.io16 != 0;
        f.subgroup_size = c.lanes, f.subgroup_max = c.most, f.sg32_stages = c.sg32;
        dev = Dev{};
        dev.driver_id = c.driver, dev.facts = f, dev.policy = device_policy(f);
        test_env = c.env;
        const bool got = wanted();
        if (got != c.want) {
            std::printf("FAIL: %s: the mesh fallback's warp %s, expected %s\n", c.name, got ? "in workgroup memory" : "on the subgroup",
                        c.want ? "in workgroup memory" : "on the subgroup");
            failed++;
        }
        // (the policy block says what this PC got, so a report's log shows it)
        if (!c.env) {
            std::string all;
            device_policy_log(dev.policy, f, [&](const char* line) { all += line, all += "\n"; });
            const std::string named = std::string(" mesh_shared_warp_default=") + (c.want ? "1" : "0");
            if (all.find(named) == std::string::npos) {
                std::printf("FAIL: %s: the policy block does not say%s\n", c.name, named.c_str());
                failed++;
            }
        }
    }
    if (failed) return 1;
    std::printf("PASS: %zu PCs and switch settings\n", sizeof cases / sizeof cases[0]);
    return 0;
}
"""


def cut():
    """what coopmat_init sets mesh_fallback_shared_warp to, and the functions of vk_coopmat.cpp it calls for it"""
    coopmat = (VK / "vk_coopmat.cpp").read_text(encoding="utf-8")
    init = block(coopmat, "void coopmat_init() {")
    found = re.search(r"Shader::Backend::SPIRV::mesh_fallback_shared_warp\s*=\s*([^;]+);", init)
    if not found:
        raise SystemExit("FAIL: coopmat_init does not set mesh_fallback_shared_warp")
    rhs = " ".join(found.group(1).split())
    if '"RX_VK_MESH_SHARED_WARP"' not in rhs:
        raise SystemExit("FAIL: coopmat_init does not read RX_VK_MESH_SHARED_WARP for the shared warp")
    functions = []
    for name in dict.fromkeys(re.findall(r"\b([a-z_][a-z0-9_]*)\(", rhs)):
        marker = "bool %s(" % name
        if marker in coopmat:
            functions.append(block(coopmat, marker))
    return rhs.replace("std::getenv(", "test_getenv("), "\n".join(functions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    rhs, functions = cut()
    rows = []
    for name, vendor, device, driver, version, windows, io16, lanes, most, sg32, env, want in CASES:
        env_c = "nullptr" if env is None else '"%s"' % env
        rows.append('        {"%s", 0x%X, 0x%X, VK_DRIVER_ID_%s, 0x%Xu, %d, %d, %du, %du, 0x%Xu, %s, %s},'
                    % (name, vendor, device, driver, version, windows, io16, lanes, most, sg32, env_c, "true" if want else "false"))
    source = FIXTURE.replace("@@FUNCTIONS@@", functions).replace("@@RHS@@", rhs).replace("@@CASES@@", "\n".join(rows))
    with tempfile.TemporaryDirectory(prefix="gcn-shared-warp-") as tmp:
        tmp = Path(tmp)
        cpp, exe = tmp / "fixture.cpp", tmp / "fixture.exe"
        cpp.write_text(source, encoding="utf-8")
        r = subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-I" + str(VK), "-I" + str(args.headers), str(cpp), "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("the fixture did not build")
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        print(run.stdout.strip())
        if run.returncode:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
