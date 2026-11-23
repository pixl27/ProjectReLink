"""Exercise Hookshot's Windows RDNA 2 mesh routing and program-cache stamp.

Compile the actual device setup, shader cache key and per-program selector.
Simulate both platform branches without needing a Vulkan device or a game.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import between, block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

HARNESS = r'''
#include "mesh_mode.h"
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

struct Device {
    struct { unsigned vendorID, deviceID; } props;
    bool mesh_shader = true;
    bool mesh_fallback = false;
    bool task_mesh_fallback = false;
    bool mesh_only_fallback = false;
    @SELECTOR@
};

Device choose(unsigned vendor, unsigned device, int mesh_mode, const char* override_value,
              bool can_address = true, bool disable_fallback = false) {
    Device dev{{vendor, device}};
    const struct { bool bufferDeviceAddress; } f12{can_address};
    const auto env_on = [disable_fallback](const char*) { return disable_fallback; };
    const auto getenv = [override_value](const char*) { return override_value; };
    @SETUP@
    return dev;
}

uint32_t cache_mesh_key(const Device& dev) {
    @STAMP@;
    return mesh;
}

void expect(const char* description, bool value, bool expected) {
    if (value != expected) {
        std::fprintf(stderr, "%s: expected %d, got %d\n", description, expected, value);
        std::exit(1);
    }
}

int main() {
    constexpr unsigned amd = 0x1002, nvidia = 0x10de;
    constexpr unsigned rx6800xt = 0x73bf, rx6750xt = 0x73df, rx6600 = 0x73ff, rx6500xt = 0x743f;
    for (const auto id : {rx6800xt, rx6750xt, rx6600}) {
        const Device selected = choose(amd, id, 0, nullptr);
        expect("RDNA 2 mesh-only draw", selected.uses_mesh_fallback(false), TEST_WINDOWS);
        expect("RDNA 2 task + mesh draw", selected.uses_mesh_fallback(true), TEST_WINDOWS);
        expect("Native mode", choose(amd, id, 1, nullptr).uses_mesh_fallback(false), false);
        expect("Native task + mesh", choose(amd, id, 1, nullptr).uses_mesh_fallback(true), false);
        expect("explicit off", choose(amd, id, 0, "0").uses_mesh_fallback(false), false);
        expect("explicit off for task + mesh", choose(amd, id, 0, "0").uses_mesh_fallback(true), false);
        expect("no buffer addresses", choose(amd, id, 0, nullptr, false).uses_mesh_fallback(false), false);
        expect("fallback disabled", choose(amd, id, 0, nullptr, true, true).uses_mesh_fallback(false), false);
    }
    const Device other = choose(nvidia, rx6800xt, 0, "1");
    expect("forced fallback on another vendor remains task-only", other.uses_mesh_fallback(false), false);
    expect("forced task fallback on another vendor", other.uses_mesh_fallback(true), true);
    expect("Navi 24 retains native mesh-only", choose(amd, rx6500xt, 0, nullptr).uses_mesh_fallback(false), false);
    const Device rdna2 = choose(amd, rx6800xt, 0, "1");
    expect("the cache separates mesh-only and task-only fallback", cache_mesh_key(rdna2) != cache_mesh_key(other),
           TEST_WINDOWS);
    Device compute = rdna2;
    compute.mesh_fallback = true;
    expect("global compute fallback handles mesh-only", compute.uses_mesh_fallback(false), true);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    selector = block((VK / "vk_internal.h").read_text(encoding="utf-8"), "bool uses_mesh_fallback(")
    setup = between(
        (VK / "vk_main.cpp").read_text(encoding="utf-8"),
        "bool task_mesh_fallback = false;",
        "if (dev.task_mesh_fallback)",
    ).replace("#ifdef _WIN32", "#if TEST_WINDOWS")
    stamp = between((VK / "vk_shader.cpp").read_text(encoding="utf-8"), "const uint32_t mesh =", ";")
    code = HARNESS.replace("@SELECTOR@", selector).replace("@SETUP@", setup).replace("@STAMP@", stamp)
    with tempfile.TemporaryDirectory(prefix="rdna2-mesh-only-") as temporary:
        source = Path(temporary) / "routing.cpp"
        executable = Path(temporary) / "routing.exe"
        source.write_text(code, encoding="utf-8")
        for windows in (1, 0):
            subprocess.run(
                [args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", f"-DTEST_WINDOWS={windows}",
                 "-I", str(VK), str(source), "-o", str(executable)],
                check=True, timeout=60,
            )
            subprocess.run([str(executable)], check=True, timeout=30)
    print("PASS: RDNA 2 mesh routing, platform and override controls, and program-cache separation")


if __name__ == "__main__":
    main()
