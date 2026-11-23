"""The shader preparation's graphics pipelines go side by side through Intel's Windows compiler on the Arc A-series and
the Core Ultra GPUs (row on Auto, driver 101.9175 or newer); its compute pipelines and play's builds stay one at a time
(no game/GPU required).

Reports from an Arc A730M and an Arc iGPU (Core Ultra 7 155H), both on driver 101.9175: the preparation took hours, its
CPU and GPU at 5-8%, as IntelDriverGate (gpu/vk/vk_internal.h) let one thread at a time into the compiler there: the
row's Auto is side by side only on the Arc B-series (intel_compiles.h). The fault the gate was put in for was the
pipeline cache's data fetched beside builds, which pcache_fetch_lock keeps apart by itself.

The real gate (vk_internal.h: intel_gate_on, IntelDriverGate and, where there, IntelPrepareLane and intel_prepare_open),
the real row questions (gpu/vk/vk_menu.cpp) and the real cached_graphics and cached_compute (gpu/vk/vk_draw.cpp) are cut
out and compiled with stand-ins: a "driver" that counts the builds inside it at once. Four threads build, on a preparation
thread (IntelPrepareLane) or not, a graphics or a compute pipeline, for an Intel GPU, driver and row value each run. And
intel_prepare_workers (intel_compiles.h): the workers held to the free memory (a 16 GB A730M at its commit limit).

    python -I -B tests/test_intel_prepare_side_by_side.py [--cxx g++]
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block  # noqa: E402

VK = ROOT / "gpu" / "vk"
D9154, D9175, D7088 = 101 << 14 | 9154, 101 << 14 | 9175, 101 << 14 | 7088   # (Intel's Windows driverVersion)
B580, A730M, A770, MTL, IRIS_XE, UHD = 0xE20B, 0x5691, 0x56A0, 0x7D55, 0x9A49, 0x9A78
# (row: 0 Auto, 1 Side by side, 2 One at a time; device; driver; lane: a preparation thread; kind: g graphics, c compute;
# the most builds inside the driver at once: 1, or 2 for "side by side"; why)
SIDE, ONE = 2, 1
CASES = [
    (0, A730M, D9175, 1, "g", SIDE, "the preparation's graphics pipelines are one at a time on an Arc A730M, driver 101.9175, "
                                    "on Auto (hours, the GPU at 5%)"),
    (0, MTL, D9175, 1, "g", SIDE, "the preparation's graphics pipelines are one at a time on a Core Ultra's Arc iGPU, driver "
                                  "101.9175, on Auto (over 7 hours, the CPU at 8%)"),
    (0, A770, D9175 + 30, 1, "g", SIDE, "a newer driver than 101.9175 keeps the preparation one at a time"),
    (0, B580, D9154, 1, "g", SIDE, "the Arc B-series (Auto side by side already) is held in the preparation"),
    (1, UHD, D7088, 1, "g", SIDE, "the row's Side by side does not hold in the preparation"),
    (0, A730M, D9175, 1, "c", ONE, "the preparation's compute pipelines go side by side (four at once froze a PC)"),
    (0, A730M, D9175, 0, "g", ONE, "play's graphics pipelines go side by side on Auto off the Arc B-series"),
    (0, MTL, D9175, 0, "c", ONE, "play's compute pipelines go side by side"),
    (2, A730M, D9175, 1, "g", ONE, "the row's One at a time does not hold in the preparation"),
    (2, B580, D9154, 1, "g", ONE, "the row's One at a time does not hold in the preparation on the Arc B-series"),
    (0, UHD, D7088, 1, "g", ONE, "an older driver (the UHD's 101.7088, where the gate's fault was seen) is not one at a time "
                                 "in the preparation"),
    (0, IRIS_XE, D9175 - 1, 1, "g", ONE, "a driver before 101.9175 is not one at a time in the preparation"),
]
# (want, MB free, workers: intel_prepare_workers)
WORKERS = [(16, 32768, 16), (16, 15000, 14), (16, 16384, 16), (16, 12000, 10), (16, 9000, 6), (16, 4096, 1), (16, 1000, 1), (16, 0, 1),
           (20, 1 << 20, 20), (1, 32768, 1), (0, 32768, 1), (16, (1 << 64) - 1, 16)]

STANDINS = r"""
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include "intel_compiles.h"
using VkResult = int;
using VkPipeline = void*;
using VkPipelineCache = void*;
#define VK_NULL_HANDLE nullptr   // (main's cached_compute names it)
using VkPipelineCreateFlags = uint32_t;
constexpr VkResult VK_SUCCESS = 0, VK_PIPELINE_COMPILE_REQUIRED = 1000297000, VK_ERROR_OUT_OF_HOST_MEMORY = -1;
constexpr VkPipelineCreateFlags VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT = 0x100;
struct VkGraphicsPipelineCreateInfo { VkPipelineCreateFlags flags = 0; };
struct VkComputePipelineCreateInfo { VkPipelineCreateFlags flags = 0; };
void logf(const char*, ...) {}
struct { struct { uint32_t vendorID = 0x8086, deviceID = 0, driverVersion = 0; } props; } dev;
struct { int intel_compiles = 0; } opts;
VkResult intel_gate_busy() { return VK_PIPELINE_COMPILE_REQUIRED; }
struct ComputeTurn {
    explicit ComputeTurn(const VkComputePipelineCreateInfo&) {}
    bool ok() const { return true; }
};
// (the driver: how many builds are inside it at once. A build waits up to 150 ms for a second to come in beside it, so
// side by side shows as 2 however the threads are scheduled, and one at a time as 1)
std::atomic<int> inside{0}, most{0};
VkResult driver(VkPipeline* pl) {
    const int now = ++inside;
    for (int seen = most.load(); now > seen && !most.compare_exchange_weak(seen, now);) {}
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
    while (inside.load() < 2 && most.load() < 2 && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    --inside;
    *pl = (VkPipeline)1;
    return VK_SUCCESS;
}
VkResult guarded_graphics_held(VkPipelineCache, const VkGraphicsPipelineCreateInfo*, VkPipeline* pl) { return driver(pl); }
VkResult guarded_compute_held(VkPipelineCache, const VkComputePipelineCreateInfo*, VkPipeline* pl) { return driver(pl); }
template <class CI>
VkResult create_cached(VkResult (*create)(VkPipelineCache, const CI*, VkPipeline*), const CI* ci, VkPipeline* pl) {
    return create(nullptr, ci, pl);
}
"""

MAIN = r"""
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    opts.intel_compiles = atoi(argv[1]);
    dev.props.deviceID = (uint32_t)strtoul(argv[2], nullptr, 0);
    dev.props.driverVersion = (uint32_t)strtoul(argv[3], nullptr, 0);
    const bool lane = atoi(argv[4]) != 0, graphics = argv[5][0] == 'g';
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++)
        threads.emplace_back([&] {
            const IntelPrepareLane on(lane);
            for (int k = 0; k < 2; k++) {
                VkPipeline pl = nullptr;
                if (graphics) {
                    VkGraphicsPipelineCreateInfo ci;
                    cached_graphics(&ci, &pl);
                } else {
                    VkComputePipelineCreateInfo ci;
                    cached_compute(&ci, &pl);
                }
            }
        });
    for (std::thread& t : threads) t.join();
    printf("%d\n", most.load());
    return 0;
}
"""

WORKERS_MAIN = r"""
#include <cstdio>
#include <cstdlib>
#include "intel_compiles.h"
int main(int argc, char** argv) {
    for (int i = 1; i + 2 < argc; i += 3) {
        const unsigned want = (unsigned)strtoul(argv[i], nullptr, 0), wanted = (unsigned)strtoul(argv[i + 2], nullptr, 0);
        const uint64_t avail = strtoull(argv[i + 1], nullptr, 0);
        const unsigned got = intel_prepare_workers(want, avail);
        if (got != wanted) {
            printf("FAIL: intel_prepare_workers(%u, %llu MB free) = %u, not %u\n", want, (unsigned long long)avail, got, wanted);
            return 1;
        }
    }
    return 0;
}
"""


def need(cond, why):
    if not cond:
        raise SystemExit("FAIL: " + why)


def gate_source():
    """vk_internal.h's gate, from intel_gate_on to the end of IntelDriverGate (with the lane between where it is there),
    Windows' half taken; the lane a stand-in that does nothing where vk_internal.h has none (main before this fix)"""
    internal = (VK / "vk_internal.h").read_text(encoding="utf-8")
    start = internal.index("inline bool intel_gate_on() {")
    gate = block(internal, "struct IntelDriverGate {")
    end = internal.index(gate) + len(gate) + 1   # (its ;)
    text = internal[start:end].replace("#ifdef _WIN32", "#if 1")
    if "struct IntelPrepareLane" not in text:
        text = "struct IntelPrepareLane { explicit IntelPrepareLane(bool = true) {} };\n" + text
    return "bool intel_compiles_side_by_side();\n" + text + "\n"


def menu_source():
    intel = (VK / "vk_menu.cpp").read_text(encoding="utf-8")
    out = [block(intel, "bool intel_compiles_side_by_side() {")]
    if "bool intel_prepare_side_by_side() {" in intel:
        out.append(block(intel, "bool intel_prepare_side_by_side() {"))
    return "\n".join(out) + "\n"


def draw_source():
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    return "\n".join([block(draw, "bool compiles_nothing(VkPipelineCreateFlags flags) {"),
                      block(draw, "VkResult cached_graphics(const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {"),
                      block(draw, "VkResult cached_compute(const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {")]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="intel-prepare-") as tmp:
        src, exe = Path(tmp) / "gate.cpp", Path(tmp) / "gate.exe"
        src.write_text(STANDINS + gate_source() + menu_source() + draw_source() + MAIN, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++17", "-pthread", "-I", str(VK), str(src), "-o", str(exe)], check=True)
        for row, device, version, lane, kind, want, why in CASES:
            r = subprocess.run([str(exe), str(row), hex(device), str(version), str(lane), kind], capture_output=True,
                               text=True, timeout=120)
            need(r.returncode == 0, "the fixture ended with %d: %s" % (r.returncode, r.stderr.strip()))
            got = int(r.stdout.strip())
            need((got >= 2) == (want == SIDE),
                 "%s (row %d, device 0x%04x, driver %d.%d, %s, %s: %d at once)" %
                 (why, row, device, version >> 14, version & 0x3FFF, "a preparation thread" if lane else "play",
                  "graphics" if kind == "g" else "compute", got))

        wsrc, wexe = Path(tmp) / "workers.cpp", Path(tmp) / "workers.exe"
        wsrc.write_text(WORKERS_MAIN, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++17", "-I", str(VK), str(wsrc), "-o", str(wexe)], check=True)
        argv = [str(wexe)]
        for want, avail, workers in WORKERS:
            argv += [str(want), str(avail), str(workers)]
        r = subprocess.run(argv, capture_output=True, text=True)
        need(r.returncode == 0, r.stdout.strip() or "the workers' fixture ended with %d" % r.returncode)

    # (the preparation's threads are on the lane, play's are not; the workers are held to the memory there)
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    for marker, step in (("void precompile_warm(void (*progress)(size_t done, size_t total)) {", '"the noted pipelines"'),
                         ("void precompile_run(void (*progress)(size_t done, size_t total)) {", '"the precompile"'),
                         ("void predict_run(void (*progress)(size_t done, size_t total)) {", '"the predicted pipelines"')):
        body = block(draw, marker)
        need("const IntelPrepareLane lane;" in body, marker + ": its workers are not preparation threads")
        need("prepare_pipeline_threads(" in body and step in body, marker + ": its workers are not held to the memory")
    prepare = block(draw, "bool prepare_run(void (*progress)(size_t done, size_t total), bool background, bool refresh_states) {")
    need("const IntelPrepareLane lane(!background);" in prepare, "prepare_run's workers in play are preparation threads")
    module = block((VK / "vk_shader.cpp").read_text(encoding="utf-8"), "VkShaderModule make_module(")
    need("IntelDriverGate gate(busy != nullptr, true);" in module, "the preparation's shader modules wait for the gate")
    print("ok: %d gate cases (the preparation's graphics side by side on Auto from driver 101.9175, compute and play one at "
          "a time), %d worker counts" % (len(CASES), len(WORKERS)))


if __name__ == "__main__":
    main()
