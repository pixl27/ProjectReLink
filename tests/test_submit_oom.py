"""A heap past its own budget no longer gets host-visible buffers, and a submit the driver refuses for want of memory is made
again instead of dropped (a Legion Go, Radeon 780M with 8 GB UMA on AMD's Windows driver; the ROG Ally too).

On that device the driver has a second device-local heap of 256 MB the CPU can write, of which it allows 243 MB. VMA put the
staging arenas there (host-visible, read by the GPU's ASTC decoder) whatever that heap held: a save load's staging took it to
252-277 MB while VRAM as a whole stood at 1.9 of 7.8 GB, and from then on every vkQueueSubmit2 came back
VK_ERROR_OUT_OF_DEVICE_MEMORY, which the replay logged (`vk: submit ... failed (-2)`) and dropped: a black screen with the
sound going on. No game or GPU required:

- create_buffer is cut out of gpu/vk/vk_vma.cpp and compiled against a stand-in VMA that places a buffer as VMA does (the
  memory type it prefers, the next one when asked to stay within the budgets and that heap is full, past the budget when not
  asked). A 16 MB staging arena with that heap at 236 of 243 MB must land where there is room; with room, where it went before;
  with no heap that has room, still made; RX_VK_BUFFER_BUDGET=0 and a driver without VK_EXT_memory_budget: as before.
- vk_replay.cpp's submit_or_retry (with gpu/vk/submit_retry.h) is cut out and compiled against a stand-in queue: a refusal for
  want of memory waits for the GPU and tells the next present to free memory, then submits again, up to three tries; other
  errors are not tried again; RX_VK_SUBMIT_RETRY=0 drops it as before; RX_VK_TEST_SUBMIT_REFUSE refuses the first try.
- Every vkQueueSubmit2 of vk_replay.cpp goes through it, nothing is destroyed under the queue's lock, and texture_frame frees
  the idle staging arenas after a refusal.

python -I -B tests/test_submit_oom.py [--cxx g++]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block, between   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

PLACEMENT = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef int VkResult;
#define VK_SUCCESS 0
#define VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
typedef uint32_t VkBufferUsageFlags;
#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT 0x1u
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT 0x2u
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x20u
struct Obj { int heap = -1; uint64_t size = 0; };
typedef Obj* VkBuffer;
typedef Obj* VmaAllocation;
struct VkBufferCreateInfo { int sType; const void* pNext; uint32_t flags; uint64_t size; VkBufferUsageFlags usage; };
enum VmaMemoryUsage { VMA_MEMORY_USAGE_UNKNOWN = 0, VMA_MEMORY_USAGE_AUTO = 7 };
#define VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT 0x1u
#define VMA_ALLOCATION_CREATE_MAPPED_BIT 0x4u
#define VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT 0x100u
#define VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT 0x400u
struct VmaAllocationCreateInfo { uint32_t flags; VmaMemoryUsage usage; uint32_t requiredFlags, preferredFlags, memoryTypeBits;
                                 void* pool; void* pUserData; float priority; };
struct VmaAllocationInfo { uint32_t memoryType; void* deviceMemory; uint64_t offset, size; void* pMappedData; void* pUserData; };

constexpr uint64_t kMB = 1ull << 20;
// (the Legion Go's heaps: 0 the carve-out, 1 system memory, 2 the carve-out's 256 MB window the CPU writes. Memory type t is
// in heap t: 0 DEVICE_LOCAL, 1 HOST_VISIBLE, 2 DEVICE_LOCAL | HOST_VISIBLE)
static uint64_t used[3], budget[3];
static const bool host_visible_type[3] = {false, true, true};
static char mapped_bytes[64];
static void* vma = nullptr;
static struct { bool memory_budget = true; } dev;
struct Tiers { float rest = 0.5f; };
static Tiers memory_tiers() { return {}; }
static void buffer_count(VmaAllocation, bool) {}
static int mem_kind_tag = 0;   // (hookshot's create_buffer tags what it makes: vk_vma.cpp)
static void tag_mapped(const VmaAllocationInfo&, int) {}
static void note_alloc(VmaAllocation, int) {}
static bool budget_switch = true;
static bool buffer_budget_on() { return budget_switch; }
static int calls = 0;

// VMA's choice for VMA_MEMORY_USAGE_AUTO: a host-visible buffer the GPU reads (any usage but transfers) prefers DEVICE_LOCAL |
// HOST_VISIBLE, one only copied from prefers system memory, a device-only one DEVICE_LOCAL. Asked to stay within the budgets
// (VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT), a type whose heap has no room is passed over for the next; not asked, the first
// is taken whatever its heap holds (the driver gives the memory: past the budget)
static VkResult vmaCreateBuffer(void*, const VkBufferCreateInfo* bi, const VmaAllocationCreateInfo* ai, VkBuffer* buffer,
                                VmaAllocation* allocation, VmaAllocationInfo* info) {
    calls++;
    *buffer = nullptr, *allocation = nullptr;
    int order[3] = {0, 2, 1};
    if (ai->flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) {
        const bool gpu_reads = (bi->usage & ~(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) != 0;
        order[0] = gpu_reads ? 2 : 1, order[1] = gpu_reads ? 1 : 2, order[2] = -1;
    }
    for (int t : order) {
        if (t < 0) break;
        if ((ai->flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) && used[t] + bi->size > budget[t]) continue;
        used[t] += bi->size;
        Obj* o = new Obj{t, bi->size};
        *buffer = o, *allocation = o;
        info->memoryType = (uint32_t)t;
        info->pMappedData = (ai->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) && host_visible_type[t] ? mapped_bytes : nullptr;
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

/* CREATE_BUFFER */

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) failures++, std::printf("FAIL: %s\n", what);
}
static void heaps(uint64_t h0, uint64_t h1, uint64_t h2) {
    used[0] = h0 * kMB, used[1] = h1 * kMB, used[2] = h2 * kMB;
    budget[0] = 7539 * kMB, budget[1] = 3741 * kMB, budget[2] = 243 * kMB;
}
static int arena(bool host_visible = true) {   // (a staging arena: vk_texture.cpp staging(), 16 MB, TRANSFER_SRC | STORAGE)
    VkBuffer b = nullptr;
    VmaAllocation a = nullptr;
    void* m = nullptr;
    if (!create_buffer(16 * kMB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_visible, b, a, &m, false))
        return -2;
    if (host_visible && !m) return -3;   // (made, but not mapped: of no use to staging())
    return a->heap;
}

int main() {
    // (the save load on the Legion Go: the 256 MB heap at 236 of its 243 MB, system memory and the carve-out with room)
    heaps(1625, 2574, 236);
    int heap = arena();
    expect(heap >= 0, "the staging arena is made");
    if (heap == 2)
        std::printf("FAIL: the staging arena went in the 256 MB DEVICE_LOCAL | HOST_VISIBLE heap past its budget: %llu of %llu MB "
                    "(the driver then refuses every submit:)\n", (unsigned long long)(used[2] / kMB),
                    (unsigned long long)(budget[2] / kMB)), failures++;
    expect(used[2] <= budget[2], "the 256 MB heap stays within its budget");
    expect(heap == 1, "the arena goes to system memory, the next type it can use");

    // (room in that heap: where VMA prefers it, as before)
    heaps(1625, 2574, 100);
    expect(arena() == 2, "with room, the arena still goes in the DEVICE_LOCAL | HOST_VISIBLE heap");

    // (no heap with room: made all the same, as before, where VMA prefers it)
    heaps(1625, 3740, 240);
    expect(arena() == 2, "with no heap that has room, the arena is still made where it was");

    // (RX_VK_BUFFER_BUDGET=0, and a driver without VK_EXT_memory_budget: as before)
    budget_switch = false;
    heaps(1625, 2574, 236);
    expect(arena() == 2, "RX_VK_BUFFER_BUDGET=0: as before");
    budget_switch = true;
    dev.memory_budget = false;
    heaps(1625, 2574, 236);
    calls = 0;
    expect(arena() == 2 && calls == 1, "without VK_EXT_memory_budget: as before, one call");
    dev.memory_budget = true;

    // (a device-only buffer: placed as before, whatever the budgets say)
    heaps(7600, 2574, 236);
    calls = 0;
    expect(arena(false) == 0 && calls == 1, "a device-only buffer: as before, one call");

    if (failures) return 1;
    std::printf("placement: ok\n");
    return 0;
}
"""

SUBMIT = r"""
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "submit_retry.h"

enum VkResult { VK_SUCCESS = 0, VK_ERROR_OUT_OF_HOST_MEMORY = -1, VK_ERROR_OUT_OF_DEVICE_MEMORY = -2, VK_ERROR_DEVICE_LOST = -4 };
struct VkSubmitInfo2 { int sType; };
typedef void* VkFence;
typedef void* VkQueue;
static struct { VkQueue queue = nullptr; } dev;
static std::string log_text;
#define logf fixture_logf   // (not <cmath>'s logf)
static void logf(const char* f, ...) {
    char line[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(line, sizeof line, f, ap);
    va_end(ap);
    log_text += line, log_text += "\n";
}
static bool retry_switch = true;
static bool env_on_or(const char* name, int) { return std::strcmp(name, "RX_VK_SUBMIT_RETRY") || retry_switch; }
static int waits = 0, told = 0, driver_calls = 0;
static VkResult script[8];
static int script_at = 0;
static void vkQueueWaitIdle(VkQueue) { waits++; }
static void texture_memory_refused() { told++; }
static VkResult vkQueueSubmit2(VkQueue, int, const VkSubmitInfo2*, VkFence) {
    driver_calls++;
    return script[std::min(script_at++, 7)];
}

/* SUBMIT_OR_RETRY */

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) failures++, std::printf("FAIL: %s\n", what);
}
static VkResult run(std::initializer_list<VkResult> answers) {
    int i = 0;
    for (VkResult r : answers) script[i++] = r;
    for (; i < 8; i++) script[i] = VK_SUCCESS;
    script_at = 0, waits = 0, told = 0, driver_calls = 0;
    VkSubmitInfo2 si{};
    return submit_or_retry(si, nullptr);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "on";
    if (mode == "on") {
        expect(run({VK_SUCCESS}) == VK_SUCCESS && driver_calls == 1 && waits == 0, "a submit taken: made once, no wait");
        // (the Legion Go's: refused for want of device memory, taken once the GPU is idle and memory given back)
        VkResult r = run({VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_SUCCESS});
        if (driver_calls == 1)
            std::printf("FAIL: a submit refused for want of memory (-2) is dropped, not made again\n"), failures++;
        expect(r == VK_SUCCESS, "refused twice, then taken: the submit's result is VK_SUCCESS");
        expect(driver_calls == 3 && waits == 2 && told == 2, "each try after a refusal waits for the GPU and tells the next present");
        r = run({VK_ERROR_OUT_OF_HOST_MEMORY, VK_SUCCESS});
        expect(r == VK_SUCCESS && driver_calls == 2, "a refusal for want of host memory is tried again too");
        r = run({VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_SUCCESS});
        expect(r == VK_ERROR_OUT_OF_DEVICE_MEMORY && driver_calls == submit_retry::kTries,
               "refused every time: kTries tries in all, then the refusal returned");
        r = run({VK_ERROR_DEVICE_LOST, VK_SUCCESS});
        expect(r == VK_ERROR_DEVICE_LOST && driver_calls == 1 && waits == 0, "a lost device is not tried again");
        expect(log_text.find("refused for want of memory") != std::string::npos, "the refusal is logged");
    } else if (mode == "off") {   // (RX_VK_SUBMIT_RETRY=0)
        retry_switch = false;
        const VkResult r = run({VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_SUCCESS});
        expect(r == VK_ERROR_OUT_OF_DEVICE_MEMORY && driver_calls == 1 && waits == 0, "RX_VK_SUBMIT_RETRY=0: dropped, as before");
    } else if (mode == "refuse") {   // (RX_VK_TEST_SUBMIT_REFUSE=1: the first try of every submit refused without the driver)
        for (int i = 0; i < 3; i++) {
            const VkResult r = run({VK_SUCCESS});
            expect(r == VK_SUCCESS && driver_calls == 1 && waits == 1, "a test refusal: the second try made, and taken");
        }
    }
    if (failures) return 1;
    std::printf("submit (%s): ok\n", mode.c_str());
    return 0;
}
"""

START_SUBMIT = "bool submit_retry_on() {"
END_SUBMIT = "// ---- (hookshot) recordings by number"


def compile_and_run(cxx, source, temp, name, runs, includes=()):
    cpp, exe = Path(temp) / (name + ".cpp"), Path(temp) / (name + ".exe")
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                    "-Wno-unused-function", "-Wno-unused-parameter", "-Wno-unused-variable", *["-I" + str(i) for i in includes],
                    str(cpp), "-o", str(exe)], check=True)
    failed = 0
    for args, env in runs:
        r = subprocess.run([str(exe)] + args, timeout=60, env={**os.environ, **env})
        failed |= r.returncode
    return failed


def check_source():
    """The rules the compiled cuts cannot see: every submit of the replay through submit_or_retry, nothing destroyed under the
    queue's lock, and the present after a refusal freeing what it can"""
    problems = []
    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    direct = len(re.findall(r"\bvkQueueSubmit2\(dev\.queue", replay))
    if direct != 1:
        problems.append("vk_replay.cpp calls vkQueueSubmit2 directly %d times: only submit_or_retry should" % direct)
    give = block(replay, "void submit_give_back(int tried, VkResult res)")
    for destroy in ("reap_now", "destroy_", "vkDestroy", "vmaFree", "vmaDestroy"):
        if destroy in give:
            problems.append("submit_give_back destroys under the queue's lock (%s): a destroy's rec::boundary can wait for the "
                            "recorder, whose part's submit waits for that lock" % destroy)
    if "vkQueueWaitIdle(dev.queue);" not in give or "texture_memory_refused();" not in give:
        problems.append("submit_give_back waits for the GPU and tells the next present")
    tex = (VK / "vk_texture.cpp").read_text(encoding="utf-8")
    frame = block(tex, "void texture_frame(uint64_t completed)")
    took = frame.find("if (memory_refused.exchange(false, std::memory_order_relaxed)) {")
    arenas = frame.find("for (auto it = arenas.begin(); it != arenas.end();) {")
    if took < 0 or arenas < 0 or took > arenas:
        problems.append("texture_frame takes the refusal before its look at the staging arenas")
    else:
        taken = frame[took:arenas]
        for part in ("linux_entered_pressure = true;",):   # (hookshot: its reaper has no queue to empty at once)
            if part not in taken:
                problems.append("texture_frame after a refusal: %s" % part)
    vma = (VK / "vk_vma.cpp").read_text(encoding="utf-8")
    if 'env_on_or("RX_VK_BUFFER_BUDGET", 1)' not in block(vma, "static bool buffer_budget_on()"):
        problems.append("RX_VK_BUFFER_BUDGET is read, on when unset")
    if 'env_on_or("RX_VK_SUBMIT_RETRY", 1)' not in block(replay, "bool submit_retry_on()"):
        problems.append("RX_VK_SUBMIT_RETRY is read, on when unset")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = p.parse_args()
    failed = 0
    vma = (VK / "vk_vma.cpp").read_text(encoding="utf-8")
    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="hookshot-submit-oom-") as temp:
        placement = PLACEMENT.replace("/* CREATE_BUFFER */", block(vma, "bool create_buffer(uint64_t size, VkBufferUsageFlags usage"))
        failed |= compile_and_run(a.cxx, placement, temp, "placement", [([], {})])
        submit = SUBMIT.replace("/* SUBMIT_OR_RETRY */", between(replay, START_SUBMIT, END_SUBMIT))
        env_clear = {"RX_VK_TEST_SUBMIT_REFUSE": ""}
        failed |= compile_and_run(a.cxx, submit, temp, "submit",
                                  [(["on"], env_clear), (["off"], env_clear), (["refuse"], {"RX_VK_TEST_SUBMIT_REFUSE": "1"})],
                                  includes=[VK])
    for problem in check_source():
        print("FAIL:", problem)
        failed = 1
    if failed:
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
