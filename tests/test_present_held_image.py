"""A window's image held and never given back makes the swapchain again, so the picture does not freeze for good.

An image the system has not given back in 250 ms is held (present_image: the acquire's fence, image_free) and
looked at again, without waiting, at each present; nothing else is presented meanwhile. With frame generation on an RX 9070 XT
(Windows, full screen) an image held after Alt+Tab was not given back once the game was in front again either: the system
lets an image go when another is shown, and none was. The game and its sound went on and the picture stayed as it was until
F11 made the swapchain again (the reporter's log: "not given back in 250 ms", then no "given back" until "display:
windowed"). Five reports, also without Alt+Tab (a hold the system never ended).

present_image's acquire, from its image index to the frame's command pool reset, and its copy's submit info, are cut out of
gpu/vk/vk_replay.cpp with the helpers before present_image (the image's fence, the held image, the fences let go), and
compiled against a model of a presentation engine that hands an image back at once and lets it go later (AMD's driver on
Windows), with a clock that the frames and the waits move. Each platform's side of the source's _WIN32 lines is compiled on
its own. Checked:

- a hold the system ends itself (the window hidden under a second): no swapchain made again for it;
- the window seen again but the held image never let go: the swapchain is made again within 1.5 s of the hold, with
  the presenter drained and the queue's lock held, and the frames are copied again after it, one each present;
- the window hidden 20 s: the swapchain made again a few times, further apart each time (at most 4 s), not at every present;
  frames shown within 5 s of the window being seen again;
- a swapchain that cannot be made (a window with no size): not tried again at every present.

Broken variants that must fail are in BROKEN below.

No GPU.

python -I -B tests/test_present_held_image.py [--cxx clang++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import between, block   # noqa: E402
from test_present_image_free import platform_side   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

FIXTURE_HEAD = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
using VkResult = int;
using VkFence = uint64_t; using VkSemaphore = uint64_t; using VkSwapchainKHR = uint64_t; using VkCommandBuffer = uint64_t;
using VkCommandPool = uint64_t;
constexpr uint64_t VK_NULL_HANDLE = 0;
constexpr VkResult VK_SUCCESS = 0, VK_NOT_READY = 1, VK_TIMEOUT = 2, VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_SURFACE_LOST_KHR = -1000000000, VK_SUBOPTIMAL_KHR = 1000001003, VK_ERROR_OUT_OF_DATE_KHR = -1000001004;
constexpr int VK_TRUE = 1, VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8, VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO = 1000314005,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO = 1000314006, VK_STRUCTURE_TYPE_SUBMIT_INFO_2 = 1000314004;
constexpr uint64_t VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT = 0x10000;
struct VkFenceCreateInfo { int sType; const void* pNext = nullptr; int flags = 0; };
struct VkSemaphoreSubmitInfo { int sType; const void* pNext = nullptr; VkSemaphore semaphore = 0; uint64_t value = 0;
    uint64_t stageMask = 0; uint32_t deviceIndex = 0; };
struct VkCommandBufferSubmitInfo { int sType; const void* pNext = nullptr; VkCommandBuffer commandBuffer = 0; uint32_t deviceMask = 0; };
struct VkSubmitInfo2 { int sType; const void* pNext = nullptr; int flags = 0; uint32_t waitSemaphoreInfoCount = 0;
    const VkSemaphoreSubmitInfo* pWaitSemaphoreInfos = nullptr; uint32_t commandBufferInfoCount = 0;
    const VkCommandBufferSubmitInfo* pCommandBufferInfos = nullptr; uint32_t signalSemaphoreInfoCount = 0;
    const VkSemaphoreSubmitInfo* pSignalSemaphoreInfos = nullptr; };
struct Extent { uint32_t width = 1920, height = 1080; };
struct QueueLock {   // (dev.present_mu, the queue's lock: whether it is held, for the remake's check)
    bool held = false;
    void lock() { held = true; }
    void unlock() { held = false; }
};
enum class Sw { acquire_ahead };
struct Policy { bool acquire_ahead = false; bool set(Sw) const { return false; } };
struct Device { int device = 1; VkSwapchainKHR swapchain = 1; uint64_t swapchain_generation = 0; Extent swap_extent;
    std::vector<VkSemaphore> swap_rendered{901, 902, 903, 904}; QueueLock present_mu; Policy policy; } dev;
struct Frame { VkCommandPool pool = 11; VkCommandBuffer cmd = 12; VkFence done = 13; };
namespace fps_stats { enum { kGpuWait, kPace }; }
namespace gpu_wait_stats { struct Exclude { Exclude() {} }; }
void fps_wait(int, int64_t) {}
constexpr int64_t kMs = 1000000;
int64_t now_ns = 0;   // (the model's clock: the frames and the waits move it)
int64_t fps_clock() { return now_ns; }
int remake_lines = 0;
template<class... T> void logf(const char* f, T...) {
    if (std::strstr(f, "the swapchain made again")) ++remake_lines;
}
int problems = 0;
void problem(const char* what) { if (++problems <= 12) std::printf("FAIL (%s): %s\n", PLATFORM, what); }
bool drained = false;   // (present_thread_drain since the last acquire)
void present_thread_drain() { drained = true; }
bool invalidate_lost_surface(VkResult) { return true; }
struct Lost {};
[[noreturn]] void device_lost() { throw Lost{}; }

// ---- the presentation engine: an acquire hands an image back at once; its fence is signaled when the engine lets the image
// go. shown: the window seen, images let go at once. Hidden: kept until engine_lets_go, or never (only another image
// shown would let it go, and none is) ----
constexpr int kImages = 4;
bool shown = true;
int next_image = 0, submits = 0, remakes = 0, remake_tries = 0;
bool remake_fails = false;   // (create_swapchain makes none: a window with no size)
std::vector<VkFence> signaled, in_use;
uint64_t next_fence = 500;
template<class T> bool has(const std::vector<T>& v, T x) { for (T y : v) if (y == x) return true; return false; }
template<class T> void drop(std::vector<T>& v, T x) { for (size_t i = 0; i < v.size(); i++) if (v[i] == x) { v.erase(v.begin() + (long)i); return; } }
void engine_lets_go() { for (VkFence f : in_use) signaled.push_back(f); in_use.clear(); }
bool create_swapchain(uint32_t, uint32_t) {
    ++remake_tries;
    if (!drained) problem("the swapchain made again with the presenter not drained first");
    if (!dev.present_mu.held) problem("the swapchain made again without the queue's lock");
    if (remake_fails) return false;
    ++remakes;
    dev.swapchain_generation++, dev.swapchain++, next_image = 0;
    in_use.clear();   // (the old swapchain's images: never let go, their fences never signaled)
    return true;
}
VkResult vkAcquireNextImageKHR(int, VkSwapchainKHR sc, uint64_t, VkSemaphore, VkFence fence, uint32_t* index) {
    drained = false;
    if (sc != dev.swapchain) problem("an acquire on a swapchain made again since");
    if (!fence || has(in_use, fence) || has(signaled, fence))
        problem("the acquire's fence is signaled or still in use (VUID-vkAcquireNextImageKHR-fence-01287)");
    *index = (uint32_t)next_image;
    next_image = (next_image + 1) % kImages;
    if (shown) signaled.push_back(fence);
    else in_use.push_back(fence);
    return VK_SUCCESS;
}
VkResult vkWaitForFences(int, uint32_t, const VkFence* f, int, uint64_t timeout) {
    if (*f == 13 || has(signaled, *f)) return VK_SUCCESS;   // (13: the frame slot's own, done)
    if (timeout == UINT64_MAX) { problem("a wait without end on an image not let go"); return VK_TIMEOUT; }
    now_ns += (int64_t)timeout;
    return VK_TIMEOUT;
}
VkResult vkResetFences(int, uint32_t, const VkFence* f) {
    if (*f == 13) return VK_SUCCESS;
    if (has(in_use, *f)) problem("a fence reset while the presentation engine may still signal it");
    drop(signaled, *f);
    return VK_SUCCESS;
}
VkResult vkCreateFence(int, const VkFenceCreateInfo*, void*, VkFence* made) { *made = ++next_fence; return VK_SUCCESS; }
VkResult vkEndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
'''

FIXTURE_TAIL = r'''
void present(Frame& f) {
    const int64_t fence_t0 = 0;
    (void)fence_t0;
@@ACQUIRE@@
@@SUBMIT@@
    (void)si;
    ++submits;
}

Frame fr;
void step() { now_ns += 16 * kMs; present(fr); }   // (a frame of 16 ms, then its present)
/// Steps until a frame is copied or `limit_ns` of the clock has gone: the time it took (-1: none copied)
int64_t until_copied(int64_t limit_ns) {
    const int64_t t0 = now_ns;
    for (int s = submits; now_ns - t0 < limit_ns;) {
        step();
        if (submits != s) return now_ns - t0;
    }
    return -1;
}

int main() {
    // 1. the window seen: a copy each present, no swapchain made again
    for (int i = 0; i < 10; i++) step();
    if (submits != 10 || remake_tries) problem("the window seen: not one copy each present, or a swapchain made again");
    // 2. a hold the system ends itself: hidden 0.5 s, then the image let go
    shown = false;
    for (int64_t t0 = now_ns; now_ns - t0 < 500 * kMs;) step();
    shown = true;
    engine_lets_go();
    if (until_copied(100 * kMs) < 0) problem("the held image let go: no copy after it");
    if (remake_tries) problem("a hold the system ended itself within a second made the swapchain again");
    // 3. hidden (Alt+Tab), then seen again, the held image never let go
    shown = false;
    const int64_t hold_t0 = now_ns;
    step();
    const int s3 = submits;
    shown = true;   // (in front again: new images let go at once, the held one never)
    const int64_t took = until_copied(3000 * kMs);
    if (took < 0)
        problem("the window seen again, its held image never let go: no frame copied in 3 s, the picture frozen until the "
                "swapchain is made again by hand (F11)");
    else {
        if (now_ns - hold_t0 > 1500 * kMs) problem("the swapchain made again more than 1.5 s after the hold began");
        if (remakes != 1) problem("not one swapchain made again for the hold");
        if (!remake_lines) problem("the swapchain made again for a hold without a log line saying so");
        const int s = submits;
        for (int i = 0; i < 10; i++) step();
        if (submits != s + 10) problem("after the swapchain was made again: not one copy each present");
    }
    (void)s3;
    // 4. hidden 20 s: a few remakes, further apart; then seen again, frames shown within 5 s
    shown = false;
    const int r4 = remakes, t4 = remake_tries;
    for (int64_t t0 = now_ns; now_ns - t0 < 20000 * kMs;) step();
    const int made = remakes - r4;
    if (remake_tries - t4 > 7) problem("the window hidden 20 s: the swapchain made again more than 7 times (not kept apart)");
    if (made < 3) problem("the window hidden 20 s: the swapchain made again fewer than 3 times");
    shown = true;
    if (until_copied(5000 * kMs) < 0) problem("the window seen again after 20 s hidden: no frame in 5 s");
    for (int i = 0; i < 5; i++) step();
    // 5. a swapchain that cannot be made: not tried at every present
    shown = false;
    remake_fails = true;
    const int t5 = remake_tries;
    for (int64_t t0 = now_ns; now_ns - t0 < 5000 * kMs;) step();
    if (remake_tries - t5 > 4) problem("a swapchain that cannot be made tried again at every present");
    if (remake_tries == t5) problem("a held image never let go: no swapchain asked for in 5 s");
    if (problems) return 1;
    std::printf("PASS (%s): a hold the system ends itself left as it is; a held image never let go makes the swapchain again "
                "within 1.5 s, frames copied after it; remakes kept apart while hidden; a failed remake not retried at "
                "each present\n", PLATFORM);
    return 0;
}
'''

# (what it breaks, the source's text, what it is replaced with): each must fail
BROKEN = [
    ("no remake for a held image (main before this fix)",
     "if (held_remake_due(now - held_since, remakes)) {", "if (false) {"),
    ("the remakes not kept further apart",
     "return held_ns >= kHeldRemakeNs << (remakes < 2 ? remakes : 2);", "return held_ns >= kHeldRemakeNs;"),
    ("a failed remake tried again at every present",
     "                held_since = now;   // (not made,", "                (void)0;   // (not made,"),
]


def fixture(replay, windows):
    present = block(replay, "void present_image(VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImageView view) {")
    acquire = between(present, "    uint32_t index = 0;", "    if (vkResetCommandPool(dev.device, f.pool, 0)")
    submit = between(present, "    if (vkEndCommandBuffer(f.cmd) == VK_ERROR_DEVICE_LOST) device_lost();",
                     "    std::unique_lock qk(dev.present_mu);")
    helpers = between(replay, "static VkFence acquire_fence = VK_NULL_HANDLE;", "void present_image(VkImage src,")
    body = (FIXTURE_TAIL.replace("@@ACQUIRE@@", platform_side(acquire, windows))
            .replace("@@SUBMIT@@", platform_side(submit, windows)))
    return ('#define PLATFORM "%s"\n' % ("Windows" if windows else "Linux") + FIXTURE_HEAD
            + platform_side(helpers, windows) + body)


def build_and_run(cxx, replay, windows, temp, name, quiet=False):
    """Whether the fixture compiled and passed (None: it did not compile)"""
    cpp, exe = Path(temp) / (name + ".cpp"), Path(temp) / (name + ".exe")
    cpp.write_text(fixture(replay, windows), encoding="utf-8")
    c = subprocess.run([cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable",
                        "-Wno-unused-parameter", "-Wno-unused-function", "-Wno-missing-field-initializers", str(cpp), "-o",
                        str(exe)], capture_output=quiet, text=True)
    if c.returncode:
        print("FAIL (%s): the fixture does not compile" % name)
        return None
    return subprocess.run([str(exe)], timeout=60, capture_output=quiet, text=True).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--cxx", default="clang++")
    a = p.parse_args()
    replay = (a.source / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
    failed = False
    with tempfile.TemporaryDirectory(prefix="present-held-image-") as temp:
        for windows in (True, False):
            failed |= not build_and_run(a.cxx, replay, windows, temp, "Windows" if windows else "Linux")
        for what, old, new in BROKEN:
            if replay.count(old) != 1:
                print("FAIL: the broken variant '%s' finds its place %d times, not once (the source changed?)" % (what, replay.count(old)))
                failed = True
                continue
            ran = build_and_run(a.cxx, replay.replace(old, new), True, temp, "broken", quiet=True)
            if ran is None or ran:
                print("FAIL: the broken variant '%s' %s" % (what, "does not compile" if ran is None else
                                                            "passes: the check does not see it"))
                failed = True
            else:
                print("ok: the broken variant '%s' fails" % what)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
