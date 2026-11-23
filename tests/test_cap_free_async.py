"""Capture buffers destroyed off the replay's thread (gpu/vk/vk_memory.cpp, RX_VK_CAP_FREE_ASYNC): a capture buffer retired
past the spares (cap_buf_free's destroy) is handed to a thread of its own, not destroyed inside the replay's take with
pools_mu held. Measured in teleport tours on 16 cores: takes of 25-49 ms that were the retire loop alone.

destroy_buffer_later and its queue are taken from the source as they are and run with a stand-in destroy_buffer that notes the
thread it ran on:
  - every buffer handed over is destroyed once, in the order handed over, on one thread that is not the caller's;
  - at the process's exit (cap_shutting_down) what is still queued is left, never destroyed;
cap_buf_free is read as text: the destroy goes to destroy_buffer_later only with the switch and not at the exit, after the
spare list was offered the buffer (a spare is never destroyed), and the switch is on unless RX_VK_CAP_FREE_ASYNC=0.

python -I -B tests/test_cap_free_async.py [--cxx g++]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block, between   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include "gpu/vk/compat/hookshot_env.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)
void logf(const char*, ...) {}
using VkBuffer = uintptr_t;
struct VmaAllocationT {};
using VmaAllocation = VmaAllocationT*;
std::atomic<bool> cap_shutting_down{false};
std::mutex seen_mu;
std::vector<std::pair<VkBuffer, std::thread::id>> seen;
void destroy_buffer(VkBuffer b, VmaAllocation) {
    std::lock_guard lk(seen_mu);
    seen.push_back({b, std::this_thread::get_id()});
}
@ON@
@QUEUE@
size_t seen_n() { std::lock_guard lk(seen_mu); return seen.size(); }
bool wait_for(size_t n) {
    for (int i = 0; i < 2000 && seen_n() < n; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return seen_n() == n;
}
int main() {
    CHECK(cap_free_async_on());   // (on unless RX_VK_CAP_FREE_ASYNC=0)
    for (VkBuffer b = 1; b <= 50; b++) destroy_buffer_later(b, nullptr);
    CHECK(wait_for(50));
    {
        std::lock_guard lk(seen_mu);
        for (size_t i = 0; i < seen.size(); i++) {
            CHECK(seen[i].first == i + 1);                              // (each once, in order)
            CHECK(seen[i].second != std::this_thread::get_id());        // (not on the caller's thread)
            CHECK(seen[i].second == seen[0].second);                    // (one thread)
        }
    }
    cap_shutting_down = true;   // (the exit: what is queued now is left)
    for (VkBuffer b = 51; b <= 60; b++) destroy_buffer_later(b, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(seen_n() == 50);
    std::puts("ok");
    std::fflush(stdout);
    std::_Exit(0);   // (the worker waits on its queue for ever, as in the game)
}
'''


def need(text, what, needles):
    for n in needles:
        if n not in text:
            raise SystemExit("%s no longer has: %s" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = ap.parse_args()
    src = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    on = block(src, "bool cap_free_async_on() {")
    need(on, "cap_free_async_on", ['env_on_or("RX_VK_CAP_FREE_ASYNC", 1)'])
    queue = between(src, "std::mutex cap_gone_mu;", "void cap_buf_free(")
    need(queue, "destroy_buffer_later", ["void destroy_buffer_later(VkBuffer buffer, VmaAllocation alloc) {",
                                         "if (!cap_shutting_down.load()) destroy_buffer(b, a);"])
    free = block(src, "void cap_buf_free(VkBuffer buffer, VmaAllocation alloc, void* map, uint64_t size, bool from_worker = false) {")
    spare = free.find("cap_free.push_back({buffer, alloc, map, size")   # (main's spares carry their age too)
    later = free.find("if (cap_free_async_on() && !cap_shutting_down.load()) destroy_buffer_later(buffer, alloc);")
    if spare < 0 or later < 0 or later < spare:
        raise SystemExit("cap_buf_free: the destroy no longer goes to destroy_buffer_later after the spares were offered it")
    need((ROOT / "gpu/vk/switches.def").read_text(encoding="utf-8"), "switches.def", ["SW(RX_VK_CAP_FREE_ASYNC,"])
    code = FIXTURE.replace("@ON@", on).replace("@QUEUE@", queue)
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "t.cpp")
        exe = os.path.join(d, "t.exe")
        open(c, "w", encoding="utf-8").write(code)
        r = subprocess.run([a.cxx, "-std=c++20", "-O1", "-pthread", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), c, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("compile failed")
        env = dict(os.environ)
        env.pop("RX_VK_CAP_FREE_ASYNC", None)
        r = subprocess.run([exe], capture_output=True, text=True, env=env, timeout=60)
        print(r.stdout.strip() + r.stderr.strip())
        if r.returncode:
            raise SystemExit("FAIL")
    print("ok: test_cap_free_async")


if __name__ == "__main__":
    main()
