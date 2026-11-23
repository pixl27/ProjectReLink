"""Run pool_lookup's import attempts (vk_memory.cpp) against a driver that refuses, and check what they count and log.

import_pool returns false at once, with no driver call, while ImportHolds holds a range that failed within kGapMs
(import_cooling set). pool_lookup counted every call as an import and logged "imported in" every 256th, so a range the
driver refused (AMD on Windows: VK_ERROR_INVALID_EXTERNAL_HANDLE after ~1.15 GB imported) read as 1.25 million imports in
0.0 ms, and the refusal's own "not imported" line was hidden. Here the part of pool_lookup that imports (its two passes and
what follows a failure) is cut out of the source and run with import_on_hold and import_remember_failure cut out too, over
an import_pool stand-in that asks a stub driver; the checks:

- a refusal logs "not imported" with its VkResult, counted as refused, never as an import; the lines stay capped at 16;
- a lookup of the held range asks no driver, logs no "imported in", and is counted apart (one line for the first);
- once the hold ends the import is made, and the "imported in" line counts only made imports (1, not 1000-odd), with
  the refused and held counts beside them; with nothing refused or held the line is as before;
- import_stats_log("close") says how many were made, refused and held.

No Vulkan library, device or game: python -I -B tests/test_import_log.py --headers <Vulkan headers> [--cxx clang++] [--source <tree>]
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import between, block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "import_hold.h"

namespace mkwvk {
static int fails;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); fails++; }
}
static std::vector<std::string> lines;
static void logf(const char* fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    lines.push_back(b);
}
static size_t count(const char* what) {
    size_t n = 0;
    for (const std::string& l : lines) n += l.find(what) != std::string::npos;
    return n;
}
static bool any(const char* a, const char* b) {
    for (const std::string& l : lines)
        if (l.find(a) != std::string::npos && l.find(b) != std::string::npos) return true;
    return false;
}
struct Pool { uint64_t lo = 0, hi = 0; };
/* GUESTPOOL */;
[[maybe_unused]] static int failures;
[[maybe_unused]] static std::atomic<int64_t> st_imports_made{0}, st_imports_refused{0}, st_imports_held{0};
static ImportHolds import_holds;
static std::mutex import_holds_mu;
static thread_local bool import_cooling;
static thread_local VkResult import_result = VK_SUCCESS;
static int64_t now_ms = 1000;
static int64_t import_now_ms() { return now_ms; }
/* HOLDS */
// (import_pool's contract, as vk_memory.cpp keeps it: held, it returns false at once without asking the driver; the driver
// refusing leaves its VkResult and holds the range)
static unsigned driver_calls;
static VkResult driver_answer = VK_SUCCESS;
static bool import_pool(uint64_t lo, uint64_t hi, Pool& p) {
    import_cooling = false;
    import_result = VK_SUCCESS;
    if (import_on_hold(lo, hi)) return false;
    driver_calls++;
    if (driver_answer != VK_SUCCESS) {
        import_result = driver_answer;
        import_remember_failure(lo, hi);
        return false;
    }
    p.lo = lo, p.hi = hi;
    return true;
}
// (pool_lookup with its range planned: one range, the retry pass's plan the same, so the second pass stops)
static bool attempt(uint64_t at, uint64_t size) {
    const GuestPool range{at, at + size};
    uint64_t lo = at & ~0xffffull, hi = (at + size + 0xffff) & ~0xffffull;
    auto plan = [&](bool) { return false; };
    const bool prefer_narrow = false;
    const uint64_t first_lo = lo, first_hi = hi;
    Pool p;
    bool ok = false;
    auto cut = [&]() -> Pool* {
        /* ATTEMPT */
        static Pool made;
        made = p;
        return &made;
    };
    return cut() != nullptr;
}
/* CLOSE */
} // namespace mkwvk

int main() {
    using namespace mkwvk;
    const uint64_t base = 0x80000000ull, mb = 1ull << 20;
    // The driver refuses one range (its VkResult): logged as not imported, counted as refused, not as an import.
    driver_answer = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    check(!attempt(base, mb), "a refused import gave a pool");
    check(driver_calls == 1, "the first lookup did not ask the driver once");
    check(any("not imported", "VkResult -1000072003"), "a refusal's line does not say not imported with its VkResult");
    check(count("imported in") == 0, "a refusal was logged as imported");
    check(st_imports_refused == 1 && st_imports_made == 0, "a refusal not counted as refused, or counted as made");
    // Lookups of the range while it is held: no driver call, no import counted or logged, counted apart.
    for (int i = 0; i < 1000; i++) attempt(base, mb);
    check(driver_calls == 1, "a held range asked the driver again");
    check(count("imported in") == 0, "lookups of a held range were logged as imports (every 256th, in 0.0 ms)");
    check(st_imports_held == 1000, "the lookups of a held range were not counted apart");
    check(st_imports_made == 0 && st_imports_refused == 1, "a held lookup counted as made or refused");
    check(count("held range") == 1, "the held lookups were not said once (the first), or said for each");
    // The hold ends and the driver takes it: one import, counted as one.
    now_ms += ImportHolds::kGapMs;
    driver_answer = VK_SUCCESS;
    check(attempt(base, mb), "the import after the hold was not made");
    check(driver_calls == 2 && st_imports_made == 1, "the import after the hold not counted as one made");
    for (uint64_t i = 1; i < 256; i++) attempt(base + i * 16 * mb, mb);
    check(any("imported in", "(256 imports,"), "the 256th import's line does not count 256 made imports");
    check(any("(256 imports,", "1 refused, 1000 lookups held"), "the imports line does not say how many were refused and held");
    // More refusals than the cap: each counted, the lines capped at 16 as before.
    driver_answer = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    for (uint64_t i = 0; i < 20; i++) attempt(base + (300 + i) * 16 * mb, mb);
    check(count("not imported") == 16, "the not-imported lines are not capped at 16");
    check(st_imports_refused == 21 && st_imports_made == 256, "refusals past the cap not counted");
    lines.clear();
    import_stats_log("close");
    check(any("vk: imports at close: 256 made", "21 refused by the driver, 1000 lookups waited on a held range"),
          "the close line does not say made, refused and held apart");
    if (fails) return 1;
    std::puts("PASS: import attempts counted apart: made, refused by the driver, held");
    return 0;
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--headers", type=Path, required=True)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    mem = (a.source / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    lookup = block(mem, "Pool* pool_lookup(uint64_t guest, uint64_t size) {")
    # (the two passes and what follows a failure, up to where the made pool goes into the table)
    cut = between(lookup, "for (int pass = 0; pass < 2 && !ok; pass++) {", "for (uint64_t k : overlapped) {")
    holds = block(mem, "bool import_on_hold(") + "\n" + block(mem, "void import_remember_failure(")
    # (the close line: on a tree without it, a stand-in that says nothing, so the checks above still run and fail first)
    close = block(mem, "void import_stats_log(const char* when) {") if "void import_stats_log(const char* when) {" in mem \
        else "static void import_stats_log(const char*) {}"
    fixture = (FIXTURE.replace("/* GUESTPOOL */", block(mem, "struct GuestPool {")).replace("/* HOLDS */", holds)
               .replace("/* ATTEMPT */", cut).replace("/* CLOSE */", close))
    with tempfile.TemporaryDirectory(prefix="import-log-") as temp:
        cpp = Path(temp) / "fixture.cpp"
        cpp.write_text(fixture, encoding="utf-8")
        exe = Path(temp) / "fixture.exe"
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-variable",
                        "-Wno-unused-function", "-pthread", "-I" + str(a.headers), "-I" + str(a.source / "gpu/vk"),
                        str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    main()
