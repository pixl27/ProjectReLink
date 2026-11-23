"""The start-up screen through the first start after a preparation (Hookshot #1522). Under RX_VK_LOAD=create that start runs
two passes on the bar's one part: precompile_warm (every noted pipeline into the pipeline cache) and then
precompile_at_create (each mesh fallback program's compute pipelines, then the kept programs' noted pipelines: minutes on an
integrated GPU, 91 s on a Steam Deck). The warm reported its own total and reached it, and the bar never goes back
(PrepareBar), so the screen read 100% for the whole second pass. The at-create branch of precompile_run (gpu/vk/vk_draw.cpp)
is cut from the source and run with stand-ins for the two passes, its progress into the real PrepareBar (gpu/vk/vk_progress.h):

- after the warm, while the second pass works, the bar is under 100% and still moving; it reaches 100% at its end;
- with no steps counted for the second pass before it (none found, or the count wrong), still under 100% until it ends;
- a start whose notes are covered already runs the second pass alone, from 0;
- a --prepare-shaders run (no second pass) still ends its warm at 100%;
- a start that falls through to RX_VK_LOAD=start's path gets its own progress back;
- the second pass's steps counted before it (start_steps::create_steps) as precompile_at_create counts them: every variant
  of each key whose records program_cache_load_some(..., fallbacks) takes, those in memory already under that key, and the
  mesh fallback programs in memory under other keys (fallback_loaded.h); and the snapshot takes the records load_some takes.
No game or GPU required.

python -I -B tests/test_start_progress_at_create.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

DRIVER = r'''#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "vk_progress.h"
#include <atomic>
#include <memory>
#include <unordered_map>
#if __has_include("start_steps.h")
#include "start_steps.h"   // (the fix's one count over both passes)
#define HAVE_STEPS 1
#endif
using namespace mkwvk;
static int fails;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); fails++; } } while (0)
typedef void (*Progress)(size_t, size_t);

// (the start: its notes covered or not, a --prepare-shaders run or not; each pass a number of steps)
static bool covered, prepare_only, warm_leaves_marker = true;
static size_t warm_n = 42475, create_n = 4568, steps_guess = 4568;
static PrepareBar bar;
static std::vector<double> after_warm;   // (the bar's fraction at each tick of the second pass)
static bool in_create;
static size_t last_done, last_total, fell_through_total;

static void tick(size_t done, size_t total) {
    bar.at(2, done, total);
    last_done = done, last_total = total;
    const double f = bar.fraction();   // (the painter reads it at every frame)
    if (in_create) after_warm.push_back(f);
}
static void start_tick(size_t done, size_t total) { fell_through_total = total, (void)done; }

namespace stub {
static bool programs_load_at_create() { return true; }
static void pcache_init() {}
static bool pcache_file_rejected = false;
static bool pcache_drop_stale() { return false; }
static int remove(const char*) { return 0; }
struct Path { const char* c_str() const { return ""; } };
static Path precompile_built_path() { return {}; }
static bool precompile_covers_notes() { return covered; }
static bool env_on(const char* name) { return prepare_only && !std::strcmp(name, "HOOKSHOT_PREPARE_ONLY"); }
static bool notes_lost_by_driver() { return false; }
struct StartNoteRefs {};   // (vk_draw.cpp: the start's kept references of the notes, let go when the load at create is not run)
static StartNoteRefs start_note_refs_kept;
static bool prepare_stopped() { return false; }
static size_t program_cache_fallback_steps() { return steps_guess; }
static void precompile_warm(Progress progress) {
    for (size_t d = 0; d <= warm_n; d += 997) if (progress) progress(d, warm_n);
    if (progress) progress(warm_n, warm_n);
    covered = warm_leaves_marker && !prepare_only;
}
static void precompile_at_create(Progress progress) {
    if (!create_n) return;   // (no workers, no progress: bounded_wait's kDone)
    in_create = true;
    for (size_t d = 0; d <= create_n; d += 101) if (progress) progress(d, create_n);
    if (progress) progress(create_n, create_n);
    in_create = false;
}

static void run(Progress progress) {
    /* BRANCH */
    if (progress) progress(1, 7);   // (RX_VK_LOAD=start's own path: its own count)
}
}  // namespace stub
using stub::run;

static void reset(bool c, bool p) {
    covered = c, prepare_only = p, bar = PrepareBar{}, after_warm.clear(), last_done = last_total = 0;
    bar.skip(0), bar.skip(1), bar.guess(2, 1);
}

int main() {
    // (the first start after a preparation: the warm, then the second pass; under 100% until its end, and moving)
    reset(false, false);
    run(tick);
    CHECK(!after_warm.empty());
    bool under = true;
    for (size_t i = 0; i + 1 < after_warm.size(); i++) under = under && after_warm[i] < 1.0;
    CHECK(under);
    CHECK(after_warm.size() > 2 && after_warm[after_warm.size() - 2] > after_warm.front());
    CHECK(bar.fraction() == 1.0 && last_done == last_total);
    // (no steps counted before the second pass, while it has some: under 100% through it, 100% at its end)
    steps_guess = 0;
    reset(false, false);
    run(tick);
    CHECK(!after_warm.empty());
    under = true;
    for (size_t i = 0; i + 1 < after_warm.size(); i++) under = under && after_warm[i] < 1.0;
    CHECK(under);
    CHECK(bar.fraction() == 1.0 && last_done == last_total);
    // (and none, with nothing to do: the bar ends at 100% when the pass returns)
    create_n = 0;
    reset(false, false);
    run(tick);
    CHECK(after_warm.empty() && bar.fraction() == 1.0 && last_done == last_total);
    create_n = 4568, steps_guess = 4568;
    // (notes covered: the second pass alone, from 0)
    reset(true, false);
    run(tick);
    CHECK(!after_warm.empty() && after_warm.front() < 0.05 && bar.fraction() == 1.0);
    // (a --prepare-shaders run: the warm alone, at 100% when it ends)
    reset(false, true);
    run(tick);
    CHECK(after_warm.empty() && bar.fraction() == 1.0);
    // (a warm that leaves no marker in a start: RX_VK_LOAD=start's path, with the start's own progress)
    warm_leaves_marker = false, fell_through_total = 0;
    reset(false, false);
    run(start_tick);
    CHECK(fell_through_total == 7);
#ifdef HAVE_STEPS
    // (the second pass's steps: a step for each program given to it, one more for each mesh fallback program)
    struct Record { uint32_t kind; };
    struct Prog { bool fallback = true, failed = false; std::atomic<bool> ready{true}; };
    std::unordered_map<uint64_t, std::vector<Record>> taken = {
        {1, {{4}, {4}, {5}}},   // (three variants on disk, each a fallback: 6)
        {2, {{4}}},             // (one: 2)
        {3, {{4}, {0}}},        // (a fallback and a variant that is not: 3)
    };
    std::unordered_map<uint64_t, std::vector<std::unique_ptr<Prog>>> programs;
    const auto add = [&](uint64_t key, bool fallback, bool failed, bool ready) {
        auto p = std::make_unique<Prog>();
        p->fallback = fallback, p->failed = failed, p->ready = ready;
        programs[key].push_back(std::move(p));
    };
    add(1, true, false, true);    // (key 1's variant in memory already: given with its records, 2)
    add(3, false, false, false);  // (key 3's, not a fallback, not ready: given all the same, 1)
    add(10, true, false, true), add(10, true, false, true);   // (in memory under a key not taken: fallback_loaded, 4)
    add(11, true, false, false);  // (not ready, 0)
    add(12, true, true, true);    // (failed, 0)
    add(13, false, false, true);  // (not a fallback, 0)
    CHECK(start_steps::create_steps(taken, programs) == 6 + 2 + 3 + 2 + 1 + 4);
    CHECK(start_steps::create_steps(std::unordered_map<uint64_t, std::vector<Record>>{}, programs) == 6);   // (nothing taken: keys 1 and 10 in memory, 3 programs)
#endif
    return fails ? 1 : 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    run = block(draw, "void precompile_run(void (*progress)(size_t done, size_t total)) {")
    branch = block(run, "if (programs_load_at_create()) {")
    code = DRIVER.replace("/* BRANCH */", branch)
    with tempfile.TemporaryDirectory(prefix="start-progress-") as tmp:
        cpp, exe = Path(tmp) / "start.cpp", Path(tmp) / ("start.exe" if sys.platform == "win32" else "start")
        cpp.write_text(code, encoding="utf-8")
        r = subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                            "-Wno-unused-variable", "-I" + str(VK), str(cpp), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("FAIL: the start's progress fixture does not build:\n" + r.stdout + r.stderr)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("FAIL: the start-up bar through a first start's two passes:\n" + r.stdout + r.stderr)
    # (the snapshot takes what load_some(..., fallbacks) takes, and precompile_at_create counts what load_some gives it)
    cache = (VK / "vk_progcache.cpp").read_text(encoding="utf-8")
    some = block(cache, "size_t program_cache_load_some(")
    steps = block(cache, "size_t program_cache_fallback_steps() {")
    at_create = block(draw, "void precompile_at_create(void (*progress)(size_t done, size_t total)) {")
    for ok, what in (
            ("(fallbacks && (it->second[0].kind & 4))" in some and "fallback_loaded::each(programs, given, fn)" in some,
             "program_cache_load_some no longer takes the kind-4 records and the fallback programs in memory"),
            ("if (!kv.second.empty() && (kv.second[0].kind & 4)) taken.emplace(kv.first, kv.second);" in steps and
             "start_steps::create_steps(taken, programs)" in steps,
             "program_cache_fallback_steps does not count what program_cache_load_some(..., fallbacks) takes"),
            ("}, false, true);" in at_create and "const size_t total = pass->fallbacks.size() + pass->kept.size();" in at_create,
             "precompile_at_create counts its steps another way than start_steps::create_steps")):
        if not ok:
            raise SystemExit("FAIL: " + what)
    print("PASS: the bar counts the load-at-create pass after the warm: under 100% until it ends")


if __name__ == "__main__":
    main()
