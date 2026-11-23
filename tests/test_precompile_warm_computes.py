# SPDX-License-Identifier: GPL-3.0-or-later
"""Which compute programs the start's warm builds (gpu/vk/vk_draw.cpp precompile_warm, RX_VK_LOAD=create): the precompile's plan
where RX_VK_PRECOMPILE_COMPUTES says which, or its default does ("listed" on Most and on Mesa's Intel driver),
and every kept compute program where nothing does, as before.

On an Iris Xe with 16 GB (Mesa 26.2.6, #1344) the warm built every kept compute program at a game's first start, 4,059 of
them, the driver's compiles on 12 threads, while the preparation's own warm had built the listed ones alone; Linux killed the
game for memory at 1,852. The piece of precompile_warm that picks the programs is cut out of the source and compiled with
stand-ins for the program cache and the plan:
- an unrestricted plan: every kept program;
- "listed": the plan's programs this install has, and no others; a key the plan names with no program is not one;
- a plan of every program ("all", or "listed" with no list): every kept program;
- "none": none.
No game or GPU is needed.

python -I -B tests/test_precompile_warm_computes.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import between, block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <string>
#include <vector>

@MIX@
@PLAN@
static ComputePlan plan_now;
const ComputePlan& compute_warm_plan() { return plan_now; }
static bool preparing = false;
static uint64_t program_cache_stamp() { return 42; }
static std::string precompile_built_path() { return "notes.built"; }
static bool env_on(const char*) { return preparing; }
@WRITE@
@READ@
std::vector<uint64_t> program_cache_compute_keys() { return {11, 22, 33, 44}; }


std::vector<uint64_t> warm_keys() {
@CUT@
    return computes;
}

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) std::printf("FAIL: %s\n", what), fails++;
}
static bool is(const std::vector<uint64_t>& got, std::vector<uint64_t> want) {
    std::vector<uint64_t> g = got;
    std::sort(g.begin(), g.end());
    std::sort(want.begin(), want.end());
    return g == want;
}

int main() {
    plan_now = {};
    plan_now.all = true;
    plan_now.keys = {22};
    check(is(warm_keys(), {11, 22, 33, 44}), "no switch and no default: not every kept compute program (as before)");

    plan_now = {};
    plan_now.keys = {22, 44, 99};
    check(is(warm_keys(), {22, 44}),
          "listed (Mesa's Intel driver): not the plan's programs alone (#1344: every kept one, 4,059 on an Iris Xe)");

    plan_now = {};
    plan_now.all = true;   // ("listed" with nothing listed, or "all")
    check(is(warm_keys(), {11, 22, 33, 44}), "a plan of every program: not every kept one");

    plan_now = {};
    plan_now.none = true;
    plan_now.keys = {22};
    check(warm_keys().empty(), "none: some compute programs warmed");

    plan_now = {};
    plan_now.keys = {22, 44};
    precompile_built_write(3, plan_now.coverage());
    preparing = true;
    check(precompile_covers_notes(), "same listed set misses its marker");
    plan_now.keys.insert(99);
    check(!precompile_covers_notes(), "new listed compute reused the smaller warm");
    plan_now.all = true;
    check(!precompile_covers_notes(), "All reused the Most warm");
    preparing = false;
    precompile_built_write(3, plan_now.coverage());
    preparing = true;
    plan_now.all = false;
    check(precompile_covers_notes(), "Most missed the full warm");

    FILE* f = std::fopen(precompile_built_path().c_str(), "wb");
    const uint64_t legacy[2] = {42, 3};
    std::fwrite(legacy, sizeof legacy, 1, f);
    std::fclose(f);
    check(!precompile_covers_notes(), "Builder trusted legacy compute coverage");
    preparing = false;
    check(precompile_covers_notes(), "ordinary start discarded its legacy marker");

    std::printf(fails ? "precompile warm computes: FAILED\n" : "precompile warm computes: ok\n");
    return fails ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    warm = block(draw, "void precompile_warm(void (*progress)(size_t done, size_t total)) {")
    # (from the kept compute programs to the count the progress is given: what picks them)
    cut = between(warm, "std::vector<uint64_t> computes = program_cache_compute_keys();", "const size_t all_n")
    with tempfile.TemporaryDirectory(prefix="precompile-warm-computes-") as tmp:
        src, exe = Path(tmp) / "warm.cpp", Path(tmp) / "warm"
        fixture = (FIXTURE.replace("@CUT@", cut)
                   .replace("@MIX@", block((ROOT / "gpu/vk/draw_key.h").read_text(), "inline uint64_t mix("))
                   .replace("@PLAN@", block(draw, "struct ComputePlan {") + ";")
                   .replace("@WRITE@", block(draw, "void precompile_built_write("))
                   .replace("@READ@", block(draw, "bool precompile_covers_notes() {")))
        src.write_text(fixture, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], cwd=tmp, check=True, timeout=60)


if __name__ == "__main__":
    main()
