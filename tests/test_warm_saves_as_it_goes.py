"""A warm writes the pipeline cache as it goes, every ten minutes, not only at its end (gpu/vk/vk_draw.cpp: warm_save_due,
warm_save_tick). No game or GPU.

Why: on a cold driver cache a preparation is hours of compiles (AMD: nightly-622's, 80 minutes in with no
hookshot_vk_pipelines.bin written yet), and the cache was written once, after the last build. A preparation killed, crashed or
stopped before its end (make_game's silence timeout, the player closing it) kept nothing, and the next one compiled it all again.

How: the real warm_save_due() is cut out of vk_draw.cpp and run over a made-up clock: the first call only sets the next save,
none is due before ten minutes, one at ten, the next ten minutes after that, and a tick that comes late saves once and counts
on from then. Then the three loops that build for many minutes (precompile_warm's noted and compute pipelines, the start's
noted pipelines in precompile_run) are checked to tick it, and warm_save_tick to save in the background, not forced (the builds
go on).

python -I -B tests/test_warm_saves_as_it_goes.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
#include <chrono>
#include <cstdio>
CUT
int main() {
    using namespace std::chrono;
    const steady_clock::time_point t0 = steady_clock::now();
    steady_clock::time_point next{};
    struct Step { int at_s; bool want; };
    // (the first tick sets the next save; none before ten minutes; one at ten; the next at twenty; a tick that comes late (at
    // 33) saves once, and counts ten minutes on from there)
    const Step steps[] = {{0, false}, {1, false}, {300, false}, {599, false}, {600, true}, {601, false}, {1199, false},
                          {1200, true}, {1980, true}, {1981, false}, {2579, false}, {2580, true}};
    int bad = 0;
    for (const Step& s : steps) {
        const bool due = warm_save_due(t0 + seconds(s.at_s), next);
        if (due != s.want) {
            std::printf("FAIL at %d s: %s\n", s.at_s, due ? "a save, none due" : "no save, one due");
            bad++;
        }
    }
    if (!bad) std::printf("ok   saves at 10, 20, 33 and 43 minutes of ticks every few seconds\n");
    return bad ? 1 : 0;
}
"""

LOOPS = (   # (the loops that build for many minutes: each ticks the save)
    ("void precompile_warm(void (*progress)(size_t done, size_t total))",
     ('prepare_progress_log("noted pipelines", done.load(), total, t0, log_at);\n        warm_save_tick(save_at);',
      'prepare_progress_log("compute pipelines", computes_done.load(), computes.size(), c0, log_at);\n        warm_save_tick(save_at);')),
    ("void precompile_run(void (*progress)(size_t done, size_t total))",
     ('prepare_progress_log("noted pipelines", done.load(), total, built0, log_at);\n        warm_save_tick(save_at);',)),
)


def check_source(draw):
    bad = []
    for marker, needs in LOOPS:
        body = block(draw, marker)
        for need in needs:
            if need not in body:
                bad.append("%s does not write the cache as it goes (%s)" % (marker.split("(")[0], need.split("(")[0]))
    if "bool warm_save_due(" not in draw:
        return bad + ["vk_draw.cpp has no warm_save_due: the cache is written at the warm's end only"]
    tick = block(draw, "void warm_save_tick(std::chrono::steady_clock::time_point& next)")
    if "pipeline_cache_save(false)" not in tick or "pipeline_cache_save(true)" in tick:
        bad.append("warm_save_tick does not save in the background (pipeline_cache_save(false)): a forced save holds the builds' loop")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    draw = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    failed = check_source(draw)
    for f in failed:
        print("FAIL " + f)
    if "bool warm_save_due(" in draw:
        start = draw.index("constexpr auto kWarmSaveEvery")
        cut = draw[start:draw.index("void warm_save_tick(", start)]
        with tempfile.TemporaryDirectory(prefix="warm-save-") as tmp:
            src = Path(tmp) / "due.cpp"
            exe = Path(tmp) / "due.exe"
            src.write_text(HARNESS.replace("CUT", cut), encoding="utf-8")
            subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
            ran = subprocess.run([str(exe)], capture_output=True, text=True)
            print(ran.stdout.strip())
            if ran.returncode != 0:
                failed.append("warm_save_due does not save every ten minutes")
    if failed:
        raise SystemExit("a warm writes the pipeline cache at its end only: " + "; ".join(failed))


if __name__ == "__main__":
    main()
