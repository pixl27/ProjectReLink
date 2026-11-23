"""The shader preparation's bar moves while the compute programs are prepared (gpu/vk/vk_progress.h PrepareWork, gpu/vk/vk_draw.cpp
prepare_run, gpu/vk/vk_main.cpp).

prepare_run takes the compute programs first, each with its texture variants compiled by the driver: seconds each, minutes on
Intel's drivers. It counted each as one program of the seed's 38,809, so the 2,186 of them, most of the time on such a PC, moved
the programs' part of the bar by under 6%. With Compile shaders on Most (the default) the predicted pipelines' guess, half again
the programs, also stayed in the bar's total until the programs were done, though no prediction runs. The launcher's "Prepare
shaders" row showed 0% after 43 minutes on an Arc A730.

Checks, with vk_progress.h compiled on its own and the bar as the launcher reads it (PrepareBar's fraction):
- the compute programs, all done, are a good share of the programs' count, and their driver compiles move it variant by variant
- the count is under the programs until all the work is done, never goes back, and is all of them at the end
- a run that compiles nothing for the driver counts as before: a program each
- on Most, with the prediction out of the total from the start, the compute programs done show at least a tenth of the step
And in the source: prepare_run reports its count through PrepareWork, its pass apart onto the same count, and both starts (the
--prepare-shaders run and the screen's) leave the prediction's guess out on Most before the programs run. No game or GPU.

python -I -B tests/test_prepare_work.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block, between  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''#include "vk_progress.h"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
using namespace mkwvk;
int main() {
    // (the shipped seed: 38,809 programs, 2,186 of them compute programs, prepared first)
    const size_t programs = 38809, computes = 2186;
    const PrepareWork w(programs, computes, true);
    // the compute programs done, translated and compiled: a good share of the count, not one program each
    const uint64_t computes_done = computes * (PrepareWork::kGraphics + w.warm_each);
    CHECK(w.done(computes_done) * 3 >= programs);
    // half of them done moves it too, and a few compiled variants do
    CHECK(w.done(computes_done / 2) > 0 && w.done(computes_done / 2) < w.done(computes_done));
    CHECK(w.done(PrepareWork::kGraphics * 13) > w.done(0));
    // under the programs until all the work is done, never back, all of them at the end
    size_t last = 0;
    for (uint64_t at = 0; at < w.whole; at += 97) {
        const size_t d = w.done(at);
        CHECK(d < programs && d >= last);
        last = d;
    }
    CHECK(w.done(w.whole - 1) < programs && w.done(w.whole) == programs && w.done(w.whole + 5) == programs);
    // nothing compiled for the driver (in play, or a start after one that crashed in it): a program each, as before
    const PrepareWork plain(programs, computes, false);
    CHECK(plain.warm_each == 0 && plain.whole == 3 * programs);
    for (size_t k : {size_t(0), size_t(1), size_t(2186), size_t(20000), size_t(38808)}) CHECK(plain.done(3 * k) == k);
    CHECK(PrepareWork(0, 0, true).done(0) == 0);
    // the bar the launcher shows, on Most: no prediction in the total, the compute programs done are a tenth of the step or more
    {
        PrepareBar bar;
        bar.guess(0, programs), bar.guess(1, programs + programs / 2), bar.guess(2, 30000);
        bar.skip(1);
        bar.at(0, w.done(computes_done), programs);
        CHECK(bar.fraction() >= 0.1);
    }
    std::printf("PASS\n");
}
'''


def check_source():
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    run = block(draw, "bool prepare_run(void (*progress)(size_t done, size_t total), bool background, bool refresh_states) {")
    for need, why in (("const PrepareWork work_of(programs, warmed_computes, warm_cs);", "prepare_run does not count by PrepareWork"),
                      ("progress(work_of.done(work.load()), programs)", "prepare_run reports a program each, not its cost"),
                      ("warm_computes_apart(apart_progress, warm_compute, only)",
                       "the pass apart reports a count of its own on the programs' part")):
        if need not in run:
            raise SystemExit("FAIL: " + why)
    if "progress(done.load() / 3, programs)" in run:
        raise SystemExit("FAIL: prepare_run still reports the programs one each")
    main = (ROOT / "gpu/vk/vk_main.cpp").read_text(encoding="utf-8")
    screen = between(block(main, "void prepare_with_screen() {"), "std::thread worker(", "prepare_run(prep_progress, false)")
    if "if (!compile_all()) prep_skip[1].store(true);" not in screen:
        raise SystemExit("FAIL: the screen's start keeps the prediction's guess in the total on Most while the programs run")
    only = between(block(main, "bool started_up() {"), "if (prepare_only()) {", "prepare_run(progress, false, true)")
    if "if (!compile_all()) bar.skip(1);" not in only:
        raise SystemExit("FAIL: --prepare-shaders keeps the prediction's guess in the total on Most while the programs run")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="prepare-work-") as tmp:
        cpp, exe = Path(tmp) / "work.cpp", Path(tmp) / "work.exe"
        cpp.write_text(FIXTURE, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "gpu/vk"), str(cpp), "-o", str(exe)],
                       check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("FAIL: the preparation's count:\n" + r.stdout + r.stderr)
    print("PASS: the shader preparation's bar counts the compute programs at their cost, and leaves the prediction out on Most")


if __name__ == "__main__":
    main()
