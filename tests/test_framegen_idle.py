"""When frame generation's images are let go (gpu/vk/framegen_idle.h): after 120 presents in a row with it off, never on a present
that recorded made frames, and the call after the present's queue (gpu/vk/vk_replay.cpp). FSR 3.1's own images by the present's
own use of FSR (fsr3fi::generate called), not by any recorded set.

Compiles the rule with a small fixture and runs it through present sequences, among them the one a review found: off for
119 presents, then a present that records made frames with it on and reads it Off before the idle check. Before, that present
counted as the 120th off and drained, waiting for its own set, which only the queue call after it would hand to the display thread.
And the one its second review found: FSR's count was fed `recorded`, true for a set the built-in interpolation or DLSS-G made, so
after FSR had made its images and another one was chosen, its count stayed 0 and they were kept for the run.

python tests/test_framegen_idle.py [--cxx clang++]
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''#include <cstdio>
#include "framegen_idle.h"
using namespace framegen_idle_rule;
static int fails;
static void expect(bool got, bool want, const char* what) {
    if (got != want) { std::printf("FAIL: %s\n", what); fails++; }
}
// A run of presents: on (the row), rec (made frames recorded); true where the images are let go at the last one
static bool run(int n_off, bool last_on, bool last_rec) {
    uint32_t off = 0;
    for (int i = 0; i < n_off; i++) off = next_off(off, false, false);
    off = next_off(off, last_on, last_rec);
    return release_due(off, true);
}
int main() {
    expect(run(119, false, false), true, "the 120th present off lets them go");
    expect(run(118, false, false), false, "the 119th does not");
    expect(run(119, false, true), false, "a present that recorded made frames, the row Off by the check: not let go (the hang)");
    expect(run(119, true, false), false, "a present with the row on: not let go");
    expect(release_due(500, false), false, "nothing made: nothing to let go");
    {   // (and after a recorded present the count starts again: 120 more off are needed)
        uint32_t off = 0;
        for (int i = 0; i < 119; i++) off = next_off(off, false, false);
        off = next_off(off, false, true);
        for (int i = 0; i < 119; i++) off = next_off(off, false, false);
        expect(release_due(off, true), false, "119 off after a recorded present");
        off = next_off(off, false, false);
        expect(release_due(off, true), true, "the 120th off after a recorded present");
    }
    // FSR 3.1's own images (step: framegen_idle's wiring of the two counts). other: a present recorded by the built-in
    // interpolation or DLSS-G with FSR not chosen (on, FSR not wanted, recorded, FSR not used)
    const Present fsr{true, true, true, true, true}, other{true, false, true, false, true};
    {   // (FSR for 1 present, then another one: 119 presents keep FSR's images, the 120th lets them go; the shared ones stay)
        Counts c;
        step(c, fsr);
        bool early = false;
        for (int i = 0; i < 119; i++) early |= step(c, other).fsr;
        expect(early, false, "FSR, then 119 presents by another: FSR's images kept");
        const Release r = step(c, other);
        expect(r.fsr, true, "FSR, then the 120th present by another: FSR's images let go (the counter the review found)");
        expect(r.shared, false, "... and the shared images, in use by the other, kept");
    }
    {   // (an FSR present at 119 that made nothing shown: a reset, or its history alone, generate called all the same)
        Counts c;
        step(c, fsr);
        for (int i = 0; i < 119; i++) step(c, other);
        const Present history{true, false, true, true, true};   // (the row moved on by the check; generate was called)
        expect(step(c, history).fsr, false, "an FSR reset or history-only present at 119: kept, its count started again");
        bool early = false;
        for (int i = 0; i < 119; i++) early |= step(c, other).fsr;
        expect(early, false, "119 presents by another after it: kept");
        expect(step(c, other).fsr, true, "the 120th: let go");
    }
    {   // (the row changed while the present recorded with FSR: Off, or another one, by the check)
        Counts c;
        for (int i = 0; i < 119; i++) step(c, other);
        c.fsr_off = 119;
        const Release to_off = step(c, Present{false, false, true, true, true});
        expect(to_off.fsr, false, "FSR used, the row Off by the check: FSR's images kept (in this present's recording)");
        expect(to_off.shared, false, "... and the shared ones (its set not shown yet)");
        Counts d;
        d.fsr_off = 119;
        expect(step(d, Present{true, false, true, true, true}).fsr, false, "FSR used, another one chosen by the check: kept");
    }
    {   // (FSR chosen and asked, but no inputs this present: generate not called; the row keeps them)
        Counts c;
        c.fsr_off = 500;
        expect(step(c, Present{true, true, true, false, true}).fsr, false, "FSR chosen, no inputs this present: kept");
    }
    {   // (frame generation off: both go after 120 presents, FSR's too)
        Counts c;
        step(c, fsr);
        const Present off{false, false, false, false, true};
        for (int i = 0; i < 119; i++) step(c, off);
        const Release r = step(c, off);
        expect(r.fsr && r.shared, true, "off for 120 after FSR: both let go");
    }
    if (!fails) std::puts("PASS: frame generation's images go after 120 presents off, never on a present that recorded; "
                          "FSR 3.1's by its own use");
    return fails ? 1 : 0;
}
'''


def check_order():
    """The idle check after the queue call, with this present's recording, in the present's path"""
    replay = (ROOT / "gpu" / "vk" / "vk_replay.cpp").read_text(encoding="utf-8")
    queue = replay.index("framegen_queue(value);")
    idle = replay.index("framegen_idle(interpolated);")
    if not queue < idle < queue + 900:
        raise SystemExit("test_framegen_idle: framegen_idle(interpolated) is not right after framegen_queue(value)")
    fg = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    idle_fn = block(fg, "void framegen_idle(bool recorded)")
    # (the rule's step, with this present's use of FSR in Present's fsr_used place, not recorded; and the flag cleared for the next)
    for part in ("std::exchange(fsr_given, false)",
                 "step(counts, {on, on && fsr3fi::frames_wanted(), recorded, fsr_used, fg.w != 0})",
                 "if (due.fsr) fsr3fi::release();", "if (!due.shared) return;"):
        if part not in idle_fn:
            raise SystemExit("test_framegen_idle: framegen_idle does not use the rule: %s" % part)
    # (the flag set in fsr3fi_frames once its inputs are taken, before each of its generate calls: a reset or its history alone too)
    frames = block(fg, "int fsr3fi_frames(")
    given = frames.find("fsr_given = true;")
    taken = frames.find("fsr3_inputs::take(f)")
    calls = [i for i in range(len(frames)) if frames.startswith("fsr3fi::generate(", i)]
    if given < 0 or not 0 <= taken < given or len(calls) != 2 or not all(given < i for i in calls):
        raise SystemExit("test_framegen_idle: fsr3fi_frames does not mark FSR's use before each generate call")
    if fg.count("fsr3fi::generate(") != 2:
        raise SystemExit("test_framegen_idle: fsr3fi::generate called outside fsr3fi_frames: mark its use there too")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="clang++", help="C++ compiler")
    a = p.parse_args()
    check_order()
    cxx = shutil.which(a.cxx)
    if not cxx:
        raise SystemExit("test_framegen_idle: no C++ compiler %r (pass --cxx)" % a.cxx)
    with tempfile.TemporaryDirectory(prefix="framegen-idle-") as temp:
        d = Path(temp)
        src, exe = d / "fixture.cpp", d / "fixture.exe"
        src.write_text(FIXTURE, encoding="utf-8")
        subprocess.run([cxx, "-std=c++17", "-O2", "-I", str(ROOT / "gpu" / "vk"), str(src), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit("test_framegen_idle: failed")


if __name__ == "__main__":
    main()
