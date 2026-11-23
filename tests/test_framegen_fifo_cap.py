"""Under FIFO the made frames do not queue: FIFO's catch-up, and NVIDIA Reflex's minimum between the game's frames (opt-in).

FIFO shows one frame a refresh. When the game's frame rate times a fixed frame generation multiplier reached the refresh,
the frames queued (on main in the driver's queue, with the timeline in the sets taken ahead and their refresh slots) and the
game frame reached the screen later and later: an RTX 5090 at 165 Hz, 3x, 40 to 51 ms after its hand-over on main and 65
to 82 ms on the timeline (the timeline's latency A/B, RX_VK_FG_LATENCY_LOG).

- vk_reflex.cpp fifo_min_interval_us gives Reflex's sleep a minimum interval: the multiplier's share of refresh -
  refresh^2/3600 (157.4 a second at 165 Hz, the rate NVIDIA's own Reflex keeps a G-SYNC display under with V-Sync), only
  under FIFO, with a fixed multiplier, frame generation on and making frames (not DLSS-G paused in the game's menus and
  movies), the refresh known and RX_VK_FG_FIFO_CAP=1 (off by default: a fixed-refresh display would repeat a frame several
  refreshes a second, and the game cannot tell VRR). reflex_sleep asks for it before Reflex's lock and
  apply_mode passes it to the driver, again when it changes.
- vk_framegen.cpp fifo_catch_up: a set whose game frame (its last) would be shown over a refresh after its own time keeps
  the most frames that bring it within a refresh, the ones left out spread over the set (fifo_pick: 6x with 3 shown, the
  2nd, 4th and 6th), at most all but one made frame out (one frame alone would leave the timeline), so 2x keeps its one
  made frame. 6x at 43 game frames a second on 165 Hz: 1 to 3 of 5 out a set, where without it the game was held to 27.5
  frames a second and each game frame shown 138 ms after its hand-over.

The real functions are cut out and run with stubs. docs/design.md section 13, rule 6.

    python -I -B tests/test_framegen_fifo_cap.py [--cxx clang++]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

STUBS = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
enum VkPresentModeKHR { VK_PRESENT_MODE_IMMEDIATE_KHR = 0, VK_PRESENT_MODE_MAILBOX_KHR = 1, VK_PRESENT_MODE_FIFO_KHR = 2 };
enum class FgMaker { None, BuiltIn, Dlss, Fsr3, DlssPaused };
struct { VkPresentModeKHR swap_mode = VK_PRESENT_MODE_FIFO_KHR; } dev;
struct { int framegen = 3; } opts;
static bool fg_on = true;
static int mhz = 165000;
static FgMaker maker = FgMaker::Dlss;
bool framegen_on() { return fg_on; }
static int runtime_most = 0;   /* (DLSS FG's runtime's most + 1: 0 no cap) */
int framegen_now() { return runtime_most && opts.framegen > runtime_most ? runtime_most : opts.framegen; }
FgMaker framegen_maker() { return maker; }
int display_refresh_mhz() { return mhz; }
bool env_value_off(const char* v) { return v && v[0] == '0'; }
"""

MAIN = r"""
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)
int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "off")) {   // (RX_VK_FG_FIFO_CAP unset or 0: off by default)
        CHECK(fifo_min_interval_us() == 0);
        return failures ? 1 : 0;
    }
    /* the cap. 165 Hz: 165 - 165^2/3600 = 157.4375 frames a second out */
    CHECK(fifo_min_interval_us() == 19056);   /* 3x: 52.5 game frames a second */
    opts.framegen = 2; CHECK(fifo_min_interval_us() == 12704);   /* 78.7 */
    opts.framegen = 4; CHECK(fifo_min_interval_us() == 25407);
    mhz = 60000; opts.framegen = 2; CHECK(fifo_min_interval_us() == 33899);   /* 60 Hz: 59 out */
    mhz = 144000; opts.framegen = 3; CHECK(fifo_min_interval_us() == 21702);   /* 144 Hz: 138.2 out */
    mhz = 165000;
    /* the multiplier as it runs: 8x saved, DLSS FG's runtime at most 6x, held as 6x */
    opts.framegen = 8; runtime_most = 6;
    CHECK(fifo_min_interval_us() == (uint32_t)std::ceil(1e6 * 6 / (165.0 - 165.0 * 165.0 / 3600.0)));
    opts.framegen = 3; runtime_most = 0;
    maker = FgMaker::BuiltIn; CHECK(fifo_min_interval_us() == 19056);   /* any maker */
    maker = FgMaker::Fsr3; CHECK(fifo_min_interval_us() == 19056);
    /* none: no frames made lately (DLSS-G paused: menus, movies), not FIFO, Auto or Off, off, the refresh not known */
    maker = FgMaker::DlssPaused; CHECK(fifo_min_interval_us() == 0);
    maker = FgMaker::None; CHECK(fifo_min_interval_us() == 0);
    maker = FgMaker::Dlss;
    dev.swap_mode = VK_PRESENT_MODE_MAILBOX_KHR; CHECK(fifo_min_interval_us() == 0);
    dev.swap_mode = VK_PRESENT_MODE_IMMEDIATE_KHR; CHECK(fifo_min_interval_us() == 0);
    dev.swap_mode = VK_PRESENT_MODE_FIFO_KHR;
    opts.framegen = 1; CHECK(fifo_min_interval_us() == 0);
    opts.framegen = 0; CHECK(fifo_min_interval_us() == 0);
    opts.framegen = 3; fg_on = false; CHECK(fifo_min_interval_us() == 0);
    fg_on = true; mhz = 0; CHECK(fifo_min_interval_us() == 0);

    /* the catch-up, ns. A 165 Hz refresh; the set before's last slot at 100 ms, so this set's first slot at 106.06 */
    const int64_t R = 6060606, ms = 1000000;
    int64_t d[8];
    const auto dues = [&](int64_t d0, int64_t step, int made) {
        for (int k = 0; k <= made; k++) d[k] = d0 + k * step;
        return d;
    };
    /* 3x at 60 game frames a second (5.56 ms apart): the game frame's slot 118.18 */
    CHECK(fifo_catch_up(2, dues(105 * ms, 5555556, 2), 100 * ms, 0, R, 100 * ms) == 0);   /* 2.07 after its time */
    CHECK(fifo_catch_up(2, dues(101010100, 5555556, 2), 100 * ms, 0, R, 100 * ms) == 0);   /* exactly a refresh: kept */
    CHECK(fifo_catch_up(2, dues(101010099, 5555556, 2), 100 * ms, 0, R, 100 * ms) == 1);   /* 1 ns more: a made frame out */
    CHECK(fifo_catch_up(2, dues(80 * ms, 5555556, 2), 100 * ms, 0, R, 100 * ms) == 1);     /* far behind: at most made - 1 */
    /* 2x keeps its one made frame; nothing made: nothing out */
    CHECK(fifo_catch_up(1, dues(80 * ms, 8333333, 1), 100 * ms, 0, R, 100 * ms) == 0);
    CHECK(fifo_catch_up(0, dues(80 * ms, 0, 0), 100 * ms, 0, R, 100 * ms) == 0);
    /* which frames are shown: spread over the set, the last the game's */
    const int p63[] = {1, 3, 5}, p65[] = {0, 1, 3, 4, 5}, p32[] = {1, 2}, p64[] = {1, 2, 4, 5};
    for (int k = 0; k < 3; k++) CHECK(fifo_pick(6, 3, k) == p63[k]);
    for (int k = 0; k < 5; k++) CHECK(fifo_pick(6, 5, k) == p65[k]);
    for (int k = 0; k < 2; k++) CHECK(fifo_pick(3, 2, k) == p32[k]);
    for (int k = 0; k < 4; k++) CHECK(fifo_pick(6, 4, k) == p64[k]);
    for (int k = 0; k < 6; k++) CHECK(fifo_pick(6, 6, k) == k);
    /* 6x at 43 game frames a second (3.88 ms apart), due on its first slot: all 6 shown leave the game frame 10.9 ms
       late; the 3rd left out, 4.9 */
    CHECK(fifo_catch_up(5, dues(106060606, 3875969, 5), 100 * ms, 0, R, 100 * ms) == 1);
    /* a refresh behind: 5 shown leave it 10.9 ms late; 4 (the 2nd, 3rd, 5th and 6th), 4.9 */
    CHECK(fifo_catch_up(5, dues(100 * ms, 3875969, 5), 100 * ms, 0, R, 100 * ms) == 2);
    /* 6x at 27 a second (6.17 ms apart, over a refresh): each frame at its own time, none out */
    CHECK(fifo_catch_up(5, dues(106060606, 6172840, 5), 100 * ms, 0, R, 100 * ms) == 0);
    /* 6x far behind: at most made - 1 = 4 out, a made frame and the game's stay */
    CHECK(fifo_catch_up(5, dues(60 * ms, 3875969, 5), 100 * ms, 0, R, 100 * ms) == 4);
    /* taken early, 2 of the set before still to show from 100: its first slot 112.12 */
    CHECK(fifo_catch_up(2, dues(110 * ms, 5555556, 2), 100 * ms, 2, R, 100 * ms) == 0);
    CHECK(fifo_catch_up(2, dues(104 * ms, 5555556, 2), 100 * ms, 2, R, 100 * ms) == 1);
    /* a line fallen behind now starts again from now (fifo_slot): late against its own time only by now */
    CHECK(fifo_catch_up(2, dues(99 * ms, 5555556, 2), 50 * ms, 0, R, 100 * ms) == 0);
    CHECK(fifo_catch_up(2, dues(90 * ms, 5555556, 2), 50 * ms, 0, R, 100 * ms) == 1);
    /* no slot yet, or the refresh not known */
    CHECK(fifo_catch_up(2, dues(50 * ms, 5555556, 2), 0, 0, R, 100 * ms) == 0);
    CHECK(fifo_catch_up(2, dues(50 * ms, 5555556, 2), 100 * ms, 0, 0, 100 * ms) == 0);
    if (failures) return 1;
    std::puts("PASS: under FIFO the game's frames at most the multiplier's share of refresh - refresh^2/3600 a second while "
              "frames are made; a set whose game frame would be over a refresh late keeps the most frames that bring it "
              "within one, the ones left out spread over the set, at most all made frames but one out, 2x none");
    return 0;
}
"""


def check_sources(reflex, framegen):
    mode = block(reflex, "void apply_mode(uint32_t minimum) {", where="gpu/vk/vk_reflex.cpp")
    assert "const uint32_t interval = want > 0 ? minimum : 0;" in mode and \
        "if (want == r.mode && interval == r.interval_us) return;" in mode and \
        "m.minimumIntervalUs = interval;" in mode and "r.interval_us = interval" in mode, \
        "apply_mode: Reflex's sleep mode carries the minimum, and is set again when it changes"
    sleep = block(reflex, "void reflex_sleep() {", where="gpu/vk/vk_reflex.cpp")
    assert "const uint32_t minimum = fifo_min_interval_us();" in sleep and "apply_mode(minimum);" in sleep and \
        sleep.index("const uint32_t minimum = fifo_min_interval_us();") < sleep.index("std::lock_guard lk(r.mu);"), \
        "reflex_sleep: the minimum asked for before Reflex's lock (it may ask Windows), passed on before each frame's sleep"
    take = block(framegen, "const auto take = [&](Taken& s, const Job& j, int room, int left)", where="gpu/vk/vk_framegen.cpp")
    assert "fifo_catch_up(j.set->made, j.set->due, slot, left, fifo_refresh, now_ns())" in take and \
        "s.shown -= s.first, stat.caught += s.first;" in take, \
        "take: the catch-up's answer leaves that many made frames out, and the log counts them"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="clang++")
    a = p.parse_args()
    vk = ROOT / "gpu" / "vk"
    reflex = (vk / "vk_reflex.cpp").read_text(encoding="utf-8")
    framegen = (vk / "vk_framegen.cpp").read_text(encoding="utf-8")
    check_sources(reflex, framegen)
    fixture = (STUBS + block(reflex, "uint32_t fifo_min_interval_us() {", where="gpu/vk/vk_reflex.cpp") + "\n"
               + block(framegen, "int fifo_pick(int frames, int shown, int k) {", where="gpu/vk/vk_framegen.cpp") + "\n"
               + block(framegen, "int fifo_catch_up(int made, const int64_t* due, int64_t slot, int left, int64_t refresh, int64_t now) {",
                       where="gpu/vk/vk_framegen.cpp") + MAIN)
    with tempfile.TemporaryDirectory(prefix="framegen-fifo-cap-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        exe = Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        env = {k: v for k, v in os.environ.items() if k != "RX_VK_FG_FIFO_CAP"}
        subprocess.run([str(exe)], check=True, env={**env, "RX_VK_FG_FIFO_CAP": "1"})
        subprocess.run([str(exe), "off"], check=True, env=env)
        subprocess.run([str(exe), "off"], check=True, env={**env, "RX_VK_FG_FIFO_CAP": "0"})


if __name__ == "__main__":
    main()
