"""The Frame generation row stops at DLSS Frame Generation's runtime while DLSS FG makes the frames.

NGX tells the most frames its runtime makes between two of the game's (MultiFrameCountMax: 5 on an RTX 50's, 1 on an
RTX 40's). Past it NGX was asked for no set at all (vk_dlss.cpp dlss_fg_frames: FG_SKIP_TOO_MANY) and the renderer's own
frame generation made every frame, with DLSS FG chosen: 7x and 8x on an RTX 50 were never DLSS FG. Now the fixed
multiplier frame generation runs at, framegen_now (gpu/vk/vk_framegen.cpp), is the row's, but at most the runtime's most
+ 1 while DLSS FG makes the frames; Off and Auto as they are; a saved value past it is kept (another engine runs it) and
is shown and run as that top.

The real framegen_capped is compiled and run: 6x at most on a runtime that makes 5, 2x on one that makes 1, the row's
value while the most is not known (0: NGX not asked yet, or another engine). The text checks: vk_dlss.cpp's
dlss_fg_most gives the runtime's most only while DLSS-G is asked for and can run, from a temporal Upscaler's inputs (0 in a
build without it);
multiplier(), swapchain_framegen(), swapchain_dlssg(), create_swapchain's set_too_big and present_image's swapchain key
take framegen_now(); the row shows the capped value and its arrows step from it, up to the top (past it round to Off, as
past 8x); the control bridge's value is set as asked. At Auto the row names who makes its frames (FSR 3.1 FG or Built-in
FG: DLSS FG makes fixed multipliers only). The real hn_framegen is compiled and run too: its 2x hint names AMD FSR 3.1
only where FSR 3.1 makes the frame, not while DLSS FG makes it (an RTX 40 with DLSS FG at a fixed multiplier, on a
temporal Upscaler, runs at 2x).

    python -I -B tests/test_framegen_dlss_cap.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import between, block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

MAIN = r"""
#include <cstdio>
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)
int main() {
    /* an RTX 50's runtime: 5 made frames at most, so 6x */
    CHECK(framegen_capped(8, 5) == 6 && framegen_capped(7, 5) == 6 && framegen_capped(6, 5) == 6);
    CHECK(framegen_capped(5, 5) == 5 && framegen_capped(2, 5) == 2);
    /* an RTX 40's: 1 at most, so 2x */
    CHECK(framegen_capped(8, 1) == 2 && framegen_capped(4, 1) == 2 && framegen_capped(2, 1) == 2);
    /* not known yet, or another engine: the row's */
    CHECK(framegen_capped(8, 0) == 8 && framegen_capped(3, 0) == 3);
    /* Off and Auto as they are */
    CHECK(framegen_capped(0, 5) == 0 && framegen_capped(1, 5) == 1 && framegen_capped(1, 1) == 1);
    if (failures) return 1;
    std::puts("PASS: DLSS FG's multiplier stops at its runtime's most + 1");
    return 0;
}
"""

# (the Frame generation row's hint, the real hn_framegen: at 2x it names AMD FSR 3.1 only where FSR 3.1 makes the frame.
# FSR 3.1 is asked with DLSS FG chosen too (fsr3fi::frames_wanted), for the sets DLSS-G does not make: with DLSS-G making
# them (dlss_fg_most above 0) the 2x hint said "by AMD FSR 3.1" on every RTX 40, whose DLSS FG stops at 2x)
HINT = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)
static uint32_t fsr_wanted = 0;
static int most = 0;
namespace fsr3fi { uint32_t frames_wanted() { return fsr_wanted; } }
int dlss_fg_most() { return most; }
/* CAPPED */
/* HINT */
static bool says_fsr(const char* t) { return t && std::strstr(t, "AMD FSR 3.1"); }
int main() {
    fsr_wanted = 1, most = 1;   /* an RTX 40, DLSS FG making the frames (FSR 3.1 asked for the sets it does not make) */
    CHECK(!says_fsr(hn_framegen(2, -1)) && std::strstr(hn_framegen(2, -1), "One made frame between each two"));
    CHECK(!says_fsr(hn_framegen(8, -1)));   /* (8x saved: shown and hinted as its 2x) */
    fsr_wanted = 1, most = 5;   /* an RTX 50 */
    CHECK(!says_fsr(hn_framegen(2, -1)) && std::strstr(hn_framegen(8, -1), "Five made frames"));
    fsr_wanted = 1, most = 0;   /* FSR 3.1 makes it: FSR 3.1 FG chosen, or DLSS FG that does not run here */
    CHECK(says_fsr(hn_framegen(2, -1)));
    fsr_wanted = 0, most = 0;   /* the built-in one */
    CHECK(!says_fsr(hn_framegen(2, -1)));
    /* the texts every value can show (the help box's height, measure_help: its loop stops at nullptr): the FSR line is
       still one of them at 2x, also while DLSS FG makes the frames */
    CHECK(says_fsr(hn_framegen(2, 1)) && hn_framegen(3, 1) == nullptr && !says_fsr(hn_framegen(2, 0)));
    CHECK(hn_framegen(2, 2) == nullptr);
    fsr_wanted = 1, most = 1;
    CHECK(says_fsr(hn_framegen(2, 1)) && hn_framegen(2, 2) == nullptr && !says_fsr(hn_framegen(2, 0)));
    if (failures) return 1;
    std::puts("PASS: the 2x hint names AMD FSR 3.1 only where FSR 3.1 makes the frame");
    return 0;
}
"""


def check_sources(framegen, main_cpp, dlss, menu, replay):
    mult = block(framegen, "int multiplier() {", where="gpu/vk/vk_framegen.cpp")
    assert "framegen_now()" in mult and "opts.framegen" not in mult, \
        "multiplier: frame generation runs at framegen_now(), the row's capped by DLSS FG's runtime"
    now = block(framegen, "int framegen_now() {", where="gpu/vk/vk_framegen.cpp")
    assert "framegen_capped(opts.framegen, dlss_fg_most())" in now, "framegen_now: the row's value, capped while DLSS FG makes them"
    sf = block(main_cpp, "int swapchain_framegen() {", where="gpu/vk/vk_main.cpp")
    assert "framegen_now()" in sf and "opts.framegen" not in sf, "swapchain_framegen: the images for the capped multiplier"
    sd = block(main_cpp, "bool swapchain_dlssg() {", where="gpu/vk/vk_main.cpp")
    assert "dlss_fg_runs(framegen_now() - 1)" in sd, "swapchain_dlssg: V-Sync Auto's FIFO for DLSS-G at the capped multiplier"
    assert "(uint32_t)std::max(framegen_now(), 0) > caps.maxImageCount" in main_cpp, \
        "create_swapchain: set_too_big is the set as it runs (a saved 8x that DLSS FG makes at 6x is a set of 6)"
    key = between(replay, "const int framegen = framegen_now();", "const bool vsync_changed")
    assert "(framegen > 4 ? framegen * 256 : 0)" in key and "opts.framegen" not in key, \
        "present_image: the swapchain is made again when the multiplier as it runs changes past 4x"
    most = [b for b in dlss.split("int dlss_fg_most() {")[1:]]
    assert len(most) == 2, "vk_dlss.cpp: dlss_fg_most with NVIDIA's SDK and without it"
    real = block(dlss, "int dlss_fg_most() {\n    if (!fg_asked()", where="gpu/vk/vk_dlss.cpp")
    for part in ("fg_asked()", "dlss_why() != DLSS_OK", "!temporal_wanted()", "start_failed", "most_known"):
        assert part in real, "dlss_fg_most: " + part
    assert "int dlss_fg_most() { return 0; }" in dlss, "vk_dlss.cpp: 0 in a build without NVIDIA's SDK"
    frames = block(dlss, "int dlss_fg_frames(VkCommandBuffer cmd,", where="gpu/vk/vk_dlss.cpp")
    too_many = frames[frames.index("if ((unsigned)count > g.most) {"):]
    assert too_many.index("if (!temporal_wanted()) return g.no_inputs++, skip(FG_SKIP_INPUTS);") < \
        too_many.index("skip(FG_SKIP_TOO_MANY)"), \
        "dlss_fg_frames: without a temporal Upscaler (no cap) the reason is the inputs, not the runtime's most"
    tx = block(menu, "const char* tx_framegen(int v) {", where="gpu/vk/vk_menu.cpp")
    assert "framegen_capped(v, dlss_fg_most())" in tx, "tx_framegen: a saved value past the top shows as the top"
    assert "framegen_maker()" in tx and '"Auto (FSR 3.1 FG)"' in tx and '"Auto (Built-in FG)"' in tx, \
        "tx_framegen: Auto names who makes its frames (DLSS FG makes fixed multipliers only)"
    row = block(menu, "void set_row(Row& r, int v, int dir, bool keep = true) {", where="gpu/vk/vk_menu.cpp")
    assert "if (r.value == &opts.framegen && dir && v == *r.value + dir * r.step) {" in row and \
        "framegen_capped(*r.value, most)" in row and \
        "top = framegen_capped(r.hi, most)" in row, "set_row: Frame generation's arrows step from the value shown, up to DLSS FG's top"
    bridge = block(menu, "int menu_setting_set(", where="gpu/vk/vk_menu.cpp")
    assert "r.value == &opts.framegen ? 0 :" in bridge, "menu_setting_set: Frame generation's value as asked, not an arrow's step"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    vk = ROOT / "gpu" / "vk"
    framegen = (vk / "vk_framegen.cpp").read_text(encoding="utf-8")
    main_cpp = (vk / "vk_main.cpp").read_text(encoding="utf-8")
    dlss = (vk / "vk_dlss.cpp").read_text(encoding="utf-8")
    menu = (vk / "vk_menu.cpp").read_text(encoding="utf-8")
    replay = (vk / "vk_replay.cpp").read_text(encoding="utf-8")
    capped = block(framegen, "int framegen_capped(", where="gpu/vk/vk_framegen.cpp")
    fixture = "#include <algorithm>\n" + capped + MAIN
    hint = HINT.replace("/* CAPPED */", capped).replace(
        "/* HINT */", block(menu, "const char* hn_framegen(int v, int variant) {", where="gpu/vk/vk_menu.cpp"))
    check_sources(framegen, main_cpp, dlss, menu, replay)
    with tempfile.TemporaryDirectory(prefix="framegen-dlss-cap-") as tmp:
        for name, text in (("fixture", fixture), ("hint", hint)):
            cpp = Path(tmp) / (name + ".cpp")
            exe = Path(tmp) / name
            cpp.write_text(text, encoding="utf-8")
            subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
