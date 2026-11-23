"""DLSS's feature made before the game's first 3D frame (RX_VK_DLSS_PREWARM; no game/GPU required).

The first gameplay frame after a load made DLSS's feature (NGX's CreateFeature: 70-120 ms of the replay thread in play,
33-50 ms headless), its chain's pipelines and images and its first evaluation, all inside that frame. Now (RX_VK_DLSS_PREWARM,
on by default; 0: as before) the replay makes them in a light frame before it (the title, a loading screen), once NGX has started, for the size the game
presents; and DLSS (not DLAA) at a full-size frame takes the scaled mode that holds it, not a DLAA-mode feature made again
17 frames later. Runs with no window ask the upscalers in the background at the first recording, as the start-up screen
does (RX_VK_PROBE_HEADLESS=0: not), so test runs no longer carry NGX's ~450 ms start in their first gameplay frame.

This compiles dlss_checks.h's rule for the DLAA-mode feature and checks where the renderer calls the pieces.

python -I -B tests/test_dlss_prewarm.py [--cxx g++]
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
VK = ROOT / "gpu/vk"

RULE = r"""
#include <cstdio>
#include "dlss_checks.h"
namespace dc = mkwvk::dlss_checks;
int main() {
    int bad = 0;
    auto check = [&](const char* what, bool ok) { if (!ok) std::printf("FAIL: %s\n", what), bad++; };
    // DLAA asked: its feature at a full-size frame, with the switch or not
    check("DLAA, full size", dc::dlaa_feature(true, true, false) && dc::dlaa_feature(true, true, true));
    // DLSS at a full-size frame: DLAA's mode without the switch (as before), the scaled mode with it
    check("DLSS full size, no switch", dc::dlaa_feature(true, false, false));
    check("DLSS full size, the switch", !dc::dlaa_feature(true, false, true));
    // a smaller frame never takes DLAA's mode here (mode_for picks the scaled one)
    check("smaller frame", !dc::dlaa_feature(false, false, false) && !dc::dlaa_feature(false, true, true));
    // the scaled modes' ranges hold the full size at 2560x1440 (MaxQuality's 1280x720 to 2560x1440): taken by it
    dc::ModeRange r[2] = {{1280, 720, 2560, 1440}, {854, 480, 1280, 720}}, took;
    check("full size in MaxQuality's range", dc::mode_for(2560, 1440, 2560, 1440, r, 2, &took) == 0);
    if (bad) return 1;
    std::puts("dlss prewarm: ok");
    return 0;
}
"""


def check_source():
    fail = []
    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    temporal = (VK / "vk_temporal.cpp").read_text(encoding="utf-8")
    dlss = (VK / "vk_dlss.cpp").read_text(encoding="utf-8")
    menu = (VK / "vk_menu.cpp").read_text(encoding="utf-8")
    # made at a recording's start (its first command buffer), before the submission's commands
    submit = block(replay, "bool run_submit(const uint64_t* handles, int n, uint64_t value, uint64_t queue, bool memory_work, int cb = 0,")
    if "temporal_prewarm(first.cmd);" not in submit:
        fail.append("run_submit does not call temporal_prewarm with the recording's first command buffer")
    elif not submit.index("begin_recording(tag);") < submit.index("temporal_prewarm(first.cmd);") < submit.index("replay_handle(cmd,"):
        fail.append("temporal_prewarm is not between the recording's start and its commands")
    tick = block(replay, "void status_tick()")
    if "temporal_note_frame(draws - l_draws);" not in tick:
        fail.append("status_tick does not give the prewarm the frame's draws")
    if "temporal_note_present(t->d.width, t->d.height);" not in block(replay, "void do_present(const nvn_tex* t, const int crop[4], float cover)"):
        fail.append("do_present does not note the presented size")
    warm = block(temporal, "void temporal_prewarm(VkCommandBuffer cmd)")
    for needle in ("dev.offscreen", '"RX_VK_PROBE_HEADLESS"', "upscalers_probe_start();", "temporal_prewarm_on()",
                   "s.asked_frame.load(", ">= 200) return;", "ur::is_dlss(mode)", "dlss_ngx_ready()", "std::lock_guard lk(s.mu);",
                   "engine.e->prepare(cmd,", "s.drawing = engine.e;", "s.last_frame = 0;", "vkCmdClearColorImage(",
                   "warm.reset = true;", "engine.e->evaluate(cmd, warm)",
                   # (made again for a row or size changed before play, once the row has held 30 frames; four at most)
                   "mode == warmed_mode && pic == warmed_pic", "frame - seen_since < 30", "++warmed_n >= 4"):
        if needle not in warm:
            fail.append("temporal_prewarm no longer has: " + needle)
    # (the probe's own run must not be waited for: NGX's start holds its lock ~450 ms)
    if warm.index("dlss_ngx_ready()") > warm.index("std::lock_guard lk(s.mu);"):
        fail.append("temporal_prewarm asks NGX after taking the chain's lock")
    ready = block(dlss, "int dlss_ngx_ready() {", unique=False)   # (the first: NGX's; the second is the build without it)
    if "std::try_to_lock" not in ready or "std::lock_guard" in ready:
        fail.append("dlss_ngx_ready waits for NGX's lock")
    on = block(temporal, "bool temporal_prewarm_on()")
    if 'env_on_or("RX_VK_DLSS_PREWARM", 1)' not in on:
        fail.append("temporal_prewarm_on: not on by default with RX_VK_DLSS_PREWARM=0 (the switches' one rule) as the way back")
    prep = block(dlss, "int prepare(VkCommandBuffer cmd, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, bool native_aa,")
    if "dlss_checks::dlaa_feature(full, native_aa, temporal_prewarm_on())" not in prep:
        fail.append("prepare does not choose DLAA's mode by dlss_checks::dlaa_feature")
    if "k < 0 && full" not in prep:
        fail.append("prepare: a full-size DLSS frame no scaled mode holds no longer falls back to DLAA's mode")
    if "void upscalers_probe_start() { probe_upscalers_soon(); }" not in menu:
        fail.append("vk_menu.cpp: upscalers_probe_start is not the start-up screen's probe")
    if fail:
        raise SystemExit("\n".join(fail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dlss-prewarm-") as tmp:
        src = Path(tmp) / "t.cpp"
        src.write_text(RULE, encoding="utf-8")
        exe = Path(tmp) / "dlss-prewarm.exe"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(VK), str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    check_source()


if __name__ == "__main__":
    main()
