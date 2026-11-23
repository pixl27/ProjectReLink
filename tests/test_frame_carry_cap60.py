"""At the default Frame rate cap of 60, frames slower than 60 fps keep the game's counters at the clock (HOS_FRAME_CARRY).

With Game speed Real time, host/nn_vi.c realtime_step gives the engine the rate of the last frames (+0x2c8, 15 to 60), and the
game steps 60 / rate a frame: 1.30 at 46 fps. The counters the game cuts to whole frames (nx2/oot_patches.py patch_whole_frames:
the time of day, the water's ripple substeps, the floor web, the Scarecrow's Song) and the clock of a loop of one frame or less
(patch_loop_steps, patch_loop_pose: the frozen Zoras) carried their fractions only while the frames' own time is the step, a cap
other than 60. At a cap of 60 the cut threw the fraction away: 46 frames of 1.30 counted 46 whole frames, not the 60 that a
second at 60 fps counts, and the day ran at 77 %.

Here the host's own code (realtime_step and the carries, cut out of host/nn_vi.c) is driven by presents at a fixed frame rate,
and each frame's counters take the step the game makes from the rate it was given:
- 46 and 50 fps at cap 60: a second hands out 60 whole frames, the time of day 120 game seconds, the water's 4 substeps 240,
  and a loop of one frame takes whole steps that add up to the clock, its pose the loop's start;
- a whole step (60, 30, 20, 15 fps: steps 1, 2, 3, 4) is cut as the game cuts it, a fraction of an amount included, as before;
- HOS_FRAME_CARRY=0: cut as before (46 a second at 46 fps);
- Game speed Per frame: the step is 1, cut as before;
- a cap above 60 is unchanged: with the frames' own time (the hook) the carries on as before; with HOS_FPS_STEP=0 (the rate up to
  the cap, steps under 1) the cut as before;
- HOS_FPS_LOG: the "vi: whole-frame counters" line says what the counters took against the step (1.000 kept, 0.77 cut at 46 fps).

python -I -B tests/test_frame_carry_cap60.py [--cc gcc]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cc", "gcc"]
ROOT = Path(__file__).resolve().parents[1]
ENV_ON_H = str(ROOT / "gpu" / "vk" / "compat" / "hookshot_env.h")

FIXTURE = r'''#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
static int fails;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); fails++; } } while (0)
typedef void Ctx;
typedef int SRWLOCK;   /* (one thread here) */
#define SRWLOCK_INIT 0
#define AcquireSRWLockExclusive(l) ((void)(l))
#define ReleaseSRWLockExclusive(l) ((void)(l))
static uint8_t guest[0x4000];
#define GP(a) ((void *)(guest + (a)))
#define MOD 0x1000u
static int64_t now = 1000000000LL;
static char whole_line[256];   /* (the last "vi: whole-frame counters" line: HOS_FPS_LOG) */
static void hos_log(const char *fmt, ...) {
    char text[256];
    va_list args; va_start(args, fmt); vsnprintf(text, sizeof text, fmt, args); va_end(args);
    if (!strncmp(text, "vi: whole-frame counters", 24)) memcpy(whole_line, text, sizeof text);
}
int hos_ini_option(const char *key, int otherwise) { (void)key; return otherwise; }
volatile uint64_t hos_gfx_module = MOD;
volatile int hos_fixed_resolution, hos_dynres_runs, hos_scene_rgba16f;
volatile uint32_t hos_fixed_scale_bits;
volatile float hos_text_scale, hos_camera_scale;
@OPTS@
int rx_option(int id, int otherwise) { return id == 0 && getenv("T_PER_FRAME") ? 0 : otherwise; }
volatile uint64_t hos_step_probe = 0x2000, hos_res_module;
volatile uint32_t hos_frame_seconds_bits;
static int64_t hos_now_ns(void) { return now; }
@HOST@
static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static float real(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
/* (the game's step, as the engine makes it at its two divides: 60 / the rate it was given, or 60 * the frame's seconds with the
   hook) */
static float game_step(void) {
    const uint32_t s = hos_frame_seconds_bits;
    return s ? 60.0f * real(s) : 60.0f / (float)*(int *)GP(MOD + 0x2c8);
}
typedef struct { uint32_t frames, day, water; float loop, loop_steps; int loop_whole, pose_start; } Second;
/* (one second of frames at `fps`: each frame the counters take this frame's step, then the frame is presented) */
static Second second(int fps, int count) {
    Second s = {0, 0, 0, 0.0f, 0.0f, 1, 1};
    for (int i = 0; i < count; i++) {
        const float step = game_step();
        s.frames += hos_frame_whole(0x7000, bits(step));                /* (a counter of whole frames: the floor web, the song) */
        s.day += hos_frame_whole(0x7100, bits(2.0f * step));            /* (the time of day: 2 game seconds a 60th) */
        s.water += hos_frame_whole_times(0x7200, bits(step), 4);        /* (the water's 4 ripple substeps a frame) */
        const float taken = real(hos_loop_step(0x7300, bits(step), bits(0.0f), bits(1.0f)));   /* (a loop of one frame) */
        s.loop_steps += taken;
        s.loop_whole &= taken == (float)(int)taken;
        s.loop += taken;
        while (s.loop >= 1.0f) s.loop -= 1.0f;
        s.pose_start &= hos_loop_pose(bits(s.loop), bits(0.0f), bits(1.0f)) == bits(0.0f);
        now += (int64_t)(1e9 / fps);
        realtime_step(now);
    }
    return s;
}
static void warm(int fps) { for (int i = 0; i < 30; i++) now += (int64_t)(1e9 / fps), realtime_step(now); }
int main(int argc, char **argv) {
    int *limit = (int *)GP(MOD + 0xfc), *rate = (int *)GP(MOD + 0x2c8);
    const char *mode = argc > 1 ? argv[1] : "";
    *limit = 60, *rate = 60;
    realtime_step(now);
    if (!strcmp(mode, "carry")) {   /* (cap 60, Real time, the switch's default) */
        warm(46);
        CHECK(*rate == 46 && hos_frame_seconds_bits == 0);
        Second s = second(46, 46);
        printf("46 fps: %u frames, %u day, %u water substeps, loop steps %.3f\n", s.frames, s.day, s.water, s.loop_steps);
        CHECK(s.frames == 60);            /* (today 46) */
        CHECK(s.day == 120);              /* (today 92: the day at 77 %) */
        CHECK(s.water == 240);            /* (today 184) */
        CHECK(s.loop_whole && s.loop_steps > 59.5f && s.loop_steps < 60.5f && s.pose_start);   /* (the Zoras keep their pose) */
        s = second(46, 460);              /* (ten seconds) */
        CHECK(s.frames >= 599 && s.frames <= 601 && s.day >= 1199 && s.day <= 1201 && s.water >= 2399 && s.water <= 2401);
        warm(50);
        CHECK(*rate == 50);
        s = second(50, 50);
        printf("50 fps: %u frames, %u day, %u water substeps\n", s.frames, s.day, s.water);
        CHECK(s.frames == 60 && s.day == 120 && s.water == 240);
    }
    if (!strcmp(mode, "carry") || !strcmp(mode, "off") || !strcmp(mode, "per frame")) {
        /* (whole steps 1, 2, 3, 4: cut as the game cuts them, an amount's fraction too (the pick-up indicator's step * distance /
           50), and a loop of one frame keeps the game's step and time) */
        static const int whole[] = {60, 30, 20, 15};
        for (int k = 0; k < 4; k++) {
            warm(whole[k]);
            CHECK(*rate == (getenv("T_PER_FRAME") ? 60 : whole[k]));
            const float step = game_step();
            CHECK(step == (float)(int)step);
            uint32_t sum = 0;
            for (int i = 0; i < 10; i++) sum += hos_frame_whole(0x7400 + k, bits(0.7f * step));
            CHECK(sum == 10 * (uint32_t)(0.7f * step));
            CHECK(hos_loop_step(0x7500, bits(step), bits(0.0f), bits(1.0f)) == bits(step));
            CHECK(hos_loop_pose(bits(0.25f), bits(0.0f), bits(1.0f)) == bits(0.25f));
        }
    }
    if (!strcmp(mode, "off") || !strcmp(mode, "per frame")) {   /* (HOS_FRAME_CARRY=0, or Per frame: cut as before) */
        warm(46);
        const Second s = second(46, 46);
        printf("%s, 46 fps: %u frames, %u day, %u water substeps\n", mode, s.frames, s.day, s.water);
        CHECK(s.frames == 46 && s.day == 92 && s.water == 184);
        CHECK(!s.loop_whole || getenv("T_PER_FRAME"));
    }
    if (!strcmp(mode, "cap 120")) {   /* (the frames' own time: carried as before) */
        warm(90);
        CHECK(*rate == 60 && hos_frame_seconds_bits != 0);
        const Second s = second(90, 90);
        printf("cap 120, 90 fps: %u frames, %u day, %u water substeps\n", s.frames, s.day, s.water);
        CHECK(s.frames >= 59 && s.frames <= 61 && s.day >= 119 && s.day <= 121 && s.water >= 239 && s.water <= 241);
        CHECK(s.loop_whole && s.pose_start);
    }
    if (!strcmp(mode, "cap 120 step off")) {   /* (HOS_FPS_STEP=0: the rate up to the cap, steps under 1, cut as before) */
        warm(90);
        CHECK(*rate == 90 && hos_frame_seconds_bits == 0);
        const Second s = second(90, 90);
        printf("cap 120 step off, 90 fps: %u frames, %u day\n", s.frames, s.day);
        CHECK(s.frames == 0 && s.day == 90);
    }
    if (!strncmp(mode, "log", 3)) {   /* (HOS_FPS_LOG: what the counters took against the step, 46 fps at cap 60) */
        warm(46);
        second(46, 1000);   /* (3000 calls: one line) */
        printf("%s\n", whole_line);
        const char *kept = strstr(whole_line, "fractions kept"), *cut = strstr(whole_line, "fractions cut");
        double ratio = 0;
        const char *open = strchr(whole_line, '(');
        CHECK(open && sscanf(open, "(%lf)", &ratio) == 1);
        if (!strcmp(mode, "log")) CHECK(kept && ratio > 0.999 && ratio < 1.001);
        else CHECK(cut && ratio > 0.76 && ratio < 0.78);   /* (46 / 60: what a second of 46 frames loses) */
    }
    if (!fails) printf("PASS: %s\n", mode);
    return fails != 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    vi = (ROOT / "host" / "nn_vi.c").read_text(encoding="utf-8")
    fps = vi[vi.index("enum { FPS_UNCAPPED"):]   # (the cap's values, fps_refresh, fps_max)
    fps = fps[:fps.index(block(vi, "static int fps_max(void) {")) + len(block(vi, "static int fps_max(void) {"))]
    movie = block(vi, "uint32_t hos_movie_frames(")
    carries = vi[vi.index(movie) + len(movie):vi.index("volatile uint64_t hos_res_module")]   # (frame_whole, loop_step and theirs)
    host = "\n".join([block(vi, "static float text_scale("), block(vi, "static float camera_scale("), fps, carries] +
                     [block(vi, m) for m in ("static int realtime_rate(", "static void realtime_step(")])
    opts = "\n".join(re.findall(r"enum \{ HOS_OPT_[^}]*\};", (ROOT / "gpu" / "vk" / "vk_api.h").read_text(encoding="utf-8")))
    fixture = FIXTURE.replace("@OPTS@", opts).replace("@HOST@", host)
    with tempfile.TemporaryDirectory(prefix="frame-carry-") as tmp:
        c = Path(tmp) / "fixture.c"
        exe = Path(tmp) / "fixture.exe"
        c.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cc, "-std=c11", "-include", ENV_ON_H, "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-I", str(ROOT / "host"), str(c), "-o", str(exe)], check=True)
        failed = []
        for mode, switches in (("carry", {}), ("off", {"HOS_FRAME_CARRY": "0"}), ("per frame", {"T_PER_FRAME": "1"}),
                               ("cap 120", {"HOS_FPS_MAX": "120"}), ("cap 120 step off", {"HOS_FPS_MAX": "120", "HOS_FPS_STEP": "0"}),
                               ("log", {"HOS_FPS_LOG": "1"}), ("log off", {"HOS_FPS_LOG": "1", "HOS_FRAME_CARRY": "0"})):
            env = {k: v for k, v in os.environ.items() if not k.startswith("HOS_") and k != "T_PER_FRAME"}
            if subprocess.run([str(exe), mode], env=dict(env, **switches)).returncode:
                failed.append(mode)
        if failed:
            raise SystemExit("FAIL: " + ", ".join(failed))


if __name__ == "__main__":
    main()
