"""Quality presets preserve the player's frame-generation choice.

Compile the menu's actual preset and row handlers with small stand-ins.
Check that the game FPS cap keeps its saved key when its label changes. Preset
and frame-generation changes do not reset temporal failures; the three scaling
rows do, using their real row definitions and scaling helpers. While NVIDIA DLSS
FG makes the frames, the Frame generation row's arrows step from the value shown
and end at its runtime's most + 1 (the real framegen_capped): 6x on an RTX 50's,
2x on an RTX 40's, round to Off past it as past 8x; a value set as it is (the
control bridge's) is kept. No game or GPU is needed.
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def block(text, start):
    opening = text.index("{", start)
    depth = 0
    for at in range(opening, len(text)):
        depth += text[at] == "{"
        depth -= text[at] == "}"
        if depth == 0:
            return text[start:at + 1]
    raise SystemExit("unterminated block at %d" % start)


def definition(text, name):
    match = re.search(re.escape(name) + r"\([^;{]*\)\s*\{", text)
    if not match:
        raise SystemExit("missing definition: " + name)
    return block(text, match.start())


FIXTURE = r"""
#include "upscaler_rows.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)
/* OPTIONS */
Options opts;
bool debug_warning = false;
int debug_warning_choice = 0;
void sync_transition_autosave() {}
bool menu_refresh = false;

/* TABS */
/* ROW */
int saves = 0;
void save() { ++saves; }
void apply_pad_layout(int) {}
int detect_pad_layout() { return opts.pad_layout; }
void gfx_limits() {}   // (detail rows' caps: not what this checks)
constexpr int kAutoPreset = 6;   /* (the Quality preset Auto (this PC): apply_auto, not reached here) */
void apply_auto() {}
void apply_deck_preset(int) {}
void env_release(const int*) {}   // (the environment's hold on a row: none here)
bool row_fixed(const int*) { return false; }
namespace ur = mkwvk::upscaler_rows;
int upscaler() { return opts.dlss; }
unsigned upscalers_available() { return ur::bit(ur::GAME) | ur::dlss_bits(true); }
namespace mr { constexpr int kMax = 8; }
/* (the Game language row's hook in set_row: language.txt, not written here) */
namespace i18n { inline void set_game_language(int) {} }
void restart_values(int*) {}
struct { void note(const int*) {} } restart_state;
bool can_mic() { return false; }
void mic_chosen() {}
int output_pick;
int monitor_pick;
bool output_forced() { return false; }
bool framegen_by_dlss_offered() { return true; }   // (vk_menu.cpp: Frame generation by offers DLSS FG)
void output_chosen() {}   // Playback selection is covered by test_audio_output_devices.py.
void monitor_chosen() {}  // Fullscreen selection is covered by test_fullscreen_monitor_menu.py.
void samplers_changed() {}
void menu_debug_rows(int, int) {}   // (the Debug rows that apply in play: vk_draw.cpp)
int temporal_resets = 0;
void temporal_reset_failure() { ++temporal_resets; }
void prompts_at_once(const Row&) {}
int step_output_choice(int id, int dir) { return id + dir; }
// (the Aspect ratio row moves the Output resolution to one of its shape's sizes, output_for_aspect: not what this checks)
int output_for_aspect(int) { return opts.output_res; }
const char* tx_framegen(int) { return ""; }
const char* hn_framegen(int, int) { return ""; }
int fg_most = 0;   // (DLSS FG's runtime's most made frames while it makes the frames, vk_dlss.cpp; 0: no cap)
int dlss_fg_most() { return fg_most; }
/* CAPPED */
const char* tx_preset(int) { return ""; }
const char* hn_preset(int, int) { return ""; }
const char* tx_fps_cap(int) { return ""; }
const char* tx_upscaler(int) { return ""; }
const char* hn_upscaler(int, int) { return ""; }
const char* tx_resolution(int) { return ""; }
const char* hn_resolution(int, int) { return ""; }
const char* tx_dlss_model(int) { return ""; }
const char* hn_dlss_model(int, int) { return ""; }
/* INI_KEY */
/* STOPS */
/* PRESET */
/* SET_ROW */

int main() {
    Row framegen = /* FRAMEGEN_ROW */;
    Row quality = /* QUALITY_ROW */;
    Row fps_cap = /* FPS_CAP_ROW */;
    CHECK(std::strcmp(fps_cap.name, "Game frame rate cap") == 0);
    CHECK(std::strcmp(ini_key(fps_cap), "frame_rate_cap") == 0);
    CHECK(std::strstr(fps_cap.help, "Generated frames are additional") != nullptr);
    CHECK(opts.framegen == 0);
    for (int previous = 0; previous <= 8; ++previous) {
        for (int preset = 0; preset <= 5; ++preset) {
            opts = Options{};
            opts.framegen = previous;
            set_row(quality, preset, 1);
            CHECK(opts.preset == preset);
            CHECK(opts.framegen == previous);
        }
    }
    CHECK(framegen.preset_exempt && !framegen.restart);
    for (int preset = 1; preset <= 5; ++preset) {
        opts = Options{};
        opts.preset = preset;
        set_row(framegen, 2, 1);
        CHECK(opts.framegen == 2 && opts.preset == preset);
        set_row(framegen, 0, -1);
        CHECK(opts.framegen == 0 && opts.preset == preset);
    }
    // (Frame generation while NVIDIA DLSS FG makes the frames: the arrows step from the value shown, end at its runtime's
    // most + 1 and go round to Off past it, as past 8x; down from Off comes to that top)
    struct Step { int most, from, dir, to; } steps[] = {
        {5, 8, 1, 0}, {5, 8, -1, 5}, {5, 7, -1, 5}, {5, 5, 1, 6}, {5, 6, 1, 0}, {5, 0, -1, 6}, {5, 1, 1, 2},   // (6x)
        {1, 3, 1, 0}, {1, 3, -1, 1}, {1, 1, 1, 2}, {1, 2, 1, 0}, {1, 0, -1, 2},                                // (2x)
        {0, 7, 1, 8}, {0, 8, 1, 0}, {0, 0, -1, 8}, {0, 2, -1, 1},   // (no cap: as before)
    };
    for (const Step& s : steps) {
        opts = Options{};
        fg_most = s.most, opts.framegen = s.from;
        set_row(framegen, s.from + s.dir * framegen.step, s.dir);
        if (opts.framegen != s.to)
            return std::fprintf(stderr, "FAIL: Frame generation %d, DLSS FG's runtime making %d, arrow %d: %d, wanted %d\n",
                                s.from, s.most, s.dir, opts.framegen, s.to), 1;
    }
    opts = Options{};
    fg_most = 5, opts.framegen = 2;
    set_row(framegen, 7, 0);   // (set as it is, the control bridge's: kept as asked, run as the top)
    CHECK(opts.framegen == 7);
    fg_most = 0;
    CHECK(std::strstr(quality.help, "Frame generation") != nullptr);
    CHECK(std::strstr(quality.help, "stay yours") != nullptr);
    CHECK(saves > 0);
    CHECK(temporal_resets == 0);

    // Only the Upscaler, Render resolution and DLSS model rows reset temporal failures.
    Row temporal_rows[] = {/* UPSCALER_ROW */, /* RESOLUTION_ROW */, /* DLSS_MODEL_ROW */};
    for (Row& r : temporal_rows) {
        opts = Options{};
        if (r.value == &opts.dlss_model) opts.dlss = ur::DLSS;
        const int previous = *r.value, resets = temporal_resets;
        set_row(r, previous + r.step, 1);
        CHECK(*r.value != previous);
        CHECK(temporal_resets == resets + 1);
    }
    std::puts("PASS: quality presets preserve frame generation and its row preserves the preset");
}
"""


def fixture(source):
    menu = (source / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
    header = (source / "gpu" / "vk" / "vk_menu.h").read_text(encoding="utf-8")
    framegen = (source / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    options = re.search(r"struct Options \{.*?\n\};", header, re.S)
    tabs = re.search(r"enum MenuTab \{.*?\};", menu, re.S)
    row = re.search(r"struct Row \{.*?\n\};", menu, re.S)
    if not options or not tabs or not row:
        raise SystemExit("the menu's options, tabs, or row definition is missing")
    fps_cap = re.search(r'\{TAB_GRAPHICS, "[^"]+", &opts\.fps_cap,', menu)
    if not fps_cap:
        raise SystemExit("missing game frame rate cap row")
    replacements = {
        "/* OPTIONS */": options.group(0),
        "/* TABS */": tabs.group(0),
        "/* ROW */": row.group(0),
        "/* STOPS */": definition(menu, "bool stops_at_ends"),
        "/* INI_KEY */": definition(menu, "const char* ini_key"),
        "/* FPS_CAP_ROW */": block(menu, fps_cap.start()),
        "/* PRESET */": definition(menu, "void apply_preset"),
        "/* SET_ROW */": definition(menu, "void set_row"),
        "/* CAPPED */": block(framegen, framegen.index("int framegen_capped(")),
        "/* FRAMEGEN_ROW */": block(menu, menu.index('{TAB_GRAPHICS, "Frame generation"')),
        "/* QUALITY_ROW */": block(menu, menu.index('{TAB_GRAPHICS, "Quality preset"')),
        "/* UPSCALER_ROW */": block(menu, menu.index('{TAB_GRAPHICS, "Upscaler"')),
        "/* RESOLUTION_ROW */": block(menu, menu.index('{TAB_GRAPHICS, "Render resolution"')),
        "/* DLSS_MODEL_ROW */": block(menu, menu.index('{TAB_GRAPHICS, "DLSS model"')),
    }
    result = FIXTURE
    for marker, value in replacements.items():
        result = result.replace(marker, value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    includes = str((args.source / "gpu" / "vk").resolve())
    with tempfile.TemporaryDirectory(prefix="framegen-preset-") as tmp:
        src = Path(tmp) / "framegen_preset.cpp"
        exe = Path(tmp) / ("framegen_preset.exe" if os.name == "nt" else "framegen_preset")
        src.write_text(fixture(args.source), encoding="utf-8")
        if Path(args.cxx).name.lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe"):
            command = [args.cxx, "/nologo", "/std:c++20", "/EHsc", "/I" + includes, str(src),
                       "/Fo" + str(Path(tmp) / "framegen_preset.obj"), "/Fe" + str(exe)]
        else:
            command = [args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", includes,
                       str(src), "-o", str(exe)]
        subprocess.run(command, cwd=tmp, check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
