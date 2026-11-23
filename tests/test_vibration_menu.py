"""The Controls page's Vibration, Vibration strength and Test vibration rows, and how the host hears them.

gpu/vk/vk_menu.cpp's Vibration row, set_row, ini_key and the Test vibration row's texts are compiled from the source with
stubs and run: Vibration's three choices (Auto, Basic, Off) go round and are saved as vibration=; Test vibration is an
action with no value, whose cell and help say, for each of host/vibration.h's VIB_SAY_* the host can report, what plays
the controller used last (a table one short would show the wrong line for the last ones). The option ids the host reads
them by (host/pad_sdl.c) are checked against the renderer's option table (gpu/vk/vk_replay.cpp); the Test vibration press
is checked to ask the host's rumble_test, as a strength change does; host/pad_sdl.c is checked to read the rows, choose
each pad's backend with host/vibration.h and report what plays the pad used last.
tests/test_rumble_menu.py checks the Vibration strength row; tests/test_vibration_pick.py host/vibration.h itself.

    python -I -B tests/test_vibration_menu.py [--cxx g++]
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
MENU = ROOT / "gpu" / "vk" / "vk_menu.cpp"


def read(*parts):
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)


def definition(text, head):
    found = re.search(re.escape(head) + r"\([^;{]*\)\s*\{", text)
    check(found is not None, "no definition of " + head)
    return block(text, head + text[found.start() + len(head):found.end()])


def one_line(text, head):
    found = re.search(r"^" + re.escape(head) + r"\(.*\}\s*$", text, re.M)
    check(found is not None, "no " + head)
    return found.group(0)


FIXTURE = r"""
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include "vibration.h"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %%d %%s\n", __LINE__, #x); return 1; } } while (0)
%(options)s
Options opts;
%(tabs)s
%(row)s
int saves = 0, presets = 0;
int debug_warning_choice = 0, output_pick = 0, monitor_pick = 0;
bool debug_warning = false, menu_refresh = false;
void save() { saves++; }
void apply_pad_layout(int) {}
int detect_pad_layout() { return opts.pad_layout; }
void apply_preset(int) { presets++; }
void sync_preset_budgets(int) {}
void env_release(const int*) {}
void vram_budget_sync() {}
bool output_forced() { return false; }
void output_chosen() {}
void monitor_chosen() {}
bool framegen_by_dlss_offered() { return false; }
int dlss_fg_most() { return 0; }
int framegen_capped(int v, int) { return v; }
void sync_transition_autosave() {}
void gfx_limits() {}   // (hookshot: the detail rows' caps, set_row's: not what this checks)
bool row_fixed(const int*) { return false; }
int upscaler() { return 0; }
namespace ur {
int step_upscaler(int, int, bool) { return 0; }
int step(int, int v, int dir) { return v + dir; }
int snap(int, int v) { return v; }
int snap_render(int, int v) { return v; }
bool full_scale(int) { return false; }
int step_engine(int, int, unsigned) { return 0; }
void step_render(int up, int v, int dir, int* u, int* r) { *u = up, *r = v + dir; }
}
namespace mr { constexpr int kMax = 8; }
namespace i18n { inline void set_game_language(int) {} }
void restart_values(int*) {}
struct { void note(const int*) {} } restart_state;
%(is_slider)s
%(stops)s
%(ini_key)s
bool can_mic() { return false; }
unsigned upscalers_available() { return 1; }
void mic_chosen() {}
void samplers_changed() {}
int temporal_resets = 0;
void temporal_reset_failure() { ++temporal_resets; }
void prompts_at_once(const Row&) {}
void menu_debug_rows(bool, bool) {}
int step_output_choice(int id, int dir) { return id + dir; }
int output_for_aspect(int) { return opts.output_res; }
%(set_row)s
void arrow(Row& r, bool right) { %(arrow)s }
%(texts)s
%(is_test)s
int main() {
    Row mode = {TAB_CONTROLS, "Vibration", &opts.vibration, %(lo)s, %(hi)s, %(step)s, tx_vibration, nullptr, nullptr, false, true};
    CHECK(mode.lo == 0 && mode.hi == 2 && mode.step == 1);
    CHECK(opts.vibration == VIB_MODE_AUTO);
    CHECK(!std::strcmp(ini_key(mode), "vibration"));
    CHECK(!std::strcmp(tx_vibration(VIB_MODE_AUTO), "Auto") && !std::strcmp(tx_vibration(VIB_MODE_BASIC), "Basic") &&
          !std::strcmp(tx_vibration(VIB_MODE_OFF), "Off"));
    /* the three go round, as a choice row's do */
    arrow(mode, true);
    CHECK(opts.vibration == VIB_MODE_BASIC);
    arrow(mode, true);
    CHECK(opts.vibration == VIB_MODE_OFF);
    arrow(mode, true);
    CHECK(opts.vibration == VIB_MODE_AUTO);
    arrow(mode, false);
    CHECK(opts.vibration == VIB_MODE_OFF);
    /* it is a control: the quality preset stays */
    opts.preset = 3;
    arrow(mode, true);
    CHECK(opts.preset == 3 && presets == 0 && saves > 0 && temporal_resets == 0);
    /* Test vibration: an action, no value saved; its cell and help say what plays the controller, for every report */
    Row test = {TAB_CONTROLS, "Test vibration", nullptr, 0, 0, 0, nullptr, nullptr, hn_vibration_test};
    CHECK(is_vibration_test_row(test) && !is_vibration_test_row(mode));
    static const char* cells[] = {"No controller", "Off", "Motors", "HD rumble", "Fine haptics (cable)",
                                  "Fine haptics (Bluetooth)", "Four motors", "HD rumble (RenHDrum)", "Motors: no sound card",
                                  "Motors: no RenHDrum", "Motors (Bluetooth)"};
    static const int says[] = {VIB_SAY_NO_PAD, VIB_SAY_OFF, VIB_SAY_MOTORS, VIB_SAY_HD, VIB_SAY_FINE_USB, VIB_SAY_FINE_BT,
                               VIB_SAY_FOUR, VIB_SAY_HD_RH, VIB_SAY_WAITING, VIB_SAY_NO_LIB, VIB_SAY_BLUETOOTH};
    for (int i = 0; i < 11; i++)
        for (int kind = VIB_PAD_OTHER; kind <= VIB_PAD_SWITCH; kind++) {
            vibration_status_cache.store(vibration_status(kind, says[i]));
            CHECK(!std::strcmp(vibration_cell(), cells[i]));
            CHECK(test.hint(0, -1) != nullptr && test.hint(0, -1) == vibration_why(says[i]));
        }
    /* every line the help can show, for its height (measure_help), and no more */
    int lines = 0;
    while (test.hint(0, lines)) lines++;
    CHECK(lines == 11);
    vibration_status_cache.store(0);
    CHECK(!std::strcmp(vibration_cell(), "No controller"));
    return 0;
}
"""


def rows(menu):
    """Vibration, then Vibration strength, then Test vibration, on the Controls tab; none of the rows they replaced"""
    order = sorted((menu.index('"%s"' % n), n) for n in ("Vibration", "Vibration strength", "Test vibration"))
    check([n for _, n in order] == ["Vibration", "Vibration strength", "Test vibration"],
          "the Controls page does not show Vibration, Vibration strength, Test vibration in that order")
    for gone in ('"Rumble strength"', '"Haptic feedback"', '"Haptic strength"', '"Haptic status"', '"Test haptics"'):
        check(gone not in menu, "vk_menu.cpp still has the row %s" % gone)


def fixture(menu, header):
    row = re.search(r'\{TAB_CONTROLS, "Vibration", &opts\.vibration, (-?\d+), (\d+), (\d+), tx_vibration,', menu)
    check(row is not None, "the menu lost its Vibration row")
    arrow = re.search(r"\n\s*(set_row\(r, \*r\.value \+ \(press\[4\] \? r\.step : -r\.step\), press\[4\] \? 1 : -1\);)", menu)
    check(arrow is not None, "the arrows no longer step a row with set_row")
    texts = menu[menu.index("const char* tx_vibration(int v) {"):menu.index("// (hookshot) Menu size:")]
    return FIXTURE % {
        "options": re.search(r"struct Options \{.*?\n\};", header, re.S).group(0),
        "tabs": re.search(r"enum MenuTab \{.*?\};", menu, re.S).group(0),
        "row": re.search(r"struct Row \{.*?\n\};", menu, re.S).group(0),
        "is_slider": one_line(menu, "bool is_slider"),
        "stops": one_line(menu, "bool stops_at_ends"),
        "ini_key": definition(menu, "const char* ini_key"),
        "set_row": definition(menu, "void set_row"),
        "arrow": "bool press[5] = {false, false, false, !right, right}; " + arrow.group(1),
        "texts": texts,
        "is_test": one_line(menu, "bool is_vibration_test_row"),
        "lo": row.group(1), "hi": row.group(2), "step": row.group(3),
    }


def presses(menu):
    """Test vibration's Confirm asks the host's rumble_test (a strength change does too), and is labelled Test"""
    tick = definition(menu, "void menu_tick")
    at = tick.index("else if (press[7] && is_vibration_test_row(r)) {")
    branch = tick[at:tick.index("}", at)]
    check("host->rumble_test()" in branch and "host->version >= 33" in branch,
          "Test vibration's Confirm does not ask the host's rumble_test")
    check(menu.count('is_vibration_test_row(selr) ? "Test"') + menu.count('is_vibration_test_row(r) ? "Test"') >= 2,
          "Test vibration's Confirm is not labelled Test")
    check("host->rumble_test()" in definition(menu, "void menu_rumble_tick"),
          "a Vibration strength change no longer gives a short buzz")


def option_ids():
    """63 is the Vibration row, 64 what the host reports plays the pad; both read by host/pad_sdl.c alone"""
    replay, api = read("gpu", "vk", "vk_replay.cpp"), read("gpu", "vk", "vk_api.h") + read("gpu", "vk", "oot_wheel_state.h")   # (hookshot: the item wheel's ids too)
    value = {}
    for first, start, rest in re.findall(r"enum \{ (HOS_OPT_\w+) = (\d+)([^}]*)\}", api):
        names = [first] + [n.strip() for n in rest.split(",") if n.strip()]
        value.update({n: int(start) + i for i, n in enumerate(names)})
    # (hookshot: the game side's own ids and the Travel pictures', as tests/test_rumble_menu.py reads them)
    value.update({n: int(v) for n, v in re.findall(r"#define (OOT_CHEAT_\w+) (\d+)", read("gpu", "vk", "oot_cheat_items.h"))})
    value.update({n: int(v) for n, v in re.findall(r"#define (HOS_OPT_\w+) (\d+)", read("gpu", "vk", "oot_teleport_capture.h"))})
    check(value.get("HOS_OPT_VIBRATION") == 63 and value.get("HOS_OPT_VIBRATION_STATUS") == 64,
          "gpu/vk/vk_api.h: the Vibration options are not 63 and 64")
    check(not any(n.startswith("HOS_OPT_HAPTIC") for n in value), "gpu/vk/vk_api.h still has the haptics options")
    table = block(replay, "t.option = [](int id) -> int {")
    ids = [int(n) if n.isdigit() else value[n] for n in re.findall(r"\bid == (\w+)", table)]
    check(len(ids) == len(set(ids)), "vk_replay.cpp: an option id is read twice: %s" % sorted(ids))
    check("if (id == HOS_OPT_VIBRATION) return opts.vibration;" in table, "vk_replay.cpp: t.option does not read Vibration")
    check("if (id == HOS_OPT_VIBRATION_STATUS) return menu_vibration_status_option();" in table,
          "vk_replay.cpp: t.option does not give back what the host reported")
    users = []
    for folder in ("host", "gpu"):
        for path in (ROOT / folder).rglob("*.c*"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\brx_option(_set)?\(\s*HOS_OPT_VIBRATION", text):
                users.append(path.relative_to(ROOT).as_posix())
    check(users == ["host/pad_sdl.c"], "the Vibration options are read outside host/pad_sdl.c: %s" % users)


def host():
    """host/pad_sdl.c reads the rows, chooses with host/vibration.h, plays each backend and reports the pad used last;
    host/nn_hid.c hands RenHDrum the game's own values, and the game's end lets it go"""
    pads = read("host", "pad_sdl.c")
    poll = definition(pads, "static void pads_poll")
    for need, why in (("rx_option(HOS_OPT_VIBRATION, VIB_MODE_AUTO)", "the Vibration row does not reach the pads"),
                      ("if (strength <= 0) mode = VIB_MODE_OFF", "an options file's Rumble strength Off is not Off"),
                      ("rh_tick(now, testing, strength);", "RenHDrum's backends do not get the test and the strength"),
                      ("vib_choose(i, mode, now);", "the pads' backends are not chosen each poll"),
                      ("if (pads[i].vib == VIB_NONE) continue;", "Off still plays"),
                      ("rx_option_set(HOS_OPT_VIBRATION_STATUS, st);", "what plays the pad is not reported")):
        check(need in poll, why)
    choose = definition(pads, "static void vib_choose")
    check("vibration_pick(" in choose, "vib_choose does not ask host/vibration.h")
    # (a backend that would not open is not opened again at every poll)
    check("vib_failed" in choose and "vib_retry_ns" in choose, "a backend that failed is tried again every poll")
    check("vib_leave(i, now);" in definition(pads, "static void pads_poll"),
          "a pad that goes away is not let go by what played it")
    hid = read("host", "nn_hid.c")
    check("rh_push(dev, v);" in definition(hid, "static void vibration_send"),
          "the game's own vibration values no longer reach RenHDrum")
    check("hos_sdl_vibration_shutdown();" in read("host", "hos_main.c"), "the game's end no longer lets RenHDrum go")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = p.parse_args()
    menu, header = MENU.read_text(encoding="utf-8"), read("gpu", "vk", "vk_menu.h")
    rows(menu)
    presses(menu)
    option_ids()
    host()
    with tempfile.TemporaryDirectory(prefix="vibration-menu-") as tmp:
        src, exe = Path(tmp) / "vibration_menu.cpp", Path(tmp) / "vibration_menu"
        src.write_text(fixture(menu, header), encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "host"), str(src), "-o",
                        str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    print("PASS: Vibration's three choices, its key, and Test vibration's line for each thing the host reports")


if __name__ == "__main__":
    main()
