"""While one of the game's menus has the mouse, host/display.c says so to the game's thread (hos_mouse_in_game_menu),
where the mouse's side buttons then stay their controls' (host/nn_hid.c mouse_buttons_arm): Mouse 4 on A confirmed
nothing in the text box's choices, as the capture a mouse button needed had gone to the menu.

display.c's menus_mouse, the one place the two menus' states change, is compiled with stand-ins for the window's calls
and run: a game menu opening, the Insert menu over it and gone again, the game menu closing, and the Insert menu alone.
What read_input does with the answer is tests/test_input_background.py's menu-side-buttons case.

python -I -B tests/test_game_menu_mouse_buttons.py [--cc gcc]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block, find   # noqa: E402

RUN_ALL_ARGS = ["--cc", "gcc"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <stdint.h>
#include <stdio.h>
typedef long LONG;
typedef int64_t LONG64;
typedef void *HWND;
typedef struct { LONG x, y; } POINT;
typedef struct { int unused; } HosPointer;
static LONG InterlockedExchange(volatile LONG *t, LONG v) { const LONG was = *t; *t = v; return was; }
static int64_t now_ns = 5000000000LL;
static int64_t hos_now_ns(void) { return now_ns += 1000000; }
static int toggles_cleared, captures, releases;
void hos_input_clear_toggles(void) { toggles_cleared++; }
static void GetCursorPos(POINT *at) { at->x = 300, at->y = 200; }
static HWND GetForegroundWindow(void) { return (HWND)(uintptr_t)1; }
static volatile LONG mouse_captured, gm_shown;
static LONG gm_mx, gm_my;
static HosPointer pointer, game_pointer;
static int menu_open, captured_before_menu, esc_had_capture, game_menu_open;
static void mouse_capture(HWND w, int on) { (void)w; mouse_captured = on; if (on) captures++; else releases++; }
static void gm_cursor_off(HWND w) { (void)w; }
static void hos_pointer_reset(HosPointer *p) { (void)p; }
static void pointer_update(HWND w, uint32_t pressed, int wheel) { (void)w, (void)pressed, (void)wheel; }
@STATE@
@MENUS@
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main(void) {
    const HWND w = (HWND)(uintptr_t)1;
    CHECK(!hos_mouse_in_game_menu());                            /* play */
    mouse_captured = 1;
    menus_mouse(w, 0, 1);
    CHECK(hos_mouse_in_game_menu() && !mouse_captured);          /* a game menu opened: it has the mouse */
    menus_mouse(w, 1, 1);
    CHECK(!hos_mouse_in_game_menu());                            /* the Insert menu over it: the mouse is that one's */
    menus_mouse(w, 0, 1);
    CHECK(hos_mouse_in_game_menu() && !mouse_captured);          /* and the game menu's again */
    menus_mouse(w, 0, 0);
    CHECK(!hos_mouse_in_game_menu() && mouse_captured);          /* closed: the capture is back */
    menus_mouse(w, 1, 0);
    CHECK(!hos_mouse_in_game_menu() && !mouse_captured);         /* the Insert menu alone */
    menus_mouse(w, 0, 0);
    CHECK(!hos_mouse_in_game_menu() && mouse_captured);
    printf("PASS\n");
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    display = (ROOT / "host" / "display.c").read_text(encoding="utf-8")
    at = find(display, "static volatile LONG game_menu_mouse;")
    state = display[at:display.index("\n", find(display, "int hos_mouse_in_game_menu(void)")) + 1]
    source = FIXTURE.replace("@STATE@", state).replace("@MENUS@", block(display, "static void menus_mouse(HWND w,"))
    with tempfile.TemporaryDirectory(prefix="hookshot-game-menu-mouse-") as tmp:
        c, exe = Path(tmp) / "fixture.c", Path(tmp) / "fixture.exe"
        c.write_text(source, encoding="utf-8")
        subprocess.run([a.cc, "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror", str(c), "-o", str(exe)], check=True)
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        print(run.stdout, end="")
        if run.returncode:
            raise SystemExit(run.stdout + run.stderr)


if __name__ == "__main__":
    main()
