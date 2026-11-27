"""The start-up screen's end: a fade to black of about 0.4 s before the game's first frame, none after a screen of under
1.5 s or when the window is not shown. gpu/vk/vk_main.cpp's painter draws the fade itself, so prepare_with_screen must let it
finish (paint_stop only after the wait has run out) before it hides the screen; vk_menu.cpp draws the fade over everything and
puts it back to 0 when the screen goes.

    python -I -B tests/test_startup_fade.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def need(text, pattern, what):
    if not re.search(pattern, text, re.S):
        raise SystemExit("FAIL: " + what)


def main():
    main_cpp = (ROOT / "gpu/vk/vk_main.cpp").read_text(encoding="utf-8")
    menu = (ROOT / "gpu/vk/vk_menu.cpp").read_text(encoding="utf-8")
    started = main_cpp[main_cpp.index("bool started_up() {"):main_cpp.index("int start(void* hwnd) {")]
    igpu = started[started.index("if (!prepare_all && igpu_prepares_in_play() && precompile_wanted()) {"):]
    if 'view.title = "Starting game";' not in igpu or "startup_screen(&view);" not in igpu:
        raise SystemExit("the integrated-GPU in-play shader path does not show the startup screen")
    if "startup_transient_at.compare_exchange_strong" not in menu or "Loading game" not in menu:
        raise SystemExit("the in-play startup screen does not start its animation on its first visible frame")
    if "v.fraction = 0.18 + 0.12" not in menu or "startup_transient_at.store(0" not in menu:
        raise SystemExit("the in-play startup bar is not visibly animated")
    if "ui::rgba(28, 10, 12, (int)(255 * (1.0f - fade_in)))" not in menu:
        raise SystemExit("the in-play startup screen does not fade in over a warm background")
    if "transient_fade_out = std::clamp((elapsed - 1.0f) / 0.42f" not in menu or \
            "ui::rgba(0, 0, 0, (int)(transient_fade_out * 255 + 0.5f))" not in menu or \
            "if (elapsed >= 1.42f) startup_screen(nullptr);" not in menu:
        raise SystemExit("the in-play startup screen does not fade away")
    prep = main_cpp[main_cpp.index("void prepare_with_screen() {"):main_cpp.index("bool started_up() {")]
    need(prep, r"if \(shown >= 1\.5 && !paint_stop\.load\(\)\)", "the fade is not skipped after a short screen")
    need(prep, r"/ 0\.4;.*?startup_fade_to\(\(float\)std::min\(f, 1\.0\)\).*?present_image", "the fade does not run 0.4 s and present")
    join = prep[prep.index("worker.join();"):]
    if join.index("paint_stop.store(true)") < join.index("WaitForSingleObject(painted"):
        raise SystemExit("FAIL: the painter is stopped before its fade can run")
    need(join, r"WaitForSingleObject\(painted, (\d+)\)", "no wait for the painter")
    wait = int(re.search(r"WaitForSingleObject\(painted, (\d+)\)", join).group(1))
    if wait < 1000 + 400:
        raise SystemExit("FAIL: the wait for the painter (%d ms) is shorter than a present and the fade" % wait)
    if join.index("WaitForSingleObject(painted") > join.index("startup_screen(nullptr)"):
        raise SystemExit("FAIL: the screen is hidden before the painter ends")
    draw = menu[menu.index("void draw_startup(ui::Ctx& c) {"):menu.index("void startup_fade_to(float f)")]
    tail = draw[draw.rindex("startup_fade.load"):]
    need(tail, r"AddRectFilled\(ImVec2\(0, 0\), ImVec2\(w, h\), ui::rgba\(0, 0, 0,", "the fade is not drawn over the whole screen")
    need(menu, r"startup_on\.store\(false, std::memory_order_release\);\s*startup_fade\.store\(0", "the fade is not reset when the screen goes")
    print("PASS: integrated-GPU startup shows an animated Red Tunic screen, fades smoothly, and clears; prepared startup fades to black")


if __name__ == "__main__":
    main()
