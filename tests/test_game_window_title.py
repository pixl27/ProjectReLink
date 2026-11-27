"""The game title is live on Windows and SDL: Red Tunic version, area and Link age."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (ROOT / "host/display.c").read_text(encoding="utf-8")
POSIX = (ROOT / "host/posix/display_sdl.c").read_text(encoding="utf-8")
SDL = (ROOT / "host/hos_sdl.c").read_text(encoding="utf-8")
SDL_H = (ROOT / "host/hos_sdl.h").read_text(encoding="utf-8")

TITLE = '"THE LEGEND OF ZELDA OCARINA OF TIME - Retunic Recomp (%s) - %s - %s Link"'
for source, platform in ((WINDOWS, "Windows"), (POSIX, "SDL")):
    if TITLE not in source:
        raise SystemExit(f"{platform}: title does not include version, area and Link age")
    if "hos_oot_current_map" not in source or "hos_oot_area_name" not in source or "hos_oot_state" not in source:
        raise SystemExit(f"{platform}: title does not use the live area and player state")
if "hos_sdl_set_window_title" not in SDL_H or "SDL_SetWindowTitle" not in SDL:
    raise SystemExit("SDL: no portable window-title update")
print("PASS: Windows and SDL window titles show Red Tunic version, current area and Link age")
