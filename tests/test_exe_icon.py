"""The game's exe has Red Tunic's icon (nx2/build.py exe_resources): builder/setup/red-tunic.ico, as work/hookshot_brand.res,
in every build, beside the name Windows shows for it when the builder read the game's from its .nup (work/hookshot_game.res,
nx2/game_title.py: version strings, no icon). A release's exe, built where there is no .nup, had no icon: none in its window
(host/display.c loads icon 1) or its taskbar. The .res is read back here: icon group 1 with every size of the .ico, each the
.ico's own PNG; the game's .res no icon of its own; a developer's hookshot.res not linked. No game, no compiler.

    python -I -B tests/test_exe_icon.py
"""
import importlib.util
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import game_title  # noqa: E402
import make_icon  # noqa: E402


def fail(message):
    print("FAIL: " + message)
    sys.exit(1)


def entries(res):
    """(type, name, data) of each entry of a .res"""
    out, at = [], 0
    while at < len(res):
        size, header = struct.unpack("<II", res[at:at + 8])
        kind, name = struct.unpack("<xxHxxH", res[at + 8:at + 16])
        out.append((kind, name, res[at + header:at + header + size]))
        at += header + size + (-size & 3)
    return out


def main():
    ico = (ROOT / "assets" / "red-tunic.ico").read_bytes()
    images = make_icon.images_of_ico(ico)
    if [s for s, _ in images] != list(make_icon.SIZES):
        fail("builder/setup/red-tunic.ico's sizes are %s, not %s" % ([s for s, _ in images], list(make_icon.SIZES)))
    if make_icon.ico_of(images) != ico:
        fail("make_icon.ico_of does not write red-tunic.ico back as it is: images_of_ico read it wrong")
    build = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
    if "exe_resources() +" not in build:
        fail("nx2/build.py does not link exe_resources() into the exe")
    spec = importlib.util.spec_from_file_location("build", ROOT / "nx2" / "build.py")
    build_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_mod)
    with tempfile.TemporaryDirectory(prefix="exe-icon-") as work:
        # (a developer's picture is not linked: Red Tunic's icon is the exe's)
        Path(work, "hookshot.res").write_bytes(make_icon.res_of(images[:1]))
        got = build_mod.exe_resources(work)
        if got != [os.path.join(work, build_mod.BRAND_RES)]:
            fail("with no name from a .nup, exe_resources gives %s, not Red Tunic's icon alone" % got)
        res = Path(got[0]).read_bytes()
        found = entries(res)
        icons = [(n, d) for k, n, d in found if k == 3]
        groups = [(n, d) for k, n, d in found if k == 14]
        if len(groups) != 1 or groups[0][0] != 1:
            fail("the .res has icon groups %s, not group 1 (what host/display.c loads)" % [n for n, _ in groups])
        if [d for _, d in icons] != [png for _, png in images]:
            fail("the .res's icons are not red-tunic.ico's images")
        if struct.unpack("<HHH", groups[0][1][:6])[2] != len(images):
            fail("the icon group does not name every image of the .ico")
        if build_mod.exe_resources(work) != got or Path(got[0]).read_bytes() != res:
            fail("a second build makes the .res again differently")
        # (the game's name, read from a .nup, kept in the folder: its .res the name alone, linked beside Red Tunic's icon)
        folder = Path(work, "game")
        folder.mkdir()
        (folder / game_title.TITLE_FILE).write_text("A game\n", encoding="utf-8")
        named = Path(work, game_title.RES_FILE)
        if not game_title.write_res(str(folder), str(named)):
            fail("game_title.write_res wrote nothing for a game with a name")
        kinds = {k for k, _, _ in entries(named.read_bytes()) if k}
        if kinds != {16}:
            fail("the game's .res holds resource types %s, not its version strings alone (16): its icon would clash" % kinds)
        if build_mod.exe_resources(work) != [os.path.join(work, build_mod.BRAND_RES), str(named)]:
            fail("with the game's name, exe_resources gives %s, not Red Tunic's icon and the name" % build_mod.exe_resources(work))
        legacy = make_icon.res_of(images[:1], {"FileDescription": "Cached title"})
        named.write_bytes(legacy)
        linked = build_mod.exe_resources(work)
        version = [entry for entry in entries(legacy) if entry[0] == 16]
        actual = [entry for entry in entries(Path(linked[1]).read_bytes()) if entry[0]]
        if actual != version or named.read_bytes() != legacy:
            fail("cached game resource still links its old icon or loses its title")
        named.write_bytes(make_icon.res_of(images[:1]))
        if build_mod.exe_resources(work) != [linked[0]]:
            fail("an icon-only cached resource was linked beside the brand")
    print("PASS: the exe's icon is Red Tunic's (%d sizes), beside the game's name when it has one" % len(images))


if __name__ == "__main__":
    main()
