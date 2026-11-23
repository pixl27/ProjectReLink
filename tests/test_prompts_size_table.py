"""Game-free: the resource size table the button prompts serve is the game's own (host/prompts.c).

The prompts' font is larger than the game's, so the size table the game opens is served with the font's entry raised.
Before, that was prompts/rsizetable.zs, made at the build from the RomFS's table whose name sorts first, and served for any
size table the game opened. A RomFS with another version's table beside the game's (a base dump with an update over it), or
another dump than the build read, got a table that is not the game's: the game dropped files it needs and crashed at the
start (guest 0x1030 on sead::MainThread). Deleting prompts/rsizetable.zs and External.bfotf was the players' way round it.

Now the table served is the RomFS's own table of the name the game opens, with the font's entry raised, made at the start;
without a table of that name or a font entry in it, the game gets its own table and its own icons; the prompts' copies of
the game's layouts and textures are served only when they are the size of the RomFS's. host/mods.c, host/prompts.c,
host/texts.c, host/zs.c and host/restbl.c as they are, over real folders (tests/test_mods_size_table.py's fixture), on
made-up tables, no game data:

- a table the build made from another dump (prompts/rsizetable.zs) is never served: the RomFS's own, its font entry raised,
  with the table or the font opened first;
- two versions' tables in the RomFS: the one the game opens, not the one that sorts first;
- a table name the RomFS has not (another game version): the game's own table and its own icons;
- a table without a font entry: the game's own table and its own icons;
- a font entry with room already (the prompts' font no larger than the game's): the game's own table, and the prompts' font;
- the layouts (prompts/Common.blarc): served when the size of the RomFS's, the game's own when not;
- (hookshot) the layouts with the pause menu's Recomp options button (prompts/Common.blarc.added: the bytes it adds): served
  when the RomFS's size and those bytes, their entry in the table raised by them, the font's with it.

python -I -B tests/test_prompts_size_table.py [--cc gcc]
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the test's folder off the path)
import test_mods_size_table as fixture  # noqa: E402 (its fixture: mods.c and prompts.c over folders)

RUN_ALL_ARGS = ["--cc", "gcc"]
RUN_ALL_SKIP = {"windows": "the prompts' files over folders run on Linux (host/posix's file calls): checked on Linux"}

ROOT = Path(__file__).resolve().parents[1]
TABLE = fixture.TABLE   # (the game's: Product.111)
OTHER = fixture.OTHER   # (another version's: Product.110, which sorts first)
NEWER = "System/Resource/ResourceSizeTable.Product.112.Nin_Ounce_NVN2.rsizetable.zs"
FONT = fixture.FONT
FONT_CRC = zlib.crc32(FONT[:-3].encode())   # (the table's key: the path without .zs)
LAYOUTS = "UI/Layout2Archive/Common.Product.100.Nin_Ounce_NVN2.blarc.zs"
LAYOUTS_CRC = zlib.crc32(LAYOUTS[:-3].encode())
OUR_FONT = struct.pack("<I", 0x18029A7F) + bytes(28)   # (prompts/External.bfotf: 32 bytes, every face's)
GAME_FONT = 16   # (the RomFS's font, decompressed)
CHECK = fixture.check


def table(font=300, others=((0x10, 100), (0x20, 200), (0x40, 400)), named=(("Collide/A", 50),)):
    """a made-up size table with the font's entry (None: none), sorted as the game's"""
    hashed = list(others) + ([(FONT_CRC, font)] if font is not None else [])
    return fixture.restbl(sorted(hashed), list(named))


def need(entry):
    """the font's entry for ours: our font's size with the allowance the game's entry gives its own font"""
    return len(OUR_FONT) + entry - GAME_FONT


def case(exe, tmp, name, romfs, opens, stale=None, blarc=None, added=None, game_font=GAME_FONT):
    """a game folder with the prompts and no mods: romfs {path: bytes}; stale: a prompts/rsizetable.zs the build made (from
    another table); blarc: prompts/Common.blarc's bytes; added: prompts/Common.blarc.added's number; game_font:
    the RomFS's font's size. {path: what was served}, the log, the folder"""
    top = tmp / name
    game = top / "game"
    (game / "prompts/xbox").mkdir(parents=True)
    (game / "hookshot_options.ini").write_text("button_prompts=2\n")   # (Xbox: no controller asked for)
    (game / "prompts/External.bfotf").write_bytes(OUR_FONT)
    if stale is not None:
        (game / "prompts/rsizetable.zs").write_bytes(fixture.frame(stale))
    if blarc is not None:
        (game / "prompts/Common.blarc").write_bytes(blarc)
    if added is not None:
        (game / "prompts/Common.blarc.added").write_text("%d\n" % added)
    files = {FONT: fixture.frame(bytes(game_font)), **romfs}
    for rel, data in files.items():
        (game / "romfs" / rel).parent.mkdir(parents=True, exist_ok=True)
        (game / "romfs" / rel).write_bytes(data)
    (top / "mods").mkdir()
    env = dict(os.environ, TEST_GAME_EXE=str(game / "hookshot"), HOS_MODS=str(top / "mods"), TEST_ROMFS=str(game / "romfs"))
    out = subprocess.run([str(exe), *opens], env=env, check=True, capture_output=True, text=True, timeout=60).stdout
    served = {}
    for line in out.splitlines():
        if line.startswith("open "):
            rel, _, what = line[5:].partition(": ")
            what = what.replace("\\", "/")
            served[rel] = what if what == "(the game's)" else os.path.relpath(what, top).replace("\\", "/")
    return served, out, top


def checks(exe, tmp):
    ours = "game/cache/prompts/rsizetable.zs"
    font = "game/cache/prompts/External.bfotf.zs"
    errors = []

    def expect(what, got, want, log):
        if got != want:
            errors.append("%s: served %s, not %s\n%s" % (what, got, want, log))

    def content(top, what, want):
        path = top / ours
        if not path.is_file():
            return
        got = fixture.unframe(path.read_bytes())
        if got != want:
            errors.append("%s: the table served is not the RomFS's own with the font's entry raised" % what)

    # a table the build made from another dump or version: never served; the RomFS's own, the font's entry raised to fit
    stale = table(font=900, others=((0x10, 50), (0x20, 80), (0x40, 90)))
    for i, opens in enumerate(((TABLE, FONT), (FONT, TABLE))):
        what = "a build's table from another dump, %s first" % ("the table" if i == 0 else "the font")
        served, log, top = case(exe, tmp, "stale-%d" % i, {TABLE: fixture.frame(table())}, opens, stale=stale)
        expect(what, served, {TABLE: ours, FONT: font}, log)
        content(top, what, table(font=need(300)))
        if served.get(TABLE) == ours and "raised from 300 to %d bytes" % need(300) not in log:
            errors.append("%s: no log line of the entry raised\n%s" % (what, log))
    # two versions' tables in the RomFS, the build's made from the one that sorts first: the one the game opens
    old = table(font=200, others=((0x10, 40), (0x20, 60)))
    served, log, top = case(exe, tmp, "versions", {OTHER: fixture.frame(old), TABLE: fixture.frame(table())}, (TABLE, FONT),
                            stale=table(font=need(200), others=((0x10, 40), (0x20, 60))))
    expect("two versions' tables", served, {TABLE: ours, FONT: font}, log)
    content(top, "two versions' tables", table(font=need(300)))
    # a table the RomFS has not (another game version): the game's own, and its own icons
    served, log, _ = case(exe, tmp, "missing", {TABLE: fixture.frame(table())}, (NEWER, FONT), stale=table(font=need(300)))
    expect("a table name the RomFS has not", served, {NEWER: "(the game's)", FONT: "(the game's)"}, log)
    # a table with no entry for the font: the game's own, and its own icons
    served, log, _ = case(exe, tmp, "no-entry", {TABLE: fixture.frame(table(font=None))}, (TABLE, FONT))
    expect("a table without a font entry", served, {TABLE: "(the game's)", FONT: "(the game's)"}, log)
    # a font entry with room for ours (the entry is the font's size and an allowance, hos_prompts_font_entry: room when ours
    # is no larger than the game's): the game's own table, and the font
    served, log, _ = case(exe, tmp, "room", {TABLE: fixture.frame(table(font=100000))}, (TABLE, FONT), game_font=len(OUR_FONT))
    expect("a font entry with room", served, {TABLE: "(the game's)", FONT: font}, log)
    # the layouts: the build's copy served when it is the size of the RomFS's, the game's own when not
    for name, size, want in (("layouts-same", 64, "game/cache/prompts/Common.blarc.zs"), ("layouts-other", 80, "(the game's)")):
        served, log, _ = case(exe, tmp, name, {TABLE: fixture.frame(table()), LAYOUTS: fixture.frame(bytes(64))}, (LAYOUTS,),
                              blarc=bytes(size))
        expect("prompts/Common.blarc of %d bytes, the RomFS's of 64" % size, served, {LAYOUTS: want}, log)
    # (hookshot) the layouts with the Recomp options button: served at the RomFS's size and its bytes, the table's entry raised
    with_button = table(others=((0x10, 100), (0x20, 200), (0x40, 400), (LAYOUTS_CRC, 500)))
    for name, size, want in (("button-same", 64 + 24, "game/cache/prompts/Common.blarc.zs"), ("button-other", 64, "(the game's)")):
        served, log, top = case(exe, tmp, name, {TABLE: fixture.frame(with_button), LAYOUTS: fixture.frame(bytes(64))},
                                (TABLE, LAYOUTS), blarc=bytes(size), added=24)
        expect("prompts/Common.blarc of %d bytes, the RomFS's of 64 and the button's 24" % size, served,
               {TABLE: ours, LAYOUTS: want}, log)
        content(top, "the layouts' button", table(font=need(300), others=((0x10, 100), (0x20, 200), (0x40, 400),
                                                                           (LAYOUTS_CRC, 524))))
        if "the layouts' entry raised from 500 to 524 bytes" not in log:
            errors.append("%s: no log line of the layouts' entry raised\n%s" % (name, log))
    CHECK(not errors, "\n".join(errors))
    print("ok: 10 cases: the RomFS's own table of the name the game opens, its font entry raised (a build's table from "
          "another dump never served, either order; two versions' tables), the game's own table and icons for a name the "
          "RomFS has not and for a table without a font entry, room already, the layouts by their size, with the Recomp "
          "options button: their entry raised")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--source", type=Path, default=ROOT / "host", help="host source directory to exercise")
    a = p.parse_args()
    if os.name == "nt" or sys.platform == "darwin":
        print("SKIP: the prompts' files over folders run on Linux (host/posix's file calls)")
        return
    with tempfile.TemporaryDirectory(prefix="prompts-size-table-") as tmp:
        exe = fixture.build_fixture(a.cc, Path(tmp) / "build", a.source)
        checks(exe, Path(tmp))


if __name__ == "__main__":
    main()
