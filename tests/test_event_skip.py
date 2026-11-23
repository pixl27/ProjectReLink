"""Every cutscene can be skipped: the skip setting's IsDisableSkip set false (host/event_skip.c; no game or GPU).

The game's skip screen leaves an event unskippable when its setting list holds the component DisableSkip...SkipSetting.bgyml,
a BYML of one entry {IsDisableSkip: true}. hos_event_skip_byml is compiled on its own and run on BYMLs this test writes in
that layout (a key table of one name, a root dictionary of one bool): true becomes false and nothing else moves; false
already, another key, another type, a value of another kind, a truncated file and a file that is no BYML are left as they are.
The serving path is checked in the source: the lookup is mods.c's, after the button prompts and the texts, with a mod's own
file first; HOS_SKIP_ANY_EVENT is its switch and has a row.

python -I -B tests/test_event_skip.py [--cc gcc]
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cc", "gcc"]
ROOT = Path(__file__).resolve().parents[1]


def byml(name=b"IsDisableSkip", kind=0xD0, value=1):
    """A BYML v7 of one dictionary entry, laid out as the game's skip setting: header, key table, root dictionary"""
    keys_at, root_at = 0x10, 0x2C
    head = b"YB" + struct.pack("<HIII", 7, keys_at, 0, root_at)
    table = bytes([0xC2]) + (1).to_bytes(3, "little") + struct.pack("<II", 12, 12 + len(name) + 1) + name + b"\0"
    table += b"\0" * (root_at - keys_at - len(table))
    root = bytes([0xC1]) + (1).to_bytes(3, "little") + (0).to_bytes(3, "little") + bytes([kind]) + struct.pack("<I", value)
    return head + table + root


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
/* CUT */
int main(int argc, char **argv) {
    (void)argc;
    FILE *f = fopen(argv[1], "rb");
    unsigned char d[4096];
    size_t n = fread(d, 1, sizeof d, f);
    fclose(f);
    const int r = hos_event_skip_byml(d, n);
    f = fopen(argv[2], "wb");
    fwrite(d, 1, n, f);
    fclose(f);
    printf("%d\n", r);
    return 0;
}
"""


def cut(src, start, end_marker="\n}\n"):
    a = src.index(start)
    return src[a:src.index(end_marker, a) + len(end_marker)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="gcc")
    a = ap.parse_args()
    if not shutil.which(a.cc):
        raise SystemExit("no C compiler %r (pass --cc)" % a.cc)
    src = (ROOT / "host/event_skip.c").read_text(encoding="utf-8")
    code = src[src.index("static uint32_t rd24("):src.index("static int skip_off(void)")]
    mods = (ROOT / "host/mods.c").read_text(encoding="utf-8")
    if "hos_texts_romfs(key, NULL, out, cap) || hos_event_skip_romfs(key, NULL, out, cap)" not in mods:
        raise SystemExit("FAIL: mods.c does not look the skip setting up after the prompts and the texts")
    if "if (strcmp(key, SKIP_KEY) || mod || !hos_romfs_dir || skip_off()) return 0;" not in src:
        raise SystemExit("FAIL: event_skip.c serves its file over a mod's own, or without its switch")
    if "SW(HOS_SKIP_ANY_EVENT," not in (ROOT / "gpu/vk/switches.def").read_text(encoding="utf-8"):
        raise SystemExit("FAIL: HOS_SKIP_ANY_EVENT has no row in gpu/vk/switches.def")
    with tempfile.TemporaryDirectory(prefix="event-skip-") as tmp:
        t = Path(tmp)
        (t / "skip.c").write_text(HARNESS.replace("/* CUT */", code), encoding="utf-8")
        exe = t / ("skip.exe" if os.name == "nt" else "skip")
        subprocess.run([a.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(t / "skip.c"), "-o", str(exe)], check=True)

        def run(data):
            (t / "in").write_bytes(data)
            r = subprocess.run([str(exe), str(t / "in"), str(t / "out")], capture_output=True, text=True, check=True)
            return int(r.stdout), (t / "out").read_bytes()

        on = byml()
        if len(on) != 56:
            raise SystemExit("FAIL: the test's BYML is %d bytes, not the skip setting's 56" % len(on))
        r, out = run(on)
        if r != 1 or out != byml(value=0):
            raise SystemExit("FAIL: IsDisableSkip true was not made false, or more changed")
        for what, data in (("false already", byml(value=0)), ("another key", byml(name=b"IsDisableSkiq")),
                           ("another type", byml(kind=0xD1)), ("truncated", on[:40]), ("no BYML", b"XB" + on[2:])):
            r, out = run(data)
            if r != 0 or out != data:
                raise SystemExit("FAIL: a BYML %s was changed" % what)
    print("PASS: the skip setting's IsDisableSkip made false, and nothing else; the file served by mods.c after a mod's own")


if __name__ == "__main__":
    main()
