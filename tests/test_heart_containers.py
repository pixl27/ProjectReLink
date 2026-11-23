"""The Discord status's pieces of heart in all (host/oot_heart_containers.c): 4 * (hearts - 3 - boss containers taken) +
pieces held, the containers' ids read from the map files.

No game file: the SARC (Pack/ZsDic.pack.zs's dictionaries) and the BYML map files (an Actors list of dictionaries with
Gyaml and Hash) are made here, in both byte orders, with made-up names and ids. The readers find the named file and the
named actors' 64-bit (and 32-bit) ids, skip the others, keep to cap, and refuse what is not one of them; each file cut at
every length is refused or read without reading past its end (the harness is built with -fsanitize=address where the
compiler has it). A SARC whose string table passes 256 KiB is read too: the name's offset in an SFAT node is 24 bits,
and its bits 16 to 23 are the ones a 16-bit mask loses. The total adds up, and is -1 (not shown) when a part is not known
or the parts do not add up.
host/hos_main.c, host/discord_rpc.c, host/oot_state_tap.c and nx2/build.py are checked in their source for using them.

    python -I -B tests/test_heart_containers.py [--cc gcc]
"""
import argparse
import os
import struct
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cc", "gcc"]

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "host"


def u24(v, be):
    return v.to_bytes(3, "big" if be else "little")


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


def byml(root_actors, be, extra_root_key=None):
    """A BYML of {"Actors": [{"Gyaml": name, "Hash": (type, value)}, ...]} (and one more root key, to be skipped)"""
    e = ">" if be else "<"
    keys = sorted({"Actors", "Gyaml", "Hash"} | ({extra_root_key} if extra_root_key else set()))
    strs = sorted({a[0] for a in root_actors})

    def table(items):
        body = b"".join(s.encode() + b"\0" for s in items)
        head = 4 + 4 * (len(items) + 1)
        offs, at = [], head
        for s in items:
            offs.append(at)
            at += len(s.encode()) + 1
        offs.append(at)
        return pad4(bytes([0xc2]) + u24(len(items), be) + b"".join(struct.pack(e + "I", o) for o in offs) + body)

    out = bytearray(16)
    key_at = len(out)
    out += table(keys)
    str_at = len(out)
    out += table(strs)
    # (the 64-bit values, then each actor's dictionary, then the list, then the root)
    actor_offs = []
    for name, (t, v) in root_actors:
        if t == 0xd5:
            v_at = len(out)
            out += struct.pack(e + "Q", v)
            word = v_at
        else:
            word = v
        entries = sorted([(keys.index("Gyaml"), 0xa0, strs.index(name)), (keys.index("Hash"), t, word)])
        actor_offs.append(len(out))
        out += bytes([0xc1]) + u24(len(entries), be)
        for k, t2, w in entries:
            out += u24(k, be) + bytes([t2]) + struct.pack(e + "I", w)
    list_at = len(out)
    out += pad4(bytes([0xc0]) + u24(len(actor_offs), be) + bytes([0xc1] * len(actor_offs)))
    out += b"".join(struct.pack(e + "I", o) for o in actor_offs)
    root_at = len(out)
    entries = [(keys.index("Actors"), 0xc0, list_at)]
    if extra_root_key:
        entries.append((keys.index(extra_root_key), 0xd1, 7))
    entries.sort()
    out += bytes([0xc1]) + u24(len(entries), be)
    for k, t, w in entries:
        out += u24(k, be) + bytes([t]) + struct.pack(e + "I", w)
    out[0:16] = (b"BY" if be else b"YB") + struct.pack(e + "HIII", 7, key_at, str_at, root_at)
    return bytes(out)


def sarc(files, be):
    """A SARC of {name: bytes}"""
    e = ">" if be else "<"
    names = b""
    name_off = {}
    for n in files:
        name_off[n] = len(names)
        names += pad4(n.encode() + b"\0")
    sfat = b"SFAT" + struct.pack(e + "HHI", 12, len(files), 0x65)
    data, nodes = b"", b""
    for n, body in files.items():
        start = len(data)
        data += pad4(body)
        nodes += struct.pack(e + "IIII", 0, 0x01000000 | (name_off[n] // 4), start, start + len(body))
    sfnt = b"SFNT" + struct.pack(e + "HH", 8, 0) + names
    data_at = 0x14 + len(sfat) + len(nodes) + len(sfnt)
    data_at += -data_at % 8
    head = b"SARC" + struct.pack(e + "H", 0x14) + (b"\xfe\xff" if be else b"\xff\xfe")
    head += struct.pack(e + "IIHH", data_at + len(data), data_at, 0x100, 0)
    out = head + sfat + nodes + sfnt
    return out + b"\0" * (data_at - len(out)) + data


def deep_sarc(be):
    """A SARC whose wanted names sit past 256 KiB of string table, where the name offset needs its bits 16 to 23"""
    files = {}
    for i in range(1024):   # (each filler name is 255 bytes and a terminator: 1024 * 256 = 256 KiB, word 0x10000)
        files["f%04d" % i + "y" * 250] = b"filler"
    files["zs.zsdic"] = b"the-zs-d"
    files["bcett.byml.zsdic"] = b"the-bcett-d"
    return sarc(files, be)


def c_array(name, data):
    return "static const unsigned char %s[%d] = {%s};\n" % (name, max(len(data), 1), ",".join(str(b) for b in data) or "0")


HARNESS = r"""
#include "oot_heart_containers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int failed;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); failed = 1; } } while (0)
/* (each file copied to a heap block of its own size, so a read past its end is one the sanitizer sees) */
static unsigned char *copy(const unsigned char *d, size_t n) {
    unsigned char *p = malloc(n ? n : 1);
    memcpy(p, d, n);
    return p;
}
int main(void) {
    for (int be = 0; be < 2; be++) {
        const unsigned char *m = be ? MAP_BE : MAP_LE;
        const size_t mn = be ? sizeof MAP_BE : sizeof MAP_LE;
        unsigned long long ids[4] = {0};
        unsigned char *p = copy(m, mn);
        CHECK(heart_byml_actor_ids(p, mn, "Obj_Heart_Container", (uint64_t *)ids, 4) == 2);
        CHECK(ids[0] == 0x8a3c0f1e2d4b6977ull && ids[1] == 0x12345678ull);   /* (a 64-bit id, then a 32-bit one) */
        CHECK(heart_byml_actor_ids(p, mn, "Obj_Heart_Container", (uint64_t *)ids, 1) == 1);
        CHECK(heart_byml_actor_ids(p, mn, "Obj_Heart_Piece", (uint64_t *)ids, 4) == 0);
        CHECK(heart_byml_actor_ids(p, mn, "Obj_Other", (uint64_t *)ids, 4) == 1 && ids[0] == 0x0102030405060708ull);
        free(p);
        for (size_t k = 0; k < mn; k++) {   /* (cut short: refused or read within, never past) */
            p = copy(m, k);
            const int r = heart_byml_actor_ids(p, k, "Obj_Heart_Container", (uint64_t *)ids, 4);
            CHECK(r >= -1 && r <= 2);
            free(p);
        }
        const unsigned char *s = be ? PACK_BE : PACK_LE;
        const size_t sn = be ? sizeof PACK_BE : sizeof PACK_LE;
        size_t a = 0, b = 0;
        p = copy(s, sn);
        CHECK(heart_sarc_find(p, sn, "bcett.byml.zsdic", &a, &b) && b - a == 11 && !memcmp(p + a, "the-bcett-d", 11));
        CHECK(heart_sarc_find(p, sn, "zs.zsdic", &a, &b) && b - a == 8 && !memcmp(p + a, "the-zs-d", 8));
        CHECK(!heart_sarc_find(p, sn, "bcett.byml", &a, &b) && !heart_sarc_find(p, sn, "pack.zsdic", &a, &b));
        free(p);
        for (size_t k = 0; k < sn; k++) {
            p = copy(s, k);
            if (heart_sarc_find(p, k, "bcett.byml.zsdic", &a, &b)) CHECK(b <= k && a <= b);
            free(p);
        }
        const unsigned char *q = be ? PACK_DEEP_BE : PACK_DEEP_LE;
        const size_t qn = be ? sizeof PACK_DEEP_BE : sizeof PACK_DEEP_LE;
        p = copy(q, qn);
        CHECK(heart_sarc_find(p, qn, "bcett.byml.zsdic", &a, &b) && b - a == 11 && !memcmp(p + a, "the-bcett-d", 11));
        CHECK(heart_sarc_find(p, qn, "zs.zsdic", &a, &b) && b - a == 8 && !memcmp(p + a, "the-zs-d", 8));
        free(p);
    }
    unsigned long long ids[2];
    CHECK(heart_byml_actor_ids((const unsigned char *)"not a map file", 14, "Obj_Heart_Container", (uint64_t *)ids, 2) == -1);
    CHECK(heart_byml_actor_ids(NO_ACTORS, sizeof NO_ACTORS, "Obj_Heart_Container", (uint64_t *)ids, 2) == -1);
    size_t a, b;
    CHECK(!heart_sarc_find((const unsigned char *)"SARC but no more of it", 22, "zs.zsdic", &a, &b));
    /* (the total: 3 hearts to start, a container each, four pieces a heart) */
    CHECK(heart_pieces_total(36, 5, 1) == 5);    /* (the Water Temple save: 9 hearts, five containers, 1 held) */
    CHECK(heart_pieces_total(12, 0, 0) == 0);    /* (a new game) */
    CHECK(heart_pieces_total(12, 0, 3) == 3);
    CHECK(heart_pieces_total(80, 8, 0) == 36);   /* (20 hearts: all 36 pieces) */
    CHECK(heart_pieces_total(32, 5, 2) == 2);
    CHECK(heart_pieces_total(36, -1, 1) == -1);  /* (the containers not known) */
    CHECK(heart_pieces_total(-1, 5, 1) == -1 && heart_pieces_total(36, 5, -1) == -1);
    CHECK(heart_pieces_total(36, 5, 4) == -1);   /* (four held would be a heart) */
    CHECK(heart_pieces_total(35, 5, 1) == -1);   /* (most life is whole hearts) */
    CHECK(heart_pieces_total(28, 5, 0) == -1);   /* (7 hearts with five containers: does not add up) */
    CHECK(heart_pieces_total(36, 9, 0) == -1);
    if (!failed) puts("ok");
    return failed;
}
"""


def check_source():
    main = (HOST / "hos_main.c").read_text(encoding="utf-8")
    if "heart_pieces_total(stats[1], hos_oot_heart_containers_taken(), oot_pouch_count(c, \"Obj_Heart_Piece\"))" not in main:
        raise SystemExit("host/hos_main.c: the pieces of heart are not the total (heart_pieces_total) any more")
    rpc = (HOST / "discord_rpc.c").read_text(encoding="utf-8")
    if "hos_oot_heart_containers_load();" not in rpc:
        raise SystemExit("host/discord_rpc.c: the boss containers' ids are not loaded on the Discord status's thread")
    tap = (HOST / "oot_state_tap.c").read_text(encoding="utf-8")
    if "int hos_oot_gamedata_bool64(uint64_t id, int *at)" not in tap or "#define STATE_BOOL64 0xc58u" not in tap:
        raise SystemExit("host/oot_state_tap.c: the Bool64bitKey flag reader is gone")
    build = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
    if '"oot_heart_containers.c")' not in build:
        raise SystemExit("nx2/build.py: host/oot_heart_containers.c is not in ZSTD_HOST (it needs zstd.h)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    a = p.parse_args()
    check_source()
    actors = [("Obj_Heart_Container", (0xd5, 0x8a3c0f1e2d4b6977)), ("Obj_Other", (0xd5, 0x0102030405060708)),
              ("Obj_Heart_Container", (0xd3, 0x12345678))]
    data = ""
    for be in (0, 1):
        tag = "BE" if be else "LE"
        data += c_array("MAP_" + tag, byml(actors, be, extra_root_key="Version"))
        data += c_array("PACK_" + tag, sarc({"zs.zsdic": b"the-zs-d", "bcett.byml.zsdic": b"the-bcett-d"}, be))
        data += c_array("PACK_DEEP_" + tag, deep_sarc(be))
    no_actors = bytearray(byml(actors, 0))
    root = struct.unpack_from("<I", no_actors, 12)[0]
    no_actors[root] = 0xc0   # (the root no longer a dictionary)
    data += c_array("NO_ACTORS", bytes(no_actors))
    with tempfile.TemporaryDirectory(prefix="heart-containers-") as tmp:
        root_dir = Path(tmp).resolve()
        src, exe = root_dir / "heart_test.c", root_dir / "heart_test"
        src.write_text(data + HARNESS, encoding="utf-8")
        windows = os.name == "nt"
        extra = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS} if windows else {}
        base = [a.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-DHEART_CONTAINERS_NO_GAME", "-I", str(HOST), str(src),
                str(HOST / "oot_heart_containers.c"), "-o", str(exe)]
        build = subprocess.run(base[:1] + ["-fsanitize=address", "-g"] + base[1:], capture_output=True, **extra)
        if build.returncode:   # (no sanitizer on this compiler: the plain build)
            build = subprocess.run(base, **extra)
            if build.returncode:
                raise SystemExit("the host did not compile: see the compiler's output above")
        run = subprocess.run([str(exe)], capture_output=True, text=True, **extra)
        print(run.stderr, end="")
        if run.returncode or run.stdout.strip() != "ok":
            raise SystemExit("the heart containers check failed")
    print("PASS: the pieces of heart in all add up from the hearts, the boss containers taken and the pieces held; the "
          "containers' ids are read from a SARC's dictionary and BYML map files of either byte order, cut short or not")


if __name__ == "__main__":
    main()
