"""Exefs mods (nx2/exefs_mods.py, host/mods_exefs.c), game-free: python -I nx2/test_exefs_mods.py

Synthetic modules made here (no game data): the two IPS kinds and pchtxt read, offsets at the segments' edges, patches for
another build, conflicts, the "no patch: the game's own exefs, nothing written" promise, the patched NSO read back by
a64-kit's nso.py and the readers make_game.py runs, the lift's checks on a stand-in lifter, ReLink's own patches when an
exefs mod is in their way, and the patch files' hash equal in Python and in the game's C (built with llvm-mingw's clang:
RELINK_MINGW, or F:/Switch2build/llvm-mingw; skipped without it).
"""
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "third_party", "a64-kit", "tools")]
import exefs_mods  # noqa: E402
from exefs_mods import PatchError, parse_ips, parse_pchtxt  # noqa: E402

try:
    import lz4.block
except ImportError:   # (the builder's Python has it: nso.py needs it)
    lz4 = None

PAGE = 0x1000
MAIN_ID = bytes.fromhex("11223344556677889900aabbccddeeff01020304") + bytes(12)
SDK_ID = bytes.fromhex("a0a1a2a3a4a5a6a7a8a9b0b1b2b3b4b5b6b7b8b9") + bytes(12)
RTLD_ID = bytes.fromhex("0f0e0d0c0b0a09080706050403020100ffeeddcc") + bytes(12)
NOP = struct.pack("<I", 0xD503201F)


def segments(text_size=0x2000, ro_size=0x800, data_size=0x400, seed=1):
    """A module's three segments: .text starts with its MOD0 pointer (nso.py needs one), .data holds an empty dynamic section"""
    text = bytearray(NOP * (text_size // 4))
    ro_off = (text_size + PAGE - 1) // PAGE * PAGE
    data_off = ro_off + (ro_size + PAGE - 1) // PAGE * PAGE
    struct.pack_into("<I", text, 4, 0x10)   # (MOD0 at .text+0x10)
    struct.pack_into("<4s6i", text, 0x10, b"MOD0", data_off - 0x10, data_off + data_size - 0x10,
                     data_off + data_size + 0x100 - 0x10, 0, 0, 0)
    ro = bytearray((seed * 7 + i) & 0xFF for i in range(ro_size))
    ro[0x100:0x110] = b"ThreadName\0\0\0\0\0\0"
    data = bytearray(data_size)   # (the dynamic section: DT_NULL first)
    return [(0, bytes(text)), (ro_off, bytes(ro)), (data_off, bytes(data))]


def make_nso(path, segs, build_id, lz4_segments=(), name=b"main"):
    """An NSO file: the 0x100 header, the module name, then the segments (LZ4 where asked), with their SHA-256"""
    hdr = bytearray(0x100)
    hdr[0:4] = b"NSO0"
    body = bytearray(name)
    flags = 0
    stored = []
    for i, (_, d) in enumerate(segs):
        if i in lz4_segments:
            stored.append(lz4.block.compress(d, store_size=False))
            flags |= 1 << i
        else:
            stored.append(d)
        flags |= 8 << i
    struct.pack_into("<I", hdr, 0x1C, 0x100)            # (module name offset)
    struct.pack_into("<I", hdr, 0x2C, len(name))        # (module name size)
    for i, ((moff, d), s) in enumerate(zip(segs, stored)):
        struct.pack_into("<3I", hdr, 0x10 + 0x10 * i, 0x100 + len(body), moff, len(d))
        struct.pack_into("<I", hdr, 0x60 + 4 * i, len(s))
        hdr[0xA0 + 0x20 * i:0xC0 + 0x20 * i] = hashlib.sha256(d).digest()
        body += s
    struct.pack_into("<I", hdr, 0x3C, 0x100)   # (bss)
    struct.pack_into("<I", hdr, 0xC, flags)
    hdr[0x40:0x60] = build_id
    with open(path, "wb") as f:
        f.write(hdr + body)


def image_of(segs):
    end = segs[2][0] + len(segs[2][1])
    img = bytearray(end)
    for moff, d in segs:
        img[moff:moff + len(d)] = d
    return img


def ips32(records):
    out = bytearray(b"IPS32")
    for off, value in records:
        if isinstance(value, tuple):   # (a run: count, byte)
            out += struct.pack(">IHHB", off, 0, value[0], value[1])
        else:
            out += struct.pack(">IH", off, len(value)) + value
    return bytes(out + b"EEOF")


def ips(records):
    out = bytearray(b"PATCH")
    for off, value in records:
        if isinstance(value, tuple):
            out += off.to_bytes(3, "big") + struct.pack(">HHB", 0, value[0], value[1])
        else:
            out += off.to_bytes(3, "big") + struct.pack(">H", len(value)) + value
    return bytes(out + b"EOF")


def read(path):
    with open(path, "rb") as f:
        return f.read()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def quiet(fn, *args, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Formats(unittest.TestCase):
    def test_ips32_records_runs_and_end(self):
        data = ips32([(0x100, b"\x01\x02\x03"), (0x2000, (5, 0xAB)), (0x01020304, b"\xff")])
        self.assertEqual(parse_ips(data), [(0x100, b"\x01\x02\x03"), (0x2000, b"\xab" * 5), (0x01020304, b"\xff")])

    def test_ips_records_runs_end_and_truncation_ignored(self):
        data = ips([(0x123, b"\xaa\xbb"), (0x10, (3, 0x00))]) + b"\x00\x10\x00"   # (the old truncation size after EOF)
        self.assertEqual(parse_ips(data), [(0x123, b"\xaa\xbb"), (0x10, b"\x00" * 3)])

    def test_ips_empty_and_bad(self):
        self.assertEqual(parse_ips(b"IPS32EEOF"), [])
        self.assertEqual(parse_ips(b"PATCHEOF"), [])
        for bad in (b"NOTIPS", b"IPS32\x00\x00\x01\x00\x00\x04ab", b"PATCH\x00\x01\x00\x00\x05abcde",
                    b"IPS32\x00\x00\x01\x00\x00\x00\x00", b"PATCH\x00\x00\x10\x00\x02"):
            with self.assertRaises(PatchError, msg=bad):
                parse_ips(bad)

    def test_pchtxt_rules(self):
        text = "\n".join([
            "# A title line",
            "@nsobid-1122334455667788990000000000",
            "@flag print_values",
            "0000 DEADBEEF        // before @enabled: not applied",
            "@enabled",
            "00000110 1F2003D5    // a NOP, bytes as written",
            "@flag offset_shift 0x100",
            "0x20 \"Name // not a comment\\0\"",
            "@flag be",
            "30 D503201F",
            "@flag le",
            "40 0102",
            "@disabled",
            "50 not hex at all, but disabled",
            "@enabled",
            "60 \"a\\x41\\n\\t\\\\\\\"\"",
            "@stop",
            "70 this is never read",
        ])
        parts, notes = parse_pchtxt(text)
        self.assertEqual(list(parts), ["1122334455667788990000000000"])
        self.assertEqual(parts["1122334455667788990000000000"], [
            (0x110, bytes.fromhex("1F2003D5")), (0x120, b"Name // not a comment\0"), (0x130, bytes.fromhex("1F2003D5")),
            (0x140, b"\x01\x02"), (0x160, b"aA\n\t\\\"")])
        self.assertEqual(notes, [])

    def test_pchtxt_shift_forms(self):
        for shift in ("0x100", "256", "0100"):
            parts, _ = parse_pchtxt("@nsobid-ab\n@flag offset_shift %s\n@enabled\n10 00\n" % shift)
            self.assertEqual(parts["ab"], [(0x110, b"\0")], shift)

    def test_pchtxt_notes_and_bad_lines(self):
        _, notes = parse_pchtxt("@nsobid-ab\n@flag something_new\n@whatever\n@enabled\n10 00\n")
        self.assertEqual(len(notes), 2)
        for bad in ("@nsobid-ab\n@enabled\n10 0\n",            # (odd hex digits)
                    "@nsobid-ab\n@enabled\nzz 00\n",           # (offset not hex)
                    "@nsobid-ab\n@enabled\n10\n",              # (no value)
                    "@nsobid-ab\n@enabled\n10 \"open\n",       # (string not closed)
                    "@nsobid-ab\n@enabled\n10 \"a\\q\"\n",     # (unknown escape)
                    "@enabled\n10 00\n",                       # (before any @nsobid-)
                    "@nsobid-xyz\n",                           # (not a build id)
                    "@nsobid-ab\n@flag offset_shift -0x100\n@enabled\n10 00\n"):   # (below 0)
            with self.assertRaises(PatchError, msg=bad):
                parse_pchtxt(bad)

    def test_same_build(self):
        self.assertTrue(exefs_mods.same_build("11223344556677889900AABBCCDDEEFF01020304", MAIN_ID.hex()))
        self.assertTrue(exefs_mods.same_build("1122334455667788990000", "11223344556677889900"))
        self.assertFalse(exefs_mods.same_build("1122334455667788", MAIN_ID.hex()))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="relink_exefs_")
        self.exefs = os.path.join(self.tmp, "exefs")
        self.work = os.path.join(self.tmp, "work")
        self.mods = os.path.join(self.tmp, "mods")
        os.makedirs(self.exefs)
        os.makedirs(self.work)
        self.main = segments(seed=1)
        self.sdk = segments(text_size=0x3000, seed=2)
        make_nso(os.path.join(self.exefs, "main"), self.main, MAIN_ID)
        make_nso(os.path.join(self.exefs, "sdk"), self.sdk, SDK_ID, lz4_segments=(0, 1, 2) if lz4 else (), name=b"sdk")
        make_nso(os.path.join(self.exefs, "rtld"), segments(seed=3), RTLD_ID, name=b"rtld")
        write(os.path.join(self.exefs, "main.npdm"), b"META" + bytes(60))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def prepare(self):
        return quiet(exefs_mods.prepare, self.exefs, self.mods, self.work)


class NoPatch(Fixture):
    def check_untouched(self, r):
        self.assertEqual(r.exefs, self.exefs)
        self.assertIsNone(r.description)
        self.assertFalse(os.path.exists(os.path.join(self.work, exefs_mods.COPY)))
        self.assertFalse(os.path.exists(os.path.join(self.work, exefs_mods.DESCRIPTION)))
        self.assertEqual(os.listdir(self.work), [])

    def test_no_mods_folder(self):
        r = self.prepare()
        self.check_untouched(r)
        self.assertEqual(r.hash, "none")
        self.assertEqual(r.lines, [])

    def test_romfs_mods_and_off_mods_only(self):
        write(os.path.join(self.mods, "textures", "romfs", "a.bin"), b"x")
        write(os.path.join(self.mods, "_off", "exefs", MAIN_ID.hex()[:40] + ".ips"), ips32([(0x200, b"\x01")]))
        write(os.path.join(self.mods, "-off2", "exefs", MAIN_ID.hex()[:40] + ".ips"), ips32([(0x200, b"\x01")]))
        r = self.prepare()
        self.check_untouched(r)
        self.assertEqual(r.hash, "none")

    def test_patches_left_out_or_writing_the_same_bytes(self):
        write(os.path.join(self.mods, "other", "exefs", "99" * 20 + ".ips"), ips32([(0x200, b"\x01")]))
        same = self.main[0][1][0x200:0x204]
        write(os.path.join(self.mods, "same", "exefs", MAIN_ID.hex()[:40] + ".ips"), ips32([(0x300, same)]))
        r = self.prepare()
        self.check_untouched(r)
        self.assertNotEqual(r.hash, "none")
        self.assertTrue(any(l.startswith("left out  other/exefs/") and "another version" in l for l in r.lines), r.lines)
        self.assertTrue(any(l.startswith("applied   same/exefs/") and "0 bytes changed" in l for l in r.lines), r.lines)


class Applied(Fixture):
    def setUp(self):
        super().setUp()
        bid = MAIN_ID.hex()[:40]
        text_end = len(self.main[0][1])
        ro_off, ro = self.main[1]
        data_off, data = self.main[2]
        # 01-a: IPS32 into main's .text, a run, and the text's very last bytes (an edge, applied)
        write(os.path.join(self.mods, "01-a", "exefs", bid + ".ips"),
              ips32([(0x100 + 0x40, b"\x11\x22\x33\x44"), (0x100 + 0x80, (8, 0xCC)), (0x100 + text_end - 4, b"\xAA\xBB\xCC\xDD")]))
        # 02-b: pchtxt, image offsets (offset_shift 0x100): the .rodata name, a conflict with 01-a, the .data's first and last bytes
        write(os.path.join(self.mods, "02-b", "exefs", "patch.pchtxt"), "\n".join([
            "@nsobid-" + bid.upper() + "000000",
            "@flag offset_shift 0x100",
            "@enabled",
            "%x \"Threadname\"" % (ro_off + 0x100),
            "40 11229944   // byte 2 differs from 01-a's",
            "%x 01" % data_off,
            "%x 02" % (data_off + len(data) - 1),
            ""]))
        # 03-c: IPS (old kind) into sdk (LZ4 in the game's exefs), its name in capitals with trailing zeros
        write(os.path.join(self.mods, "03-c", "exefs", SDK_ID.hex().upper() + ".ips"), ips([(0x100 + 0x1000, b"\x5A" * 4)]))
        # deep in the layout emulators use: found (depth 3)
        write(os.path.join(self.mods, "contents", "0100000000000000", "04-deep", "exefs", bid + ".ips"),
              ips32([(0x100 + 0x60, b"\x77")]))
        # too deep (depth 5): not a mod
        write(os.path.join(self.mods, "a", "b", "c", "d", "05-toodeep", "exefs", bid + ".ips"), ips32([(0x100 + 0x64, b"\x66")]))
        # edges, each left out whole (its first record is fine and must not be applied either)
        edges = {"06-header": 0xFF, "07-cross": 0x100 + text_end - 2, "08-gap": 0x100 + ro_off + len(ro) + 0x10,
                 "09-bss": 0x100 + data_off + len(data)}
        for name, off in edges.items():
            write(os.path.join(self.mods, name, "exefs", bid + ".ips"), ips32([(0x100 + 0x500, b"\x99"), (off, b"\x01\x02\x03\x04")]))
        write(os.path.join(self.mods, "10-rtld", "exefs", RTLD_ID.hex()[:40] + ".ips"), ips32([(0x200, b"\x01")]))
        write(os.path.join(self.mods, "11-badname", "exefs", "notabuildid.ips"), ips32([(0x200, b"\x01")]))
        write(os.path.join(self.mods, "12-broken", "exefs", "bad.pchtxt"), "@nsobid-%s\n@enabled\n500 0g\n" % bid)
        self.r = self.prepare()
        self.copy = os.path.join(self.work, exefs_mods.COPY)

    def test_result_and_lines(self):
        r = self.r
        self.assertEqual(r.exefs, self.copy)
        self.assertEqual(r.description, os.path.join(self.work, exefs_mods.DESCRIPTION))
        text = "\n".join(r.lines)
        for name in ("06-header", "07-cross", "08-gap", "09-bss", "10-rtld", "11-badname", "12-broken"):
            self.assertIn("left out  %s/exefs/" % name, text)
        self.assertIn("the NSO header", text)
        self.assertIn("past its end", text)
        self.assertIn("gap between two segments", text)
        self.assertIn("bss", text)
        self.assertIn("rtld", text)
        self.assertNotIn("05-toodeep", text)
        self.assertIn("conflict  main+0x42 (1 byte): 02-b/exefs/patch.pchtxt wins over 01-a/exefs/", text)
        self.assertIn("applied   contents/0100000000000000/04-deep/exefs/", text)
        self.assertIn("applied   03-c/exefs/", text)

    def test_patched_main_read_back(self):
        import nso
        m = nso.Nso(os.path.join(self.copy, "main"))
        want = image_of(self.main)
        ro_off = self.main[1][0]
        data_off, data = self.main[2]
        text_end = len(self.main[0][1])
        want[0x40:0x44] = b"\x11\x22\x99\x44"           # (02-b wins at +0x42)
        want[0x80:0x88] = b"\xCC" * 8
        want[text_end - 4:text_end] = b"\xAA\xBB\xCC\xDD"
        want[ro_off + 0x100:ro_off + 0x10A] = b"Threadname"
        want[data_off] = 1
        want[data_off + len(data) - 1] = 2
        want[0x60] = 0x77
        self.assertEqual(bytes(m.image[:len(want)]), bytes(want))
        self.assertEqual(m.build_id, MAIN_ID.rstrip(b"\0").hex())
        self.assertEqual([s[:2] for s in m.segments], [(o, len(d)) for o, d in self.main])
        raw = read(os.path.join(self.copy, "main"))
        self.assertEqual(struct.unpack_from("<I", raw, 0xC)[0] & 7, 0)
        for i, (foff, moff, size) in enumerate(struct.unpack_from("<3I", raw, 0x10 + 0x10 * i) for i in range(3)):
            self.assertEqual(raw[0xA0 + 0x20 * i:0xC0 + 0x20 * i], hashlib.sha256(raw[foff:foff + size]).digest())

    def test_patched_lz4_sdk_and_copies(self):
        import nso
        if lz4 is None:
            self.skipTest("lz4 is not installed (the builder's Python has it)")
        m = nso.Nso(os.path.join(self.copy, "sdk"))
        want = image_of(self.sdk)
        want[0x1000:0x1004] = b"\x5A" * 4
        self.assertEqual(bytes(m.image[:len(want)]), bytes(want))
        for name in ("rtld", "main.npdm"):
            self.assertEqual(read(os.path.join(self.copy, name)), read(os.path.join(self.exefs, name)))
        self.assertEqual(sorted(os.listdir(self.copy)), sorted(os.listdir(self.exefs)))

    def test_game_exefs_untouched(self):
        self.assertEqual(nso_bytes(os.path.join(self.exefs, "main")), nso_bytes_expected(self.main, MAIN_ID))

    def test_make_game_readers(self):
        import make_game
        make_game.check_modules(self.copy)
        self.assertEqual(make_game.build_id(os.path.join(self.copy, "main")), make_game.build_id(os.path.join(self.exefs, "main")))
        import kit  # noqa: F401
        self.assertEqual(kit.game_modules(self.copy), ["main", "sdk"])
        import nso2  # noqa: F401  (the lift's NSO reader: DT_RELR)
        import nso
        self.assertEqual([m.name for m in nso.load_exefs(self.copy)], ["rtld", "main", "sdk"])
        import image
        img = image.EdenImage(self.copy)
        main = next(m for m in img.modules if m.name == "main")
        self.assertEqual(img.read(main.base + 0x40, 4), b"\x11\x22\x99\x44")

    def test_description(self):
        with open(self.r.description) as f:
            desc = json.load(f)
        by_file = {p["file"]: p for p in desc["patches"]}
        a = next(p for f, p in by_file.items() if f.startswith("01-a/"))
        self.assertEqual(a["module"], "main")
        text_end = len(self.main[0][1])
        self.assertEqual(a["ranges"], [[0x80, 8], [text_end - 4, 4]])   # (+0x40..0x43: 02-b wrote them last)
        b = by_file["02-b/exefs/patch.pchtxt"]
        self.assertIn([0x40, 4], b["ranges"])
        # (the 4 text bytes, 1 of "Threadname" over "ThreadName", the 2 data bytes: only the bytes that differ from the game's)
        self.assertEqual(sum(n for _, n in b["ranges"]), 4 + 1 + 2)

    def test_report(self):
        out = os.path.join(self.tmp, "game")
        os.makedirs(out)
        exefs_mods.write_report(out, self.r, ["warning   x: y"])
        lines = read(os.path.join(out, exefs_mods.REPORT)).decode("utf-8").splitlines()
        self.assertEqual(lines[0], self.r.hash)
        self.assertEqual(exefs_mods.read_report_hash(out), self.r.hash)
        self.assertEqual(lines[-1], "warning   x: y")
        self.assertRegex(self.r.hash, "^[0-9a-f]{16}$")

    def test_hash_changes_with_contents_and_set(self):
        h = self.r.hash
        files = exefs_mods.patch_files(self.mods)
        contents = {rel: read(p) for rel, p, _ in files}
        self.assertEqual(exefs_mods.files_hash(contents), h)
        changed = dict(contents)
        changed["02-b/exefs/patch.pchtxt"] += b" "
        self.assertNotEqual(exefs_mods.files_hash(changed), h)
        fewer = dict(contents)
        del fewer["02-b/exefs/patch.pchtxt"]
        self.assertNotEqual(exefs_mods.files_hash(fewer), h)

    def test_prepare_again_same_bytes(self):
        before = {n: read(os.path.join(self.copy, n)) for n in os.listdir(self.copy)}
        r = self.prepare()
        self.assertEqual(r.hash, self.r.hash)
        self.assertEqual({n: read(os.path.join(self.copy, n)) for n in os.listdir(self.copy)}, before)


class CodeCave(Fixture):
    """(Discord, 19/11) mods that put their own code in the rest of the .text's last page: applied, the .text that much longer"""
    CAVE = b"\x1F\x20\x03\xD5\xC0\x03\x5F\xD6"   # (nop; ret)

    def setUp(self):
        super().setUp()
        self.main = segments(text_size=0x1D40, seed=1)   # (.text ends 0x1d40, .rodata at 0x2000)
        make_nso(os.path.join(self.exefs, "main"), self.main, MAIN_ID)
        bid = MAIN_ID.hex()[:40]
        # cave: a branch into the cave, and the cave's code at 0x1e00; past: from the page's last word into the .rodata
        write(os.path.join(self.mods, "cave", "exefs", bid + ".ips"),
              ips32([(0x100 + 0x40, b"\x70\x07\x00\x14"), (0x100 + 0x1E00, self.CAVE)]))
        write(os.path.join(self.mods, "past", "exefs", bid + ".ips"), ips32([(0x100 + 0x1FFC, b"\x01\x02\x03\x04\x05")]))
        self.r = self.prepare()

    def test_cave_applied_and_the_page_end_kept(self):
        text = "\n".join(self.r.lines)
        self.assertIn("applied   cave/exefs/", text)
        self.assertIn("left out  past/exefs/", text)
        self.assertIn(".text's last page, and goes past its end (0x2000)", text)

    def test_patched_text_reaches_the_cave(self):
        import nso
        path = os.path.join(self.work, exefs_mods.COPY, "main")
        m = nso.Nso(path)
        self.assertEqual([s[:2] for s in m.segments], [(0, 0x1E08), (0x2000, 0x800), (0x3000, 0x400)])
        self.assertEqual(bytes(m.image[0x1D40:0x1E00]), bytes(0xC0))   # (zeros between the game's code and the cave)
        self.assertEqual(bytes(m.image[0x1E00:0x1E08]), self.CAVE)
        self.assertEqual(m.code_range(), (0, 0x1E08))
        self.assertEqual(bytes(m.image[0x2000:0x2800]), self.main[1][1])
        raw = read(path)
        for i in range(3):
            foff, _, size = struct.unpack_from("<3I", raw, 0x10 + 0x10 * i)
            self.assertEqual(raw[0xA0 + 0x20 * i:0xC0 + 0x20 * i], hashlib.sha256(raw[foff:foff + size]).digest())

    def test_description_counts_only_the_patch(self):
        with open(self.r.description) as f:
            desc = json.load(f)
        (cave,) = desc["patches"]
        self.assertEqual(cave["ranges"], [[0x40, 4], [0x1E00, 8]])


def nso_bytes(path):
    return read(path)


def nso_bytes_expected(segs, build_id):
    d = tempfile.mkdtemp()
    try:
        make_nso(os.path.join(d, "x"), segs, build_id)
        return read(os.path.join(d, "x"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


class Hash(Fixture):
    def clang(self):
        for c in (os.path.join(os.environ.get("RELINK_MINGW", ""), "bin", "clang.exe"), r"F:\Switch2build\llvm-mingw\bin\clang.exe",
                  shutil.which("clang") or ""):
            if c and os.path.isfile(c):
                return c
        self.skipTest("no clang to build host/mods_exefs.c with")

    def test_python_and_c_agree(self):
        cc = self.clang()
        bid = MAIN_ID.hex()[:40]
        write(os.path.join(self.mods, "B-mod", "exefs", bid + ".IPS"), ips32([(0x200, b"\x01")]))
        write(os.path.join(self.mods, "a-mod", "exefs", "z.pchtxt"), "@nsobid-%s\n@enabled\n200 02\n" % bid)
        write(os.path.join(self.mods, "a-mod", "exefs", "Y.PchTxt"), "")
        write(os.path.join(self.mods, "a-mod", "exefs", "notes.txt"), "not a patch")
        write(os.path.join(self.mods, "a-mod", "romfs", "x.bin"), b"r")
        write(os.path.join(self.mods, "Ünïcode mod", "exefs", bid + ".ips"), ips32([(0x204, b"\x03")]))
        write(os.path.join(self.mods, "contents", "0100", "deep", "exefs", bid + ".ips"), ips32([(0x208, b"\x04")]))
        write(os.path.join(self.mods, "_off", "exefs", bid + ".ips"), ips32([(0x20C, b"\x05")]))
        write(os.path.join(self.mods, ".hidden", "exefs", bid + ".ips"), ips32([(0x20C, b"\x05")]))
        os.makedirs(os.path.join(self.mods, "a-mod", "exefs", "sub.ips"))   # (a folder, not a patch)
        write(os.path.join(self.mods, "romfsonly", "romfs", "y.bin"), b"r")
        exe = os.path.join(self.tmp, "hash.exe")
        main_c = os.path.join(self.tmp, "main.c")
        write(main_c, "\n".join([
            "#include <stdio.h>", "#include <stdint.h>",
            "int hos_exefs_patch_hash(const char *root, uint64_t *hash, void (*each)(void *, const char *, uint64_t), void *arg);",
            "static void each(void *a, const char *rel, uint64_t size) { (void)a; printf(\"file %s %llu\\n\", rel, (unsigned long long)size); }",
            "int main(int argc, char **argv) { uint64_t h = 0; int n = argc > 1 ? hos_exefs_patch_hash(argv[1], &h, each, 0) : 0;",
            "  if (n) printf(\"hash %016llx\\n\", (unsigned long long)h); else printf(\"hash none\\n\"); return 0; }", ""]))
        # (the game's two ways: Windows' wide-character calls, and the POSIX ones Linux builds, here on MinGW's dirent.h, whose
        # names are in the ANSI code page: the mod with letters outside ASCII is for the first only)
        for variant, flags in (("windows", []), ("posix", ["-DMODS_EXEFS_POSIX"])):
            if variant == "posix":
                shutil.rmtree(os.path.join(self.mods, "Ünïcode mod"))
            r = subprocess.run([cc, "-O2", "-Wall", "-Werror"] + flags + [main_c, os.path.join(ROOT, "host", "mods_exefs.c"), "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = subprocess.run([exe, self.mods], capture_output=True).stdout.decode("utf-8").splitlines()
            files = exefs_mods.patch_files(self.mods)
            contents = {rel: read(p) for rel, p, _ in files}
            want = ["file %s %d" % (rel, len(contents[rel])) for rel in sorted(contents, key=lambda k: exefs_mods._fold(k))]
            want.append("hash " + exefs_mods.files_hash(contents))
            self.assertEqual(out, want, variant)
            # (B-mod, a-mod's two, Ünïcode mod, deep; not notes.txt, sub.ips, _off, .hidden)
            self.assertEqual(len(files), 5 if variant == "windows" else 4)
            empty = subprocess.run([exe, os.path.join(self.tmp, "nothing")], capture_output=True).stdout.decode().split()
            self.assertEqual(empty, ["hash", "none"], variant)
        self.assertEqual(exefs_mods.files_hash({}), "none")


# ---------------------------------------------------------------- the lift's side, on stand-ins

class FakeModule:
    def __init__(self, name, base, size):
        self.name, self.base, self.image = name, base, bytearray(size)


class FakeImage:
    def __init__(self):
        self.modules = [FakeModule("main", 0x80020000, 0x10000)]
        self.hooks = {0x80024000: "imp_nnFoo"}
        self.hook_syms = {"imp_nnFoo": "_ZN2nn3FooEv"}

    def in_code(self, a):
        return not a & 3 and 0x80020000 <= a < 0x80028000


class FakeLifter:
    def __init__(self):
        self.funcs = {0x80021000: ({0x80021000 + 4 * i: (("op",), 0) for i in range(16)}, set())}
        self.eh_starts = [0x80021000, 0x80022000, 0x80024000, 0x80025000]


class Lift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="relink_exefs_lift_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lift_check(self):
        desc = {"format": 1, "patches": [
            {"mod": "m", "file": "m/exefs/a.ips", "module": "main", "ranges": [[0x1004, 2], [0x2000, 4]]},
            {"mod": "n", "file": "n/exefs/b.pchtxt", "module": "main", "ranges": [[0x4008, 4], [0xC000, 8]]}]}
        path = os.path.join(self.tmp, "d.json")
        with open(path, "w") as f:
            json.dump(desc, f)
        img = FakeImage()
        p = exefs_mods.Patched(path, img)
        self.assertEqual(sorted(p.words), [0x80021004, 0x80022000, 0x80024008])   # (.data's 0xc000 is not code)
        touched, lines = exefs_mods.lift_check(FakeLifter(), img, p)
        self.assertEqual(touched, {"f_80021000": ["m/exefs/a.ips"]})
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("no effect n/exefs/b.pchtxt: it changes _ZN2nn3FooEv"), lines)
        self.assertTrue(lines[1].startswith("not translated m/exefs/a.ips: 1 instruction word"), lines)
        self.assertIn("main+0x2000", lines[1])
        site = exefs_mods.site_lines("field of view", [0x80021004, 0x80021008], p, img)
        self.assertEqual(site, ["warning   m/exefs/a.ips: it changes main+0x1004, where ReLink's field of view goes: left out there"])


class OwnPatches(unittest.TestCase):
    def setUp(self):
        self.gen = tempfile.mkdtemp(prefix="relink_exefs_gen_")

    def tearDown(self):
        shutil.rmtree(self.gen, ignore_errors=True)

    def code(self, text):
        write(os.path.join(self.gen, "code_000.c"), '#include "a64.h"\n' + text)

    def test_output_size_stops_with_the_mods_name(self):
        import oot_patches
        self.code("void f_810a7b70(Ctx *__restrict A64_NOESC c) {\n}\n")
        with self.assertRaises(oot_patches.ExefsModConflict) as e:
            quiet(oot_patches.apply_output_size, self.gen, required=True, modded={"f_810a7b70": ["res/exefs/x.ips"]})
        self.assertIn("res/exefs/x.ips", str(e.exception))
        self.assertIn("f_810a7b70", str(e.exception))
        with self.assertRaises(SystemExit) as e:   # (no mod in the way: the stop as before)
            quiet(oot_patches.apply_output_size, self.gen, required=True, modded={"f_80000000": ["other"]})
        self.assertNotIsInstance(e.exception, oot_patches.ExefsModConflict)
        quiet(oot_patches.apply_output_size, self.gen, required=False, modded={"f_810a7b70": ["res/exefs/x.ips"]})

    def test_native_replacement_left_out_and_warnings(self):
        import oot_patches
        self.code(oot_patches.ZSTD_HEAD + "}\n" + oot_patches.TEXT_HEAD + "\n}\n")
        notes = []
        modded = {"f_801102e0": ["z/exefs/z.ips"], "f_80b1e140": ["t/exefs/t.pchtxt"]}
        with self.assertRaises(SystemExit):   # (the output size places are not in this file: required stops, after the rest)
            quiet(oot_patches.apply, self.gen, protect_hearts=True, required=True, modded=modded, notes=notes)
        s = read(os.path.join(self.gen, "code_000.c")).decode()
        self.assertNotIn("hos_zstd_seq", s)
        self.assertTrue(any("z/exefs/z.ips" in n and "natively" in n for n in notes), notes)
        self.assertTrue(any("t/exefs/t.pchtxt" in n and "text speed block" in n for n in notes), notes)
        self.code(oot_patches.ZSTD_HEAD + "}\n")
        quiet(oot_patches.apply, self.gen, protect_hearts=True, required=False)
        self.assertIn("hos_zstd_seq", read(os.path.join(self.gen, "code_000.c")).decode())


class Run(unittest.TestCase):
    def test_plain_stop(self):
        import make_game
        d = tempfile.mkdtemp()
        try:
            stop = os.path.join(d, "stop.txt")
            script = os.path.join(d, "s.py")
            write(script, "import sys\nopen(sys.argv[1], 'w').write('The plain reason.')\nprint('Traceback: noise')\nsys.exit(1)\n")
            buf = io.StringIO()
            with redirect_stdout(buf), self.assertRaises(SystemExit):
                make_game.run([sys.executable, "-I", script, stop], plain_stop=stop)
            self.assertIn("@@error The plain reason.", buf.getvalue())
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
