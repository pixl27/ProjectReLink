"""nx2/build.py's object keys follow every file an #include names, whatever its extension: gpu/vk/switches.h includes
switches.def twice, and before, the keys counted only .h, .hpp and .inc (HEADER_EXTS and the folders' blob, tree_headers). A
new switch row then compiled nothing again: an update with the old object cache, or a pack over a base, kept the old switch
table in vk_main.o and vk_shader.o, and the renderer said "not a switch of this build" for the new switch. Now a .def or .inl
edit compiles the files that read it, through the same depfile keys (renderer_deps, rekey_depfiles) the build uses, and an
object that reads none of them keeps the key it had, and the sources' fingerprint (nx2/version.py) counts them too. The
broken variants (the old extensions) must keep the old object and the old fingerprint.

Nothing here runs a real compiler or needs the game.

python -I -B tests/test_header_ext_deps.py
"""
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "nx2"), str(ROOT / "third_party" / "a64-kit" / "tools")]
spec = importlib.util.spec_from_file_location("build_header_exts", ROOT / "nx2" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
import version  # noqa: E402  (nx2/version.py, on the path above)

OLD_EXTS = (".h", ".hpp", ".inc")   # (what the keys counted before: .def and .inl left out)

# (a compiler for the scenario: the "object" is the source with every #include "..." replaced by that file, followed into the
# included files, as the preprocessor does; the depfile names the source and every file it read, as clang -MD writes it)
FAKE_CC = r'''
import os, re, sys
args = sys.argv[1:]
src, obj = args[args.index("-c") + 1], args[args.index("-o") + 1]
read = []
def opened(path, mode="rb", **kw):   # (Windows: a deep temporary folder past 260 characters, as an extended-length path)
    full = os.path.abspath(path)
    if os.name == "nt" and len(full) >= 248 and not full.startswith("\\\\?\\"):
        path = "\\\\?\\UNC\\" + full[2:] if full.startswith("\\\\") else "\\\\?\\" + full   # (a share's path: \\?\UNC\server\...)
    return open(path, mode, **kw)
def expand(path):
    out = b""
    for line in opened(path).read().splitlines(keepends=True):
        m = re.match(rb'\s*#include "([^"]+)"', line)
        if not m:
            out += line
            continue
        inc = os.path.join(os.path.dirname(path), m.group(1).decode())
        if inc not in read:
            read.append(inc)
        out += expand(inc)
    return out
opened(obj, "wb").write(expand(src))
# (as clang escapes: '$' doubled, ' ' and '#' after a backslash; a backslash in a POSIX name stays)
esc = lambda p: p.replace(os.sep, "/").replace("$", "$$").replace(" ", "\\ ").replace("#", "\\#")
opened(obj[:-2] + ".d", "w", encoding="utf-8").write("%s: %s\n" % (esc(obj), " ".join(esc(p) for p in [src] + read)))
'''

INCLUDE = re.compile(r'#\s*include\s*"([^"]+)"')


def uncounted(texts):
    """The files these sources #include "..." that the object keys would not count: matched as the build matches them
    (str.endswith, so case counts: an "x.DEF" is not counted), and a name with no extension is not counted either"""
    return {name for text in texts for name in INCLUDE.findall(text) if not name.endswith(build.HEADER_EXTS)}


class HeaderExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hookshot header exts ")   # (a space in the path, as on Windows)
        self.addCleanup(self.temporary.cleanup)
        d = Path(self.temporary.name)
        self.vk, self.obj_dir = d / "gpu" / "vk", d / "obj"
        for p in (self.vk, self.obj_dir):
            p.mkdir(parents=True)
        self.roots = (str(self.vk),)
        cc = d / "fake_cc.py"
        cc.write_text(FAKE_CC)
        self.fl = [str(cc), "-MD"]
        # (switches.h's shape: the table included twice, once for the rows and once for the gone ones)
        (self.vk / "switches.h").write_bytes(b'#define SW(name) name,\n#include "switches.def"\n#undef SW\n'
                                             b'#define GONE(name) name,\n#include "switches.def"\n#undef GONE\n')
        self.table = self.vk / "switches.def"
        self.table.write_bytes(b"SW(RX_OLD_ROW)\n")
        (self.vk / "vk_main.cpp").write_bytes(b'#include "switches.h"\n')
        (self.vk / "vk_draw.h").write_bytes(b"int draw;\n")
        (self.vk / "vk_draw.cpp").write_bytes(b'#include "vk_draw.h"\n')
        self.inline = self.vk / "vk_math.inl"
        self.inline.write_bytes(b"int v1;\n")
        # (and a header: with .inl left out its depfile key is then vk_draw.h's, not the folders' blob, which an .inl edit
        # changes anyway)
        (self.vk / "vk_math.cpp").write_bytes(b'#include "vk_draw.h"\n#include "vk_math.inl"\n')
        for p in (mock.patch.object(build, "OBJ_CACHE", str(d / "objcache")), mock.patch.object(build, "GATE", None),
                  mock.patch.dict(build.DEPFILE_JOBS, {}), mock.patch.dict(build.PLANNED, {})):
            p.start()
            self.addCleanup(p.stop)

    def build_one(self, name):
        """One build of gpu/vk/<name> as nx2/build.py's back end builds its own files: keyed by its last depfile (or the folders'
        blob without one), compiled when that key is new, then re-keyed by its new depfile. Its object and its planned key"""
        src, obj = self.vk / name, str(self.obj_dir / (name[:-4] + ".o"))
        build._HEADER_DIGESTS.clear()   # (a new build)
        build.PLANNED.clear()
        blob = build.tree_headers(*self.roots)
        deps = build.renderer_deps(blob, self.roots, obj)
        job = (sys.executable, str(src), obj, self.fl, deps)
        build.DEPFILE_JOBS[obj] = (blob, self.roots, b"")
        key = build.object_key(str(src), self.fl, deps)
        build.PLANNED[(str(src), obj)] = key
        _, _, err = build.compile_one(job)
        self.assertIsNone(err)
        build.rekey_depfiles([job])
        return Path(obj).read_bytes(), key

    def new_row_scenario(self):
        """A build, a new row in switches.def, the next build: vk_main's object and vk_draw's keys before and after"""
        main1, _ = self.build_one("vk_main.cpp")
        self.build_one("vk_draw.cpp")   # (its first build is keyed by the folders' blob; the next by its depfile)
        _, draw_key1 = self.build_one("vk_draw.cpp")
        self.assertIn(b"SW(RX_OLD_ROW)", main1)
        self.table.write_bytes(b"SW(RX_NEW_ROW)\nSW(RX_OLD_ROW)\n")
        main2, _ = self.build_one("vk_main.cpp")
        _, draw_key2 = self.build_one("vk_draw.cpp")
        return main2, draw_key1, draw_key2

    def test_a_new_switch_row_compiles_its_includer_again(self):
        main2, draw_key1, draw_key2 = self.new_row_scenario()
        self.assertIn(b"SW(RX_NEW_ROW)", main2, "a new switches.def row kept the object built with the old table")
        self.assertEqual(draw_key1, draw_key2, "a file that does not read switches.def changed its key")

    def inl_scenario(self):
        self.build_one("vk_math.cpp")
        first, _ = self.build_one("vk_math.cpp")   # (keyed by its depfile now)
        self.assertIn(b"int v1;", first)
        self.inline.write_bytes(b"int v2;\n")
        second, _ = self.build_one("vk_math.cpp")
        return second

    def test_an_inl_edit_compiles_its_includer_again(self):
        self.assertIn(b"int v2;", self.inl_scenario(), "an .inl edit kept the object built with the old one")

    def test_broken_variant_without_inl_keeps_the_old_object(self):
        with mock.patch.object(build, "HEADER_EXTS", tuple(x for x in build.HEADER_EXTS if x != ".inl")):
            self.assertNotIn(b"int v2;", self.inl_scenario(), "the broken variant compiled it again: the .inl case proves nothing")

    def test_the_folders_blob_counts_def_and_inl(self):
        """A file with no depfile yet is keyed by the folders' blob: a .def or .inl edit changes it too"""
        obj = str(self.obj_dir / "vk_new.o")
        first = build.renderer_deps(build.tree_headers(*self.roots), self.roots, obj)
        self.table.write_bytes(b"SW(RX_NEW_ROW)\nSW(RX_OLD_ROW)\n")
        second = build.renderer_deps(build.tree_headers(*self.roots), self.roots, obj)
        self.assertNotEqual(first, second, "a switches.def edit left the folders' blob as it was")
        self.inline.write_bytes(b"int v2;\n")
        self.assertNotEqual(build.renderer_deps(build.tree_headers(*self.roots), self.roots, obj), second,
                            "an .inl edit left the folders' blob as it was")

    def test_an_object_that_reads_neither_keeps_its_key(self):
        """No object rebuilds on the update that ships this unless it reads a .def or .inl: an object whose depfile names only
        .h files, and a folder with none of them, are keyed as before. The source itself is never one of the key's files"""
        self.build_one("vk_draw.cpp")
        obj = str(self.obj_dir / "vk_draw.o")
        build._HEADER_DIGESTS.clear()
        now = build.renderer_deps(b"blob", self.roots, obj)
        self.assertTrue(now.startswith(build.DEPFILE_DEPS))
        with mock.patch.object(build, "HEADER_EXTS", OLD_EXTS):
            build._HEADER_DIGESTS.clear()
            self.assertEqual(build.renderer_deps(b"blob", self.roots, obj), now)
        plain = Path(self.temporary.name) / "plain"
        plain.mkdir()
        (plain / "a.h").write_bytes(b"int a;\n")
        (plain / "b.cpp").write_bytes(b'#include "a.h"\n')
        now_blob = build.tree_headers(str(plain))
        with mock.patch.object(build, "HEADER_EXTS", OLD_EXTS):
            self.assertEqual(build.tree_headers(str(plain)), now_blob)

    def test_broken_variant_the_old_extensions_keep_the_old_table(self):
        """The negative control: with the old extension list the same steps bring back the object with the old table"""
        with mock.patch.object(build, "HEADER_EXTS", OLD_EXTS):
            main2, _, _ = self.new_row_scenario()
        self.assertNotIn(b"SW(RX_NEW_ROW)", main2, "the broken variant compiled the includer again: the test proves nothing")

    def test_the_source_fingerprint_counts_def_and_inl(self):
        """The game's first log line names the sources' fingerprint (nx2/version.py): a build with another switch row is
        another build. The old extension list is the broken variant"""
        d = Path(self.temporary.name)
        first = version.fingerprint(str(d))
        self.table.write_bytes(b"SW(RX_NEW_ROW)\nSW(RX_OLD_ROW)\n")
        second = version.fingerprint(str(d))
        self.assertNotEqual(first, second, "a switches.def edit kept the fingerprint")
        self.inline.write_bytes(b"int v2;\n")
        self.assertNotEqual(version.fingerprint(str(d)), second, "an .inl edit kept the fingerprint")
        old = tuple(x for x in version.SOURCE_EXT if x not in (".def", ".inl"))
        with mock.patch.object(version, "SOURCE_EXT", old):
            before = version.fingerprint(str(d))
            self.table.write_bytes(b"SW(RX_OTHER_ROW)\n")
            self.assertEqual(version.fingerprint(str(d)), before, "the broken variant saw the edit: the check proves nothing")

    def test_every_extension_the_sources_include_is_counted(self):
        """Every file host/ and gpu/ #include "..." by name has an extension the keys count, so a new kind of included file
        (an .ipp, a .tbl) is caught here instead of keeping stale objects"""
        texts = []
        for top in ("host", "gpu"):
            for d, _, files in os.walk(ROOT / top):
                for f in files:
                    if f.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".inl", ".def")):
                        texts.append(Path(d, f).read_text(encoding="utf-8", errors="replace"))
        self.assertGreater(len(texts), 100, "found almost no sources under host/ and gpu/")
        self.assertIn(".def", {os.path.splitext(n)[1] for t in texts for n in INCLUDE.findall(t)},
                      "no source includes a .def any more: gpu/vk/switches.h's switches.def was the reason for this check")
        self.assertEqual(uncounted(texts), set(), "included by name, but not in nx2/build.py's HEADER_EXTS")
        # (the check itself: an extension the keys do not count is found)
        self.assertEqual(uncounted(['#include "vk_table.ipp"\n#include "a.h"\n#include "extra.DEF"\n#include "table"\n']),
                         {"vk_table.ipp", "extra.DEF", "table"})


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=1))
