"""An incomplete dump's damaged textures are read as a whole texture of their kind, whoever unpacked the romfs
(host/nn_fs.c OpenFile, host/oot_damaged_files.c).

The 23.3 GB dumps hold three .zs textures that start with zeros instead of a zstd frame. The builder copies a stand-in
over each (nx2/nup.py stand_in_damaged), but only in a romfs it unpacked itself; a romfs another tool unpacked kept them,
and the game asked for the model that uses the Great Fairy Fountain's floor about 15 times a second for as long as Link
stood in Kokiri or the Market. The actual OpenFile is compiled with host/oot_damaged_files.c and a made-up romfs in a
temporary folder (no game bytes), and checked:

- a damaged texture (zeros where the frame starts) opens its stand-in's bytes, and one warning names both, once;
- the same name whole (a dump that is complete) is read as it is, with no line;
- another file is read as it is;
- a damaged texture whose stand-in is missing (or damaged too) is read as it is, with a warning, once;
- each pair in host/oot_damaged_files.c's table is what nx2/nup.py's stand_in_for picks for that damaged file, so the
  host and the builder put the same texture in its place.

python -I -B tests/test_damaged_rom_files.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import nup   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ENV_ON_H = str(ROOT / "gpu" / "vk" / "compat" / "hookshot_env.h")
HOST_FILE = ROOT / "host" / "oot_damaged_files.c"

FAIRY = "Tex/CmnTex_Map_Soil_FairyFountain_Floor_D_Nrm.bntx.zs"
FAIRY_STAND_IN = "Tex/CmnTex_Map_Sand_FairyFountain_Sand_A_Nrm.bntx.zs"
GATE = "Tex/Rock_WaterTemple_Gate_M_01_Nrm.bntx.zs"
GANON = "Tex/RBake_Scene_GanonStage_G_2_1_d0_t0.bntx.zs"
GANON_STAND_IN = "Tex/RBake_Scene_GanonStage_G_2_2_d0_t0.bntx.zs"
ZSTD = b"\x28\xb5\x2f\xfd"

HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <strings.h>
#define _stricmp strcasecmp
typedef void *HANDLE;
typedef unsigned long DWORD;
typedef int BOOL;
typedef struct { DWORD Offset, OffsetHigh; } OVERLAPPED;
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 1
#define FILE_SHARE_WRITE 2
#define OPEN_EXISTING 3
#define FILE_ATTRIBUTE_NORMAL 0x80
#endif
typedef struct { uint64_t x[32]; } Ctx;
#define HLE(sym) void imp_##sym(Ctx *c)
#define ARG(n) (c->x[n])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define GP(a) ((void *)(uintptr_t)(a))
#define G64(a) (*(uint64_t *)GP(a))
#define HOS_FS_PATH 1024
#define R_PATH_NOT_FOUND 0x202
enum { HOS_LOG_ERROR, HOS_LOG_WARN, HOS_LOG_INFO, HOS_LOG_DEBUG, HOS_LOG_TRACE };
static int hos_trace;
static int fs_trace(void) { return 0; }
static char logged[8192];
static int warnings;
static void log_line(int level, const char *f, va_list ap) {
    char line[2048];
    vsnprintf(line, sizeof line, f, ap);
    if (level == HOS_LOG_WARN) warnings++;
    strncat(logged, line, sizeof logged - strlen(logged) - 2);
    strcat(logged, "\n");
}
void hos_log(const char *f, ...) { va_list ap; va_start(ap, f); log_line(HOS_LOG_INFO, f, ap); va_end(ap); }
void hos_log_at(int level, const char *f, ...) { va_list ap; va_start(ap, f); log_line(level, f, ap); va_end(ap); }
static const char *gstr(uint64_t a) { return (const char *)(uintptr_t)a; }
int hos_vcore_away(void) { return 0; }
void hos_vcore_back(void) {}

/* the romfs mount: "content:/<rel>" is <romfs>/<rel> */
static char romfs[HOS_FS_PATH], mod[HOS_FS_PATH];
static int host_path(const char *g, char *out, size_t cap, int *writable) {
    const char *rel = strstr(g, ":/");
    if (!rel) return 0;
    snprintf(out, cap, "%s/%s", mod[0] && strstr(rel, "Soil_FairyFountain") ? mod : romfs, rel + 2);
    if (writable) *writable = 0;
    return 1;
}
int hos_ultrawide_redirect(const char *guest, char *out, size_t cap) { (void)guest, (void)out, (void)cap; return 0; }
static int hos_blank_cached_icon(const char *guest, const char *host) { (void)guest, (void)host; return 0; }
#ifdef _WIN32
@WIDE@
static thread_local wchar_t fs_wbuf[HOS_FS_PATH + 8];
#define FS_W(p) fs_wide((p), fs_wbuf, HOS_FS_PATH + 8)
@CREATE@
#define CreateFileA fs_CreateFile
static DWORD fs_GetFileAttributes(const char *p) { return GetFileAttributesW(FS_W(p)); }
#define GetFileAttributesA fs_GetFileAttributes
#else
static DWORD GetFileAttributesA(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return INVALID_FILE_ATTRIBUTES;
    fclose(f);
    return 0;
}
/* files as stdio streams: what the host's open and read calls see */
static HANDLE CreateFileA(const char *p, DWORD a, DWORD s, void *sa, DWORD d, DWORD f, HANDLE t) {
    (void)a, (void)s, (void)sa, (void)d, (void)f, (void)t;
    FILE *file = fopen(p, "rb");
    return file ? (HANDLE)file : INVALID_HANDLE_VALUE;
}
static BOOL ReadFile(HANDLE h, void *buf, DWORD n, DWORD *got, OVERLAPPED *ov) {
    FILE *file = (FILE *)h;
    fseek(file, ov ? (long)ov->Offset : 0, SEEK_SET);
    *got = (DWORD)fread(buf, 1, n, file);
    return 1;
}
static BOOL CloseHandle(HANDLE h) { return fclose((FILE *)h) == 0; }
#endif
typedef struct { HANDLE h; int writable; char path[HOS_FS_PATH]; } Handle;
static Handle *opened;
static uint64_t new_handle(Handle *h) { opened = h; return 7; }
static int writes_open;

@FS_HELPERS@
@STAND_INS@

@OPEN@

/* the first bytes of what the game reads from "content:/<rel>" ("" when it is not found), and the lines logged */
static int open_and_read(const char *rel, char *out, size_t cap) {
    char guest[512];
    snprintf(guest, sizeof guest, "content:/%s", rel);
    static uint64_t handle;
    Ctx c;
    memset(&c, 0, sizeof c);
    c.x[0] = (uint64_t)(uintptr_t)&handle, c.x[1] = (uint64_t)(uintptr_t)guest, c.x[2] = 1;
    opened = NULL;
    imp__ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci(&c);
    out[0] = 0;
    if (c.x[0] || !opened) return 0;
    static char whole[1 << 16];
    DWORD n = 0;
    ReadFile(opened->h, whole, sizeof whole, &n, NULL);
    /* (its first 4 bytes, a |, and its last 24, as text: a zero as 0, the frame's magic as dots) */
    size_t at = 0;
    for (size_t i = 0; i < n && at + 1 < cap; i++) {
        if (i == 4) out[at++] = '|';
        if (i >= 4 && i + 24 < n) continue;
        if (at + 1 >= cap) break;
        const unsigned char b = (unsigned char)whole[i];
        out[at++] = b < 32 || b > 126 ? (b ? '.' : '0') : (char)b;
    }
    out[at] = 0;
    CloseHandle(opened->h);
    free(opened);
    return 1;
}
static void unhex(char *out, size_t cap, const char *hex) {
    size_t i = 0;
    while (hex[0] && hex[1] && i + 1 < cap) {
        unsigned int c;
        if (sscanf(hex, "%2x", &c) != 1) break;
        out[i++] = (char)c;
        hex += 2;
    }
    out[i] = 0;
}
int main(int argc, char **argv) {
    unhex(romfs, sizeof romfs, argv[1]);
    unhex(mod, sizeof mod, argv[2]);
    char overlay[HOS_FS_PATH + 16] = "HOS_FS_OVERLAY=";
    unhex(overlay + strlen(overlay), sizeof overlay - strlen(overlay), argv[3]);
#ifdef _WIN32
    _putenv(overlay);
#else
    putenv(overlay);
#endif
    for (int i = 4; i + 1 < argc; i += 2) {   /* <rel> <times>: opened that many times */
        char got[64];
        for (int n = atoi(argv[i + 1]); n > 0; n--) open_and_read(argv[i], got, sizeof got);
        printf("read %s: %s\n", argv[i], got);
    }
    printf("warnings %d\n%s", warnings, logged);
    return 0;
}
"""


def write(path, data):
    if os.name == "nt":
        path = Path("\\\\?\\" + str(path.resolve()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def table():
    """host/oot_damaged_files.c's (damaged, stand-in) pairs; none before it was written"""
    if not HOST_FILE.exists():
        return []
    return re.findall(r'\{\s*"(Tex/[^"]+)",\s*"(Tex/[^"]+)"\s*\}', HOST_FILE.read_text(encoding="utf-8"))


def stand_ins_source():
    """host/oot_damaged_files.c as it is, without its includes (the harness has the declarations); none before it was
    written, when OpenFile does not call it"""
    if not HOST_FILE.exists():
        return ""
    return "\n".join(line for line in HOST_FILE.read_text(encoding="utf-8").split("\n")
                     if not line.startswith("#include") and not line.startswith("#define WIN32_LEAN_AND_MEAN"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    failures = []

    def check(what, ok):
        if not ok:
            failures.append(what)

    pairs = table()
    check("host/oot_damaged_files.c names the three damaged textures of the 23.3 GB dumps",
          sorted(d for d, _ in pairs) == sorted((FAIRY, GATE, GANON)))
    with tempfile.TemporaryDirectory(prefix="damaged-tex-") as temporary:
        # (nup.stand_in_for's choice among the six: the damaged files as in the dump, zeros first; their stand-ins whole)
        tex = Path(temporary) / "Tex"
        for damaged, stand_in in pairs:
            write(Path(temporary) / damaged, bytes(12288) + b"rest")
            write(Path(temporary) / stand_in, ZSTD + b"whole")
        damaged_paths = {os.path.normpath(os.path.join(temporary, d)) for d, _ in pairs}
        for damaged, stand_in in pairs:
            picked = nup.stand_in_for(os.path.normpath(os.path.join(temporary, damaged)), damaged_paths)
            check("%s: the host reads %s, nup.stand_in_for picks %s" % (damaged, stand_in, picked and os.path.relpath(picked, temporary)),
                  picked and Path(picked) == tex / Path(stand_in).name)

    src = (ROOT / "host" / "nn_fs.c").read_text(encoding="utf-8")
    helpers = "\n".join(block(src, marker) for marker in (
        "HANDLE hos_fs_open_read(const char *path) {",
        "static void fs_overlay_path(const char *guest, char *path, size_t cap) {",
        "int hos_fs_resolve_read(const char *guest, char *path, size_t cap) {"))
    fixture = HARNESS.replace("@WIDE@", block(src, "static const wchar_t *fs_wide("))
    fixture = fixture.replace("@CREATE@", block(src, "static HANDLE fs_CreateFile("))
    fixture = fixture.replace("@FS_HELPERS@", helpers).replace("@STAND_INS@", stand_ins_source()).replace(
        "@OPEN@", block(src, "HLE(_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci) {"))
    fixture = "#include <stdarg.h>\n" + fixture
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise SystemExit("C++ compiler not found: " + args.cxx)
    with tempfile.TemporaryDirectory(prefix="damaged-rom-files-") as temporary:
        folder = Path(temporary)
        (folder / "fixture.cpp").write_text(fixture, encoding="utf-8")
        exe = folder / ("fixture.exe" if os.name == "nt" else "fixture")
        subprocess.run([compiler, "-std=c++17", "-include", ENV_ON_H, "-O1", "-w", str(folder / "fixture.cpp"), "-o", str(exe)],
                       check=True, timeout=120)

        def run(romfs, *opens, mod=None, overlay=None):
            encode = lambda path: (str(path) if path else "").encode("utf-8").hex()
            argv = [str(exe), encode(romfs), encode(mod), encode(overlay)]
            for rel, times in opens:
                argv += [rel, str(times)]
            out = subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout
            reads = dict(re.findall(r"^read (\S+): (.*)$", out, re.M))
            warnings = int(re.search(r"^warnings (\d+)$", out, re.M).group(1))
            lines = [line for line in out.split("\n") if line.startswith("fs:")]
            return reads, warnings, lines

        # An incomplete dump as another tool unpacked it: the three textures zeros first, the stand-ins whole
        damaged_romfs = folder / "incomplete" / "romfs"
        write(damaged_romfs / FAIRY, bytes(12288) + b"after the zeros")
        write(damaged_romfs / FAIRY_STAND_IN, ZSTD + b"sand floor")
        write(damaged_romfs / GANON, bytes(4096) + b"ganon's own")
        write(damaged_romfs / GANON_STAND_IN, bytes(4096) + b"damaged stand-in")   # (damaged too: not taken)
        write(damaged_romfs / "Tex/Other_Nrm.bntx.zs", ZSTD + b"other")
        reads, warnings, lines = run(damaged_romfs, (FAIRY, 15), (GANON, 3), ("Tex/Other_Nrm.bntx.zs", 1), (GATE, 1))
        check("a damaged texture opens its stand-in's bytes (read %r)" % reads.get(FAIRY),
              reads.get(FAIRY, "").endswith("sand floor"))
        fairy_lines = [line for line in lines if FAIRY in line]
        check("one line names the damaged texture and its stand-in, once in 15 opens (%r)" % fairy_lines,
              len(fairy_lines) == 1 and Path(FAIRY_STAND_IN).name in fairy_lines[0])
        check("the line says the dump is incomplete and what to do (%r)" % fairy_lines,
              len(fairy_lines) == 1 and "incomplete" in fairy_lines[0] and "dump the game again" in fairy_lines[0].lower())
        check("a damaged texture with no whole stand-in is read as it is (read %r)" % reads.get(GANON),
              reads.get(GANON, "").endswith("ganon's own"))
        ganon_lines = [line for line in lines if GANON in line]
        check("a damaged texture with no whole stand-in is said once in 3 opens (%r)" % ganon_lines, len(ganon_lines) == 1)
        check("both lines are warnings (%d)" % warnings, warnings == 2)
        check("another file is read as it is (%r)" % reads.get("Tex/Other_Nrm.bntx.zs"),
              reads.get("Tex/Other_Nrm.bntx.zs", "").endswith("other"))
        check("a file that is not there is still not found (%r)" % reads.get(GATE), reads.get(GATE) == "")

        # A complete dump: the same names whole, read as they are and nothing said
        whole_romfs = folder / "complete" / "romfs"
        write(whole_romfs / FAIRY, ZSTD + b"fairy floor")
        write(whole_romfs / FAIRY_STAND_IN, ZSTD + b"sand floor")
        reads, warnings, lines = run(whole_romfs, (FAIRY, 2))
        check("a whole texture under a damaged name is read as it is (read %r)" % reads.get(FAIRY),
              reads.get(FAIRY, "").endswith("fairy floor"))
        check("a complete dump logs nothing (%r)" % lines, not lines and warnings == 0)

        # Both override sources can supply only the damaged file; the stand-in is in the game folder.
        for source in ("mod", "overlay"):
            override = folder / source
            write(override / FAIRY, bytes(128) + b"override")
            reads, warnings, lines = run(whole_romfs, (FAIRY, 1), **{source: override})
            check("%s damaged override uses the game's stand-in (%r)" % (source, reads),
                  reads.get(FAIRY, "").endswith("sand floor") and warnings == 1)

        # The probe and final open must both handle UTF-8 and Windows extended-length paths.
        long_romfs = folder / "\u6e38\u620f" / ("a" * 100) / ("b" * 100) / "romfs"
        write(long_romfs / FAIRY, bytes(128) + b"damaged")
        write(long_romfs / FAIRY_STAND_IN, ZSTD + b"long path stand-in")
        reads, warnings, lines = run(long_romfs, (FAIRY, 1))
        check("UTF-8 long path opens the stand-in (%r)" % reads,
              reads.get(FAIRY, "").endswith("long path stand-in") and warnings == 1)

    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        raise SystemExit("an incomplete dump's damaged textures are not read as their stand-ins")
    print("ok: damaged textures read as their stand-ins, whole ones as they are, the table as nup.stand_in_for picks")


if __name__ == "__main__":
    main()
