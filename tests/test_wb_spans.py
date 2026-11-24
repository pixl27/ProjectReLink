"""A CPU read of GPU results waits for the write-backs over its own bytes, not for every one over its 4 KB pages
(gpu/vk/vk_memory.cpp wb_pages_note / memory_writeback_value, RX_VK_WB_SPANS).

OoT's culling tables are double-buffered: two slots back to back whose seam is one page. The GPU's write of one slot noted
that page, and the CPU's read of the other slot waited for it. Compiles the production pieces (the same cut as
test_invalidate_quick.py) with a small harness and checks:
  seam       a read of slot A does not wait for the part that wrote slot B on their shared page; a read of B does
  own        a later write over A's bytes on that page is waited for
  base       a page's full-page note before its partial ones is still waited for
  full       a full-page note makes the partial ones' record stale: a read waits for it
  samepart   ... also in the same part as them (equal value and part), and a big note over a recorded page (interior)
  merge      past 4 partial notes on a page, a read waits at least for the newest note over its bytes
  model      2,000 random write-backs and reads against an exact per-byte model: the part and value a read waits for are
             never older than the newest write over its own bytes (no write is missed), and its submission never later
             than the page rule's (it waits no longer than before)
  off        RX_VK_WB_SPANS=0: the page rule, as before (the seam read waits for B's part)

python -I -B tests/test_wb_spans.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include "gpu/vk/compat/hookshot_env.h"
#include "gpu/vk/page_marks.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
using mkwvk::PageMarks;
constexpr uint64_t kPageShift = 12, kChunkPages = 64, kPages = 1024;
namespace {
#include "wb_spans.inc"   // (closes the namespace it opens, as vk_memory.cpp's does)

struct Waits { uint64_t v, part, whole; };
Waits read(uint64_t guest, uint64_t size) {
    bool ever = false;
    uint64_t part = 0, whole = 0;
    const uint64_t v = memory_writeback_value(guest, size, &ever, &part, &whole);
    return {v, part, whole};
}
void note(uint64_t lo, uint64_t hi, uint64_t value, uint64_t part) { wb_pages_note(lo, hi, value, part); }

// (two slots back to back, as OoT's: 0x9b040 bytes each from page 16)
constexpr uint64_t kA = 16ull << kPageShift, kSize = 0x9b040, kB = kA + kSize;

int seam() {
    note(kA, kA + kSize, 5, 10);    // (last frame: slot A written in part 10)
    note(kB, kB + kSize, 7, 20);    // (this frame: slot B, part 20; its first page is A's last)
    const Waits a = read(kA, kSize), b = read(kB, kSize);
    CHECK(a.part == 10 && a.v == 5);   // (A: its own write, not B's)
    CHECK(b.part == 20 && b.v == 7);
    return 0;
}
int own() {
    note(kA, kA + kSize, 5, 10);
    note(kB, kB + kSize, 7, 20);
    const uint64_t seam_page = kB & ~0xfffull;
    note(seam_page + 0x10, seam_page + 0x30, 9, 30);   // (a later write over A's bytes on the seam page)
    CHECK(read(kA, kSize).part == 30);
    CHECK(read(kA, 0x100).part == 10);   // (A's first bytes: not on that page)
    return 0;
}
int base() {
    const uint64_t pg = 40ull << kPageShift;
    note(pg, pg + 0x1000, 3, 5);            // (the whole page, part 5)
    note(pg + 0x800, pg + 0x2000, 4, 8);    // (its upper half and the next page, part 8: a record on the page)
    CHECK(read(pg, 0x100).part == 5);       // (the lower bytes: the full-page note)
    CHECK(read(pg + 0x900, 0x10).part == 8);
    CHECK(read(pg, 0x1000).part == 8);      // (the whole page: the newest)
    return 0;
}
int full() {
    note(kA, kA + kSize, 5, 10);
    note(kB, kB + kSize, 7, 20);
    const uint64_t seam_page = kB & ~0xfffull;
    note(seam_page, seam_page + 0x1000, 11, 40);   // (the whole seam page again)
    CHECK(read(kA, kSize).part == 40);
    return 0;
}
int samepart() {
    const uint64_t pg = 80ull << kPageShift;
    note(pg + 0x100, pg + 0x200, 50, 60);   // (a partial note)
    note(pg, pg + 0x1000, 50, 60);          // (then the whole page, in the same part: equal value and part)
    note(pg + 0x300, pg + 0x380, 50, 60);   // (and another partial one)
    CHECK(read(pg + 0x800, 0x10).part == 60);   // (bytes only the whole-page note wrote: waited for)
    CHECK(read(pg + 0x800, 0x10).v == 50);
    const uint64_t pg2 = 90ull << kPageShift;
    note(pg2, pg2 + 0x1000, 3, 5);
    note(pg2 + 0x800, pg2 + 0x2000, 7, 9);
    note(pg2, pg2 + 0x1000, 7, 9);          // (the whole page again, same submission and part as the partial one)
    CHECK(read(pg2, 0x100).part == 9);
    return 0;
}
int within() {
    const uint64_t pg = 110ull << kPageShift;
    note(pg + 0x40, pg + 0x2000, 4, 4);       // (a record on page 110)
    note(pg + 0x400, pg + 0x500, 6, 6);       // (a note within page 110, over other bytes)
    CHECK(read(pg + 0x40, 0x20).part == 4);   // (the read's bytes: last written in part 4)
    CHECK(read(pg + 0x400, 0x20).part == 6);
    return 0;
}
int interior() {
    // (a big note over a page with a record: the record goes stale, the read waits for the big note)
    const uint64_t pg = 100ull << kPageShift;
    note(pg + 0x40, pg + 0x2000, 4, 4);     // (partial on page 100: a record there)
    note(pg - 0x2000, pg + 0x3000, 8, 8);   // (pages 98-102 whole: 100 an interior page of it)
    CHECK(read(pg, 0x20).part == 8);
    return 0;
}
int merge() {
    const uint64_t pg = 60ull << kPageShift;
    // (12 notes from the page before into page 61, each ending earlier on it: prefixes, newest shortest, merged past 4)
    const uint64_t T = pg + 0x1000;
    for (uint64_t i = 0; i < 12; i++) note(pg + 0x800, T + 0x1000 - 0x100 * (i + 1), 100 + i, 200 + i);
    for (uint64_t j = 0; j < 15; j++) {
        uint64_t want = 0;   // (the newest note over [T + 0x100 j, + 0x80))
        for (uint64_t i = 0; i < 12; i++)
            if (T + 0x1000 - 0x100 * (i + 1) > T + 0x100 * j) want = 200 + i;
        const Waits w = read(T + 0x100 * j, 0x80);
        CHECK(w.part >= want);   // (merged ones wait for more, never for less)
    }
    return 0;
}
int model() {
    // (pages 100..115, every byte's newest note in an exact model; parts and values rise as the renderer's do)
    constexpr uint64_t lo0 = 100ull << kPageShift, n = 16ull << kPageShift;
    std::vector<uint64_t> bv(n, 0), bp(n, 0);
    std::vector<uint64_t> pv(16, 0), pp(16, 0);   // (the page rule's: each page's newest note)
    std::mt19937_64 rng(1234);
    uint64_t seq = 0;
    for (int step = 0; step < 2000; step++) {
        if (rng() % 3) {
            const uint64_t a = rng() % n, len = 1 + rng() % (rng() % 2 ? 0x300 : 0x3000);
            const uint64_t b = std::min(n, a + len);
            seq++;
            const uint64_t value = seq, part = (rng() % 8) ? seq : 0;   // (a few in place: part 0)
            note(lo0 + a, lo0 + b, value, part);
            for (uint64_t i = a; i < b; i++) bv[i] = value, bp[i] = part;
            for (uint64_t p = a >> kPageShift; p <= (b - 1) >> kPageShift; p++) pv[p] = value, pp[p] = part;
        } else {
            const uint64_t a = rng() % n, len = 1 + rng() % 0x2000, b = std::min(n, a + len);
            const Waits w = read(lo0 + a, b - a);
            // (each byte's newest write is waited for: by a part or a whole submission at least as new, as the GPU's
            // timeline finishes them in order; here parts and values are one count, as the renderer's both rise with time)
            uint64_t ev = 0, pgv = 0;
            for (uint64_t i = a; i < b; i++) {
                ev = std::max(ev, bv[i]);
                if (bv[i]) CHECK(std::max(w.part, w.whole) >= bv[i]);
            }
            for (uint64_t p = a >> kPageShift; p <= (b - 1) >> kPageShift; p++) pgv = std::max(pgv, pv[p]);
            CHECK(w.v >= ev);   // (the read's value: at least its bytes' newest)
            CHECK(w.v <= pgv);   // (no later submission than the page rule's; a part may be older than its whole)
        }
    }
    return 0;
}
int off() {
    note(kA, kA + kSize, 5, 10);
    note(kB, kB + kSize, 7, 20);
    CHECK(read(kA, kSize).part == 20);   // (the page rule: B's part on the shared page)
    return 0;
}

int main(int argc, char** argv) {
    const std::string c = argc > 1 ? argv[1] : "";
    if (c == "seam") return seam();
    if (c == "own") return own();
    if (c == "base") return base();
    if (c == "full") return full();
    if (c == "samepart") return samepart();
    if (c == "interior") return interior();
    if (c == "within") return within();
    if (c == "merge") return merge();
    if (c == "model") return model();
    if (c == "off") return off();
    std::fprintf(stderr, "unknown case %s\n", c.c_str());
    return 2;
}
'''
CASES = (("seam", {}), ("own", {}), ("base", {}), ("full", {}), ("samepart", {}), ("interior", {}), ("within", {}), ("merge", {}), ("model", {}),
         ("model", {"RX_VK_WB_SPANS": "0"}), ("off", {"RX_VK_WB_SPANS": "0"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--source", type=Path, default=ROOT)
    args = parser.parse_args()
    memory = (args.source / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    note = block(memory, "void wb_pages_note(")
    assert note.count("wb_edge_note(") == 3 and "wb_full_pages(" in note, "wb_pages_note no longer notes its pages' extents"
    assert note.index("wb_full_pages(") < note.index("wb_edge_note(") < note.index("wb_pages.set("), \
        "a page's rest must be taken before the page takes the note's value"
    assert "RX_VK_WB_SPANS" in (args.source / "gpu/vk/switches.def").read_text(encoding="utf-8"), "no switches.def row"
    # (every note into the pages goes through wb_pages_note, with the bytes it copies: what the spans rely on)
    assert memory.count("wb_pages.set(") == 1, "a page's write-back value is set outside wb_pages_note: are its bytes noted?"
    wb = (memory[memory.index("std::mutex wb_pages_mu;"):memory.index("uint64_t memory_writeback_value(")] +
          block(memory, "uint64_t memory_writeback_value("))
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise SystemExit("C++ compiler not found: " + args.cxx)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    env = dict(os.environ)
    env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
    env["RX_VK_WB_SPANS"] = "1"
    with tempfile.TemporaryDirectory(prefix="wb-spans-") as temporary:
        tmp = Path(temporary)
        (tmp / "wb_spans.inc").write_text(wb, encoding="utf-8")
        (tmp / "wb_spans_test.cpp").write_text(HARNESS, encoding="utf-8")
        exe = tmp / ("wb_spans_test" + (".exe" if os.name == "nt" else ""))
        subprocess.run([compiler, "-std=c++20", "-O1", "-w", "-I", str(args.source), "-I", str(tmp),
                        str(tmp / "wb_spans_test.cpp"), "-o", str(exe)], check=True, env=env, creationflags=flags)
        for case, extra in CASES:
            r = subprocess.run([str(exe), case], env=dict(env, **extra), capture_output=True, text=True, creationflags=flags)
            if r.returncode:
                raise SystemExit("case %s %s failed: %s" % (case, extra or "", (r.stderr or r.stdout).strip()))
    print("wb spans: %d cases pass" % len(CASES))


if __name__ == "__main__":
    main()
