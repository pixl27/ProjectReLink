"""The late pages' fix's stamps made ahead (gpu/vk/vk_memory.cpp, RX_VK_LATE_STAMPS_AHEAD): a late page whose 64 KB stamp
block is not there yet makes its capture wait (the first time, as before) and, with the switch, the stamp blocks of the rest
of its pool too, so that the late pages the next captures find in other blocks of that pool are fixed, not waited for.

stamp_blocks_around and late_stamps_ahead_on are taken from the source as they are and run with a stand-in for
stamp_block_make (which makes one block, keyed by guest >> 16, when the pool has a shadow and the 64 MB budget allows):
  - a pool of 16 MB or less: every whole 64 KB block of it, none outside it;
  - a bigger pool: the whole blocks of the 2 MB around the page, within the pool;
  - a pool whose ends are not on a block: only the blocks wholly inside it (a block across two pools stays the late page's
    own way, made when a late page is found in it);
  - the switch: on unless RX_VK_LATE_STAMPS_AHEAD=0.
late_fix_build is read as text: the blocks around are made where a late page's missing block is made (and that capture still
falls back, fail(3)), unless the switch is 0.

python -I -B tests/test_late_stamps_ahead.py [--cxx g++]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include "gpu/vk/compat/hookshot_env.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <set>
#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)
void logf(const char*, ...) {}
constexpr unsigned kStampShift = 16;
constexpr uint64_t kStampBlock = 1ull << kStampShift, kStampMask = kStampBlock - 1;
struct Pool { uint64_t lo = 0, hi = 0; };
std::set<uint64_t> made;   // (block keys)
void stamp_block_make(Pool& p, uint64_t guest) {
    CHECK(guest >= p.lo && guest + kStampBlock <= p.hi);   // (never a block reaching outside the pool)
    CHECK((guest & kStampMask) == 0);
    made.insert(guest >> kStampShift);
}
@ON@
@AROUND@
int main() {
    {   // (on unless RX_VK_LATE_STAMPS_AHEAD=0)
        CHECK(late_stamps_ahead_on());
    }
    {   // an 8 MB pool: all its 128 blocks
        Pool p{0x103d000000ull, 0x103d800000ull};
        made.clear();
        stamp_blocks_around(p, 0x103d3a2000ull);
        CHECK(made.size() == 128);
        CHECK(*made.begin() == (0x103d000000ull >> 16) && *made.rbegin() == ((0x103d800000ull >> 16) - 1));
    }
    {   // a 16 MB pool: all of it too
        Pool p{0x1000000000ull, 0x1001000000ull};
        made.clear();
        stamp_blocks_around(p, 0x1000ff0000ull);
        CHECK(made.size() == 256);
    }
    {   // a 132 MB pool: the 2 MB around the page (32 blocks), within the pool
        Pool p{0x1178400000ull, 0x1180800000ull};
        made.clear();
        stamp_blocks_around(p, 0x11789a4000ull);
        CHECK(made.size() == 32);
        CHECK(*made.begin() == (0x1178800000ull >> 16) && *made.rbegin() == ((0x1178a00000ull >> 16) - 1));
        made.clear();   // (a page in the pool's first 2 MB region, which starts before the pool: clipped to it)
        Pool q{0x1178500000ull, 0x1180800000ull};
        stamp_blocks_around(q, 0x1178510000ull);
        CHECK(made.size() == 16);
        CHECK(*made.begin() == (0x1178500000ull >> 16));
    }
    {   // a small pool not on 64 KB: only the blocks wholly inside it
        Pool p{0x2000008000ull, 0x2000038000ull};   // (192 KB, from 32 KB into a block)
        made.clear();
        stamp_blocks_around(p, 0x2000010000ull);
        CHECK(made.size() == 2);   // (0x2000010000 and 0x2000020000; 0x2000000000 and 0x2000030000 reach outside)
        CHECK(made.count(0x2000010000ull >> 16) && made.count(0x2000020000ull >> 16));
    }
    {   // a pool smaller than a block: none
        Pool p{0x3000001000ull, 0x3000005000ull};
        made.clear();
        stamp_blocks_around(p, 0x3000002000ull);
        CHECK(made.empty());
    }
    std::puts("ok");
    return 0;
}
'''


def need(text, what, needles):
    for n in needles:
        if n not in text:
            raise SystemExit("%s no longer has: %s" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = ap.parse_args()
    src = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    on = block(src, "bool late_stamps_ahead_on() {")
    around = block(src, "void stamp_blocks_around(Pool& p, uint64_t guest) {")
    need(on, "late_stamps_ahead_on", ['env_on_or("RX_VK_LATE_STAMPS_AHEAD", 1)'])
    build = block(src, "bool late_fix_build(Capture& c) {")
    need(build, "late_fix_build", ["stamp_block_make(*p, a);\n                    if (late_stamps_ahead_on()) stamp_blocks_around(*p, a);\n                    fail(3);"])
    sw = (ROOT / "gpu/vk/switches.def").read_text(encoding="utf-8")
    need(sw, "switches.def", ["SW(RX_VK_LATE_STAMPS_AHEAD,"])
    code = FIXTURE.replace("@ON@", on).replace("@AROUND@", around)
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "t.cpp")
        exe = os.path.join(d, "t.exe")
        open(c, "w", encoding="utf-8").write(code)
        r = subprocess.run([a.cxx, "-std=c++20", "-O1", "-pthread", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), c, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("compile failed")
        env = dict(os.environ)
        env.pop("RX_VK_LATE_STAMPS_AHEAD", None)
        r = subprocess.run([exe], capture_output=True, text=True, env=env)
        print(r.stdout.strip() + r.stderr.strip())
        if r.returncode:
            raise SystemExit("FAIL")
    print("ok: test_late_stamps_ahead")


if __name__ == "__main__":
    main()
