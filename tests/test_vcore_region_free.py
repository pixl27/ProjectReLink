"""FreeAddressRegion gives its virtual core up while the region's pages are discarded, and the region stays listed until
they are (host/hos_mem.c hos_region_free).

The discard hands the pages back to the system (DiscardVirtualMemory on Windows, MADV_REMOVE on Linux): milliseconds for
a large region. With the core held through it, the threads that share the core waited for the whole discard, often
threads of a higher priority than the one freeing (the game frees on its resource memory thread). And the region left the
list before its discard: an AllocateAddressRegion meanwhile could be handed the same pages and lose what it wrote to the
discard. Kept listed, the region is marked as being freed: a second free of the same address meanwhile (the guest freeing
it twice) must do nothing, not unlist it while the first discard runs. The actual hos_region_free and
hos_region_alloc are compiled with stand-ins that record, in the discard, whether the thread holds its core, whether the
region is still listed and whether the list's lock is held:

- the discard runs once, on the region's pages, without the core and outside the lock, with the region still listed;
- an allocation of the same size during the discard, with the region the highest hole, is not handed its pages;
- after it the region is gone from the list, the others are kept in order, and the core is held again, taken once;
- an address that is no region discards nothing and keeps the core;
- a thread that holds no core when it frees takes none;
- a second free of the region during its discard does nothing and leaves it listed;
- a region of no size leaves the list;
- a region allocated while another is being freed, in the slot that one moved out of, can itself be freed.

python -I -B tests/test_vcore_region_free.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the list of address regions and its lock, as host/hos_mem.c keeps them (room for 8: hos_region_alloc never grows it) */
typedef struct { uint64_t addr, size; int freeing; } Range;
static Range room[8];
static Range *regions = room;
static int nregions, capregions = 8;
static int locked;
#define LOCK() (locked++)
#define UNLOCK() (locked--)
static void pages_mark_used_locked(uint64_t addr, uint64_t size) { (void)addr, (void)size; }

/* what hos_region_alloc needs besides: the window, and the heap below the regions (no holes under it) */
#define HOS_SPACE_END 0x8000000000ULL
#define HOS_HEAP_BASE 0x1000000000ULL
static uint64_t hos_heap_size = 0x100000000ULL, region_floor = ~0ULL, low_hole[2][2];
static int hos_commit(uint64_t addr, uint64_t size) { (void)addr, (void)size; return 1; }
static int env_on(const char *name) { (void)name; return 0; }   /* (hookshot_env.h's switches: all off here) */
void rx_guest_release(uint64_t lo, uint64_t hi) { (void)lo, (void)hi; }   /* (host/rx_backend.c: the renderer told; none here) */
static void hos_fatal(void *c, const char *what, ...) { (void)c; printf("FAIL: %s\n", what); exit(1); }

/* the thread's virtual core, as hos_vcore_away / hos_vcore_back keep it (host/hos_thread.c) */
static int held, takes;
int hos_vcore_away(void) { if (!held) return 0; held = 0; return 1; }
void hos_vcore_back(void) { if (!held) held = 1, takes++; }

/* the discard: what it was given, and what held while it ran. With twice set, the address is freed once more while it
   is discarded, as a second thread freeing it would, and whether the region is still listed after that is kept. With
   alloc_size set, a region of that size is allocated while it is discarded, as another thread would, and where kept. */
void hos_region_free(uint64_t addr);
uint64_t hos_region_alloc(uint64_t size, uint64_t align);
static int discards, held_in_discard, listed_in_discard, locked_in_discard, twice, listed_after_second;
static uint64_t discard_addr, discard_size, alloc_size, alloc_in_discard;
static int listed(uint64_t addr, uint64_t size) {
    for (int i = 0; i < nregions; i++)
        if (regions[i].addr == addr && regions[i].size == size) return 1;
    return 0;
}
static void discard_pages(uint64_t addr, uint64_t size) {
    discards++;
    discard_addr = addr, discard_size = size;
    held_in_discard = held;
    locked_in_discard = locked;
    listed_in_discard = listed(addr, size);
    if (twice) {
        twice = 0;
        hos_region_free(addr);
        listed_after_second = listed(addr, size);
    }
    if (alloc_size) {
        const uint64_t want = alloc_size;
        alloc_size = 0;
        alloc_in_discard = hos_region_alloc(want, 0x10000);
    }
}

@ALLOC@

@FREE@

static int bad;
static void check(int ok, const char *what) { if (!ok) { printf("FAIL: %s\n", what); bad = 1; } }
static void three_regions(void) {   /* (by address, descending, as hos_region_alloc keeps them) */
    nregions = 3;
    regions[0].addr = 0x7200000000ULL, regions[0].size = 0x10000;
    regions[1].addr = 0x7100000000ULL, regions[1].size = 0x2000000;
    regions[2].addr = 0x7000000000ULL, regions[2].size = 0x400000;
    for (int i = 0; i < 3; i++) regions[i].freeing = 0;
}
static void free_region(uint64_t addr, int core) {
    held = core, takes = 0, discards = 0, held_in_discard = locked_in_discard = listed_in_discard = -1;
    listed_after_second = -1;
    hos_region_free(addr);
}
int main(void) {
    three_regions();
    free_region(0x7100000000ULL, 1);
    check(discards == 1 && discard_addr == 0x7100000000ULL && discard_size == 0x2000000, "the region's pages are discarded, once");
    check(held_in_discard == 0, "the discard runs without the virtual core");
    check(locked_in_discard == 0, "the discard runs outside the list's lock");
    check(listed_in_discard == 1, "the region stays listed through its discard (no allocation is handed its pages)");
    check(nregions == 2 && regions[0].addr == 0x7200000000ULL && regions[1].addr == 0x7000000000ULL,
          "after it the region is gone from the list, the others kept in order");
    check(held == 1 && takes == 1 && locked == 0, "the core is held again after it, taken once, and the lock let go");

    /* the region to free just under the highest one, so that once it is off the list it is the highest hole that fits */
    nregions = 3;
    regions[0].addr = 0x7FFFFF0000ULL, regions[0].size = 0x10000;
    regions[1].addr = 0x7FFDFF0000ULL, regions[1].size = 0x2000000;
    regions[2].addr = 0x7000000000ULL, regions[2].size = 0x400000;
    for (int i = 0; i < 3; i++) regions[i].freeing = 0;
    alloc_size = 0x2000000, alloc_in_discard = 0;
    free_region(0x7FFDFF0000ULL, 1);
    check(alloc_in_discard && (alloc_in_discard + 0x2000000 <= 0x7FFDFF0000ULL || alloc_in_discard >= 0x7FFFFF0000ULL),
          "an allocation while the region is discarded is not handed its pages");
    check(nregions == 3 && listed(alloc_in_discard, 0x2000000) && !listed(0x7FFDFF0000ULL, 0x2000000),
          "after the discard the region allocated meanwhile stays listed and the freed one is gone");

    three_regions();
    free_region(0x7300000000ULL, 1);
    check(discards == 0 && nregions == 3, "an address that is no region discards nothing and keeps the list");
    check(held == 1 && takes == 0, "an address that is no region keeps the core");

    three_regions();
    free_region(0x7000000000ULL, 0);
    check(discards == 1 && nregions == 2 && regions[1].addr == 0x7100000000ULL, "a thread with no core frees the region");
    check(held == 0 && takes == 0, "a thread that held no core takes none");

    three_regions();
    twice = 1;
    free_region(0x7100000000ULL, 1);
    check(discards == 1, "a second free of the region while it is discarded (the guest freeing it twice) discards nothing");
    check(listed_after_second == 1,
          "a second free of the region while it is discarded leaves it listed (its address is not handed out again yet)");
    check(nregions == 2 && regions[0].addr == 0x7200000000ULL && regions[1].addr == 0x7000000000ULL,
          "after both frees the region has left the list, once, the others kept");
    check(held == 1 && takes == 1 && locked == 0, "after both frees the core is held again, taken once, and the lock let go");

    three_regions();
    regions[1].size = 0;
    free_region(0x7100000000ULL, 1);
    check(nregions == 2 && regions[0].addr == 0x7200000000ULL && regions[1].addr == 0x7000000000ULL,
          "a region of no size leaves the list");
    check(held == 1 && locked == 0, "a region of no size: the core is held after it, and the lock let go");

    three_regions();
    regions[0].freeing = 1;   /* (0x7200000000 is being freed) */
    const uint64_t fresh = hos_region_alloc(0x1000, 0x1000);
    check(fresh == 0x7FFFFFF000ULL && nregions == 4 && regions[0].addr == fresh && regions[1].addr == 0x7200000000ULL,
          "(the new region goes on top, in the slot the one being freed moved out of)");
    free_region(fresh, 1);
    check(discards == 1 && discard_addr == fresh && nregions == 3 && regions[0].addr == 0x7200000000ULL,
          "a region allocated while another is being freed can itself be freed (it does not carry the other's mark)");
    if (!bad) puts("ok");
    return bad;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--source", type=Path, default=ROOT / "host/hos_mem.c")
    args = parser.parse_args()
    src = args.source.read_text(encoding="utf-8")
    # (Hookshot's hos_region_alloc is region_alloc with the pages committed: both, the wrapper last)
    alloc = block(src, "uint64_t hos_region_alloc(uint64_t size, uint64_t align) {")
    if "region_alloc(size, align, 1)" in alloc and "static uint64_t region_alloc(uint64_t size, uint64_t align, int commit) {" in src:
        alloc = (block(src, "static uint64_t region_alloc(uint64_t size, uint64_t align, int commit) {") + "\n" + alloc)
    fixture = (HARNESS.replace("@ALLOC@", alloc)
               .replace("@FREE@", block(src, "void hos_region_free(uint64_t addr) {")))
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise SystemExit("C++ compiler not found: " + args.cxx)
    with tempfile.TemporaryDirectory(prefix="vcore-region-free-") as temporary:
        folder = Path(temporary)
        (folder / "fixture.cpp").write_text(fixture, encoding="utf-8")
        exe = folder / ("fixture.exe" if os.name == "nt" else "fixture")
        subprocess.run([compiler, "-std=c++17", "-O1", "-w", str(folder / "fixture.cpp"), "-o", str(exe)], check=True,
                       timeout=120)
        result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        print(result.stdout, end="")
        if result.returncode or "ok" not in result.stdout:
            raise SystemExit("FreeAddressRegion holds its virtual core through the discard, lets its region go before it, "
                             "or acts on a second free of it")


if __name__ == "__main__":
    main()
