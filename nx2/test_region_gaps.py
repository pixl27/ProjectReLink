"""Test the address-region gap planner used before Linux/Windows guest-page release.

No game data, renderer, Vulkan device, or platform decommit call is used.

python nx2/test_region_gaps.py [--cc clang]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    fixture = r'''
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "host/region_gaps.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); exit(1); } } while (0)

typedef struct Seen { uint64_t lo[8], hi[8]; size_t count; } Seen;

static uint64_t note(uint64_t lo, uint64_t hi, void *opaque) {
    Seen *seen = opaque;
    CHECK(seen->count < 8 && hi > lo);
    seen->lo[seen->count] = lo;
    seen->hi[seen->count++] = hi;
    return hi - lo;
}

int main(void) {
    const HosRegionGap regions[] = {{800, 100}, {500, 100}, {200, 100}};
    Seen seen = {0};
    uint64_t bytes = hos_region_gaps_release(regions, 3, 250, 850, note, &seen);
    CHECK(bytes == 400 && seen.count == 2);
    CHECK(seen.lo[0] == 600 && seen.hi[0] == 800);
    CHECK(seen.lo[1] == 300 && seen.hi[1] == 500);

    seen = (Seen){0};
    bytes = hos_region_gaps_release(NULL, 0, 100, 200, note, &seen);
    CHECK(bytes == 100 && seen.count == 1 && seen.lo[0] == 100 && seen.hi[0] == 200);

    seen = (Seen){0};
    bytes = hos_region_gaps_release(regions, 3, 520, 580, note, &seen);
    CHECK(bytes == 0 && seen.count == 0);
    CHECK(hos_region_gaps_release(regions, 3, 10, 10, note, &seen) == 0);
    CHECK(hos_region_gaps_release(regions, 3, 10, 20, NULL, &seen) == 0);

    puts("PASS: active regions are excluded from descending address-space gaps");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="region-gaps-", dir=work) as temp:
        directory = Path(temp).resolve()
        directory.relative_to(work.resolve())
        source = directory / "fixture.c"
        executable = directory / "fixture"
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
