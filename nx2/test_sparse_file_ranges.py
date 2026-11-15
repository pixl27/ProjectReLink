"""Test that freeing sparse memfd data zeros allocated extents without instantiating holes.

No Vulkan device or game data is used. Linux memfd SEEK_DATA/SEEK_HOLE is the production mechanism.

python nx2/test_sparse_file_ranges.py [--cc clang]
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
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "host/sparse_file_ranges.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE), size = 4 * page;
    const int fd = (int)syscall(SYS_memfd_create, "sparse-ranges-test", 1u);
    CHECK(fd >= 0 && ftruncate(fd, (off_t)size) == 0);
    uint8_t *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(map != MAP_FAILED);
    CHECK(hos_sparse_file_has_data(fd, 0, size) == 0);
    map[17] = 0xa5;
    map[2 * page + 29] = 0x5a;
    CHECK(hos_sparse_file_has_data(fd, 0, page) == 1);
    CHECK(hos_sparse_file_has_data(fd, page, 2 * page) == 0);
    CHECK(hos_sparse_file_has_data(fd, 2 * page, 3 * page) == 1);
    CHECK(hos_sparse_file_has_data(-1, 0, page) == -1);
    CHECK(lseek(fd, (off_t)page, SEEK_DATA) == (off_t)(2 * page));
    CHECK(hos_sparse_file_zero_data(fd, map, 0, 3 * page) == 0);
    CHECK(map[17] == 0 && map[2 * page + 29] == 0);
    CHECK(lseek(fd, (off_t)page, SEEK_DATA) == (off_t)(2 * page));
    CHECK(hos_sparse_file_zero_data(fd, map, page, page) == 0);
    CHECK(hos_sparse_file_zero_data(-1, map, 0, page) == -1);
    munmap(map, size);
    close(fd);
    puts("PASS: sparse guest zeroing preserves unallocated memfd holes");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparse-file-ranges-", dir=work) as temp:
        source = Path(temp) / "fixture.c"
        executable = Path(temp) / "fixture"
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
