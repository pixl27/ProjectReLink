"""Test sparse guest-memory block rounding, merging, accounting, and already-bound filtering.

No Vulkan device, game data, or Linux udmabuf device is used.

python nx2/test_sparse_guest_ranges.py [--cxx clang++]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    fixture = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gpu/vk/sparse_guest_ranges.h"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    constexpr uint64_t base = 0x1000000000ull, block = 65536;
    struct Live { uint64_t lo, hi; };
    std::vector<uint8_t> wanted(8);
    std::vector<uint64_t> bound(1);

    std::vector<sparse_guest::Run> merged;
    CHECK(sparse_guest::append_merged(merged, {block, block}));
    CHECK(sparse_guest::append_merged(merged, {2 * block, block}));
    CHECK(sparse_guest::append_merged(merged, {2 * block + 1, 2 * block}));
    CHECK(merged.size() == 1 && merged[0].offset == block && merged[0].size == 3 * block + 1);
    CHECK(sparse_guest::append_merged(merged, {5 * block, block}));
    CHECK(merged.size() == 2 && merged[1].offset == 5 * block && merged[1].size == block);
    CHECK(!sparse_guest::append_merged(merged, {4 * block, block}));
    CHECK(!sparse_guest::append_merged(merged, {UINT64_MAX, 2}));

    sparse_guest::mark(wanted, base, block, base + 4096, base + block + 4096);
    sparse_guest::mark(wanted, base, block, base + 3 * block, base + 4 * block);
    auto runs = sparse_guest::runs(wanted, bound, block);
    CHECK(runs.size() == 2);
    CHECK(runs[0].offset == 0 && runs[0].size == 2 * block);
    CHECK(runs[1].offset == 3 * block && runs[1].size == block);

    sparse_guest::set_bound(bound, 1, 1);
    runs = sparse_guest::runs(wanted, bound, block);
    CHECK(runs.size() == 2);
    CHECK(runs[0].offset == 0 && runs[0].size == block);
    CHECK(runs[1].offset == 3 * block && runs[1].size == block);
    CHECK(sparse_guest::bound(bound, 1) && !sparse_guest::bound(bound, 2));

    sparse_guest::set_bound(bound, 0, 1);
    sparse_guest::set_bound(bound, 3, 1);
    CHECK(sparse_guest::runs(wanted, bound, block).empty());
    CHECK(sparse_guest::overlaps_bound(bound, base, block, base + 1, base + block));
    CHECK(!sparse_guest::overlaps_bound(bound, base, block, base + 2 * block, base + 3 * block));
    CHECK(sparse_guest::overlaps_bound(bound, base, block, base + 2 * block, base + 3 * block + 1));
    CHECK(!sparse_guest::overlaps_bound(bound, base, block, base - block, base));
    CHECK(!sparse_guest::overlaps_bound(bound, base, 0, base, base + 1));
    sparse_guest::clear_bound(bound, 3, 1);
    CHECK(!sparse_guest::bound(bound, 3) && sparse_guest::bound(bound, 0));
    sparse_guest::clear_bound(bound, 99, 1);

    sparse_guest::clear_bound(bound, 1, 1);
    sparse_guest::set_bound(bound, 3, 1);
    auto unbound = sparse_guest::unbound_runs(bound, block, 8 * block, 1, 5 * block + 1);
    CHECK(unbound.size() == 2);
    CHECK(unbound[0].offset == block && unbound[0].size == 2 * block);
    CHECK(unbound[1].offset == 4 * block && unbound[1].size == 2 * block);
    sparse_guest::set_bound(bound, 1, 2);
    sparse_guest::set_bound(bound, 4, 2);
    CHECK(sparse_guest::unbound_runs(bound, block, 8 * block, 1, 5 * block + 1).empty());
    CHECK(sparse_guest::unbound_runs(bound, 0, 8 * block, 0, block).empty());
    CHECK(sparse_guest::unbound_runs(bound, block, 8 * block, 8 * block, 9 * block).empty());
    sparse_guest::clear_bound(bound, 1, 2);
    sparse_guest::clear_bound(bound, 3, 1);
    sparse_guest::clear_bound(bound, 4, 2);
    CHECK(sparse_guest::scan_advance(base, block, base + 17, base + 4096, base + 8 * block) == base + block);
    CHECK(sparse_guest::scan_advance(base, block, base + 17, base + 3 * block, base + 8 * block) == base + 3 * block);
    CHECK(sparse_guest::scan_advance(base, block, base + 7 * block, base + 9 * block, base + 8 * block) ==
          base + 8 * block);
    CHECK(sparse_guest::scan_advance(base, 0, base, base, base + block) == base + block);

    std::fill(wanted.begin(), wanted.end(), 0);
    sparse_guest::mark(wanted, base, block, base + 7 * block + 1, base + 8 * block);
    runs = sparse_guest::runs(wanted, bound, block);
    CHECK(runs.size() == 1 && runs[0].offset == 7 * block && runs[0].size == block);
    sparse_guest::mark(wanted, base, block, base, base);
    sparse_guest::mark(wanted, base, 0, base, base + 1);

    const std::vector<sparse_guest::Run> allocation{{block, block}, {4 * block, 2 * block}};
    CHECK(sparse_guest::bytes(allocation) == 3 * block);
    CHECK(sparse_guest::bytes({{0, UINT64_MAX}, {0, 1}}) == UINT64_MAX);
    CHECK(sparse_guest::touches(allocation, base, base + block, base + 2 * block));
    CHECK(sparse_guest::touches(allocation, base, base + 5 * block, base + 7 * block));
    CHECK(!sparse_guest::touches(allocation, base, base + 2 * block, base + 4 * block));
    CHECK(!sparse_guest::touches(allocation, base, base, base));
    CHECK(sparse_guest::all_outside(allocation, base, std::vector<Live>{{base, base + block}}));
    CHECK(!sparse_guest::all_outside(allocation, base, std::vector<Live>{{base + 5 * block, base + 7 * block}}));
    CHECK(!sparse_guest::all_outside(allocation, base,
                                     std::vector<Live>{{base, base + block}, {base + block, base + 2 * block}}));
    CHECK(sparse_guest::all_outside(allocation, base,
                                    std::vector<Live>{{base, base + block}, {base + 2 * block, base + 4 * block}}));

    const auto batches = sparse_guest::batches(
        std::vector<sparse_guest::Run>{{0, 150 * block}, {200 * block, 3 * block}}, 64 * block, 2, block);
    CHECK(batches.size() == 3);
    CHECK(batches[0].size() == 1 && batches[0][0].offset == 0 && batches[0][0].size == 64 * block);
    CHECK(batches[1].size() == 1 && batches[1][0].offset == 64 * block && batches[1][0].size == 64 * block);
    CHECK(batches[2].size() == 2);
    CHECK(batches[2][0].offset == 128 * block && batches[2][0].size == 22 * block);
    CHECK(batches[2][1].offset == 200 * block && batches[2][1].size == 3 * block);
    CHECK(sparse_guest::batches({{1, block}}, 64 * block, 2, block).empty());
    CHECK(sparse_guest::batches({{0, block}}, block - 1, 2, block).empty());
    CHECK(sparse_guest::batches({{UINT64_MAX - block + 1, block}}, 64 * block, 2, block).empty());

    std::puts("PASS: sparse guest ranges round, merge, account, release, and exclude live blocks");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparse-guest-ranges-", dir=work) as temp:
        directory = Path(temp).resolve()
        directory.relative_to(work.resolve())
        cpp = directory / "fixture.cpp"
        exe = directory / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(cpp), "-o", str(exe)],
            check=True,
        )
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
