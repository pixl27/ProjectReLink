"""Test integrated/discrete texture-policy allowances without Vulkan or a device.

python nx2/test_texture_memory_policy.py [--cxx clang++]
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
#include "gpu/vk/astc_bc_policy.h"
#include "gpu/vk/texture_memory_policy.h"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    using namespace texture_memory_policy;
    constexpr uint64_t raw = 9594ull * MiB;
    constexpr uint64_t explicit_budget = 768ull * MiB;
    CHECK(effective_allowance(raw, false) == raw);
    CHECK(effective_allowance(raw, true) == IntegratedDefault);
    CHECK(effective_allowance(2048ull * MiB, true) == 2048ull * MiB);
    CHECK(effective_allowance(raw, false, explicit_budget) == explicit_budget);
    CHECK(effective_allowance(raw, true, explicit_budget) == explicit_budget);
    CHECK(effective_allowance(0, true) == IntegratedDefault);
    CHECK(effective_allowance(0, false, explicit_budget) == explicit_budget);
    CHECK(effective_allowance(0, false) == 0);

    CHECK(!astc_bc_policy::all_by_vram(raw));
    CHECK(astc_bc_policy::all_by_vram(effective_allowance(raw, true)));
    CHECK(astc_bc_policy::all_by_vram(effective_allowance(raw, false, explicit_budget)));
    CHECK(staging_keep(raw) == 256ull * MiB);
    CHECK(staging_keep(effective_allowance(raw, true)) == 64ull * MiB);
    CHECK(staging_keep(explicit_budget) == 64ull * MiB);
    std::puts("PASS: integrated/configured texture allowance drives ASTC BC3 and 64 MB staging policies");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="texture-memory-policy-", dir=work) as temp:
        source = Path(temp) / "fixture.cpp"
        executable = Path(temp) / "fixture"
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
