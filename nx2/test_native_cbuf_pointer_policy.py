"""Test native non-c0 storage-candidate validation without a GPU or game data.

python nx2/test_native_cbuf_pointer_policy.py [--cxx clang++]
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
#include "gpu/vk/eden/shader_recompiler/ir_opt/native_cbuf_pointer_policy.h"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    using native_cbuf_pointer_policy::leave_global;
    // Native vertex/task/mesh: reject an unmapped scalar; preserve a mapped pointer.
    CHECK(leave_global(true, true, true, 5, false, false));
    CHECK(!leave_global(true, true, true, 5, false, true));
    // The pre-existing memory-origin rule remains active for geometry, native or not.
    CHECK(leave_global(true, true, true, 5, true, true));
    CHECK(leave_global(false, true, true, 5, true, false));
    CHECK(!leave_global(false, true, true, 5, false, false));
    // Fragment rejects only the unmapped native scalar, not the older memory-origin case.
    CHECK(leave_global(true, true, false, 5, false, false));
    CHECK(!leave_global(true, true, false, 5, false, true));
    CHECK(!leave_global(false, true, false, 5, true, false));
    // Compute and c0 keep their existing paths.
    CHECK(!leave_global(true, false, false, 5, true, false));
    CHECK(!leave_global(true, true, true, 0, true, false));
    std::puts("PASS: native graphics rejects scalar non-c0 storage candidates without widening fragment memory-origin fallback");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-cbuf-pointer-", dir=work) as temp:
        directory = Path(temp).resolve()
        directory.relative_to(work.resolve())
        source = directory / "fixture.cpp"
        executable = directory / "fixture"
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
             str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
