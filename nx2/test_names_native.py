"""Test native name-search comparison state and guest scheduler checkpoints.

No game data, guest memory, renderer, or platform thread is used.

python nx2/test_names_native.py [--cc clang]
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    fixture = r'''
#include <stdio.h>
#include <stdlib.h>
#include "host/hos.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); exit(1); } } while (0)

static int checkpoints;
void hos_vcore_checkpoint(void) { checkpoints++; }
void hos_log(const char *fmt, ...) { (void)fmt; }

int hos_names_lower(Ctx *c);
int hos_names_upper(Ctx *c, int at_head);
int hos_names_test_str_cmp(Ctx *c, uint64_t element, uint64_t target, uint32_t neg, uint64_t out[7]);

int main(void) {
    Ctx c = {0};
    c.poll = 1;
    CHECK(hos_names_lower(&c) == 0);
    CHECK(c.poll == 0);
    CHECK(checkpoints == 1);
    CHECK(hos_names_lower(&c) == 0);
    CHECK(checkpoints == 1);
    c.poll = 1;
    CHECK(hos_names_upper(&c, 1) == 0);
    CHECK(c.poll == 0);
    CHECK(checkpoints == 2);
    CHECK(hos_names_upper(&c, 0) == 0);
    CHECK(checkpoints == 2);

    uint64_t out[7] = {0};
    char equal_a[] = "abc", equal_b[] = "abc";
    CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)equal_a, (uint64_t)(uintptr_t)equal_b, 1, out) == 0);
    CHECK((uint32_t)out[0] == 0);
    CHECK(out[1] == (uint64_t)(uintptr_t)equal_b + 3);
    CHECK(out[2] == (uint64_t)(uintptr_t)equal_a + 3);
    CHECK(out[3] == 0x400000);
    CHECK((uint32_t)out[4] == 0 && (uint32_t)out[5] == 0 && (uint32_t)out[6] == UINT32_MAX);

    char lower[] = "abc", greater[] = "abd";
    CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)lower, (uint64_t)(uintptr_t)greater, 1, out) == 0);
    CHECK((uint32_t)out[0] == 1 && (uint32_t)out[4] == 'c' && (uint32_t)out[5] == 'd' && (uint32_t)out[6] == 1);
    CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)greater, (uint64_t)(uintptr_t)lower, 1, out) == 0);
    CHECK((uint32_t)out[0] == UINT32_MAX && (uint32_t)out[4] == 'd' && (uint32_t)out[5] == 'c');

    char long_a[] = "abcdefghX", long_b[] = "abcdefghY";
    c.poll = 1;
    CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)long_a, (uint64_t)(uintptr_t)long_b, 1, out) == 0);
    CHECK(c.poll == 0 && checkpoints == 3);
    CHECK(out[1] == (uint64_t)(uintptr_t)long_b + 11);
    CHECK(out[2] == (uint64_t)(uintptr_t)long_a + 11);
    CHECK(out[3] == 0x3ffff8);
    CHECK((uint32_t)out[0] == 1 && (uint32_t)out[4] == 'X' && (uint32_t)out[5] == 'Y');

    char scan_a[48], scan_b[48];
    for (int len = 0; len < 32; len++) {
        for (int i = 0; i < len; i++) scan_a[i] = scan_b[i] = (char)('A' + i % 23);
        scan_a[len] = scan_b[len] = 0;
        CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)scan_a, (uint64_t)(uintptr_t)scan_b, 1, out) == 0);
        CHECK((uint32_t)out[0] == 0 && (uint32_t)out[4] == 0 && (uint32_t)out[5] == 0);
        CHECK(out[1] == (uint64_t)(uintptr_t)scan_b + 3 + (uint64_t)(len / 8) * 8);
        CHECK(out[2] == (uint64_t)(uintptr_t)scan_a + 3 + (uint64_t)(len / 8) * 8);
        CHECK(out[3] == 0x400000 - (uint64_t)(len / 8) * 8);
        for (int different = 0; different < len; different++) {
            scan_b[different]++;
            CHECK(hos_names_test_str_cmp(&c, (uint64_t)(uintptr_t)scan_a, (uint64_t)(uintptr_t)scan_b, 1, out) == 0);
            CHECK((uint32_t)out[0] == 1);
            CHECK((uint32_t)out[4] == (uint8_t)scan_a[different]);
            CHECK((uint32_t)out[5] == (uint8_t)scan_b[different]);
            CHECK(out[1] == (uint64_t)(uintptr_t)scan_b + 3 + (uint64_t)(different / 8) * 8);
            CHECK(out[2] == (uint64_t)(uintptr_t)scan_a + 3 + (uint64_t)(different / 8) * 8);
            CHECK(out[3] == 0x400000 - (uint64_t)(different / 8) * 8);
            scan_b[different]--;
        }
    }
    puts("PASS: native name-search comparisons preserve state and virtual-core preemption");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="names-native-", dir=work) as temp:
        directory = Path(temp).resolve()
        directory.relative_to(work.resolve())
        source = directory / "fixture.c"
        executable = directory / "fixture"
        # names_native includes the generated guest model through hos.h.  Keep
        # this scheduler test independent of a previous title lift while still
        # modelling the registers and interrupt flag it touches.
        (directory / "a64.h").write_text(
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "typedef struct GuestContext {\n"
            "    uint64_t x[32];\n"
            "    uint8_t n, z, c, v;\n"
            "    uint32_t poll;\n"
            "} GuestContext;\n"
            "typedef GuestContext Ctx;\n"
            "#define A64_PTR(a) ((uint8_t *)(uintptr_t)(a))\n",
            encoding="utf-8",
        )
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [
                args.cc,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DHOS_NAMES_TEST",
                "-I",
                str(ROOT),
                "-I",
                str(directory),
                "-I",
                str(ROOT / "third_party" / "a64-kit" / "rt"),
                str(ROOT / "host" / "names_native.c"),
                str(source),
                "-o",
                str(executable),
            ],
            check=True,
        )
        env = os.environ.copy()
        env["HOS_NAMES_NATIVE"] = "0"
        subprocess.run([str(executable)], check=True, env=env)


if __name__ == "__main__":
    main()
