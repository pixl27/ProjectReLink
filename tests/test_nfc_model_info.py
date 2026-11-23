"""GetModelInfo converts the tag's big-endian model number to the SDK's native u16.

Compile the real HLE function from host/nn_nfp.c with stand-ins for guest memory and
locking. Synthetic model blocks exercise unequal bytes in both orders, zero and
equal bytes; the other fields, reserved bytes, output bounds, tag contents and
unmounted states must stay correct. No tag dumps, keys or game files are used.

    python -I -B tests/test_nfc_model_info.py   (needs a C compiler; CC overrides it)
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block

FIXTURE = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct { uint64_t x[8]; } Ctx;
#define HLE(name) static void name(Ctx *c)
#define ARG(n) (c->x[n])
#define GP(a) ((void *)(uintptr_t)(a))
#define RET(v) (c->x[0] = (uint64_t)(v))
#define ST_MOUNTED 4
#define NFP_WRONG_STATE (115u | (73u << 9))
static int nfc_lk, dev_state, acquired, released;
static uint8_t tag[540];
static void AcquireSRWLockShared(int *lock) { (void)lock; acquired++; }
static void ReleaseSRWLockShared(int *lock) { (void)lock; released++; }
/* CUT */
static uint64_t call(uint8_t *out) {
    Ctx c = {{(uint64_t)(uintptr_t)out}};
    acquired = released = 0;
    _ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE(&c);
    return c.x[0];
}
int main(void) {
    const uint8_t cases[][7] = {
        {0x12, 0x34, 0x56, 0x01, 0x12, 0x34, 0x09},
        {0x78, 0x9a, 0xbc, 0x02, 0xab, 0x01, 0x05},
        {0xde, 0xf0, 0x01, 0x00, 0x00, 0xff, 0x04},
        {0x23, 0x45, 0x67, 0x01, 0xff, 0x00, 0x03},
        {0x89, 0xab, 0xcd, 0x02, 0x7e, 0x7e, 0x01},
        {0xef, 0x01, 0x23, 0x00, 0x00, 0x00, 0x00}
    };
    int fails = 0;
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint8_t guarded[80], original[540];
        uint8_t *out = guarded + 8;
        memset(tag, 0x5a, sizeof tag);
        memcpy(tag + 0x54, cases[i], 7);
        memcpy(original, tag, sizeof tag);
        memset(guarded, 0xa5, sizeof guarded);
        dev_state = ST_MOUNTED;
        const uint64_t result = call(out);
        uint16_t number;
        memcpy(&number, out + 4, sizeof number);
        const unsigned expected = (unsigned)cases[i][4] * 256 + cases[i][5];
        if (number != expected) {
            printf("FAIL case %u: model number 0x%04x, expected 0x%04x\n", i, number, expected);
            fails++;
        }
        if (result || acquired != 1 || released != 1 || memcmp(out, cases[i], 4) || out[6] != cases[i][6]) fails++;
        for (unsigned k = 7; k < 64; k++) if (out[k]) fails++;
        for (unsigned k = 0; k < sizeof guarded; k++)
            if ((k < 8 || k >= 72) && guarded[k] != 0xa5) fails++;
        if (memcmp(tag, original, sizeof tag)) fails++;
        for (int state = 0; state <= 5; state++) if (state != ST_MOUNTED) {
            dev_state = state;
            memset(guarded, 0xa5, sizeof guarded);
            if (call(out) != NFP_WRONG_STATE || acquired != 1 || released != 1) fails++;
            for (unsigned k = 0; k < sizeof guarded; k++) if (guarded[k] != 0xa5) fails++;
        }
    }
    dev_state = ST_MOUNTED;
    if (call(NULL) || acquired != 1 || released != 1) fails++;
    printf("%s: 6 synthetic models, output guards, unchanged tag, 5 unmounted states, null output\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
"""


class NfcModelInfo(unittest.TestCase):
    def test_model_number_and_output_contract(self):
        compiler = shlex.split(os.environ.get("CC", ""))
        if not compiler:
            found = shutil.which("gcc") or shutil.which("clang")
            if not found:
                self.skipTest("no C compiler")
            compiler = [found]
        source = (ROOT / "host" / "nn_nfp.c").read_text(encoding="utf-8")
        function = block(source, "HLE(_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE)",
                         where="host/nn_nfp.c")
        scratch = ROOT / "work"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nfc-model-", dir=scratch) as folder:
            folder = Path(folder)
            test = folder / "test.c"
            test.write_text(FIXTURE.replace("/* CUT */", function), encoding="utf-8")
            binary = folder / ("test.exe" if sys.platform == "win32" else "test")
            built = subprocess.run(compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror", str(test), "-o", str(binary)],
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
