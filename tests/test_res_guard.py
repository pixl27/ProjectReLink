"""The resource lookup's invalid handles take its null path (nx2/oot_patches.py patch_res_guard; no game or GPU).

f_80cb38d0 looks a type-0 handle up in the resource manager's table. A negative id, one past the table or an empty entry
went on to read the entry's +0x150 with a null entry (guest address 0x150). The patch sends the three branches to the
function's own invalid-handle path (L_80cb3a18), which stores null for its callers to test. Checked on a fragment written
as the lift writes the function (its registers in locals, and in the context): the three branches moved, nothing else; a second run changes nothing; a file without the
function, or with the code changed (another game version), is left as it is.

python -I -B tests/test_res_guard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import oot_patches as op  # noqa: E402

FRAGMENT = """#include "a64.h"
void f_80cb38d0(Ctx *__restrict A64_NOESC c) {
    x19_ = 0;
if (((x8_ >> 31) & 1)) { goto L_80cb3940; }
    x9_=recomp_load32(c,x21_ + 2112);
if (!((fn_ == fv_) && !fz_)) { goto L_80cb3940; }
    x19_ = recomp_load64(c,(x9_ + ((x8_) << 3)));
if (x19_ == 0) { goto L_80cb3940; }
L_80cb3940:;
    x0_=recomp_load64(c,x19_ + 336);
L_80cb39e0:;
if (((x8_ >> 31) & 1)) { goto L_80cb3a18; }
L_80cb3a18:;
    recomp_store64(c,c->x[20] + 0,(uint64_t)0);
}
void f_80cb3b80(Ctx *__restrict A64_NOESC c) {
if (x19_ == 0) { goto L_80cb3940; }
}
"""


def main():
    s, n = op.patch_res_guard(FRAGMENT)
    if n != 1 or s == FRAGMENT:
        raise SystemExit("FAIL: the resource lookup's branches were not patched (%d)" % n)
    head, end = op._function(s, op.RES_HEAD)
    body, rest = s[head:end], s[end:]
    if any(old in body for old in op.RES_SITES) or body.count(op.RES_GUARD) != 3:
        raise SystemExit("FAIL: not all three invalid-handle branches go to the null path")
    if body.count("goto L_80cb3a18;") != 4 or "L_80cb3940:;" not in body:
        raise SystemExit("FAIL: the function's other code changed")
    if rest != FRAGMENT[FRAGMENT.index("\nvoid f_80cb3b80"):]:
        raise SystemExit("FAIL: another function was changed")
    again, n2 = op.patch_res_guard(s)
    if n2 != 1 or again != s:
        raise SystemExit("FAIL: a second run changed the file")
    if op.patch_res_guard("void f_1(Ctx *__restrict A64_NOESC c) {\n}\n") != ("void f_1(Ctx *__restrict A64_NOESC c) {\n}\n", 0):
        raise SystemExit("FAIL: a file without the function was changed")
    other = FRAGMENT.replace("if (x19_ == 0) { goto L_80cb3940; }\nL_80cb3940", "if (x19_ != 0) { goto L_80cb3940; }\nL_80cb3940")
    if op.patch_res_guard(other) != (other, 0):
        raise SystemExit("FAIL: a function whose code differs was changed")
    # (the lift's other form: the registers in the context, not in locals)
    ctx = FRAGMENT
    for a, b in zip(op.RES_FORMS[0], op.RES_FORMS[1]):
        ctx = ctx.replace(a, b)
    s, n = op.patch_res_guard(ctx)
    head, end = op._function(s, op.RES_HEAD)
    if n != 1 or any(old in s[head:end] for old in op.RES_FORMS[1]) or s[head:end].count(op.RES_GUARD) != 3:
        raise SystemExit("FAIL: the lift's form with the registers in the context was not patched")
    if op.patch_res_guard(s) != (s, 1):
        raise SystemExit("FAIL: a second run changed the context form")
    print("PASS: the resource lookup's three invalid-handle branches take its null path, once, and nothing else (both forms)")


if __name__ == "__main__":
    main()
