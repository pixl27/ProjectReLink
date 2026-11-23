"""The hot pieces' native float helpers are made from the run's shapes (nx2/chunks.py fill_fpx; no game or GPU).

The emit's processes lift with no texts of their own (chunks.no_texts), so the lifter writes a64_fpx.h and a64_hot_fp.inc for
no FPX shape, and fill_fpx puts in the shapes the processes found. The hot pieces call a64_fpx_nat_<fast>_<exact> for each
shape they use; an inc left for no shape defines none of them, and the game build stops at its link. Checked on a folder
written as the lifter writes it: after fill_fpx, a64_fpx.h has the shapes' helpers and a64_hot_fp.inc names every
a64_fpx_nat_* of them; with A64_NATIVE_FP=0 the inc is not written.

python -I -B tests/test_native_fp_fill.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "a64-kit", "lift"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "a64-kit", "tools"))
sys.path.insert(0, os.path.join(ROOT, "nx2"))
import a64_lift  # noqa: E402
import chunks  # noqa: E402

SHAPES = {("add32", "add32", 2, 2), ("mul64", "fma64", 2, 3), ("sqrt32", "sqrt32", 1, 1)}


def run(native):
    a64_lift.NATIVE_FP = native
    with tempfile.TemporaryDirectory(prefix="native-fill-") as d:
        with open(os.path.join(d, "a64_fpx.h"), "w", encoding="utf-8", newline="\n") as f:
            f.write(a64_lift.fpx_header(set()) + "/* the rest */\n")
        inc = os.path.join(d, a64_lift.NATIVE_INC)
        if native:
            with open(inc, "w", encoding="utf-8", newline="\n") as f:
                f.write(a64_lift.native_header(set()))
        chunks.fill_fpx(d, SHAPES)
        with open(os.path.join(d, "a64_fpx.h"), encoding="utf-8") as f:
            head = f.read()
        text = None
        if os.path.exists(inc):
            with open(inc, encoding="utf-8") as f:
                text = f.read()
        return head, text


def main():
    head, text = run(True)
    for fast, exact, _, _ in SHAPES:
        if "a64_fpx_%s_%s(" % (fast, exact) not in head:
            raise SystemExit("FAIL: a64_fpx.h has no helper for %s_%s" % (fast, exact))
        if text is None or "a64_fpx_nat_%s_%s" % (fast, exact) not in text:
            raise SystemExit("FAIL: a64_hot_fp.inc names no a64_fpx_nat_%s_%s (the hot pieces would not link)" % (fast, exact))
    if not head.endswith("/* the rest */\n"):
        raise SystemExit("FAIL: fill_fpx changed a64_fpx.h past its helpers")
    _, text = run(False)
    if text is not None:
        raise SystemExit("FAIL: A64_NATIVE_FP=0 and fill_fpx wrote a64_hot_fp.inc")
    print("PASS: fill_fpx makes a64_hot_fp.inc from the run's shapes, every a64_fpx_nat_* named; none with A64_NATIVE_FP=0")


if __name__ == "__main__":
    main()
