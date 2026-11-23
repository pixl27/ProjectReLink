"""nx2/fma_soft.h, game-free: the early exit in front of suyu's exact FMA bodies gives the same value and the same
guest state as the exact body, on every operand it takes and every one it declines.

On an x86-64 host without a hardware FMA (an x86-64-v2 build: RECOMP_FPX_FMA64 is 0) the generated runtime's
recomp_fpe_fma32 and recomp_fpe_fma64 start with nx2/fma_soft.h's correctly rounded FMA (Veltkamp/Dekker exact product,
TwoSum, rounding to odd, one rounding to nearest; binary32 through binary64), which suyu_rt_fixes.py patches in. This builds the
runtime the way the lift does (a64emit built here at -O0 into a temporary folder, a64_rtgen with nx2/suyu_rt_fixes.py) and
compiles tests/fma_soft_test.c against it at -ffp-contract=off -O2 (the build's FP flags), at -march=x86-64-v2 (the early
exit compiled in) and at -march=x86-64-v3 (compiled out: the build must say so, and run the same code twice). In the v2
build the driver calls each function twice on the same operands and the same GuestContext, with the early exit off
(recomp_fma_soft_mode = 0: suyu's exact body, what main does) and on, and compares the result and every byte of the
context (FPSR, FPCR, everything):
  - 'random': 10M calls (default; seeded) over operands of every class: zeros, subnormals, infinities, NaNs (quiet and
    signalling), the extreme and the moderate normals, exponents at the early exit's limits, fractions of few bits (exact
    products), the addend cancelling the product to a few ulps, around the product's last place and a rounding's width
    above it (the sticky cases), making the sum a power of two, and far from the product; under FPCR 0 most of the time and
    otherwise FZ, DN, every rounding mode, FIZ/AH/NEP, trap enables, FPX1's kill switch (bit 32) and random words, and
    under random FPSR flags;
  - 'edge': every exponent pair around each limit of the early exit (binary64 operands 397-403 and 1597-1603 and between,
    binary32 37-42 and 208-213, the addend at its limits, at 0, 1, the top and around the product), six fractions each,
    all eight sign patterns (~8M calls).
The early exit must be taken on at least an eighth of the random calls (a fortieth of the edge sweep's), or the run proves
nothing (about a quarter and a sixth are taken). About 20-30 s here.

It has teeth (each by hand, then restored; 2M random calls and the edge sweep):
  - the inexact flag dropped (`*ix = 0` in both): mismatches in both widths (FPSR IXC missing);
  - the rounding to odd left out (`so = s`): double rounding, mismatches in both widths;
  - the FPCR gate dropped (`!(c->fpcr & RECOMP_FPX_FPCR_MASK)` removed from suyu_rt_fixes.py): the non-nearest rounding
    modes and the kill switch mismatch.
Two guards are margins that no input here needs, and the test cannot show them: the power-of-two exclusion (removing it
changes nothing in 6M random calls with sums made power-of-two on purpose) and the operand exponent limits (the product
and result range checks catch what overflows, and an input that needs the lower limit has to be a rounding tie built
from a split part that underflows, which none of the generators makes).

    python -I -B tests/test_fma_soft.py [--cc clang] [--cxx clang++] [--trials N]

Skipped (exit 0) on a host that is not x86-64, has no C/C++ compiler (PATH, $CC/$CXX, or the build's own toolchain) or a
CPU without x86-64-v2.
"""
import argparse
import os
import platform
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fpx_sticky as fpx   # noqa: E402  (its toolchain helpers: compiler(), target_flags(), run(), build())

ROOT, KIT, FLAGS = fpx.ROOT, fpx.KIT, fpx.FLAGS
TRIALS, SEED = 10000000, 20261114
BUILDS = (("x86-64-v2", 1), ("x86-64-v3", 0))   # (-march, EXPECT_SOFT: whether the early exit is compiled in there)


def skip(why):
    print("fma soft: skipped (%s)" % why)
    sys.exit(0)


def generate(cxx, tmp, flags):
    """a64_fpx.h, a64_suyu.h and a64_suyu.c, by the generators the lift runs (a64emit built into tmp)"""
    dll = os.path.join(tmp, "a64emit.dll" if os.name == "nt" else "liba64emit.so")
    fpx.build([cxx, "-std=c++20", "-O0", "-shared", "-Wno-everything", "-o", dll, os.path.join(KIT, "lift", "a64emit.cpp")]
              + flags + ([] if os.name == "nt" else ["-fPIC"]), "a64emit")
    import build_emit
    build_emit.build = lambda force=False: dll   # (not a64emit.dll beside the source)
    import a64_lift
    import a64_rtgen
    import a64emit
    import suyu_rt_fixes
    suyu_rt_fixes.install()   # (as nx2/lift.py does)
    try:
        a64_rtgen.write(tmp)
        with open(os.path.join(tmp, "a64_fpx.h"), "w", newline="\n") as f:   # (rt/a64.h includes it: no wrapper is used here)
            f.write(a64_lift.fpx_header(set()))
    finally:
        if a64emit._lib is not None:   # (unloaded, so the folder can go)
            import _ctypes
            (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(a64emit._lib._handle)
            a64emit._lib = None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cc")
    p.add_argument("--cxx")
    p.add_argument("--trials", type=int, default=TRIALS)
    a = p.parse_args()
    if platform.machine().lower() not in ("x86_64", "amd64"):
        skip("not an x86-64 host: %s" % platform.machine())
    cc = fpx.compiler(a.cc, "CC", ("clang", "gcc", "cc"), "clang")
    cxx = fpx.compiler(a.cxx, "CXX", ("clang++", "g++", "c++"), "clang++")
    if not cc or not cxx:
        skip("no C compiler" if not cc else "no C++ compiler")
    t0 = time.time()
    failed, ran = [], []
    with tempfile.TemporaryDirectory(prefix="hookshot-fma-soft-") as tmp:
        generate(cxx, tmp, fpx.target_flags(cxx, tmp))
        with open(os.path.join(tmp, "fma_soft_probe.h"), "w", newline="\n") as f:   # (the runtime counts the early exits taken)
            f.write("#include <stdint.h>\nextern uint64_t g_fma_soft_taken[2];\n#define RECOMP_FMA_SOFT_PROBE(k) (g_fma_soft_taken[k]++)\n")
        for march, soft in BUILDS:
            exe = os.path.join(tmp, "fma_soft_test_%s.exe" % march)
            fpx.build([cc] + fpx.target_flags(cc, tmp) + ["-march=" + march] + FLAGS +
                      ["-DEXPECT_SOFT=%d" % soft, "-include", os.path.join(tmp, "fma_soft_probe.h"),
                       "-D_CRT_SECURE_NO_WARNINGS", "-Wno-everything", "-I" + os.path.join(KIT, "rt"), "-I" + tmp,
                       os.path.join(ROOT, "tests", "fma_soft_test.c"), os.path.join(tmp, "a64_suyu.c"), "-o", exe],
                      "the driver at " + march)
            if march != "x86-64-v2":
                # (an x86-64-v3 binary runs only where the CPU has AVX2/FMA: it must say the early exit is compiled out)
                r = fpx.run([exe, "random", "1000", "1"])
                if r.returncode == 3 or "Illegal" in r.stdout or r.returncode < 0:
                    print("fma soft: %s built, not run (this CPU is not %s)" % (march, march))
                    continue
                print(r.stdout.rstrip())
                ran.append(march)
                if r.returncode:
                    failed.append("%s compiled out" % march)
                continue
            for args in (["random", str(a.trials), str(SEED)], ["edge"]):
                r = fpx.run([exe] + args)
                print(r.stdout.rstrip())
                ran.append(march)
                if r.returncode:
                    failed.append("%s %s (exit %d)" % (march, args[0], r.returncode))
    if not ran:
        skip("this CPU is not x86-64-v2")
    if failed:
        raise SystemExit("fma soft: FAIL (%s) in %.1f s" % (", ".join(failed), time.time() - t0))
    print("fma soft: pass in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
