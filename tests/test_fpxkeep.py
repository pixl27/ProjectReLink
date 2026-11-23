"""nx2/fpxkeep.py (A64_FPXKEEP): NaN operands (N), infinite operands (I) and exact-zero multiply-adds (Z) kept off the
software bodies, game-free: bit-exact with the bodies, result and FPSR.

Built like tests/test_fpx_sticky.py (a64emit from its source, a64_suyu.h / a64_suyu.c by a64_rtgen with
nx2/suyu_rt_fixes.py, a64_fpx.h by a64_lift.fpx_header with nx2/fastvec.py and nx2/fpxkeep.py installed, as nx2/lift.py
installs them), and tests/fpxkeep_test.c compiled against the real rt/a64.h at the lifted code's FP flags, at
-march=x86-64-v3 (a host FMA) and x86-64-v2 (none: Z for binary64 must keep nothing). In each build every scalar wrapper
fpxkeep changes (FADD, FSUB, FMUL, FMADD family; S and D) and the kit's lane routes (2- and 4-lane FADD / FSUB, 4-lane FMUL
and FMUL by element) are compared with the exact bodies:
  - random: 2M cases of each op (16M scalar checks): operand classes weighted to NaNs (quiet, signalling, and a quiet
    NaN with payload), infinities, zeros, subnormals and the range edges, multiply-adds with (near) cancelling addends and
    products at Z's exponent edge, random FPCR (FZ, DN, RMode, other bits, the kill switch) and FPSR states;
  - sweep: every pair and triple of 30 binary32 / 26 binary64 special values under FPCR 0 and FZ and four FPSR states
    (~1.4M checks), and Z's edge (products whose exponents sum to the edge - 2 .. + 2, addends that leave 0, +-2^-1075 /
    2^-150, +-2^-1074 / 2^-149);
  - lanes: 1M random vector ops of each form, lanes weighted to a quiet NaN with payload.
The rules' results are counted inside the generated code: each rule must keep cases (Z for binary64 only at v3), none
under a closed gate. Then the generated header is checked for its edits (a rescue in each of the eight wrappers, the
lane routes' hooks) and for A64_FPXKEEP=0 (a64_fpx.h as without fpxkeep, to the byte); and deliberately broken rules,
each built at v3 and run on a smaller set, must fail: the NaN order, IOC for a signalling NaN, IDC under FZ, the
default NaN for a quiet-NaN addend with inf * 0, FSUB negating a NaN, inf - inf and inf * 0 kept, a subnormal
multiplicand not flushed beside an infinity under FZ, Z's binary64 edge one lower, Z's binary32 test in binary32; and
in the lane routes: 2 lanes treated as 4, the NaN order, IOC, FSUB's infinite b, inf * a flushed subnormal.

    python -I -B tests/test_fpxkeep.py [--cc clang] [--cxx clang++] [--random 2000000] [--lanes 1000000] [--no-faults]

Skipped (exit 0) on a host that is not x86-64 or has no C/C++ compiler; a build this CPU cannot run is skipped.
"""
import argparse
import os
import platform
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_fpx_sticky as fs  # noqa: E402  (its compiler, target, build and run helpers; its paths on sys.path)

SEED = 20261120
BUILDS = (("x86-64-v3", 1), ("x86-64-v2", 0))
SHAPES = {("add32", "add32", 2, 2), ("sub32", "sub32", 2, 2), ("add64", "add64", 2, 2), ("sub64", "sub64", 2, 2),
          ("mul32", "fma32", 2, 3), ("fma32", "fma32", 3, 3), ("mul64", "fma64", 2, 3), ("fma64", "fma64", 3, 3)}
# (name, text in the generated a64_fpx.h, its broken form): each must be found once, and its build must fail
FAULTS = (
    ("the NaN order (op2 first)", "return a64_fpxk_nan##W(a) ? a : b; }", "return a64_fpxk_nan##W(b) ? b : a; }"),
    ("no IOC for a signalling NaN", "if (a64_fpxk_snan##W(a)) { *f |= 1u; return a | (Q); }",
     "if (a64_fpxk_snan##W(a)) { return a | (Q); }"),
    ("no IDC for a flushed operand of a sum",
     "uint32_t f = (fz && (a64_fpxk_sub##W(a) || a64_fpxk_sub##W(b))) ? 0x80u : 0u;", "uint32_t f = 0u;"),
    ("a quiet-NaN addend with inf * 0 kept as the addend", "r = (DNAN);", "r = z;"),
    ("FSUB negating a NaN operand", "        r = (T)a64_fpxk_pick##W(a, b, &f); \\\n    } else if (a64_fpxk_inf##W(a) || a64_fpxk_inf##W(b)) {",
     "        r = (T)a64_fpxk_pick##W(a, op ? b ^ (SIGN) : b, &f); \\\n    } else if (a64_fpxk_inf##W(a) || a64_fpxk_inf##W(b)) {"),
    ("inf - inf kept (sums)", "&& ((a ^ sb) & (SIGN))) return 0;", "&& 0) return 0;"),
    ("inf * 0 kept", "if (za || zb) return 0;", "if (0) return 0;"),
    ("a subnormal multiplicand not flushed under FZ (inf * it kept)", "zb = !(b & ~(SIGN)) || (fz && sb);",
     "zb = !(b & ~(SIGN));"),
    ("Z's binary64 edge one lower", "ea && eb && ea + eb >= 1076 &&", "ea && eb && ea + eb >= 1075 &&"),
    ("Z's binary32 test in binary32", "(double)recomp_fpx_f(a) * (double)recomp_fpx_f(b) + (double)recomp_fpx_f(z) == 0.0",
     "recomp_fpx_f(a) * recomp_fpx_f(b) + recomp_fpx_f(z) == 0.0f"),
    ("lane route: 2 lanes treated as 4", "_mm_set_epi32(0, 0, -1, -1)", "_mm_set1_epi32(-1)"),
    ("lane route: the NaN order", "const __m128i picka = _mm_or_si128(sna, _mm_andnot_si128(snb, nana));",
     "const __m128i picka = nana;"),
    ("lane route: no IOC", "(_mm_movemask_epi8(_mm_and_si128(_mm_or_si128(sna, snb), special)) ? 1u : 0u)", "0u"),
    ("lane route: no IDC", "(_mm_movemask_epi8(_mm_and_si128(_mm_or_si128(suba, subb), special)) ? 0x80u : 0u)", "0u"),
    ("lane route: FSUB's infinite b without its flipped sign", "_mm_or_si128(_mm_and_si128(be, sgn), inf)",
     "_mm_or_si128(_mm_and_si128(b, sgn), inf)"),
    ("lane route: inf * a flushed subnormal kept", "_mm_or_si128(_mm_or_si128(z0a, suba), _mm_or_si128(z0b, subb))",
     "_mm_or_si128(z0a, z0b)"),
)


def generate(cxx, tmp, flags):
    """a64_suyu.h / .c and a64_fpx.h (fpxkeep on), and the checks of the header's text and of A64_FPXKEEP=0"""
    dll = os.path.join(tmp, "a64emit.dll" if os.name == "nt" else "liba64emit.so")
    fs.build([cxx, "-std=c++20", "-O0", "-shared", "-Wno-everything", "-o", dll, os.path.join(fs.KIT, "lift", "a64emit.cpp")]
             + flags + ([] if os.name == "nt" else ["-fPIC"]), "a64emit")
    import build_emit
    build_emit.build = lambda force=False: dll
    import a64_lift
    import a64_rtgen
    import a64emit
    import suyu_rt_fixes
    import fastvec
    import fpxkeep
    suyu_rt_fixes.install()
    fastvec.install()
    # (A64_FPXKEEP=0: install() changes nothing, so the header is main's to the byte)
    base = a64_lift.fpx_header(SHAPES)
    shape = ("mul32", "fma32", 2, 3)
    partial = a64_lift.fpx_header({shape})
    empty = a64_lift.fpx_header(set())
    saved = os.environ.get("A64_FPXKEEP")
    os.environ["A64_FPXKEEP"] = "0"
    try:
        fpxkeep.install()
        off = a64_lift.fpx_header(SHAPES)
        fpxkeep._installed = False
        os.environ["A64_FPXKEEP"] = "1"
        fpxkeep.install()
        header = a64_lift.fpx_header(SHAPES)
    finally:
        if saved is None:
            del os.environ["A64_FPXKEEP"]
        else:
            os.environ["A64_FPXKEEP"] = saved
    if off != base:
        raise SystemExit("fpxkeep: A64_FPXKEEP=0 changed a64_fpx.h")
    for (fast, exact), call in fpxkeep.WRAPPED.items():
        line = [x for x in header.split("\n") if x.startswith("A64_FPX_FN uint64_t a64_fpx_%s_%s(" % (fast, exact))]
        if len(line) != 1 or ("if (%s) return v; } return recomp_fpe_" % call) not in line[0]:
            raise SystemExit("fpxkeep: the wrapper a64_fpx_%s_%s has no rescue: %s" % (fast, exact, line))
    for hook in ("if (a64_fpxk_vspecial(c, n, m, k, #name[0] == 's', _ka, _kb, &_ks, &_kr, &_kf))",
                 "if (a64_fpxk_vspecial(c, n, m, 4, 2, _ka, _kb, &_ks, &_kr, &_kf))"):
        if header.count(hook) != 1:
            raise SystemExit("fpxkeep: the lane route's hook %r is in a64_fpx.h %d times" % (hook, header.count(hook)))
    if header == base:
        raise SystemExit("fpxkeep: not on")
    # A missing or duplicated requested declaration must fail even if its prefix changes; absent shapes are allowed.
    declaration = "A64_FPX_FN uint64_t a64_fpx_mul32_fma32("
    if partial.count(declaration) != 1:
        raise SystemExit("fpxkeep: test wrapper declaration changed")
    fpxkeep.keep_wrappers(empty, set())
    fpxkeep.keep_wrappers(partial, {shape})
    for broken in (partial.replace(declaration, "changed uint64_t a64_fpx_mul32_fma32("),
                   partial + next(x for x in partial.split("\n") if x.startswith(declaration)) + "\n"):
        try:
            fpxkeep.keep_wrappers(broken, {shape})
        except SystemExit:
            pass
        else:
            raise SystemExit("fpxkeep: missing or duplicated wrapper accepted")
    try:
        a64_rtgen.write(tmp)
    finally:
        if a64emit._lib is not None:
            import _ctypes
            (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(a64emit._lib._handle)
            a64emit._lib = None
    with open(os.path.join(tmp, "a64_fpx.h"), "w", newline="\n") as f:
        f.write(header)
    return header


def compile_driver(cc, tmp, march, fma, inc, name):
    exe = os.path.join(tmp, "fpxkeep_test_%s.exe" % name)
    fs.build([cc] + fs.target_flags(cc, tmp) + ["-march=" + march] + fs.FLAGS +
             ["-DEXPECT_FMA=%d" % fma, "-D_CRT_SECURE_NO_WARNINGS", "-I" + inc, "-I" + tmp, "-I" + os.path.join(fs.KIT, "rt"),
              os.path.join(ROOT, "tests", "fpxkeep_test.c"), os.path.join(tmp, "a64_suyu.c"), "-o", exe],
             "the driver " + name)
    return exe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cc")
    p.add_argument("--cxx")
    p.add_argument("--random", type=int, default=2000000)
    p.add_argument("--lanes", type=int, default=1000000)
    p.add_argument("--zero", type=int, default=4000000)
    p.add_argument("--no-faults", action="store_true")
    a = p.parse_args()
    if platform.machine().lower() not in ("x86_64", "amd64"):
        fs.skip("not an x86-64 host: %s" % platform.machine())
    cc = fs.compiler(a.cc, "CC", ("clang", "gcc", "cc"), "clang")
    cxx = fs.compiler(a.cxx, "CXX", ("clang++", "g++", "c++"), "clang++")
    if not cc or not cxx:
        fs.skip("no C compiler" if not cc else "no C++ compiler")
    t0 = time.time()
    failed, ran = [], []
    no_lane_routes = set()
    with tempfile.TemporaryDirectory(prefix="hookshot-fpxkeep-") as tmp:
        probe = os.path.join(tmp, "fpxkeep_cpu.exe")
        fs.build([cc] + fs.target_flags(cc, tmp) + ["-march=x86-64", os.path.join(ROOT, "tests", "fpxkeep_cpu.c"), "-o", probe],
                 "the CPU feature probe")
        supported = {march: fs.run([probe, str(fma)]).returncode == 0 for march, fma in BUILDS}
        header = generate(cxx, tmp, fs.target_flags(cxx, tmp))
        runs = (["random", str(a.random), str(SEED)], ["sweep"], ["zero", str(a.zero), str(SEED)],
                ["lanes", str(a.lanes), str(SEED)])
        for march, fma in BUILDS:
            if not supported[march]:
                print("%-10s skipped (this CPU or OS does not support %s)" % (march, march))
                continue
            exe = compile_driver(cc, tmp, march, fma, tmp, march)
            for args in runs:
                r = fs.run([exe] + args, timeout=3600)
                print("%-10s %s" % (march, r.stdout.rstrip()))
                ran.append(march)
                if (r.returncode == 4 and args[0] == "lanes" and " 0 mismatches" in r.stdout
                        and "lane routes 0;" in r.stdout):
                    # (Hookshot's kit: its own fast float code takes the lanes before the kit's lane routes, so
                    # fpxkeep's lane-route keep is never reached here; the lanes are still checked bit-exact)
                    no_lane_routes.add(march)
                    continue
                if r.returncode:
                    failed.append("%s %s (exit %d)" % (march, args[0], r.returncode))
        if not a.no_faults and "x86-64-v3" in ran:
            for i, (what, old, new) in enumerate(FAULTS):
                if what.startswith("lane route") and "x86-64-v3" in no_lane_routes:
                    print("fault %-55s skipped (no lane route on this kit)" % what)
                    continue
                if header.count(old) != 1:
                    raise SystemExit("fpxkeep: the fault %r: its text is in a64_fpx.h %d times" % (what, header.count(old)))
                d = os.path.join(tmp, "fault%d" % i)
                os.makedirs(d)
                with open(os.path.join(d, "a64_fpx.h"), "w", newline="\n") as f:
                    f.write(header.replace(old, new))
                exe = compile_driver(cc, tmp, "x86-64-v3", 1, d, "fault%d" % i)
                codes = [fs.run([exe] + args, timeout=3600).returncode
                         for args in (["sweep"], ["random", "300000", str(SEED)], ["zero", "300000", str(SEED)],
                                      ["lanes", "200000", str(SEED)])]
                caught = any(c in (1, 4) for c in codes)
                print("fault %-55s %s (exits %s)" % (what, "caught" if caught else "NOT CAUGHT", codes))
                if not caught:
                    failed.append("fault not caught: " + what)
    if not ran:
        fs.skip("this CPU is not x86-64-v2")
    if failed:
        raise SystemExit("fpxkeep: FAIL (%s) in %.1f s" % (", ".join(failed), time.time() - t0))
    print("fpxkeep: pass in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
