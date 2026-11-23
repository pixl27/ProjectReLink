"""Changes to suyu's runtime header (a64-kit/lift/a64_rtgen.py's a64_suyu.h), applied by wrapping a64_rtgen.runtime_h.

1. FPX1's binary32 FMA, FRECPS and FRSQRTS fast paths (recomp_fpx_fma32, _recps32, _rsqrts32) use the host's fused
   multiply-add on an x86-64 host that has one (RECOMP_FPX_FMA64: x86-64-v3), as they do on an AArch64 host. suyu's
   header gives that to AArch64 alone, and an x86-64-v3 build computed them in binary64 and rounded twice, falling
   back to the exact body on every binary32 midpoint. One fused operation rounds once, so the result is the same where
   both keep it (tests/test_fpx_sticky.py's "fast" run), and the midpoints are kept too.
2. FCVTZS (recomp_fp_to_int, signed, toward zero) of a normal value below 2^(int_bits-1) in magnitude takes the C
   conversion, which truncates as the guest does and cannot overflow, and sets IXC when the result differs from the
   input. Zeros, subnormals and FZ, infinities, NaNs and saturation keep the exact body (tests/test_fcvtzs_fast.py).

2. A correctly rounded FMA for x86-64 hosts without one (nx2/fma_soft.h), in front of the exact bodies recomp_fpe_fma32 and
   recomp_fpe_fma64 of the generated runtime (a64_suyu.c, wrapping a64_rtgen.runtime_c). On an x86-64-v2 build (a CPU
   without AVX2/FMA3) FPX1 has no double-precision FMA fast path and the single-precision one needs the flags set
   already, so most guest FMADD/FMSUB/FNMADD/FNMSUB/FMLA reached the exact bodies' multi-word integer arithmetic (about
   180 ns a call). The early exit gives the same value and the same FPSR in the common case (tests/test_fma_soft.py) and
   leaves every other case to the exact body. Off with HOS_FMA_FAST=0; compiled out where RECOMP_FPX_FMA64 holds.

Each change is a text substitution required to match exactly once, so a refreshed a64-kit that moved these lines fails
here loudly.
"""
import os

try:
    import a64_rtgen
except ImportError:   # (tests/test_fcvtzs_fast.py takes fcvtzs() alone, without the lifter on the path)
    a64_rtgen = None

_FMA_GUARD = "#if RECOMP_FPX_HOST == 1\n"
SUBST = [(sig + _FMA_GUARD, sig + "#if RECOMP_FPX_HOST == 1 || RECOMP_FPX_FMA64\n") for sig in (
    "static RECOMP_INLINE int recomp_fpx_fma32(uint64_t a, uint64_t b, uint64_t z, uint32_t fz, uint64_t* v) {\n",
    "static RECOMP_INLINE int recomp_fpx_recps32(uint64_t a, uint64_t b, uint32_t fz, uint64_t* v) {\n",
    "static RECOMP_INLINE int recomp_fpx_rsqrts32(uint64_t a, uint64_t b, uint32_t fz, uint64_t* v) {\n")]

FP_TO_INT = "static RECOMP_INLINE uint64_t recomp_fp_to_int(uint64_t bits, unsigned fp_bits,\n"
FP_TO_INT_AT = "    int shift;\n"
# (off where the host's float compare could be folded or kept in extended precision: FPX1's own conditions)
FCVTZS = """#if !defined(RECOMP_NO_NATIVE_FP) && !defined(__FAST_MATH__) && !defined(_M_FP_FAST) && \\
    !(defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0)
    if (rmode == 3 && is_signed && exponent - 1u < bias + int_bits - 2u) {
        if (fp_bits == 32) {
            const uint32_t narrow = (uint32_t)bits;
            float f;
            memcpy(&f, &narrow, 4);
            if (int_bits == 32) {
                const int32_t r = (int32_t)f;
                if ((float)r != f) *fpsr |= UINT64_C(16);
                return (uint32_t)r;
            }
            const int64_t r = (int64_t)f;
            if ((float)r != f) *fpsr |= UINT64_C(16);
            return (uint64_t)r;
        }
        double d;
        memcpy(&d, &bits, 8);
        if (int_bits == 32) {
            const int32_t r = (int32_t)d;
            if ((double)r != d) *fpsr |= UINT64_C(16);
            return (uint32_t)r;
        }
        const int64_t r = (int64_t)d;
        if ((double)r != d) *fpsr |= UINT64_C(16);
        return (uint64_t)r;
    }
#endif
"""


def fcvtzs(h):
    """FCVTZS's native path, after recomp_fp_to_int's locals (recomp_fp_to_fixed has the same lines)"""
    if h.count(FP_TO_INT) != 1:
        raise SystemExit("suyu_rt_fixes: recomp_fp_to_int found %d times (expected once): update nx2/suyu_rt_fixes.py"
                         % h.count(FP_TO_INT))
    at = h.find(FP_TO_INT_AT, h.find(FP_TO_INT))
    if at < 0 or at > h.find("\n}\n", h.find(FP_TO_INT)):
        raise SystemExit("suyu_rt_fixes: recomp_fp_to_int's locals moved: update nx2/suyu_rt_fixes.py")
    at += len(FP_TO_INT_AT)
    start = h.find(FP_TO_INT)
    return h[:start] + "#include <float.h>\n" + h[start:at] + FCVTZS + h[at:]


def patch(h):
    for old, new in SUBST:
        n = h.count(old)
        if n != 1:
            raise SystemExit("suyu_rt_fixes: %r found %d times in a64_suyu.h (expected once): update nx2/suyu_rt_fixes.py"
                             % (old[:70], n))
        h = h.replace(old, new)
    return fcvtzs(h)


_INCLUDE = '#include "a64.h"\n'
_FMA_HEAD = "RECOMP_FPE_COLD uint64_t recomp_fpe_fma%d(GuestContext* c,uint64_t _a,uint64_t _b,uint64_t _z){uint64_t _v=0;\n"
_FMA_EARLY = ("#if RECOMP_FMA_SOFT\n{ int _ix; if (recomp_fma_soft_on() && !(c->fpcr & RECOMP_FPX_FPCR_MASK) && "
              "recomp_fma_soft%d(_a, _b, _z, &_v, &_ix)) { RECOMP_FMA_SOFT_PROBE(%d); if (_ix) c->fpsr |= 16; return _v; } }\n"
              "#endif\n")


def fma_soft_text():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fma_soft.h"), encoding="utf-8") as f:
        return f.read()


def patch_c(c):
    """the generated runtime C with nx2/fma_soft.h and its early exits in the two FMA exact bodies"""
    subst = [(_INCLUDE, _INCLUDE + fma_soft_text())]
    for w, probe in ((32, 0), (64, 1)):
        subst.append((_FMA_HEAD % w, _FMA_HEAD % w + _FMA_EARLY % (w, probe)))
    for old, new in subst:
        n = c.count(old)
        if n != 1:
            raise SystemExit("suyu_rt_fixes: %r found %d times in a64_suyu.c (expected once): update nx2/suyu_rt_fixes.py"
                             % (old[:70], n))
        c = c.replace(old, new)
    return c


_installed = False


def install():
    global _installed
    if _installed:
        return
    if a64_rtgen is None:
        raise SystemExit("suyu_rt_fixes: a64-kit's lift folder is not on the path (a64_rtgen)")
    _installed = True
    base_runtime_h = a64_rtgen.runtime_h
    a64_rtgen.runtime_h = lambda: patch(base_runtime_h())
    base_runtime_c = a64_rtgen.runtime_c
    a64_rtgen.runtime_c = lambda: patch_c(base_runtime_c())
